import argparse
import os
from glob import glob

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio
from tqdm import tqdm

from dataloaders.test_dataset import TestInpaintingDataset
from inpainting_ldm.model import create_model, load_state_dict

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('--model_path', default=None, type=str, help='config file')
    args.add_argument('--exp_name', default=None, type=str, help='experiment name')
    args.add_argument('--test_path', default='data/masked_pairs', help='Test path')
    args.add_argument('--mask_path', default='', help='Test mask path')
    args.add_argument('--ngpu', default=1, type=int, help='gpu number')
    args.add_argument('--fp16', action='store_true', help='use FP16')
    args.add_argument('--manual_pairs_x4', action='store_true',
                      help='test for manually masked pairs (each pair is tested with x4 times)')
    args.add_argument('--test_size', type=int, default=512, help='test image size')
    args.add_argument('--metric_size', type=int, default=512, help='metric image size')
    args.add_argument('--metric_output', default='metric_outputs', help='Output metric path')
    args.add_argument('--cfg', default=1.0, type=float, help='cfg float coef')
    args.add_argument('--eta', default=0.0, type=float, help='eta float coef')
    args.add_argument('--batch_size', default=4, type=int, help='batchsize')
    args.add_argument('--output_path', default='test_outputs', type=str, help='output_path')
    args.add_argument('--save_single', action='store_true')

    args = args.parse_args()
    if args.save_single:
        args.batch_size = 1

    model_config = os.path.join(args.model_path, 'model_config.yaml')

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()

    ckpt_list = glob(os.path.join(args.model_path, 'ckpts/epoch=*.ckpt'))
    if len(ckpt_list) > 1:
        resume_path = sorted(ckpt_list, key=lambda x: int(x.split('/')[-1].split('.ckpt')[0].split('=')[-1]))[-1]
        # raise NotImplementedError("Too many ckpts", ckpt_list)
    else:
        resume_path = ckpt_list[0]
    #     resume_path = os.path.join(args.model_path, 'ckpts/last.ckpt')
    print('Load ckpt', resume_path)

    reload_weights = load_state_dict(resume_path, location='cpu')
    torch_init_model(model, reload_weights, key='none')
    if getattr(model, 'save_prompt_only', False):
        pretrained_weights = load_state_dict('pretrained_models/512-inpainting-ema.ckpt', location='cpu')
        torch_init_model(model, pretrained_weights, key='none')
    # model.cfg = config
    model.world_size = args.ngpu
    model.loss_fn_alex = lpips.LPIPS(net='alex')

    model.to("cuda")
    model.eval()

    ex_param = sum(p.numel() for p in model.cond_stage_model.special_embeddings.parameters())
    print("Ex params: %.2fM" % (ex_param / 1e6), "%.2fKB" % (ex_param / 1024))

    batch_size = 1 if args.manual_pairs_x4 else args.batch_size
    test_dataset = TestInpaintingDataset(args.test_path, img_size=args.test_size,
                                         deep_prompt=model.cond_cfg.get('deep_prompt', False), **model.data_cfg)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    output_path = f'{args.output_path}/{args.model_path.split("/")[-1]}' + f'_cfg{args.cfg}'
    output_path += f'_{args.test_path.split("/")[-1]}'
    output_path += f'_eta{args.eta}'

    if 'last.ckpt' in resume_path:
        output_path += f'_last'

    output_path += '_' + str(args.test_size)

    batch_idx = 0
    psnr = []
    lpips = []
    ssim = []
    with torch.no_grad(), torch.autocast("cuda"):
        # for each sample, run 4 times
        for batch in tqdm(test_loader):
            for k in batch:
                if type(batch[k]) == torch.Tensor:
                    batch[k] = batch[k].cuda()

                if args.manual_pairs_x4:
                    if type(batch[k]) == torch.Tensor:
                        batch[k] = batch[k].repeat(4, 1, 1, 1).cuda()
                    else:
                        batch[k] = batch[k] * args.batch_size

            N = batch['image'].shape[0]

            x = model.log_images(batch, N=N, unconditional_guidance_scale=args.cfg, ddim_eta=args.eta)
            grids = []

            # if h!=w, we need to crop the right side for the evaluation
            h, w = x['pred'].shape[2], x['pred'].shape[3]
            mask = batch['mask'].permute(0, 3, 1, 2)
            x['pred'] = x['pred'] * mask + x['origin_image'] * (1 - mask)
            if h != w:
                x['pred'] = x['pred'][:, :, :, w // 2:]
                x['origin_image'] = x['origin_image'][:, :, :, w // 2:]

            if args.metric_size < args.test_size:
                x['pred'] = F.interpolate(x['pred'], size=(args.metric_size, args.metric_size), mode='area')
                x['origin_image'] = F.interpolate(x['origin_image'], size=(args.metric_size, args.metric_size), mode='area')
                scale_factor = args.metric_size / args.test_size
                x['masked_image'] = F.interpolate(x['masked_image'], scale_factor=scale_factor, mode='area')

            for i in range(N):
                psnr_ = peak_signal_noise_ratio((x['pred'][i:i + 1] + 1) / 2, (x['origin_image'][i:i + 1] + 1) / 2, data_range=1.0)
                lpips_ = model.loss_fn_alex(x['pred'][i:i + 1], x['origin_image'][i:i + 1])  # lpips needs -1~1
                pred_np = TF.rgb_to_grayscale((x['pred'][i] + 1) / 2)[0].cpu().numpy()
                origin_np = TF.rgb_to_grayscale((x['origin_image'][i] + 1) / 2)[0].cpu().numpy()
                ssim_ = structural_similarity(pred_np, origin_np)
                psnr.append(psnr_.item())
                ssim.append(ssim_)
                lpips.append(lpips_.item())

            if not args.save_single:
                for k in x.keys():
                    grid = torchvision.utils.make_grid(x[k], nrow=1)
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.cpu().numpy()
                    grid = np.clip((grid * 255), 0, 255).astype(np.uint8)
                    grids.append(grid)
                grids = np.concatenate(grids, axis=1)
                filename = "{:06}.png".format(batch_idx)
                path = os.path.join(output_path, filename)

                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grids).save(path)
            else:
                filename = "{:06}.png".format(batch_idx)
                # save for fid metric
                for i in range(N):
                    pred = (torch.clamp(x['pred'][i], -1, 1) + 1.0) / 2.0 * 255
                    pred = pred.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    path = os.path.join(output_path, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    cv2.imwrite(path, pred[:, :, ::-1])

            # break

            batch_idx += 1

    print('EXP:', output_path.split('/')[-1])
    print('PSNR:', np.mean(psnr))
    print('SSIM:', np.mean(ssim))
    print('LPIPS:', np.mean(lpips))

    os.makedirs(args.metric_output, exist_ok=True)
    with open(args.metric_output + f'/{output_path.split("/")[-1]}.txt', 'w') as w:
        w.write('PSNR:' + str(np.mean(psnr)) + '\n')
        w.write('SSIM:' + str(np.mean(ssim)) + '\n')
        w.write('LPIPS:' + str(np.mean(lpips)) + '\n')
