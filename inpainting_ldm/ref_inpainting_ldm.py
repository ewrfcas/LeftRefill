import collections

import numpy as np
import torch
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio

from dataloaders.inpainting_crossview_dataset import InpaintingCrossViewDataset, BalancedRandomSampler
from dataloaders.inpainting_dataset import InpaintingDataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentInpaintDiffusion


class RefInpaintLDM(LatentInpaintDiffusion):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn_alex = None
        self.cfg = None
        self.optim_cfg = None
        self.data_cfg = kwargs.pop('data_config')
        self.cond_cfg = kwargs.get('cond_stage_config')['params']
        self.world_size = 1
        self.image_text_pair = False
        self.img_size = self.data_cfg.pop('img_size', 256)
        self.save_prompt_only = kwargs.pop('save_prompt_only', False)

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        if self.cond_cfg.get('deep_prompt', False):
            return self.get_learned_conditioning([[""] * N] * self.cond_cfg['cross_attn_layers'])
        else:
            return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, ddim_steps=50, ddim_eta=0.0, unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_concat, c_crossattn = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
        N = min(z.shape[0], N)
        log["masked_image"] = batch['masked_image'].permute(0, 3, 1, 2)
        log["origin_image"] = batch['image'].permute(0, 3, 1, 2)

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_concat = c_concat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_concat], "c_crossattn": [uc_cross]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_concat], "c_crossattn": [c_crossattn]},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["pred"] = x_samples_cfg
        elif unconditional_guidance_scale == 0.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_concat = c_concat  # torch.zeros_like(c_cat)
            samples_cfg, _ = self.sample_log(cond={"c_concat": [uc_concat], "c_crossattn": [uc_cross]},
                                             batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["pred"] = x_samples_cfg
        else:
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_concat], "c_crossattn": [c_crossattn]},
                                             batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["pred"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        # shape = (self.channels, h // 8, w // 8)
        shape = (self.channels, h, w)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.optim_cfg['learning_rate']
        wd = self.optim_cfg['weight_decay']
        params = list(self.cond_stage_model.special_embeddings.parameters())
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        lr_scheduler = self.optim_cfg['lr_scheduler']
        if lr_scheduler == 'cosine':
            eta_min = self.optim_cfg['eta_min']
            sche = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.trainer.max_steps, eta_min=eta_min * lr)
            sche = {'scheduler': sche, 'interval': 'step', 'frequency': 1}
            return [opt], [sche]
        else:
            print('Unknown scheduler', lr_scheduler)
            return opt

    # set custom dataloader
    def train_dataloader(self):
        if self.cfg['cross_view_inpainting']:
            train_dataset = InpaintingCrossViewDataset(image_path=self.cfg['image_path'], pair_path=self.cfg['train_pair'],
                                                       mask_path=self.cfg['train_mask_path'], mode='train', img_size=self.img_size,
                                                       deep_prompt=self.cond_cfg['deep_prompt'], **self.data_cfg)
            sampler = BalancedRandomSampler(train_dataset.image_dict, train_dataset.pairs,
                                            n_sample_per_scene=self.cfg['n_sample_per_scene'],
                                            rank=self.local_rank, num_replicas=self.world_size)
            return DataLoader(train_dataset, num_workers=8, batch_size=self.cfg['batch_size'], sampler=sampler)
        else:
            train_dataset = InpaintingDataset(image_list=self.cfg['image_path'], mask_path=self.cfg['train_mask_path'],
                                              annotation=self.cfg['annotation'], mode='train', img_size=self.img_size, **self.data_cfg)
            return DataLoader(train_dataset, num_workers=8, batch_size=self.cfg['batch_size'], shuffle=True)

    def val_dataloader(self):
        val_dataset = InpaintingCrossViewDataset(image_path=self.cfg['val_image_path'], pair_path=None,
                                                 mask_path=self.cfg['val_mask_path'], mode='val', img_size=self.img_size,
                                                 deep_prompt=self.cond_cfg['deep_prompt'], **self.data_cfg)
        return DataLoader(val_dataset, num_workers=4, batch_size=4, shuffle=False, drop_last=True)

    def validation_step(self, batch, batch_idx):
        N = batch['image'].shape[0]
        log = self.log_images(batch, N=N, unconditional_guidance_scale=self.data_cfg['cfg'])
        pred = (log['pred'].to(torch.float32) + 1) / 2.0
        origin = (log['origin_image'].to(torch.float32) + 1) / 2.0
        mask = batch['mask'].permute(0, 3, 1, 2)
        # we should focus in masked regions during the evaluation
        pred = pred * mask + origin * (1 - mask)
        _, _, _, w = origin.shape
        pred = pred[:, :, :, w // 2:]
        origin = origin[:, :, :, w // 2:]
        # we eval results with PSNR, SSIM, LPIPS
        psnr, ssim, lpips = [], [], []
        for i in range(N):
            psnr_ = peak_signal_noise_ratio(pred[i:i + 1], origin[i:i + 1], data_range=1.0)
            pred_np = TF.rgb_to_grayscale(pred[i])[0].cpu().numpy()
            origin_np = TF.rgb_to_grayscale(origin[i])[0].cpu().numpy()
            ssim_ = structural_similarity(pred_np, origin_np)
            psnr.append(psnr_.item())
            ssim.append(ssim_)
            lpips_ = self.loss_fn_alex(pred[i:i + 1] * 2 - 1., origin[i:i + 1] * 2 - 1.)  # lpips needs -1~1
            lpips.append(lpips_.item())

        self.log('val/psnr', np.mean(psnr), sync_dist=True)
        self.log('val/ssim', np.mean(ssim), sync_dist=True)
        self.log('val/lpips', np.mean(lpips), sync_dist=True)

        return {'psnr': np.mean(psnr), 'ssim': np.mean(ssim), 'lpips': np.mean(lpips)}

    def validation_epoch_end(self, outputs):
        metric_dict = collections.defaultdict(list)
        for out in outputs:
            for k in out:
                metric_dict[k].append(out[k])

        if self.local_rank == 0:
            print('Steps:', self.global_step)
            for k in metric_dict:
                print(k, np.mean(metric_dict[k]))

    def on_train_epoch_start(self):
        if self.world_size == 1 and self.cfg['cross_view_inpainting']:  # we have to set epoch manually if using ddp
            self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)


    def on_save_checkpoint(self, checkpoint):
        if self.save_prompt_only:
            # pop the backbone here using custom logic
            delete_keys = []
            for key in checkpoint['state_dict']:
                if not key.startswith('cond_stage_model') or key.startswith('cond_stage_model.model.'):
                    delete_keys.append(key)

            for key in delete_keys:
                del checkpoint['state_dict'][key]
