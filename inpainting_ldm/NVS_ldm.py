import collections
import itertools

import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from dataloaders.novel_view_synthesis_dataset import NVS_DTUDataset, WarpNVS_DTUDataset
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader
from torchmetrics.functional import peak_signal_noise_ratio

from dataloaders.obj_nvs_dataset import NVS_OBJDataset
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentInpaintDiffusion
from ldm.modules.diffusionmodules.openaimodel import Downsample, Upsample
from ldm.modules.diffusionmodules.openaimodel import UNetModel, timestep_embedding
from ldm.modules.diffusionmodules.util import conv_nd, GroupNorm32


class NVSUnetModel(UNetModel):
    def __init__(self, *args, **kwargs):
        self.use_sep = kwargs.pop('use_sep', False)
        super().__init__(*args, **kwargs)
        if self.use_sep:
            channels = [9, 320, 640, 1280, 2560, 1920, 960]
            self.sep_token = nn.ParameterDict()
            for ch in channels:
                self.sep_token[str(ch)] = nn.Parameter(torch.randn(ch), requires_grad=True)
        else:
            self.sep_token = None

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        c_input = kwargs.get('c_input', None)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)

        for i, module in enumerate(self.input_blocks):
            if self.use_sep and type(module[-1]) not in [Downsample, Upsample]:
                B, C, H, W = h.shape
                sep = self.sep_token[str(C)].reshape(1, C, 1, 1).repeat(B, 1, H, 1)
                h = torch.cat([h[..., :W // 2], sep, h[..., W // 2:]], dim=-1)

            h = module(h, emb, context)
            if i == 0 and c_input is not None:
                if c_input.shape == h.shape:
                    h += c_input
                else:
                    h[:, :, :, h.shape[-1] // 2:] += c_input

            if self.use_sep and type(module[-1]) not in [Downsample, Upsample]:
                h = torch.cat([h[..., :W // 2], h[..., -W // 2:]], dim=-1)

            hs.append(h)

        if self.use_sep:
            B, C, H, W = h.shape
            sep = self.sep_token[str(C)].reshape(1, C, 1, 1).repeat(B, 1, H, 1)
            h = torch.cat([h[..., :W // 2], sep, h[..., W // 2:]], dim=-1)

        h = self.middle_block(h, emb, context)

        if self.use_sep:
            h = torch.cat([h[..., :W // 2], h[..., -W // 2:]], dim=-1)

        for module in self.output_blocks:

            h = torch.cat([h, hs.pop()], dim=1)

            if self.use_sep and type(module[-1]) not in [Downsample, Upsample]:
                B, C, H, W = h.shape
                sep = self.sep_token[str(C)].reshape(1, C, 1, 1).repeat(B, 1, H, 1)
                h = torch.cat([h[..., :W // 2], sep, h[..., W // 2:]], dim=-1)

            h = module(h, emb, context)

            if self.use_sep and type(module[-1]) not in [Downsample, Upsample]:
                h = torch.cat([h[..., :W // 2], h[..., -W // 2:]], dim=-1)

        h = h.type(x.dtype)

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class NVSLDM(LatentInpaintDiffusion):

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
        self.mask_steps = 0
        self.warmup_mask_steps = self.data_cfg.get('warmup_mask_steps', 0)
        self.complete_mask_rate = self.data_cfg.get('complete_mask_rate', 0)
        self.save_prompt_only = kwargs.pop('save_prompt_only', False)
        self.refinement_config = kwargs.pop('refinement_config', {'use_input_refinement': False, 'only_masked_refine': False})
        if self.refinement_config['use_input_refinement']:  # refine the input with sub-pixel masked images and masks
            model_channels = kwargs['unet_config']['params']['model_channels']
            self.refinement_model = nn.Sequential(conv_nd(2, 4, 32, 3, padding=1),
                                                  nn.SiLU(),
                                                  conv_nd(2, 32, 64, 3, padding=1, stride=2),
                                                  GroupNorm32(16, 64),
                                                  nn.SiLU(),
                                                  conv_nd(2, 64, 64, 3, padding=1),
                                                  GroupNorm32(16, 64),
                                                  nn.SiLU(),
                                                  conv_nd(2, 64, 128, 3, padding=1, stride=2),
                                                  GroupNorm32(32, 128),
                                                  nn.SiLU(),
                                                  conv_nd(2, 128, 128, 3, padding=1),
                                                  GroupNorm32(32, 128),
                                                  nn.SiLU(),
                                                  conv_nd(2, 128, 256, 3, padding=1, stride=2),
                                                  GroupNorm32(32, 256),
                                                  nn.SiLU(),
                                                  conv_nd(2, 256, model_channels, 3, padding=1),
                                                  GroupNorm32(32, model_channels),
                                                  nn.SiLU())
            self.refinement_alpha = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        else:
            self.refinement_model = None
            self.refinement_alpha = None

        # LORA
        self.lora_cfg = kwargs.pop('lora', {'do_lora': False})

    def set_lora(self):
        if self.lora_cfg['do_lora']:
            from .lora import inject_trainable_lora, inject_trainable_lora_extended
            if self.lora_cfg['lora_type'] == 'default':
                unet_lora_params, lora_names = inject_trainable_lora(self.model.diffusion_model, r=self.lora_cfg['lora_rank'],
                                                                     scale=self.lora_cfg['lora_scale'], verbose=True)
            elif self.lora_cfg['lora_type'] == 'extended':
                unet_lora_params, lora_names = inject_trainable_lora_extended(self.model.diffusion_model, r=self.lora_cfg['lora_rank'],
                                                                              scale=self.lora_cfg['lora_scale'])
            else:
                raise NotImplementedError('Not implemented LoRA')

            self.unet_lora_params = unet_lora_params
        else:
            self.unet_lora_params = None

    def get_input(self, batch, k, cond_key=None, bs=None, return_first_stage_outputs=False, force_c_encode=True):
        x, c = super().get_input(batch, k, cond_key, bs, return_first_stage_outputs, force_c_encode)
        if self.refinement_config['use_input_refinement']:
            if self.refinement_config.get('only_masked_refine', False):
                masked_key = 'clean_masked_image'
                mask_key = 'clean_mask'
            else:
                masked_key = 'masked_image'
                mask_key = 'subpixel_mask'

            clean_masked_image = einops.rearrange(batch[masked_key], 'b h w c -> b c h w').to(self.device)
            clean_masked_image = clean_masked_image.to(memory_format=torch.contiguous_format).float()
            clean_mask = einops.rearrange(batch[mask_key], 'b h w c -> b c h w').to(self.device)
            clean_mask = clean_mask.to(memory_format=torch.contiguous_format).float()

            inp = torch.cat([clean_masked_image, clean_mask], dim=1)
            if bs is not None:
                inp = inp[:bs]
            c_input = self.refinement_model(inp) * self.refinement_alpha
            c['c_input'] = c_input
        else:
            c['c_input'] = None

        return x, c

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
        if 'c_input' in c and c['c_input'] is not None:
            c_input = c['c_input'][:N]
        else:
            c_input = None
        N = min(z.shape[0], N)
        log["masked_image"] = batch['masked_image'].permute(0, 3, 1, 2)
        log["origin_image"] = batch['image'].permute(0, 3, 1, 2)

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_concat = c_concat  # torch.zeros_like(c_cat)
            if c_input is not None:
                uc_input = c_input.clone()
            else:
                uc_input = None
            uc_full = {"c_concat": [uc_concat], "c_crossattn": [uc_cross]}
            c_full = {"c_concat": [c_concat], "c_crossattn": [c_crossattn]}
            if c_input is not None:
                c_full['c_input'] = c_input
                uc_full['c_input'] = uc_input
            samples_cfg, _ = self.sample_log(cond=c_full,
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["pred"] = x_samples_cfg
        else:
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_concat], "c_crossattn": [c_crossattn]},
                                             batch_size=N, ddim=use_ddim, ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log["pred"] = x_samples_cfg

        return log

    @torch.no_grad()
    def log_multi_cond_images(self, batch, N=4, ddim_steps=50, ddim_eta=0.0, unconditional_guidance_scale=9.0, **kwargs):
        use_ddim = ddim_steps is not None
        assert type(batch) == list

        log = dict()
        log["masked_image"] = batch[0]['masked_image'].permute(0, 3, 1, 2)
        log["origin_image"] = batch[0]['image'].permute(0, 3, 1, 2)

        c_fulls = []
        uc_fulls = []
        for batch_ in batch:
            z, c = self.get_input(batch_, self.first_stage_key, bs=N)
            c_concat, c_crossattn = c["c_concat"][0][:N], c["c_crossattn"][0][:N]
            if 'c_input' in c and c['c_input'] is not None:
                c_input = c['c_input'][:N]
            else:
                c_input = None

            uc_cross = self.get_unconditional_conditioning(N)
            uc_concat = c_concat  # torch.zeros_like(c_cat)
            if c_input is not None:
                uc_input = c_input.clone()
            else:
                uc_input = None
            uc_full = {"c_concat": [uc_concat], "c_crossattn": [uc_cross]}
            c_full = {"c_concat": [c_concat], "c_crossattn": [c_crossattn]}
            if c_input is not None:
                c_full['c_input'] = c_input
                uc_full['c_input'] = uc_input
            c_fulls.append(c_full)
            uc_fulls.append(uc_full)

        samples_cfg, _ = self.sample_log(cond=c_fulls,
                                         batch_size=N, ddim=use_ddim,
                                         ddim_steps=ddim_steps, eta=ddim_eta,
                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                         unconditional_conditioning=uc_fulls)
        x_samples_cfg = self.decode_first_stage(samples_cfg)
        log["pred"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        if type(cond) == list:
            b, c, h, w = cond[0]["c_concat"][0].shape
        else:
            b, c, h, w = cond["c_concat"][0].shape
        # shape = (self.channels, h // 8, w // 8)
        shape = (self.channels, h, w)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def on_train_batch_end(self, *args, **kwargs):
        super().on_train_batch_end(*args, **kwargs)
        if self.warmup_mask_steps > 0 and self.mask_steps <= self.warmup_mask_steps:
            self.trainer.train_dataloader.dataset.datasets.complete_mask_rate = \
                self.complete_mask_rate + (self.mask_steps / self.warmup_mask_steps * (1.0 - self.complete_mask_rate))
            if self.trainer.train_dataloader.dataset.datasets.complete_mask_rate > 1.0:
                self.trainer.train_dataloader.dataset.datasets.complete_mask_rate = 1.0
            self.mask_steps += 1

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        if self.warmup_mask_steps > 0 and self.trainer.train_dataloader is not None:
            items['mask_rate'] = self.trainer.train_dataloader.dataset.datasets.complete_mask_rate
        return items

    def configure_optimizers(self):
        lr = self.optim_cfg['learning_rate']
        wd = self.optim_cfg['weight_decay']
        all_trainable = self.optim_cfg.get('all_trainable', False)
        if all_trainable:
            lr_sd = self.optim_cfg.get('lr_sd', 1e-5)
            params_sd = list(self.model.diffusion_model.parameters())
            total_params = [{'params': params_sd, 'lr': lr_sd}]
        else:
            total_params = []
        params = list(self.cond_stage_model.special_embeddings.parameters())
        if getattr(self.cond_stage_model, 'rel_pos_model', None) is not None:
            params.extend(self.cond_stage_model.rel_pos_model.parameters())
        if self.refinement_model is not None and self.refinement_alpha is not None:
            params.extend(self.refinement_model.parameters())
            params.append(self.refinement_alpha)
        total_params.append({'params': params, 'lr': lr})

        if self.unet_lora_params is not None:
            total_params.append({'params': itertools.chain(*self.unet_lora_params), 'lr': self.optim_cfg['lr_lora'],
                                 'weight_decay': self.optim_cfg['wd_lora']})

        opt = torch.optim.AdamW(total_params, lr=lr, weight_decay=wd)
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
        if self.data_cfg['obj_dataset']:
            train_dataset = NVS_OBJDataset(datapath=self.cfg['datapath'], listfile=self.cfg['train_list'], mode='train',
                                           img_size=self.img_size, **self.data_cfg)
        elif self.data_cfg['warping_based']:
            train_dataset = WarpNVS_DTUDataset(datapath=self.cfg['datapath'], listfile=self.cfg['train_list'], mode='train',
                                               img_size=self.img_size, deep_prompt=self.cond_cfg['deep_prompt'], **self.data_cfg)
        else:
            train_dataset = NVS_DTUDataset(datapath=self.cfg['datapath'], listfile=self.cfg['train_list'], mode='train',
                                           img_size=self.img_size, deep_prompt=self.cond_cfg['deep_prompt'],
                                           view_prompt=self.cond_cfg.get('view_prompt', False), **self.data_cfg)
        return DataLoader(train_dataset, num_workers=8, batch_size=self.cfg['batch_size'], shuffle=True)

    def val_dataloader(self):
        if self.data_cfg['obj_dataset']:
            val_dataset = NVS_OBJDataset(datapath=self.cfg['datapath'], listfile=self.cfg['val_list'], mode='val',
                                         img_size=self.img_size, **self.data_cfg)
        elif self.data_cfg['warping_based']:
            val_dataset = WarpNVS_DTUDataset(datapath=self.cfg['datapath'], listfile=self.cfg['val_list'], mode='val',
                                             img_size=self.img_size, deep_prompt=self.cond_cfg['deep_prompt'], **self.data_cfg)
        else:
            val_dataset = NVS_DTUDataset(datapath=self.cfg['datapath'], listfile=self.cfg['val_list'], mode='val',
                                         img_size=self.img_size, deep_prompt=self.cond_cfg['deep_prompt'],
                                         view_prompt=self.cond_cfg.get('view_prompt', False), **self.data_cfg)
        return DataLoader(val_dataset, num_workers=4, batch_size=4, shuffle=False, drop_last=True)

    def validation_step(self, batch, batch_idx):
        N = batch['image'].shape[0]
        log = self.log_images(batch, N=N, unconditional_guidance_scale=self.data_cfg['cfg'])
        pred = (log['pred'].to(torch.float32) + 1) / 2.0
        origin = (log['origin_image'].to(torch.float32) + 1) / 2.0
        mask = batch['mask'].permute(0, 3, 1, 2)
        # we should consider whole image test here
        # pred = pred * mask + origin * (1 - mask)
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

    # def on_train_epoch_start(self):
    #     if self.world_size == 1 and self.cfg['cross_view_inpainting']:  # we have to set epoch manually if using ddp
    #         self.trainer.train_dataloader.sampler.set_epoch(self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        if self.save_prompt_only:
            # pop the backbone here using custom logic
            delete_keys = []
            for key in checkpoint['state_dict']:
                if key.startswith('cond_stage_model') and not key.startswith('cond_stage_model.model.'):
                    continue
                elif key.startswith('refinement_'):
                    continue
                elif 'lora_down' in key or 'lora_up' in key:
                    continue
                elif 'sep_token' in key:
                    continue
                else:
                    delete_keys.append(key)

            for key in delete_keys:
                del checkpoint['state_dict'][key]
