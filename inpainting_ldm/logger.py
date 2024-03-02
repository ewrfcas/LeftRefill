import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from matplotlib import cm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only


def mapping_color(img, vmin, vmax, cmap="rainbow"):
    np_img = img
    if vmin is not None and vmax is not None:
        np_img = plt.Normalize(vmin=vmin, vmax=vmax)(np_img)
    mapped_img = getattr(cm, cmap)(np_img)
    results = mapped_img[:, :, 0:3]
    return results


class InpaintingLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, save_dir='image_log'):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs is not None else {}
        self.log_first_step = log_first_step
        self.save_dir = save_dir
        self.special_tokens = None
        self.special_token_embeddings_old = None

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, self.save_dir, split)
        grids = []
        keys = ['masked_image', 'origin_image', 'control', 'warp_image', 'pred', 'reference']
        if 'att_scores' in images:
            keys.append('att_scores')
        for k in keys:
            if k in images:
                if k == 'att_scores':
                    for a_ in images[k]:
                        a_ = F.interpolate(a_.unsqueeze(1), size=[512, 512], mode='nearest')
                        grid = torchvision.utils.make_grid(a_, nrow=1)  # [h,w]
                        grid = grid.transpose(0, 1).transpose(1, 2)[:,:,0]
                        grid = grid.numpy()
                        grid = (mapping_color(grid, 0, 1, cmap="viridis") * 255).astype(np.uint8)
                        grids.append(grid)
                else:
                    grid = torchvision.utils.make_grid(images[k], nrow=1)
                    if self.rescale:
                        grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid * 255).astype(np.uint8)
                    grids.append(grid)
        grids = np.concatenate(grids, axis=1)
        filename = "gs-{:06}_e-{:06}_b-{:06}.jpg".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.split(path)[0], exist_ok=True)
        Image.fromarray(grids).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            # logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                if k == 'att_scores':
                    N = min(images[k][0].shape[0], self.max_images)
                    images[k] = [im[:N].detach().cpu() for im in images[k]]
                else:
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            if pl_module.local_rank == 0:
                self.log_local(pl_module.logger.save_dir, split, images,
                               pl_module.global_step, pl_module.current_epoch, batch_idx)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="train")

        if batch_idx % trainer.log_every_n_steps == 0 and hasattr(pl_module.cond_stage_model, 'special_tokens'):
            if self.special_token_embeddings_old is None:
                self.special_tokens = pl_module.cond_stage_model.special_tokens
                self.special_token_embeddings_old = pl_module.cond_stage_model.special_embeddings.weight.clone()
            else:
                for j, special_token in enumerate(self.special_tokens):
                    d = self.special_token_embeddings_old[j] - pl_module.cond_stage_model.special_embeddings.weight[j]
                    d = torch.sqrt(torch.sum(d**2)).item()
                    pl_module.log(f'special_tokens/{special_token}', d, on_step=True)
                self.special_token_embeddings_old = pl_module.cond_stage_model.special_embeddings.weight.clone()

