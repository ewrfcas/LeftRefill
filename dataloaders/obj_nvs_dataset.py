import math
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class NVS_OBJDataset(Dataset):
    def __init__(self, datapath, listfile, mode='train', img_size=512, nviews=12, token_map=None,
                 test_limit=150, dilate_size=[8, 20], pts_size=[15, 30], mask_enlarge=[0.0, 0.0], mask_file_path=None, mask_type='fix',
                 width_range=[60, 120], complete_mask_rate=0.0, use_ref_mask=False, **kwargs):
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews

        self.img_size = img_size
        self.token_map = token_map
        self.repeat_sp_token = kwargs.get('repeat_sp_token', 0)  # if repeat_sp_token>0, we only repeat the same token X times for the prompt
        self.sp_token = kwargs.get('sp_token', None)
        self.deep_prompt = kwargs.get('deep_prompt', False)
        self.cross_attn_layers = 16
        self.test_limit = test_limit
        self.dilate_size = dilate_size
        self.pts_size = pts_size
        self.mask_type = mask_type
        self.obj_mask_path = kwargs.get('obj_mask_path', None)
        self.complete_mask_rate = complete_mask_rate
        self.width_range = width_range
        self.mask_enlarge = mask_enlarge
        self.use_ref_mask = use_ref_mask

        self.metas = []
        with open(self.listfile, 'r') as f:
            fs = f.readlines()
            self.metas = [os.path.join(self.datapath, f.strip()) for f in fs]

        if self.mode == 'val' and self.test_limit < len(self.metas):
            self.metas = self.metas[::len(self.metas) // self.test_limit]

        self.mask_file_path = mask_file_path
        if self.mask_file_path is not None:
            print('Using masks from', self.mask_file_path)

    def __len__(self):
        return len(self.metas)

    def get_prompt(self):  # only used for cross-view inpainting
        if self.repeat_sp_token > 0 and self.sp_token is not None:
            text = ""
            for i in range(self.repeat_sp_token):
                text = text + self.sp_token.replace('>', f'{i}> ')
            text = text.strip()
            if self.deep_prompt:
                text_list = []
                for layer_i in range(self.cross_attn_layers):
                    text_list.append(text.replace('>', f'-layer{layer_i}>'))
                text = text_list
        else:
            left_token = self.token_map['left_token']
            right_token = self.token_map['right_token']
            task_token = self.token_map['task_token']
            real_token = self.token_map['real_token']
            templets = [f"Both {left_token} and {right_token} images show the {real_token} with different {task_token}.",
                        f"The {real_token} remains the same in both the {left_token} and {right_token} images, but the {task_token} are different.",
                        f"The {left_token} and {right_token} images depict identical {real_token}, but from different {task_token}.",
                        f"The painting depicts the {real_token}, but from two different {task_token}; one from the {left_token} and one from the {right_token}.",
                        f"Both figures capture the same {real_token}, but the {left_token} one and the {right_token} one are taken from different {task_token}.",
                        f"The two drawings show the {real_token}, but one is from the {left_token} side and the other is from the {right_token} side, and they are from different {task_token}",
                        f"Both pictures depict the same {real_token}, but the {left_token} image and the {right_token} image are captured with different {task_token}."]

            if self.mode == 'train':
                text = np.random.choice(templets, size=1)[0]
            else:
                text = templets[0]

        return text

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(T_target[None, :])

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_T

    def __getitem__(self, idx):
        batch = dict()
        filename = self.metas[idx]
        if self.mode == 'train':
            index_target, index_cond = random.sample(range(self.nviews), 2)  # without replacement
        else:
            index_target, index_cond = 0, 2

        target_im = cv2.imread(os.path.join(filename, '%03d.png' % index_target), cv2.IMREAD_UNCHANGED) / 255.
        mask = target_im[:, :, -1].copy()
        mask[mask > 0] = 1
        target_im[target_im[:, :, -1] == 0.] = [1., 1., 1., 1.]
        target_im = (target_im[:, :, :3] * 255.).astype(np.uint8)[:, :, ::-1]  # RGB to BGR

        cond_im = cv2.imread(os.path.join(filename, '%03d.png' % index_cond), cv2.IMREAD_UNCHANGED) / 255.
        cond_im[cond_im[:, :, -1] == 0.] = [1., 1., 1., 1.]
        cond_im = (cond_im[:, :, :3] * 255.).astype(np.uint8)[:, :, ::-1]  # RGB to BGR

        # resize
        target_im = cv2.resize(target_im, (self.img_size, self.img_size))
        cond_im = cv2.resize(cond_im, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        mask[mask > 0] = 1

        # get mask
        if self.mask_file_path is not None and self.mode != 'train' and self.mask_type == 'fix':
            if self.use_ref_mask:
                mask = cv2.imread(os.path.join(self.mask_file_path, filename.split('/')[-1], '%03d.png' % index_cond))[:, :, 0] / 255
            else:
                mask = cv2.imread(os.path.join(self.mask_file_path, filename.split('/')[-1], '%03d.png' % index_target))[:, :, 0] / 255
            mask = mask.astype(np.float32)
        elif self.mode != 'train' and self.mask_type == 'complete':
            mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
        else:  # training mask
            if random.random() < self.complete_mask_rate:
                mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
            else:
                kernel_size = random.randint(self.dilate_size[0], self.dilate_size[1])
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                mask = cv2.dilate(mask, kernel, iterations=1)
                if mask.sum() == 0:
                    mask = np.ones((self.img_size, self.img_size), dtype=np.float32)
                else:
                    mask_bbox = np.where(mask > 0)
                    h_min, h_max = mask_bbox[0].min(), mask_bbox[0].max()
                    w_min, w_max = mask_bbox[1].min(), mask_bbox[1].max()
                    if self.mask_enlarge[1] > self.mask_enlarge[0]:
                        enlarge_rate = random.random() * (self.mask_enlarge[1] - self.mask_enlarge[0]) + self.mask_enlarge[0]
                        max_diff = max(h_max - h_min, w_max - w_min) * enlarge_rate
                        h_min = np.clip(h_min - max_diff, 0, self.img_size - 1)
                        h_max = np.clip(h_max + max_diff, 0, self.img_size - 1)
                        w_min = np.clip(w_min - max_diff, 0, self.img_size - 1)
                        w_max = np.clip(w_max + max_diff, 0, self.img_size - 1)
                    pts_size = random.randint(self.pts_size[0], self.pts_size[1])
                    random_x = np.random.randint(w_min, w_max, size=pts_size)
                    random_y = np.random.randint(h_min, h_max, size=pts_size)
                    random_pts = np.stack([random_x, random_y], axis=1)
                    irr_mask = Image.new('L', (self.img_size, self.img_size), 0)
                    min_width = self.width_range[0] * (self.img_size / 512)
                    max_width = self.width_range[1] * (self.img_size / 512)
                    width = np.random.randint(min_width, max_width)
                    draw = ImageDraw.Draw(irr_mask)
                    pts = np.append(random_pts, random_pts[:1], axis=0)
                    pts = pts.astype(np.float32)  # FIXME: important for pillow drawing with np.float32
                    draw.line(pts, fill=1, width=width)
                    for v in pts:
                        draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)
                    irr_mask = np.asarray(irr_mask, np.float32).copy()
                    mask = np.clip(mask + irr_mask, 0, 1)

        image = np.concatenate([cond_im, target_im], axis=1)
        mask = np.concatenate([np.zeros_like(mask), mask], axis=1)
        # Normalize images to [-1, 1].
        image = (image.astype(np.float32) / 127.5) - 1.0
        mask = mask[:, :, None]
        if self.mode != 'train' and self.use_ref_mask:
            masked_image = np.concatenate([cond_im, np.ones_like(cond_im) * 255], axis=1)
            masked_image = (masked_image.astype(np.float32) / 127.5) - 1.0
            masked_image = masked_image * (mask < 0.5)
        else:
            masked_image = image * (mask < 0.5)  # here we should promise the image is masked correctly

        # load camera
        target_RT = np.load(os.path.join(filename, '%03d.npy' % index_target))
        cond_RT = np.load(os.path.join(filename, '%03d.npy' % index_cond))

        batch["image"] = image
        batch["masked_image"] = masked_image
        batch["mask"] = mask
        batch["rel_pose"] = self.get_T(target_RT, cond_RT)

        # prompt
        prompt = self.get_prompt()
        batch['txt'] = prompt

        return batch
