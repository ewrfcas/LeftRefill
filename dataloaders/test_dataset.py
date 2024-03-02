import os
from glob import glob

import cv2
import numpy as np
from torch.utils.data import Dataset


class TestInpaintingDataset(Dataset):
    def __init__(self, root_path, img_size=256, token_map=None, mask_path=None, **kwargs):
        self.img_size = img_size
        self.root_path = root_path
        self.token_map = token_map
        if os.path.isdir(root_path):
            self.pairs = glob(root_path + '/*')
            self.pairs.sort(key=lambda x: x.split('/')[-1])
        else:
            with open(root_path, 'r') as f:
                self.pairs = f.readlines()
                self.pairs = [p.strip() for p in self.pairs]
        if mask_path is not None:
            self.mask_list = glob(mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])
        else:
            self.mask_list = None

        self.repeat_sp_token = kwargs.get('repeat_sp_token', 0)  # if repeat_sp_token>0, we only repeat the same token X times for the prompt
        self.sp_token = kwargs.get('sp_token', None)
        self.deep_prompt = kwargs.get('deep_prompt', False)
        self.cross_attn_layers = 16

    def __len__(self):
        return len(self.pairs)

    def resize_and_crop(self, image):
        image = cv2.resize(image, [self.img_size, self.img_size], interpolation=cv2.INTER_AREA)
        return image

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
            if self.token_map is None:
                text = '[REFERENCE_INPAINTING]'  # test for old models
            else:
                left_token = self.token_map['left_token']
                right_token = self.token_map['right_token']
                task_token = self.token_map['task_token']
                real_token = self.token_map['real_token']
                text = f"Both {left_token} and {right_token} images show the {real_token} with different {task_token}."

        return text

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        source_filename = pair + '/source.jpg'
        target_filename = pair + '/target.jpg'
        if not os.path.exists(source_filename):
            source_filename = source_filename.replace('.jpg', '.png')
        if not os.path.exists(target_filename):
            target_filename = target_filename.replace('.jpg', '.png')

        if self.mask_list is None:
            mask_file = pair + '/mask.png'
        else:
            mask_file = self.mask_list[idx % len(self.mask_list)]

        prompt = self.get_prompt()

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # images from megadepth have very different height/width, we
        # should resize or crop them into 512x512
        source = self.resize_and_crop(source)
        target = self.resize_and_crop(target)

        image = np.concatenate([source, target], axis=1)

        # Normalize source images to [-1, 1].
        image = (image.astype(np.float32) / 127.5) - 1.0

        # load mask
        mask = cv2.imread(mask_file)[:, :, 0]
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.float32) / 255.0
        mask = mask[:, :, None]
        mask0 = np.zeros_like(mask)
        mask = np.concatenate([mask0, mask], axis=1)
        masked_image = image * (mask < 0.5)

        return dict(image=image, txt=prompt, masked_image=masked_image, mask=mask)
