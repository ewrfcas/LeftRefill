import os
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class InpaintingDataset(Dataset):
    def __init__(self, image_list, mask_path=None, mode='train', img_size=512, token_map=None,
                 test_limit=200, flip=True, outpainting=False, outpainting_min_rate=0.25,
                 outpainting_max_rate=0.75, root_path=None, **kwargs):
        print(f'Loading image list...')
        if image_list.endswith('.txt'):
            with open(image_list, 'r') as f:
                self.image_list = f.readlines()
                if root_path is None:
                    self.image_list = [im.strip() for im in self.image_list]
                else:
                    self.image_list = [os.path.join(root_path, im.strip()) for im in self.image_list]
        else:
            self.image_list = glob(image_list + '/*')
            self.image_list = sorted(self.image_list, key=lambda x: x.split('/')[-1])

        self.mask_path = mask_path
        self.mode = mode
        self.img_size = img_size
        self.token_map = token_map
        self.repeat_sp_token = kwargs.get('repeat_sp_token', 0)  # if repeat_sp_token>0, we only repeat the same token X times for the prompt
        self.sp_token = kwargs.get('sp_token', None)
        self.deep_prompt = kwargs.get('deep_prompt', False)
        self.cross_attn_layers = 16
        self.flip = flip
        self.outpainting = outpainting
        self.outpainting_min_rate = outpainting_min_rate
        self.outpainting_max_rate = outpainting_max_rate

        if self.mode == 'train':  # we mask images with both irregular and segmentation masks
            with open(mask_path[0]) as f:
                self.irregular_mask_list = f.readlines()
                self.irregular_mask_list = [i.strip() for i in self.irregular_mask_list]
            self.irregular_mask_list = sorted(self.irregular_mask_list, key=lambda x: x.split('/')[-1])
            with open(mask_path[1]) as f:
                self.segment_mask_list = f.readlines()
                self.segment_mask_list = [i.strip() for i in self.segment_mask_list]
            self.segment_mask_list = sorted(self.segment_mask_list, key=lambda x: x.split('/')[-1])
        else:
            print(f'For testing, using fixed masking...')
            if mask_path.endswith('.txt'):
                with open(mask_path) as f:
                    self.mask_list = f.readlines()
                    self.mask_list = [im.strip() for im in self.mask_list]
            else:
                self.mask_list = glob(mask_path + '/*')
                self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

        if mode == 'val':
            split_n = len(self.image_list) // test_limit
            self.image_list = self.image_list[::split_n]
            split_n_mask = len(self.mask_list) // test_limit
            self.mask_list = self.mask_list[::split_n_mask]

    def __len__(self):
        return len(self.image_list)

    def resize_and_crop(self, image):
        if self.mode == 'train':
            rng = random.random()
            if rng < 0.5:  # directly resizing
                image = cv2.resize(image, [self.img_size, self.img_size], interpolation=cv2.INTER_AREA)
            else:  # resize and random crop
                h, w, _ = image.shape
                if h < w:
                    long_side = max(self.img_size, int(w * (self.img_size / h)))
                    image = cv2.resize(image, (long_side, self.img_size), interpolation=cv2.INTER_AREA)
                else:
                    long_side = max(self.img_size, int(h * (self.img_size / w)))
                    image = cv2.resize(image, (self.img_size, long_side), interpolation=cv2.INTER_AREA)
                w_start = random.randint(0, image.shape[1] - self.img_size)
                h_start = random.randint(0, image.shape[0] - self.img_size)
                image = image[h_start:h_start + self.img_size, w_start:w_start + self.img_size, :]
        else:
            image = cv2.resize(image, [self.img_size, self.img_size], interpolation=cv2.INTER_AREA)
        return image

    def load_mask(self):
        imgh, imgw = self.img_size, self.img_size
        rdv = random.random()
        if rdv < 0.4:
            mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
            mask = cv2.imread(self.irregular_mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
        elif rdv < 0.8:
            mask_index = random.randint(0, len(self.segment_mask_list) - 1)
            mask = cv2.imread(self.segment_mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
        else:
            mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
            mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
            mask1 = cv2.imread(self.segment_mask_list[mask_index1], cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.resize(mask1, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.imread(self.irregular_mask_list[mask_index2], cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.resize(mask2, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

        # if mask.shape[0] != imgh or mask.shape[1] != imgw:
        #     mask = cv2.resize(mask, (imgw, imgh), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
        return mask

    def load_outpainting_mask(self):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        mask_idx = np.random.random() * (self.outpainting_max_rate - self.outpainting_min_rate) + self.outpainting_min_rate
        mask_idx = int(mask_idx * self.img_size)
        mask[:, mask_idx:] = 255
        return mask

    def get_prompt(self):
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
        else:  # only used for cross-view inpainting
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

    def __getitem__(self, idx):
        path = self.image_list[idx]
        if self.mode == 'test':
            img = Image.open(path).convert("RGB")
            img = img.resize((self.img_size, self.img_size), resample=Image.BICUBIC)
            img = np.array(img.convert("RGB"))
        else:
            img = cv2.imread(path)
            # Do not forget that OpenCV read images in BGR order.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.resize_and_crop(img)

        # load mask
        if self.mode == 'train':
            if self.outpainting:
                mask = self.load_outpainting_mask()
            else:
                mask = self.load_mask()
        else:
            mask = cv2.imread(self.mask_list[idx % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
        mask = mask.astype(np.float32) / 255.0

        if self.flip and self.mode == 'train':
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
            if random.random() < 0.5:
                mask = mask[:, ::-1].copy()

        # Normalize images to [-1, 1].
        image = (img.astype(np.float32) / 127.5) - 1.0
        mask = mask[:, :, None]
        masked_image = image * (mask < 0.5)

        # prompt
        prompt = self.get_prompt()

        return dict(image=image, txt=prompt, masked_image=masked_image, mask=mask)
