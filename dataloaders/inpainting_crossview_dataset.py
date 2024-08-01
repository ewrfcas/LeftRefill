import collections
from einops import repeat
import math
import os
import pickle
import random
from glob import glob

import cv2
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

mask_rate = []


class InpaintingCrossViewDataset(Dataset):
    def __init__(self, image_path, pair_path, mask_path, mode='train', img_size=256,
                 only_mask_image=False, no_padding=True, token_map=None, view_mask_rate=0.9,
                 test_limit=150, flip=False, constant_place=False, **kwargs):
        if mode == 'train':
            print(f'Loading image dict...')
            self.image_dict = pickle.load(open(image_path, 'rb'))
            print(f'Loading {mode} pairs...')
            self.pairs = pickle.load(open(pair_path, 'rb'))
        else:
            if os.path.isdir(image_path):
                self.pairs = glob(image_path + '/*')
                self.pairs.sort(key=lambda x: x.split('/')[-1])
                split_n = len(self.pairs) // test_limit
                self.pairs = self.pairs[::split_n]
            else:
                image_path_normal = image_path[0]
                image_path_sp = image_path[1]
                image_files = []
                with open(image_path_sp, 'r') as f:
                    image_files.extend(f.readlines())
                with open(image_path_normal, 'r') as f:
                    image_files.extend(f.readlines()[:test_limit - len(image_files)])
                print(f'{len(image_files)}, testing image pairs are loaded...')
                self.pairs = [im.strip() for im in image_files]

        self.mask_path = mask_path
        self.mode = mode
        self.img_size = img_size
        self.only_mask_image = only_mask_image
        self.no_padding = no_padding
        self.token_map = token_map
        self.view_mask_rate = view_mask_rate  # the rate of masking the whole another view
        self.repeat_sp_token = kwargs.get('repeat_sp_token', 0)  # if repeat_sp_token>0, we only repeat the same token X times for the prompt
        self.sp_token = kwargs.get('sp_token', None)
        self.match_mask = kwargs.get('match_mask', False)  # whether to use matching based mask
        self.match_mask_rate = kwargs.get('match_mask_rate', 0.0)  # ratio of matching based mask used in random mask
        self.match_path = kwargs.get('match_path', None)
        self.deep_prompt = kwargs.get('deep_prompt', False)
        self.cross_attn_layers = 16
        self.flip = flip
        self.constant_place = constant_place

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
            self.mask_list = glob(mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

    def __len__(self):
        return len(self.pairs)

    def resize_and_crop(self, image):
        crop_info = None
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
                resized_h, resized_w, _ = image.shape
                w_start = random.randint(0, image.shape[1] - self.img_size)
                h_start = random.randint(0, image.shape[0] - self.img_size)
                image = image[h_start:h_start + self.img_size, w_start:w_start + self.img_size, :]
                crop_info = {'w_start': w_start, 'h_start': h_start, 'w': resized_w, 'h': resized_h}
        else:
            image = cv2.resize(image, [self.img_size, self.img_size], interpolation=cv2.INTER_AREA)
        return image, crop_info

    def get_match_based_mask(self, idx, target_pos='left', target_crop_info=None, source_crop_info=None):
        # first, load matching result pkl
        pkl_name = os.path.join(self.match_path, str(idx).zfill(8) + '.pkl')
        if not os.path.exists(pkl_name):
            return None
        res = pickle.load(open(pkl_name, 'rb'))

        # hyper-parameters:
        min_width = 35
        max_width = 70
        min_area_rate = 0.2  # the area of the small rectangle divide by the whole area of the image
        max_area_rate = 0.5
        num_vertex = random.randint(15, 30)
        min_num = 10
        match_size = 832
        match_mask_size = 256
        threshold_prob = 0.8

        scores_max = res['scores'].max()
        # Decide to mask left or right. mkpts0=target, mkpts1=source
        if self.constant_place:
            rdv = 1.0  # always mask right
        else:
            rdv = random.random()
        if rdv < 0.5:
            mask_left = True
            mkpt = 'mkpts0' if target_pos == 'left' else 'mkpts1'
            crop_info = target_crop_info if target_pos == 'left' else source_crop_info
        else:
            mask_left = False
            mkpt = 'mkpts1' if target_pos == 'left' else 'mkpts0'
            crop_info = source_crop_info if target_pos == 'left' else target_crop_info

        good_pts = res[mkpt][np.where(res['scores'] > scores_max * threshold_prob)]
        if crop_info is None:
            good_pts = good_pts / match_size * match_mask_size  # normalize
        else:  # crop the pts if we have cropped the original image
            good_pts = good_pts / match_size  # 0~1
            good_pts[:, 0] *= crop_info['w']  # 0~w
            good_pts[:, 1] *= crop_info['h']  # 0~h
            good_pts[:, 0] -= crop_info['w_start']  # -w'~w-w'
            good_pts[:, 1] -= crop_info['h_start']  # -h'~h-h'
            # rescale (512-->256)
            ms = min(crop_info['w'], crop_info['h']) / match_mask_size
            good_pts /= ms
            good_pts = good_pts[(good_pts[:, 0] >= 0) * (good_pts[:, 1] >= 0) * (good_pts[:, 0] < match_mask_size) * (good_pts[:, 1] < match_mask_size)]

        if len(good_pts) < min_num:
            return None

        # get large rectangle
        x_set = good_pts[:, 0]
        y_set = good_pts[:, 1]
        x_min, x_max, y_min, y_max = x_set.min(), x_set.max(), y_set.min(), y_set.max()
        good_w = x_max - x_min
        good_h = y_max - y_min
        good_area = good_w * good_h

        if good_area == 0:
            return None

        # set area
        rate = match_mask_size * match_mask_size * (min_area_rate + (max_area_rate - min_area_rate) * random.random()) / good_area
        rate_1D = math.sqrt(rate)
        if rate < 1:
            a = good_w * rate_1D
            b = good_h * rate_1D
            x_start = x_min + np.random.randint(0, good_w - a + 1)
            y_start = y_min + np.random.randint(0, good_h - b + 1)
            temp = good_pts[np.where(good_pts[:, 0] > x_start)]
            temp = temp[np.where(temp[:, 0] < x_start + a)]
            temp = temp[np.where(temp[:, 1] > y_start)]
            temp = temp[np.where(temp[:, 1] < y_start + b)]
            picked_pts = np.random.permutation(temp)
        else:
            picked_pts = np.random.permutation(good_pts)

        if picked_pts.shape[0] < min_num:
            return None
        picked_pts = picked_pts[:num_vertex]
        width = np.random.randint(min_width, max_width)
        mask = Image.new('L', (256, 256), 0)  # these mask must be set in 256
        draw = ImageDraw.Draw(mask)
        draw.line(np.append(picked_pts, picked_pts[:1], axis=0), fill=1, width=width)
        for v in picked_pts:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)
        mask = np.asarray(mask, np.float32)
        if self.img_size != match_mask_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask_rate.append(mask.mean())

        mask0 = np.zeros_like(mask)
        if mask_left:
            mask = np.concatenate([mask, mask0], axis=1)
        else:
            mask = np.concatenate([mask0, mask], axis=1)

        return mask

    def get_inpainting_mask(self):
        rdv = random.random()
        if rdv < 0.4:
            mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
            mask = cv2.imread(self.irregular_mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        elif rdv < 0.8:
            mask_index = random.randint(0, len(self.segment_mask_list) - 1)
            mask = cv2.imread(self.segment_mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
            mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
            mask1 = cv2.imread(self.segment_mask_list[mask_index1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask1 = cv2.resize(mask1, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.imread(self.irregular_mask_list[mask_index2], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask2 = cv2.resize(mask2, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

        mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
        mask = mask.astype(np.float32) / 255.0

        mask0 = np.zeros_like(mask)
        rdv = random.random()
        if rdv < 0.5:  # masking only one side should be better
            mask = np.concatenate([mask, mask0], axis=1)
        else:
            mask = np.concatenate([mask0, mask], axis=1)

        return mask

    def load_mask(self, idx, gt_pos, target_crop_info, source_crop_info):
        if self.match_mask and random.random() < self.match_mask_rate:
            mask = self.get_match_based_mask(idx, gt_pos, target_crop_info, source_crop_info)
            if mask is None:
                mask = self.get_inpainting_mask()
        else:  # random masking as the regular inpainting task
            mask = self.get_inpainting_mask()
        return mask

    def padding(self, img):
        double_size = self.img_size * 2
        if len(img.shape) == 3:
            pad = np.zeros(((double_size - self.img_size) // 2, double_size, 3), dtype=img.dtype)
        else:
            pad = np.zeros(((double_size - self.img_size) // 2, double_size), dtype=img.dtype)
        img_pad = np.concatenate([pad, img, pad], axis=0)

        return img_pad

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

    def __getitem__(self, idx):

        pair = self.pairs[idx]

        if self.mode == 'train':
            source_filename = self.image_dict[pair['source']]
            target_filename = self.image_dict[pair['target']]
        else:
            source_filename = pair + '/source.jpg'
            target_filename = pair + '/target.jpg'
            if not os.path.exists(source_filename):
                source_filename = source_filename.replace('.jpg', '.png')
            if not os.path.exists(target_filename):
                target_filename = target_filename.replace('.jpg', '.png')

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # images from megadepth have very different height/width, we
        # should resize or crop them into 512x512
        source, source_crop_info = self.resize_and_crop(source)
        target, target_crop_info = self.resize_and_crop(target)

        # random concat img and label
        rdv = random.random()
        if self.mode == 'train' and rdv < 0.5 and not self.constant_place:
            gt_pos = 'left'
            image = np.concatenate([target, source], axis=1)
        else:  # for testing, target is always in right
            gt_pos = 'right'
            image = np.concatenate([source, target], axis=1)

        # load mask
        if self.mode == 'train':
            if self.only_mask_image:
                mask = np.zeros((self.img_size, self.img_size * 2), dtype=np.float32)
                if gt_pos == 'left':
                    mask[:, :self.img_size] = 1
                else:
                    mask[:, self.img_size:] = 1
            else:
                rdv = random.random()
                if rdv < 1.0 - self.view_mask_rate:  # for training, X% random masking
                    mask = self.load_mask(idx, gt_pos, target_crop_info, source_crop_info)
                else:  # mask the whole left or right side (view mask)
                    rdv = random.random()
                    mask = np.zeros((self.img_size, self.img_size * 2), dtype=np.float32)
                    if rdv < 0.5:
                        mask[:, :self.img_size] = 1
                    else:
                        mask[:, self.img_size:] = 1
        else:  # for testing, sp images use specific mask, normal images use random mask
            if os.path.exists(pair + '/mask.png'):
                mask = cv2.imread(pair + '/mask.png', cv2.IMREAD_GRAYSCALE)
            else:
                mask = cv2.imread(self.mask_list[idx % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            mask = mask.astype(np.float32) / 255.0
            mask0 = np.zeros_like(mask)
            mask = np.concatenate([mask0, mask], axis=1)  # for testing, mask is always in right

        # flip
        if self.mode == 'train' and self.flip:
            rdv = random.random()
            if rdv < 0.5:
                image[:, :self.img_size] = image[:, :self.img_size][:, ::-1]
                mask[:, :self.img_size] = mask[:, :self.img_size][:, ::-1]
            rdv = random.random()
            if rdv < 0.5:
                image[:, self.img_size:] = image[:, self.img_size:][:, ::-1]
                mask[:, self.img_size:] = mask[:, self.img_size:][:, ::-1]

        # padding
        if not self.no_padding:
            image = self.padding(image)
            mask = self.padding(mask)

        ## Normalize images to [-1, 1].
        image = (image.astype(np.float32) / 127.5) - 1.0
        mask = mask[:, :, None]
        masked_image = image * (mask < 0.5)

        # prompt
        prompt = self.get_prompt()

        return dict(image=image, txt=prompt, masked_image=masked_image, mask=mask)


class InpaintingMultiViewDataset(Dataset):
    def __init__(self, image_path, pair_path, mask_path, mode='train', img_size=256,
                 only_mask_image=False, no_padding=True, token_map=None, view_mask_rate=0.9,
                 test_limit=150, flip=False, constant_place=False, max_ref_view=3, **kwargs):
        if mode == 'train':
            print(f'Loading image dict...')
            self.image_dict = pickle.load(open(image_path, 'rb'))
            print(f'Loading {mode} pairs...')
            self.pairs = pickle.load(open(pair_path, 'rb'))
        else:
            if os.path.isdir(image_path):
                self.pairs = glob(image_path + '/*')
                self.pairs.sort(key=lambda x: x.split('/')[-1])
                split_n = len(self.pairs) // test_limit
                self.pairs = self.pairs[::split_n]
            else:
                image_path_normal = image_path[0]
                image_path_sp = image_path[1]
                image_files = []
                with open(image_path_sp, 'r') as f:
                    image_files.extend(f.readlines())
                with open(image_path_normal, 'r') as f:
                    image_files.extend(f.readlines()[:test_limit - len(image_files)])
                print(f'{len(image_files)}, testing image pairs are loaded...')
                self.pairs = [im.strip() for im in image_files]

        self.mask_path = mask_path
        self.mode = mode
        self.img_size = img_size
        self.only_mask_image = only_mask_image
        self.no_padding = no_padding
        self.token_map = token_map
        self.view_mask_rate = view_mask_rate  # the rate of masking the whole another view
        self.repeat_sp_token = kwargs.get('repeat_sp_token', 0)  # if repeat_sp_token>0, we only repeat the same token X times for the prompt
        self.sp_token = kwargs.get('sp_token', None)
        self.match_mask = kwargs.get('match_mask', False)  # whether to use matching based mask
        self.match_mask_rate = kwargs.get('match_mask_rate', 0.0)  # ratio of matching based mask used in random mask
        self.match_path = kwargs.get('match_path', None)
        self.deep_prompt = kwargs.get('deep_prompt', False)
        self.cross_attn_layers = 16
        self.max_ref_view = max_ref_view
        self.view_num = kwargs.get('view_num', 4)
        self.view_token_len = kwargs.get('view_token_len', 30)
        self.flip = flip
        self.constant_place = constant_place
        self.source_shuffle = kwargs.get('source_shuffle', False)
        self.concat_target = kwargs.get('concat_target', False)

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
            self.mask_list = glob(mask_path + '/*')
            self.mask_list = sorted(self.mask_list, key=lambda x: x.split('/')[-1])

    def __len__(self):
        return len(self.pairs)

    def resize_and_crop(self, image):
        crop_info = None
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
                resized_h, resized_w, _ = image.shape
                w_start = random.randint(0, image.shape[1] - self.img_size)
                h_start = random.randint(0, image.shape[0] - self.img_size)
                image = image[h_start:h_start + self.img_size, w_start:w_start + self.img_size, :]
                crop_info = {'w_start': w_start, 'h_start': h_start, 'w': resized_w, 'h': resized_h}
        else:
            image = cv2.resize(image, [self.img_size, self.img_size], interpolation=cv2.INTER_AREA)
        return image, crop_info

    def get_match_based_mask(self, idx, target_pos='left', target_crop_info=None, source_crop_info=None):
        # first, load matching result pkl
        pkl_name = os.path.join(self.match_path, str(idx).zfill(8) + '.pkl')
        if not os.path.exists(pkl_name):
            return None
        res = pickle.load(open(pkl_name, 'rb'))

        # hyper-parameters:
        min_width = 35
        max_width = 70
        min_area_rate = 0.2  # the area of the small rectangle divide by the whole area of the image
        max_area_rate = 0.5
        num_vertex = random.randint(15, 30)
        min_num = 10
        match_size = 832
        match_mask_size = 256
        threshold_prob = 0.8

        scores_max = res['scores'].max()
        # Decide to mask left or right. mkpts0=target, mkpts1=source
        if self.constant_place:
            rdv = 0.0  # always mask left
        else:
            rdv = random.random()
        if rdv < 0.5:
            mask_left = True
            mkpt = 'mkpts0' if target_pos == 'left' else 'mkpts1'
            crop_info = target_crop_info if target_pos == 'left' else source_crop_info
        else:
            mask_left = False
            mkpt = 'mkpts1' if target_pos == 'left' else 'mkpts0'
            crop_info = source_crop_info if target_pos == 'left' else target_crop_info

        good_pts = res[mkpt][np.where(res['scores'] > scores_max * threshold_prob)]
        if crop_info is None:
            good_pts = good_pts / match_size * match_mask_size  # normalize
        else:  # crop the pts if we have cropped the original image
            good_pts = good_pts / match_size  # 0~1
            good_pts[:, 0] *= crop_info['w']  # 0~w
            good_pts[:, 1] *= crop_info['h']  # 0~h
            good_pts[:, 0] -= crop_info['w_start']  # -w'~w-w'
            good_pts[:, 1] -= crop_info['h_start']  # -h'~h-h'
            # rescale (512-->256)
            ms = min(crop_info['w'], crop_info['h']) / match_mask_size
            good_pts /= ms
            good_pts = good_pts[(good_pts[:, 0] >= 0) * (good_pts[:, 1] >= 0) * (good_pts[:, 0] < match_mask_size) * (good_pts[:, 1] < match_mask_size)]

        if len(good_pts) < min_num:
            return None

        # get large rectangle
        x_set = good_pts[:, 0]
        y_set = good_pts[:, 1]
        x_min, x_max, y_min, y_max = x_set.min(), x_set.max(), y_set.min(), y_set.max()
        good_w = x_max - x_min
        good_h = y_max - y_min
        good_area = good_w * good_h

        if good_area == 0:
            return None

        # set area
        rate = match_mask_size * match_mask_size * (min_area_rate + (max_area_rate - min_area_rate) * random.random()) / good_area
        rate_1D = math.sqrt(rate)
        if rate < 1:
            a = good_w * rate_1D
            b = good_h * rate_1D
            x_start = x_min + np.random.randint(0, good_w - a + 1)
            y_start = y_min + np.random.randint(0, good_h - b + 1)
            temp = good_pts[np.where(good_pts[:, 0] > x_start)]
            temp = temp[np.where(temp[:, 0] < x_start + a)]
            temp = temp[np.where(temp[:, 1] > y_start)]
            temp = temp[np.where(temp[:, 1] < y_start + b)]
            picked_pts = np.random.permutation(temp)
        else:
            picked_pts = np.random.permutation(good_pts)

        if picked_pts.shape[0] < min_num:
            return None
        picked_pts = picked_pts[:num_vertex]
        width = np.random.randint(min_width, max_width)
        mask = Image.new('L', (256, 256), 0)  # these mask must be set in 256
        draw = ImageDraw.Draw(mask)
        draw.line(np.append(picked_pts, picked_pts[:1], axis=0), fill=1, width=width)
        for v in picked_pts:
            draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=1)
        mask = np.asarray(mask, np.float32)
        if self.img_size != match_mask_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        mask_rate.append(mask.mean())

        # mask0 = np.zeros_like(mask)
        # if mask_left:
        #     mask = np.concatenate([mask, mask0], axis=1)
        # else:
        #     mask = np.concatenate([mask0, mask], axis=1)

        return mask

    def get_inpainting_mask(self):
        rdv = random.random()
        if rdv < 0.4:
            mask_index = random.randint(0, len(self.irregular_mask_list) - 1)
            mask = cv2.imread(self.irregular_mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        elif rdv < 0.8:
            mask_index = random.randint(0, len(self.segment_mask_list) - 1)
            mask = cv2.imread(self.segment_mask_list[mask_index], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        else:
            mask_index1 = random.randint(0, len(self.segment_mask_list) - 1)
            mask_index2 = random.randint(0, len(self.irregular_mask_list) - 1)
            mask1 = cv2.imread(self.segment_mask_list[mask_index1], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask1 = cv2.resize(mask1, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask2 = cv2.imread(self.irregular_mask_list[mask_index2], cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask2 = cv2.resize(mask2, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = np.clip(mask1 + mask2, 0, 255).astype(np.uint8)

        mask = (mask > 127).astype(np.uint8) * 255  # threshold due to interpolation
        mask = mask.astype(np.float32) / 255.0

        return mask

    def load_mask(self, idx, gt_pos, target_crop_info, source_crop_info):
        if self.match_mask and random.random() < self.match_mask_rate:
            mask = self.get_match_based_mask(idx, gt_pos, target_crop_info, source_crop_info)
            if mask is None:
                mask = self.get_inpainting_mask()
        else:  # random masking as the regular inpainting task
            mask = self.get_inpainting_mask()
        return mask

    def padding(self, img):
        double_size = self.img_size * 2
        if len(img.shape) == 3:
            pad = np.zeros(((double_size - self.img_size) // 2, double_size, 3), dtype=img.dtype)
        else:
            pad = np.zeros(((double_size - self.img_size) // 2, double_size), dtype=img.dtype)
        img_pad = np.concatenate([pad, img, pad], axis=0)

        return img_pad

    def get_prompt(self):  # only used for cross-view inpainting
        if self.repeat_sp_token > 0 and self.sp_token is not None:
            text = ""
            for i in range(self.repeat_sp_token):
                text = text + self.sp_token.replace('>', f'{i}> ')
            text = text.strip()
            if self.deep_prompt:
                raise NotImplementedError()
                text_list = []
                for layer_i in range(self.cross_attn_layers):
                    text_list.append(text.replace('>', f'-layer{layer_i}>'))
                text = text_list

            text_list = []
            if self.concat_target:
                # for concat image, we have [view_num - 1] views
                for j in range(self.view_num - 1):
                    temp_text = text
                    for l in range(self.view_token_len):
                        temp_text = temp_text + f"<view_direct-{j}-{l}>"
                    text_list.append(temp_text)
            else:
                for j in range(self.view_num):
                    temp_text = text
                    for l in range(self.view_token_len):
                        temp_text = temp_text + f"<view_direct-{j}-{l}>"
                    text_list.append(temp_text)
            text = text_list

        else:
            raise NotImplementedError()
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
        pair = self.pairs[idx]

        if self.mode == 'train':
            target_filename = self.image_dict[pair['target'][0]]
            source_filenames = []
            for idx in pair['source']:
                source_filenames.append(self.image_dict[idx])
                
        else:
            source_filename_0 = pair + '/source.jpg'
            source_filename_1 = pair + '/source_1.jpg'
            source_filename_2 = pair + '/source_2.jpg'
            source_filename_3 = pair + '/source_3.jpg'
            target_filename = pair + '/target.jpg'
            if not os.path.exists(source_filename_0):
                source_filename_0 = source_filename_0.replace('.jpg', '.png')
            if not os.path.exists(source_filename_1):
                source_filename_1 = source_filename_1.replace('.jpg', '.png')
            if not os.path.exists(source_filename_2):
                source_filename_2 = source_filename_2.replace('.jpg', '.png')
            if not os.path.exists(source_filename_3):
                source_filename_3 = source_filename_3.replace('.jpg', '.png')
            if not os.path.exists(target_filename):
                target_filename = target_filename.replace('.jpg', '.png')
                
            source_filenames = [source_filename_0, source_filename_1, source_filename_2, source_filename_3]

        target = cv2.imread(target_filename)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target, target_crop_info = self.resize_and_crop(target)

        sources = []
        
        # shuffle and choose the first [view_num] ref view, always include the first view (which has highest overlap)
        if self.source_shuffle:
            random_idx = np.random.choice(self.view_num - 1, self.view_num - 1, replace=False)
        else:
            random_idx = range(self.view_num - 1)
        for i in random_idx:
            source = cv2.imread(source_filenames[i])
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source, _ = self.resize_and_crop(source) # varients "_" is a varient we do not need it anymore in MultiView
            sources.append(source)

        # Do not forget that OpenCV read images in BGR order.

        # in multiview, target image is in View 0, and source at View 1,2,3
        image = np.array([target, *sources])

        if self.mode == "train":
            gt_pos = 'right'
        else:
            gt_pos = 'left'

        # load mask, in multiveiw, we only mask the source image
        if self.mode == 'train':
            if self.only_mask_image:
                raise NotImplementedError()
                # mask = np.zeros((self.img_size, self.img_size * 2), dtype=np.float32)
                # if gt_pos == 'left':
                #     mask[:, :self.img_size] = 1
                # else:
                #     mask[:, self.img_size:] = 1
            else:
                rdv = random.random()
                if rdv < 1.0 - self.view_mask_rate:  # for training, X% random masking
                    mask = self.load_mask(pair['idx'], gt_pos, _, source_crop_info)
                else:  # mask the whole left or right side (view mask)
                    mask = np.ones([self.img_size, self.img_size], dtype=np.float32)
        else:  # for testing, sp images use specific mask, normal images use random mask
            if os.path.exists(pair + '/mask.png'):
                mask = cv2.imread(pair + '/mask.png', cv2.IMREAD_GRAYSCALE)
            else:
                mask = cv2.imread(self.mask_list[idx % len(self.mask_list)], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.uint8) * 255
            mask = mask.astype(np.float32) / 255.0

        # padding
        if not self.no_padding:
            raise NotImplementedError()

        ## Normalize images to [-1, 1].
        image = (image.astype(np.float32) / 127.5) - 1.0
        mask = mask[:, :, None]
        masked_image = image.copy()
        masked_image[0] = masked_image[0] *  (mask < 0.5)
        final_mask = repeat(mask, "h w c -> repeat h w c", repeat=len(image))
        final_mask[1:] = 0
       
        if self.concat_target:
            # expand to (view_num-1, H, 2*W, C)
            concat_image = np.zeros((self.view_num - 1, image.shape[1], image.shape[2] * 2, image.shape[3]))
            concat_masked_image = np.zeros((self.view_num - 1, masked_image.shape[1], masked_image.shape[2] * 2, masked_image.shape[3]))
            concat_mask = np.zeros((self.view_num - 1, final_mask.shape[1], final_mask.shape[2] * 2, final_mask.shape[3]))
            for i in range(len(sources)):
                # pay attention to whther the ":" is in front of or behind the img_size varient 
                concat_image[i, :, self.img_size:, :] = image[0]
                concat_image[i, :, 0:self.img_size, :] = image[i+1]
                concat_masked_image[i, :, self.img_size:, :] = masked_image[0]
                concat_masked_image[i, :, 0:self.img_size, :] = masked_image[i+1]
                concat_mask[i, :, self.img_size:, :] = final_mask[0]
                concat_mask[i, :, 0:self.img_size:, :] = final_mask[i+1]
                
            image = concat_image
            masked_image = concat_masked_image
            final_mask = concat_mask

        # prompt
        prompt = self.get_prompt()
        
        return dict(image=image, txt=prompt, masked_image=masked_image, mask=final_mask, idx=int(pair.split('/')[-1]))


# megadepth is extremely unbalanced data, so we need to sample
# subset for each epoch during the training
class BalancedRandomSampler(Sampler):
    def __init__(self, image_dict, pairs, n_sample_per_scene=100, rank=0, num_replicas=1):
        self.n_sample_per_scene = n_sample_per_scene
        self.rank = rank
        self.epoch = 0
        self.num_replicas = num_replicas
        if rank >= num_replicas or rank < 0:
            raise ValueError("Invalid rank {}, rank should be in the interval"
                             " [0, {}]".format(rank, num_replicas - 1))

        self.scene_idx = collections.defaultdict(list)
        print('Setting scene data for megadepth...')
        for i, p in enumerate(tqdm(pairs, desc='Split megadepth/scannet scene data...')):
            scene = image_dict[p['source']].split('/')[-3]
            self.scene_idx[scene].append(i)

        for scene in self.scene_idx:
            if n_sample_per_scene > len(self.scene_idx[scene]):
                raise ValueError(
                    "n_sample_per_scene should be less than the min scene sample "
                    "but got {}>{}".format(n_sample_per_scene, len(self.scene_idx[scene]))
                )
        self.n_scene = len(self.scene_idx)
        total_size = self.n_scene * self.n_sample_per_scene
        if total_size % self.num_replicas != 0:
            self.num_samples = math.ceil((total_size - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(total_size / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas
        print(f'Training dataset rank{rank + 1}/{self.num_replicas}, '
              f'scenes:{self.n_scene / self.num_replicas}, samples:{self.num_samples}')

    def __iter__(self):
        print(f'\nUsing seed epoch{self.epoch}')
        new_list = []
        # deterministically shuffle based on epoch and seed
        random.seed(self.epoch)
        for scene in self.scene_idx:
            random.shuffle(self.scene_idx[scene])
            new_list.extend(self.scene_idx[scene][:self.n_sample_per_scene])

        # global shuffle
        random.shuffle(new_list)

        # split for each process
        indices = new_list[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        print(f'\nRank{self.rank} set epoch{epoch} success!')
        self.epoch = epoch

