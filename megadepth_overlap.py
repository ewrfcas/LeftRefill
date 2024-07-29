import numpy as np
from tqdm import tqdm
from glob import glob
import os
import json
import pickle

prompt = '[REFERENCE_INPAINTING]'
overlap = [0.4, 0.7]
root_path = 'data/megadepth' # this should be the megadepth image folder
train_info_path = 'data/megadepth/index/scene_info_0.1_0.7'
test_info_path = 'data/megadepth/index/scene_info_val_1500'
out_path = f'data/megadepth_{overlap[0]}_{overlap[1]}'

# preset image pair
npz_fs = glob(f'{train_info_path}/*.npz')
train_set = []
img_name_to_id = {}
img_id_to_name = {}
img_idx = 0
for f in tqdm(npz_fs):
    scene_info = np.load(f, allow_pickle=True)
    pair_infos = scene_info['pair_infos']
    image_paths = scene_info['image_paths']
    for idx in range(len(pair_infos)):
        (idx0, idx1), score, _ = pair_infos[idx]
        if score < overlap[0] or score > overlap[1]:
            continue
        img_name0 = image_paths[idx0]
        img_name1 = image_paths[idx1]

        if img_name0 not in img_name_to_id:
            img_name_to_id[img_name0] = img_idx
            img_id_to_name[img_idx] = os.path.join(root_path, img_name0)
            img_idx += 1
        if img_name1 not in img_name_to_id:
            img_name_to_id[img_name1] = img_idx
            img_id_to_name[img_idx] = os.path.join(root_path, img_name1)
            img_idx += 1
        train_set.append({
            'source': img_name_to_id[img_name0],
            'target': img_name_to_id[img_name1],
            'prompt': prompt,
        })

print('Train Unique images:', len(img_name_to_id))
print('Train Pair num:', len(train_set))

# preset test image pair
npz_fs = glob(f'{test_info_path}/*.npz')
test_set = []
for f in tqdm(npz_fs):
    scene_info = np.load(f, allow_pickle=True)
    pair_infos = scene_info['pair_infos']
    image_paths = scene_info['image_paths']
    for idx in range(len(pair_infos)):
        (idx0, idx1), _, _ = pair_infos[idx]
        img_name0 = image_paths[idx0]
        img_name1 = image_paths[idx1]

        if img_name0 not in img_name_to_id:
            img_name_to_id[img_name0] = img_idx
            img_id_to_name[img_idx] = os.path.join(root_path, img_name0)
            img_idx += 1
        if img_name1 not in img_name_to_id:
            img_name_to_id[img_name1] = img_idx
            img_id_to_name[img_idx] = os.path.join(root_path, img_name1)
            img_idx += 1
        test_set.append({
            'source': img_name_to_id[img_name0],
            'target': img_name_to_id[img_name1],
            'prompt': prompt,
        })

print('Test Pair num:', len(test_set))

os.makedirs(out_path, exist_ok=True)

with open(f'{out_path}/image_dict.pkl', 'wb') as w:
    pickle.dump(img_id_to_name, w)

with open(f'{out_path}/train_pairs.pkl', 'wb') as w:
    pickle.dump(train_set, w)

with open(f'{out_path}/test_pairs.pkl', 'wb') as w:
    pickle.dump(test_set, w)

import random
with open(f'{out_path}/test_pairs_100.pkl', 'wb') as w:
    random.shuffle(test_set)
    pickle.dump(test_set[:100], w)