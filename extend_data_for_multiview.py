# This is an example showing how to extend the training/testing examples to multiview version.

from tqdm import tqdm
import numpy as np
import pickle
from glob import glob
import os

new_tp = []
megadepth_root = r"data/megadepth"
out_path = "data/megadepth_0.4_0.7"
overlap_threshold = 0.2
mode = "train"

# Get Scene Info of megadepth
if mode == "train":
    info_path = 'data/megadepth/index/scene_info_0.1_0.7'
else:
    info_path = 'data/megadepth/index/scene_info_val_1500'

scene_info_path = info_path
npz_fs = glob(f'{scene_info_path}/*.npy')
scene_info_dict = {}
for npz in npz_fs:
    key =  npz.split('/')[-1].split('.')[0]
    scene_info_dict[key] = npz

# Load the pairs generated from overlap.py
with open(out_path+f"/{mode}_pairs.pkl", "rb") as f:
    tp = pickle.load(f)
with open(out_path+"/image_dict.pkl", "rb") as f:
    ip = pickle.load(f)

# reversed_ip maps the image path to the data ids used in LeftRefill
# while other id-ended varients represents the id in megadepth dataset, every scene has a special id.
reversed_ip = {}
for k,v in ip.items():
    reversed_ip[v] = k

for idx_pair, t_pair in tqdm(enumerate(tp), total=len(tp)):
    # reversed source and target
    target_path = ip[t_pair['target']]
    source_path = ip[t_pair['source']]

    extended_source = [t_pair['source']]

    # We need to find more view from the scene where the source image is from
    scene_id = target_path.split('/')[-3]
    scene_info = np.load(scene_info_dict[scene_id], allow_pickle=True).item()

    source_flag = False
    target_flag = False
    for idx, image_path in enumerate(scene_info['image_paths']):
        if image_path is not None:
            if image_path.split('/')[-1] == source_path.split('/')[-1]:
                source_id_dataset = idx 
                source_flag = True
            if image_path.split('/')[-1] == target_path.split('/')[-1]:
                target_id_dataset = idx
                target_flag = True
        if source_flag and target_flag:
            break

    # We pick out view-images with enough overlap of the target image
    involved_idx = []
    for idx, (pair, score, _) in enumerate(scene_info['pair_infos']):
        if (pair[0] == target_id_dataset and pair[1] != source_id_dataset and score >= overlap_threshold):
            involved_idx.append(idx)

    if len(involved_idx) < 3:
        print(f"{idx_pair} lack enough images")
        continue

    index_list = np.random.choice(len(involved_idx), 3, replace=False)

    for index in index_list:
        selected_idx = involved_idx[index]
        new_source_id = scene_info['pairs'][selected_idx][1]
        if new_source_id != source_id_dataset:
            new_source_path = scene_info['image_paths'][new_source_id]
            new_source_path = os.path.join(megadepth_root, new_source_path)
            if new_source_path in reversed_ip.keys():
                extended_source.append(reversed_ip[new_source_path])
            else:
                # Here the LeftRefill ID is used.
                idx = len(reversed_ip)
                ip[idx] = new_source_path
                reversed_ip[new_source_path] = idx
                extended_source.append(idx)

    new_tp.append({'idx': idx_pair, 'target': [t_pair['target']], 'source': extended_source})

with open(out_path + '/image_dict.pkl', 'wb') as w:
    pickle.dump(ip, w)

with open(out_path + f'/extended_{mode}_pairs.pkl', 'wb') as w:
    pickle.dump(new_tp, w)