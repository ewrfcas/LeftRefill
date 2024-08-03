# LeftRefill
LeftRefill: Filling Right Canvas based on Left Reference through Generalized Text-to-Image Diffusion Model (CVPR2024)

[Project Page](https://ewrfcas.github.io/LeftRefill/),
[Paper](https://arxiv.org/abs/2305.11577)


## Abstract

This paper introduces LeftRefill, an innovative approach to efficiently harness large Text-to-Image (T2I) diffusion models for reference-guided image synthesis. 
As the name implies, LeftRefill horizontally stitches reference and target views together as a whole input. 
The reference image occupies the left side, while the target canvas is positioned on the right.
Then, LeftRefill paints the right-side target canvas based on the left-side reference and specific task instructions. 
Such a task formulation shares some similarities with contextual inpainting, akin to the actions of a human painter.

This novel formulation efficiently learns both structural and textured correspondence between reference and target without other image encoders or adapters.
We inject task and view information through cross-attention modules in T2I models, and further exhibit multi-view reference ability via the re-arranged self-attention modules.
These enable LeftRefill to perform consistent generation as a generalized model without requiring test-time fine-tuning or model modifications.
Thus, LeftRefill can be seen as a simple yet unified framework to address reference-guided synthesis. 
As an exemplar, we leverage LeftRefill to address two different challenges: reference-guided inpainting and novel view synthesis, based on the pre-trained StableDiffusion.

![demo](asserts/ref_inpainting_demo.gif)


## TODOList

- [x] Releasing training, inference codes of Ref-inpainting and NVS (1-reference).
- [x] Releasing pre-trained Ref-inpainting weights.
- [x] Releasing codes of data processing.
- [x] Releasing codes of multi-view LeftRefill.

## Models

### Ref-inpainting

Since our Ref-inpainting model only contains 50 trainable tokens (704KB), weights are saved in ```checkpoints/ref_guided_inpainting```.
Before the inference, you need to download SD2-inpainting model from [here](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt), 
and place it at ```./pretrained_models/512-inpainting-ema.ckpt```

## Start with Gradio

You can simply use gradio to test reference-inpainting with custom masks.

```bash
python ref_inpainting_gradio.py
```

## Training

### Preprocessing from MegaDepth

Please preset data root path in ```megadepth_overlap.py```, you could download megadepth data from [link](https://www.cs.cornell.edu/projects/megadepth/dataset/Megadepth_v1/MegaDepth_v1.tar.gz).
Data index (scene_info_0.1_0.7, scene_info_val_1500) could be downloaded from LoFTR ([train link](https://drive.google.com/file/d/1YMAAqCQLmwMLqAkuRIJLDZ4dlsQiOiNA/view?usp=drive_link), [test link](https://drive.google.com/file/d/12yKniNWebDHRTCwhBNJmxYMPgqYX3Nhv/view?usp=drive_link)).

```bash
python megadepth_overlap.py
```
### Extending Dataset for Multi-View training and testing

Please refer to ```extend_data_for_multiview.py```, which provide an example of extending MegaDepth data for multi-view training and testing.

```bash
python extend_data_for_multiview.py
```

### Matching mask

You could directly training LeftRefill without matching mask for a fast try, which would achieve slightly worse results. If you need matching results, please refer to some matching works for details ([CasMTR](https://github.com/ewrfcas/CasMTR.git), [RoMa](https://github.com/Parskatt/RoMa.git)).

### Reference-guided inpainting (1-reference)

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_inpainting.py \
  --config_file configs/ref_inpainting_training_config.yaml \
  --exp_name RefInpainting_V0 \
  --ngpu 2 \
  --fp16
```

### Reference-guided inpainting (multi-reference)

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_inpainting.py \
  --config_file configs/multiview_ref_inpainting_training_config.yaml \
  --exp_name RefInpainting_4View_V0 \
  --ngpu 2 \
  --fp16
```

### Novel view synthesis (1-reference)

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_inpainting.py --config_file configs/nvs_training_config.yaml \
  --exp_name NVS_OBJ_V0 \
  --ngpu 2 \
  --fp16
```

## Testing

### Reference-guided inpainting

If you want to test multi-view version of reference-guided inpainting, please execute ```test_multiview_inpainting.py``` instead of ```test_inpainting.py```.

```bash
CUDA_VISIBLE_DEVICES=0 python test_inpainting.py --model_path check_points/ref_guided_inpainting \
                                                 --batch_size 1 \
                                                 --test_path ./data/megadepth_0.4_0.7/match_test_image_pairs \ 
                                                 --cfg 2.5 \
                                                 --test_size 512 \
                                                 --metric_size 512 \
                                                 --eta 1.0 \
                                                 --output_path test_outputs_compare
```

### Multi-View Reference-guided inpainting

```bash
CUDA_VISIBLE_DEVICES=0 python test_inpainting.py --model_path check_points/ref_guided_inpainting \
                                                 --batch_size 1 \
                                                 --test_path ./data/megadepth_0.4_0.7/match_test_image_pairs \ 
                                                 --cfg 2.5 \
                                                 --test_size 512 \
                                                 --metric_size 512 \
                                                 --eta 1.0 \
                                                 --output_path test_outputs_compare
```

## Cite

If you found our project helpful, please consider citing:

```
@inproceedings{cao2024leftrefill,
      title={LeftRefill: Filling Right Canvas based on Left Reference through Generalized Text-to-Image Diffusion Model}, 
      author={Chenjie Cao and Yunuo Cai and Qiaole Dong and Yikai Wang and Yanwei Fu},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2024},
}
```