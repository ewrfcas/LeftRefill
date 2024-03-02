# LeftRefill
LeftRefill: Filling Right Canvas based on Left Reference through Generalized Text-to-Image Diffusion Model (CVPR2024)

Codes and datasets will be released soon.

[Project Page](https://ewrfcas.github.io/LeftRefill/)


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

## TODOList

- [x] Releasing training, inference codes of Ref-inpainting and NVS (1-reference).
- [x] Releasing pre-trained Ref-inpainting weights.
- [ ] Releasing codes of data processing (Megadepth, Objaverse).
- [ ] Releasing codes of multi-view LeftRefill.

## Models

### Ref-inpainting

Since our Ref-inpainting model only contains 50 trainable tokens (704KB), weights are saved in ```checkpoints/ref_guided_inpainting```.
Before the inference, you need to download SD2-inpainting model from [here](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.ckpt), 
and place it at ```./pretrained_models/512-inpainting-ema.ckpt```

## Start with Gradio

```
python ref_inpainting_gradio.py
```

## Training

### Reference-guided inpainting (1-reference)

```
CUDA_VISIBLE_DEVICES=0,1 python train_inpainting.py \
  --config_file configs/training_config.yaml \
  --exp_name RefInpainting_V0 \
  --ngpu 2 \
  --fp16
```

### Novel view synthesis (1-reference)

```
CUDA_VISIBLE_DEVICES=0,1 python train_inpainting.py --config_file configs/nvs_training_config.yaml \
  --exp_name NVS_OBJ_V0 \
  --ngpu 2 \
  --fp16
```

## Testing

### Reference-guided inpainting

```
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
@misc{cao2023harnessing,
      title={Harnessing Text-to-Image Attention Prior for Reference-based Multi-view Image Synthesis}, 
      author={Chenjie Cao and Yunuo Cai and Qiaole Dong and Yikai Wang and Yanwei Fu},
      year={2023},
      eprint={2305.11577},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```