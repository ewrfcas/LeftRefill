import argparse
import os
import shutil

import lpips
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from inpainting_ldm.logger import InpaintingLogger
from inpainting_ldm.model import create_model, load_state_dict


def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Config')
    args.add_argument('--config_file', default=None, type=str, help='config file')
    args.add_argument('--exp_name', default=None, type=str, help='experiment name')
    args.add_argument('--save_path', default='./check_points', type=str)
    args.add_argument('--ngpu', default=1, type=int, help='gpu number')
    args.add_argument('--fp16', action='store_true', help='use FP16')
    args.add_argument('--restore', action='store_true', help='restore training')
    args.add_argument('--no_restore', action='store_true', help='restore training')

    args = args.parse_args()
    if args.restore:
        config = yaml.load(open(args.save_path + f'/{args.exp_name}/training_config.yaml'), Loader=yaml.FullLoader)
        model = create_model(open(args.save_path + f'/{args.exp_name}/model_config.yaml')).cpu()
        if args.restore and getattr(model, 'save_prompt_only', False) and not os.path.exists(args.save_path + f'/{args.exp_name}/ckpts/last_resave.ckpt'):
            reload_weights = load_state_dict('pretrained_models/512-inpainting-ema.ckpt', location='cpu')
            torch_init_model(model, reload_weights, key='none')
    else:  # load new training config
        config = yaml.load(open(args.config_file), Loader=yaml.FullLoader)
        # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
        model = create_model(config['model_config']).cpu()
        if not args.no_restore:
            reload_weights = load_state_dict(config['resume_path'], location='cpu')
            torch_init_model(model, reload_weights, key='none')

    if hasattr(model, 'lora_cfg'):
        model.set_lora()

    # model.learning_rate = config['learning_rate']
    model.cfg = config
    model.optim_cfg = config['optim_cfg']
    model.world_size = args.ngpu
    model.image_text_pair = config['image_text_pair']
    model.loss_fn_alex = lpips.LPIPS(net='alex').cpu()

    if args.restore and getattr(model, 'save_prompt_only', False):  # if save prompt only we need to reload&resave models
        if not os.path.exists(args.save_path + f'/{args.exp_name}/ckpts/last_resave.ckpt'):
            print('Saving new ckpt...')
            resave_ckpt = torch.load(args.save_path + f'/{args.exp_name}/ckpts/last.ckpt', map_location='cpu')
            old_weights = resave_ckpt['state_dict']
            new_weights = model.state_dict()
            for key in new_weights:
                if key not in old_weights:
                    old_weights[key] = new_weights[key]
            resave_ckpt['state_dict'] = old_weights
            torch.save(resave_ckpt, args.save_path + f'/{args.exp_name}/ckpts/last_resave.ckpt')

    logger = loggers.TestTubeLogger(
        save_dir=f'{args.save_path}',
        name=args.exp_name,
        debug=False,
        create_git_tag=False
    )

    log_images_kwargs = {'return_attn': config.get('return_attn', False), 'unconditional_guidance_scale': model.data_cfg['cfg']}
    img_callback = InpaintingLogger(batch_frequency=config['logger_freq'], save_dir=f'{args.exp_name}/samples',
                                    log_images_kwargs=log_images_kwargs)
    monitor = f'val/{config.get("monitor", "lpips")}'
    monitor_mode = config.get('monitor_mode', 'min')
    checkpoint_callback = ModelCheckpoint(f'{args.save_path}/{args.exp_name}/ckpts', save_top_k=config['save_top_k'],
                                          monitor=monitor, mode=monitor_mode, save_last=True)  # every_n_train_steps=config['save_freq'],
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # copy configs
    if not args.restore:
        os.makedirs(f'{args.save_path}/{args.exp_name}', exist_ok=True)
        shutil.copy(args.config_file, f'{args.save_path}/{args.exp_name}/training_config.yaml')
        shutil.copy(config['model_config'], f'{args.save_path}/{args.exp_name}/model_config.yaml')

    if args.restore:  # only support resume from the last ckpt
        if model.save_prompt_only:
            resume_ckpt = args.save_path + f'/{args.exp_name}/ckpts/last_resave.ckpt'
        else:
            resume_ckpt = args.save_path + f'/{args.exp_name}/ckpts/last.ckpt'
    else:
        resume_ckpt = None
    max_steps = config.get('max_steps', None)
    max_epochs = config.get('max_epochs', None)

    check_val_every_n_epoch = config.get('check_val_every_n_epoch', 1)
    val_check_interval = config.get('val_check_interval', 1.0)
    trainer = pl.Trainer(logger=logger, gpus=args.ngpu, precision=16 if args.fp16 else 32,
                         callbacks=[img_callback, lr_monitor, checkpoint_callback],
                         max_steps=max_steps,
                         max_epochs=max_epochs,
                         log_every_n_steps=50,
                         resume_from_checkpoint=resume_ckpt,
                         accumulate_grad_batches=config.get('accumulate_grad_batches', None),
                         val_check_interval=val_check_interval,
                         check_val_every_n_epoch=check_val_every_n_epoch,
                         benchmark=True, accelerator='ddp' if args.ngpu > 1 else None,
                         replace_sampler_ddp=False if config['cross_view_inpainting'] else True,
                         num_sanity_val_steps=2)  # use custom sampler

    # Train!
    trainer.fit(model)
