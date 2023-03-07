# https://github.com/victoresque/pytorch-template#data-loader

import argparse
import collections
import os
import random
from datetime import datetime

import numpy as np
import torch
import wandb

import week1.metrics.loss as module_loss
import week1.models as module_arch
import week1.metrics.metric as module_metric

import week1.datamodules.dataloader as module_data
from week1.train.trainer.trainer import Trainer
from week1.utils import prepare_device, read_json, concat_jsons, add_dict_to_argparser, str2bool
from week1.utils.parser.parse_config import ConfigParser

# fix random seeds for reproducibility
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# fix random seeds for reproducibility
DEFAULT_SEED = 6

# this will be overriden in config file only when set as arguments
ARGS_CONFIGPATH = dict(  # alias for CLI argument: route in config file
    name=("name",),
    pred_length=("pred_length",),
    obs_length=("obs_length",),
    wandb=("wandb", "store"),
    project=("wandb", "project"),

    lr=("optimizer", "args", "lr"),
    lr_gamma=("lr_scheduler", "args", "gamma"),
    lr_step=("lr_scheduler", "args", "step_size"),
    weight_decay=("optimizer", "args", "weight_decay"),
    steps=("diffusion", "args", "steps"),
    noise=("diffusion", "args", "noise_std"),

    archtype=("arch", "type"),

    normalize_data=("normalize_data",),
    normalize_type=("normalize_type",),
    batch_size=("trainer", "batch_size"),
    use_amp=("trainer", "use_amp"),
    samples_epoch=("trainer", "samples_epoch"),
    epochs=("trainer", "epochs"),
    output=("trainer", "save_dir"),
    vf=("trainer", "validation_frequency"),
    es=("trainer", "early_stop"),
    sp=("trainer", "save_period"),
    v=("trainer", "verbosity"),
)
ARGS_TYPES = dict(
    name=str,
    wandb=str2bool,
    project=str,

    lr=float,
    lr_gamma=float,
    lr_step=int,
    weight_decay=float,
    diffusion_predict=str,
    steps=int,
    noise=float,

    archtype=str,

    normalize_data=str2bool,
    normalize_type=str,
    batch_size=int,
    use_amp=str2bool,
    samples_epoch=int,
    epochs=int,
    output=str,
    vf=int,
    es=int,
    sp=int,
    v=int
)
# double check
for k in ARGS_CONFIGPATH:
    assert k in ARGS_TYPES, f"'{k}' not in ARGS_TYPES"
for k in ARGS_TYPES:
    assert k in ARGS_CONFIGPATH, f"'{k}' not in ARGS_CONFIGPATH"


def init_wandb(config, resume, wandb_id=None, overriden_id=False):
    store_to_wandb = "wandb" in config and config["wandb"]["store"]
    if store_to_wandb and resume:
        if wandb_id == None:
            raise Exception(
                f"[ERROR] You need to specify the wandb id for the run you want to attach.")

        wandb.init(project=config["wandb"]["project"],
                   name=config["name"],
                   id=config["unique_id"] if not overriden_id else wandb_id,
                   entity="",
                   notes=config["wandb"]["description"],
                   tags=config["wandb"]["description"],
                   config=config,
                   resume="must" if resume else "never")
        # logger.info(f"Wandb was resumed successfully. Run_ID = {wandb.run.id}")
        print(f"Wandb was resumed successfully. Run_ID = {wandb.run.id}")
        config = wandb.config.as_dict()
    elif store_to_wandb:
        wandb.init(project=config["wandb"]["project"],
                   name=config["name"],
                   id=wandb_id,
                   entity="",
                   notes=config["wandb"]["description"],
                   tags=config["wandb"]["description"],
                   config=config)
        # logger.info(f"Wandb was started successfully. Run_ID = {wandb.run.id}")
        print(f"Wandb was started successfully. Run_ID = {wandb.run.id}")
        config = wandb.config.as_dict()
    return config


def main(config_dict, resume, only_eval, deterministic=False, benchmark_mode=False, wid=None,
         debug=False):
    unique_id = datetime.now().strftime(
        r'%y%m%d_%H%M%S') + f"_{random.randint(0, 1000):03d}"  # we need it before initializing config (wandb sweeps update config dict)
    config_dict = init_wandb(config_dict, resume, wandb_id=unique_id if wid is None else wid,
                             overriden_id=wid is not None)
    config = ConfigParser(config_dict, resume=args.resume, run_id=unique_id)

    logger = config.get_logger('train')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    for i in range(torch.cuda.device_count()):
        logger.info(f"> GPU {i} ready: {torch.cuda.get_device_name(i)}")
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    use_amp = config['trainer']['use_amp'] if 'use_amp' in config[
        'trainer'] else False  # default to config["dtype"]

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler,
                      use_amp=use_amp)

    trainer.train()

    del trainer
    del model
    torch.cuda.empty_cache()


def create_argparser():
    """
    for key in defaults_to_config.keys():
        assert key in defaults, f"[code error] key '{key}' has no config path associated."
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-d', '--dataset', default=None, type=str,
                        help='dataset config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-s', '--seed', default=DEFAULT_SEED, type=int,
                        help='random seed')
    parser.add_argument('-e', '--eval', action='store_true',
                        help='only evaluation of resumed checkpoint')  # only evaluation of resumed checkpoint. Skips training
    parser.add_argument('-debug', '--debug', action='store_true',
                        help='(only diffusion) debug training with extra wandb plots')  # only evaluation of resumed checkpoint. Skips training
    parser.add_argument('-det', '--deterministic', action='store_true',
                        help='only 1 sample will be used for evaluation')  # launch deterministic eval
    parser.add_argument('-wid', '--wandb_id', default=None, type=str,
                        help='id of wandb run to re-attach (only if it DOES NOT match the unique_id from the config file)')

    add_dict_to_argparser(parser, ARGS_TYPES)

    return parser


def update_config_with_arguments(config_dict, args):
    for key in vars(args):
        new_value = getattr(args, key)
        if new_value is not None and key in ARGS_TYPES:
            # print(f"'{key}' will be overriden by '{new_value}'")

            # we find the place inside the nested dict where the new value needs to be put
            config_tree = config_dict
            for subtree_key in ARGS_CONFIGPATH[key][:-1]:
                config_tree = config_tree[subtree_key]

            key_tobereplaced = ARGS_CONFIGPATH[key][-1]
            # if key_tobereplaced not in config_tree:
            #    print(f"Not found. No overriden then.")
            # else:
            #    print(config_tree[key_tobereplaced])
            config_tree[key_tobereplaced] = new_value


if __name__ == '__main__':
    args = create_argparser().parse_args()

    if args.resume:
        config_path = os.path.join(os.path.dirname(args.resume), "config.json")
        config_dict = read_json(config_path)
    else:
        config = read_json(args.config)
        config_dataset = read_json(args.dataset)
        config_dict = concat_jsons(config, config_dataset)

    update_config_with_arguments(config_dict, args)
    config_dict["seed"] = args.seed

    main(config_dict, resume=args.resume is not None, only_eval=args.eval,
         deterministic=args.deterministic, wid=args.wandb_id, debug=args.debug)
