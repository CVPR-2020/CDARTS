""" Retrain cell """
import os
import torch
import json
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import utils.genotypes as gt

from tensorboardX import SummaryWriter
from models.cdarts_controller import CDARTSController
from utils.visualize import plot
from utils import utils
from configs.config import AugmentConfig
from core.pretrain_function import train, sample_train, validate, sample_validate

# config
config = AugmentConfig()

# make apex optional
if config.distributed:
    # DDP = torch.nn.parallel.DistributedDataParallel
    try:
        import apex
        from apex.parallel import DistributedDataParallel as DDP
        from apex import amp, optimizers
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


# tensorboard
writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
writer.add_text('config', config.as_markdown(), 0)

logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
if config.local_rank == 0:
    config.print_params(logger.info)
    
if 'cifar' in config.dataset:
    from datasets.cifar import get_augment_datasets
elif 'imagenet' in config.dataset:
    from datasets.imagenet import get_augment_datasets
else:
    raise Exception("Not support dataset!")

def main():
    logger.info("Logger is set - training start")

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if config.distributed:
        config.gpu = config.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(config.gpu)
        # distributed init
        torch.distributed.init_process_group(backend='nccl', init_method=config.dist_url,
                world_size=config.world_size, rank=config.local_rank)

        config.world_size = torch.distributed.get_world_size()

        config.total_batch_size = config.world_size * config.batch_size
    else:
        config.total_batch_size = config.batch_size


    loaders, samplers = get_augment_datasets(config)
    train_loader, valid_loader = loaders
    train_sampler, valid_sampler = samplers

    net_crit = nn.CrossEntropyLoss().cuda()
    controller = CDARTSController(config, net_crit, n_nodes=4, stem_multiplier=config.stem_multiplier)

    resume_state = None
    if config.resume:
        resume_state = torch.load(config.resume_path,  map_location='cpu')
        controller.load_state_dict(resume_state['model_main'])

    controller = controller.cuda()

    controller = apex.parallel.convert_syncbn_model(controller)
    # weights optimizer
    optimizer = torch.optim.SGD([ {"params": controller.feature_extractor.parameters()},
                                  {"params": controller.super_layers_pool.parameters()},
                                  {"params": controller.fc_super.parameters()}],
                                lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
    if config.use_amp:
        controller, optimizer = amp.initialize(controller, optimizer, opt_level=config.opt_level)

    if config.distributed:
        controller = DDP(controller, delay_allreduce=True)

    best_top1 = 0.
    best_top5 = 0.
    sta_epoch = 0
    # training loop
    if config.resume:
        optimizer.load_state_dict(resume_state['optimizer'])
        lr_scheduler.load_state_dict(resume_state['lr_scheduler'])
        best_top1 = resume_state['best_top1']
        best_top5 = resume_state['best_top5']
        sta_epoch = resume_state['sta_epoch']

    for epoch in range(sta_epoch, config.epochs):
        # reset iterators
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(epoch)
        current_lr = lr_scheduler.get_lr()[0]

        if config.local_rank == 0:
            logger.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < config.warmup_epochs and config.total_batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            if config.local_rank == 0:
                logger.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        
        # training
        if config.sample_pretrain:
            sample_train(train_loader, controller, optimizer, epoch, writer, logger, config)
            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1, top5 = sample_validate(valid_loader, controller, epoch, cur_step, writer, logger, config)
        else:
            train(train_loader, controller, optimizer, epoch, writer, logger, config)
            # validation
            cur_step = (epoch+1) * len(train_loader)
            top1, top5 = validate(valid_loader, controller, epoch, cur_step, writer, logger, config)

        if 'cifar' in config.dataset:
            lr_scheduler.step()
        elif 'imagenet' in config.dataset:
            lr_scheduler.step()
            # current_lr = utils.adjust_lr(optimizer, epoch, config)
        else:
            raise Exception('Lr error!')
            
        # save
        if best_top1 < top1:
            best_top1 = top1
            best_top5 = top5
            is_best = True
        else:
            is_best = False

        # save
        if config.local_rank == 0:
            torch.save({
                "controller":controller.module.state_dict(),
                "optimizer":optimizer.state_dict(),
                "lr_scheduler":lr_scheduler.state_dict(),
                "best_top1":best_top1,
                "best_top5":best_top5,
                "sta_epoch":epoch + 1
            }, os.path.join(config.path, "retrain_resume.pth.tar"))
            utils.save_checkpoint(controller.module.state_dict(), config.path, is_best)
        
    if config.local_rank == 0:
        logger.info("Final best Prec@1 = {:.4%}, Prec@5 = {:.4%}".format(best_top1, best_top5))

    
if __name__ == "__main__":
    main()
