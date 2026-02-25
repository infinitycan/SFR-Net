import torch



def make_optimizer(cfg, model, lr_mult=0.1):
    clip_params, other_params, attention_params = [], [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        elif 'conv_branch' in pname:
            attention_params.append(p)
        elif pname.startswith('clip'):
            clip_params.append(p)
        else:
            other_params.append(p)

    # Optimizer1
    param_groups = [
        {'params': clip_params, 'lr': cfg.SOLVER.BASE_LR * lr_mult, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': other_params, 'lr': cfg.SOLVER.BASE_LR_SGD, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': attention_params, 'lr': cfg.SOLVER.BASE_LR_SGD * lr_mult * lr_mult, 'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
    ]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = torch.optim.SGD(param_groups, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(param_groups, momentum=cfg.SOLVER.MOMENTUM)

    return optimizer

