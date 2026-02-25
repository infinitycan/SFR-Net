import cv2
import torch
import numpy as np
import random
import torch.distributed as dist
from clip import clip
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import os
import errno
import pickle
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))
            self.module.temperature = deepcopy(model.temperature)
            if hasattr(model, "temperature_glb"):
                self.module.temperature_glb = deepcopy(model.temperature_glb)
            
    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def thread_flag(dist_train):
    if not dist_train:
        return True
    else:
        return dist.get_rank() == 0

def is_main_process():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True  
    return dist.get_rank() == 0

def getModelSize(model):
    param_size = 0
    param_sum = 0
    grad_param_size = 0
    grad_param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
        if param.requires_grad == True:
            grad_param_size += param.nelement() * param.element_size()
            grad_param_sum += param.nelement()
    print('total number of params:{:.3f}M'.format(param_sum / 1000 / 1000))
    print('trainable number of params:{:.3f}M ({:.5%})'.format(grad_param_sum / 1000 / 1000, grad_param_sum/param_sum))

    return (param_size, param_sum, grad_param_size)



def convert_params_to_value(params):
    if params[0] == -1:
        return [-1]    # not using
    elif params[-1] == -1:
        return list(range(params[0]))    # continuous N layers
    else:
        return params


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    if is_main_process():
        print("Build CLIP Model")
    
    model = clip.build_model(state_dict or model.state_dict(), cfg.INPUT.SIZE_TRAIN)

    return model.float()



def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def tolist_if_not(x):
    """Convert to a list."""
    if not isinstance(x, list):
        x = [x]
    return x


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    """
    mkdir_if_missing(save_dir)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    iters = state["iters"]
    if is_best:
        model_name = "model-best.pth"
    else:
        model_name = f"model-iters{iters}.pth"
    fpath = os.path.join(save_dir, model_name)

    torch.save(state, fpath)


def load_checkpoint(fpath):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        fpath = 'log/my_model/model.pth.tar-10'
        checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    map_location = None if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint



def load_pretrained_weights(model, weight_path):
    r"""Load pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        # >>> weight_path = 'log/my_model/model-best.pth.tar'
        # >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        print(
            f"Cannot load {weight_path} (check the key names manually)"
        )
    else:
        print(f"Successfully loaded pretrained weights from {weight_path}")
        if len(discarded_layers) > 0:
            print(
                f"Layers discarded due to unmatched keys or size: {discarded_layers}"
            )
    

class StepWiseBetaScheduler:
    def __init__(self, total_steps, start_beta=0.1, end_beta=0.9):
        """
        逐训练步更新beta值的调度器。
        
        Args:
            total_steps (int): 训练过程总共的步数。
            start_beta (float): 初始beta。
            end_beta (float): 最终beta。
        """
        self.total_steps = total_steps
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.current_step = 0

    def step(self):
        """
        在每个训练步后调用，更新并返回当前的beta值。
        """
        # 线性增长
        progress = self.current_step / (self.total_steps - 1)
        current_beta = self.start_beta + progress * (self.end_beta - self.start_beta)
        
        self.current_step += 1
        
        return min(current_beta, self.end_beta) # 确保不超过最大值
    

class PhasedAnnealingBetaScheduler:
    """
    一个分阶段退火的beta权重调度器类。
    它将训练过程分为三个阶段：BCE主导期，线性过渡期，Ranking Loss主导期。
    """
    def __init__(self, 
                 phase1_end_epoch: int,
                 phase2_end_epoch: int,
                 beta_phase1: float,
                 beta_phase3: float):
        """
        初始化调度器。

        Args:
            phase1_end_epoch (int): 第一阶段（BCE主导）结束的轮数。
            phase2_end_epoch (int): 第二阶段（过渡期）结束的轮数。
            beta_phase1 (float): 第一阶段的beta值。
            beta_phase3 (float): 第三阶段的beta值。
        """
        if phase1_end_epoch >= phase2_end_epoch:
            raise ValueError("phase1_end_epoch必须小于phase2_end_epoch")
            
        self.phase1_end = phase1_end_epoch
        self.phase2_end = phase2_end_epoch
        self.beta1 = beta_phase1
        self.beta3 = beta_phase3
        
        # 内部状态
        self._epoch = 0
        self._beta = self.beta1

    def step(self, epoch: int):
        # 外部传入的epoch是从1开始的，直接赋值
        self._epoch = epoch
            
        # 阶段A: BCE主导期
        # 如果 phase1_end=2, 那么 epoch 1, 2 都会进入这个分支
        if self._epoch <= self.phase1_end:
            self._beta = self.beta1
        
        # 阶段B: 过渡/退火期
        # 如果 phase2_end=4, 那么 epoch 3, 4 会进入这个分支
        elif self._epoch <= self.phase2_end:
            # 计算过渡期的总时长
            anneal_duration = self.phase2_end - self.phase1_end
            
            # 计算当前在过渡期的第几个epoch (从1开始)
            # 例如，epoch=3, phase1_end=2 -> step_in_phase=1
            step_in_phase = self._epoch - self.phase1_end
            
            # 计算线性增长的进度
            progress = step_in_phase / anneal_duration
            self._beta = self.beta1 + progress * (self.beta3 - self.beta1)
            
        # 阶段C: Ranking Loss主导期
        else:
            self._beta = self.beta3
            
        return self._beta