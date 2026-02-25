import torch
from utils.model_utils import is_main_process
import torch.nn as nn
from torch.backends import cudnn

def log_basic_info(logger):
    logger.info(f"- PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"- GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"- CUDA version: {torch.version.cuda}")
        logger.info(f"- CUDNN version: {cudnn.version()}")
    logger.info("--------------")

    
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
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    if is_main_process():
        print('total number of params:{:.3f}'.format(param_sum))
        print('total size of params:{:.3f}MB'.format(param_size / 1024 / 1024))
        print('total size of params(include buffer):{:.3f}MB'.format(all_size))
        print('trainable number of params:{:.3f}'.format(grad_param_sum))
        print('trainable size of params:{:.3f}MB'.format(grad_param_size/1024/1024))

    return (param_size, param_sum, buffer_size, buffer_sum, grad_param_size)

def print_all_trainable_parameters(model: nn.Module, model_name: str = "Model"):
    """
    Prints a summary of the model's trainable parameters, including the total 
    number of parameters, the number of trainable parameters, the percentage 
    of parameters that are trainable, and a list of the names of [all] 
    trainable parameters.

    Args:
        model (nn.Module): The PyTorch model instance to be inspected.
        model_name (str): The name of the model, used to make the print 
                        output more readable.
    """
    print("=" * 60)
    print(f"Trainable Parameters Full List for: {model_name}")
    print("=" * 60)

    total_params = 0
    trainable_params = 0
    trainable_param_names = []

    if model is None:
        print(f"{model_name} is None. Cannot summarize parameters.")
        print("=" * 60)
        return

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_param_names.append(name)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    if total_params > 0:
        percentage_trainable = (trainable_params / total_params) * 100
        print(f"Percentage of trainable parameters: {percentage_trainable:.2f}%")
    else:
        print("Model has no parameters.")

    print("-" * 60)
    if not trainable_param_names:
        print("No parameters are set to be trainable in this model.")
    else:
        print(f"Full List of {len(trainable_param_names)} Trainable Parameters:")
        for i, name in enumerate(trainable_param_names):
            print(f"  {i+1}. {name}") # Adding an index for readability
    print("=" * 60)