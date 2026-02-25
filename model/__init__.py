from model.base import BaseModel
from model.model import SFRNet
from utils.model_utils import is_main_process


def build_model(cfg, clip_model, dataset, model_name=None):

    model_factory = {
        "BaseModel": BaseModel,
        "SFRNet": SFRNet,
    }

    # check model availability
    if model_name not in model_factory:
        raise ValueError(
            f"Unsupported model: {model_name}. Already supported models are: {list(model_factory.keys())}"
        )

    # 获取对应的模型类
    ModelClass = model_factory[model_name]

    # 实例化模型
    try:
        model = ModelClass(
            cfg,
            clip_model,
            dataset.classnames_seen,
            dataset.classnames_unseen
        )
    except TypeError as e:
        # 捕获可能由于构造函数参数不匹配导致的错误
        print(f"Error instantiating model '{model_name}': {e}")
        print(f"Please check the constructor parameters for {model_name}.")
        raise

    if is_main_process():
        print(f'Build {model_name} model done!')

    return model