import torch.nn as nn
import torch

from utils.model_utils import is_main_process

class RSBlock(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, dropout_p=0.1):
        super().__init__()
        self.down_project = nn.Linear(input_dim, bottleneck_dim)
        self.non_linearity = nn.GELU() 
        self.dropout = nn.Dropout(dropout_p) 
        self.up_project = nn.Linear(bottleneck_dim, input_dim)
        
        # key initialization
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)

    def forward(self, x):
        # data stream：x -> down -> activation -> up -> output
        adapted_x = self.up_project(self.dropout(self.non_linearity(self.down_project(x))))
        return adapted_x


class RSBlockWrapper(nn.Module):
    def __init__(self, original_block, adapter):
        super().__init__()
        self.original_block = original_block
        self.adapter = adapter

    def forward(self, x: torch.Tensor):
        # forzen part
        x = x + self.original_block.attention(self.original_block.ln_1(x))
        normalized_x = self.original_block.ln_2(x)
        ffn_output = self.original_block.mlp(normalized_x)
        # learning part
        adapter_output = self.adapter(normalized_x)
    
        x = x + ffn_output + adapter_output
        return x

def _inject_adapter_to_blocks(resblocks, rs_layers, bottleneck_dim):
    """
    Generic injection logic: Iterate over resblocks and replace the specified layer 
    with a Wrapper that contains both the original block and the new RSBlock.
    """
    total_layers = len(resblocks)
    
    # get the num of injected layers
    if -1 in rs_layers:
        target_indices = list(range(total_layers))
    else:
        target_indices = [idx for idx in rs_layers if 0 <= idx < total_layers]

    for idx in target_indices:
        original_block = resblocks[idx]
        d_model = original_block.ln_1.normalized_shape[0]

        adapted_block = RSBlock(d_model, bottleneck_dim)
        resblocks[idx] = RSBlockWrapper(original_block, adapted_block)
        
    return len(target_indices)

def add_rs_blocks_to_clip(model, cfg):
    """
    Main entry: Mount RSBlock in both visual and text branches.
    """
    rs_layers = cfg.MODEL.RS_LAYERS
    bottleneck_dim = cfg.MODEL.BOTTLENECK_DIM

    # visual branch (Visual Transformer)
    if hasattr(model.visual, 'transformer'):
        v_count = _inject_adapter_to_blocks(
            model.visual.transformer.resblocks, 
            rs_layers, 
            bottleneck_dim
        )
        if is_main_process():
            print(f"[Visual] Added {v_count} RSBlocks to Visual Transformer.")
    else:
        if is_main_process():
            print("[Visual] Has no transformer, skipping RSBlock injection for visual branch.")

    # 2. text branch (Transformer)
    if hasattr(model, 'transformer'):
        t_count = _inject_adapter_to_blocks(
            model.transformer.resblocks, 
            rs_layers, 
            bottleneck_dim
        )
        if is_main_process():
            print(f"[Textual] Added {t_count} RSBlocks to Text Transformer.")

    return model