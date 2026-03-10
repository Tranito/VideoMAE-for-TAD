import torch
from torch import optim as optim
import json

def get_vit_parameter_groups(
    model: torch.nn.Module,
    base_lr: float = 2.5e-5,
    weight_decay: float = 0.05,
    layer_decay: float = 0.75,
    head_lr_mult: float = 1.0,
    skip_list: tuple = None,
    verbose: bool = False,
):
    """
    Build parameter groups for ViT with layer-wise learning rate decay.
    
    Args:
        model: VisionTransformer instance
        base_lr: Base learning rate for the highest layer
        weight_decay: Weight decay for parameters not in skip_list
        layer_decay: LR multiplier per layer (< 1.0). Earlier layers get smaller LR.
        head_lr_mult: Extra multiplier for head parameters (typically 5-20 for random init)
        skip_list: Parameter names to exclude from weight decay
        verbose: Print parameter groups for debugging
    
    Returns:
        List of parameter group dicts for optimizer
    """
    if skip_list is None:
        skip_list = ()
    
    # Extend skip_list with model's no_weight_decay() if available
    try:
        model_no_wd = getattr(model, "no_weight_decay", None)
        if callable(model_no_wd):
            skip_set = set(skip_list) | set(model_no_wd())
        else:
            skip_set = set(skip_list)
    except Exception:
        skip_set = set(skip_list)

    # Get number of transformer blocks
    num_blocks = getattr(model, "get_num_layers", lambda: None)()
    if num_blocks is None:
        num_blocks = len(getattr(model, "blocks", []))

    # Layer IDs: 0 (patch_embed), 1..num_blocks (blocks), num_blocks+1 (heads/fc_norm)
    max_layer_id = num_blocks
    head_layer_id = max_layer_id + 1
    
    # LR scales: earlier layers get decay**(max_layer_id - layer_id)
    # Last block gets scale=1.0, patch_embed gets smallest scale
    scales = [layer_decay ** (max_layer_id - i) for i in range(max_layer_id + 1)]
    scales.append(1.0 * head_lr_mult)  # heads get extra multiplier

    def _get_layer_id(var_name: str) -> int:
        if var_name.startswith("patch_embed") or var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("blocks"):
            try:
                layer_index = int(var_name.split(".")[1])
                return layer_index + 1
            except Exception:
                return max_layer_id
        elif var_name.startswith("fc_norm") or var_name.startswith("rel_pos_bias"):
            return max_layer_id
        elif var_name.startswith("head"):
            return head_layer_id
        else:
            return head_layer_id

    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_set:
            decay_type = "no_decay"
            this_wd = 0.0
        else:
            decay_type = "decay"
            this_wd = weight_decay

        layer_id = _get_layer_id(name)
        group_name = f"layer_{layer_id}_{decay_type}"
        scale = scales[min(layer_id, len(scales) - 1)]

        if group_name not in parameter_group_names:
            lr = base_lr * float(scale)
            parameter_group_names[group_name] = {
                "weight_decay": this_wd,
                "params": [],
                "lr": lr,
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_wd,
                "params": [],
                "lr": lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    if verbose:
        print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))

    return list(parameter_group_vars.values())