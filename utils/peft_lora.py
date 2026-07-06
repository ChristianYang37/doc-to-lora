import os
from typing import Any, Iterable, List, Optional

import torch.nn as nn
from omegaconf import DictConfig, ListConfig, OmegaConf


QWEN_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

M2P_PREFERRED_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "linear",
    "fc1",
    "fc2",
    "proj",
    "dense",
    "out_proj",
    "linear1",
    "linear2",
]


def _require_peft():
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT LoRA is enabled, but the `peft` package is not installed. "
            "Install it with `pip install peft`."
        ) from exc
    return LoraConfig, PeftModel, TaskType, get_peft_model


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, (DictConfig, dict)):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return list(value)


def peft_section_enabled(cfg: Any, section: str) -> bool:
    peft_cfg = cfg_get(cfg, "peft")
    section_cfg = cfg_get(peft_cfg, section)
    return bool(cfg_get(section_cfg, "enabled", False))


def unwrap_peft_model(model: nn.Module) -> nn.Module:
    base_model = getattr(model, "base_model", None)
    if base_model is not None and hasattr(base_model, "model"):
        return base_model.model
    return model


def is_peft_model(model: nn.Module) -> bool:
    return hasattr(model, "peft_config") and hasattr(model, "save_pretrained")


def freeze_non_lora_params(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name


def _delegate_custom_qwen_methods(peft_model: nn.Module) -> nn.Module:
    base = unwrap_peft_model(peft_model)
    for name in [
        "reset_mem_tokens",
        "lora_params_numel",
        "set_generate_func",
        "generate_lora_dict",
        "init_lora_dict",
        "divide_idx",
    ]:
        if hasattr(base, name) and not hasattr(peft_model, name):
            setattr(peft_model, name, getattr(base, name))
    return peft_model


def apply_lora_to_qwen(qwen_model: nn.Module, lora_config_args: Any, is_trainable: bool = True) -> nn.Module:
    LoraConfig, PeftModel, TaskType, get_peft_model = _require_peft()
    adapter_path = cfg_get(lora_config_args, "adapter_path")
    adapter_name = cfg_get(lora_config_args, "adapter_name", "default")

    if adapter_path:
        peft_model = PeftModel.from_pretrained(
            qwen_model,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(cfg_get(lora_config_args, "r", 8)),
            lora_alpha=int(cfg_get(lora_config_args, "lora_alpha", 16)),
            lora_dropout=float(cfg_get(lora_config_args, "lora_dropout", 0.05)),
            bias=cfg_get(lora_config_args, "bias", "none"),
            target_modules=as_list(cfg_get(lora_config_args, "target_modules")) or QWEN_LORA_TARGETS,
        )
        peft_model = get_peft_model(qwen_model, peft_config, adapter_name=adapter_name)

    peft_model.set_adapter(adapter_name)
    freeze_non_lora_params(peft_model)
    return _delegate_custom_qwen_methods(peft_model)


def find_linear_module_names(model: nn.Module, preferred: Optional[Iterable[str]] = None) -> List[str]:
    linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linear_names:
        return []

    preferred = list(preferred or M2P_PREFERRED_TARGETS)
    leaf_names = {name.rsplit(".", 1)[-1] for name in linear_names}
    matched = [name for name in preferred if name in leaf_names]
    if matched:
        return matched

    return linear_names


def print_named_modules(model: nn.Module, logger: Any = None, max_lines: int = 200) -> None:
    lines = []
    for idx, (name, module) in enumerate(model.named_modules()):
        if idx >= max_lines:
            lines.append(f"... truncated after {max_lines} modules")
            break
        lines.append(f"{name or '<root>'}: {module.__class__.__name__}")
    text = "\n".join(lines)
    if logger is not None:
        logger.info("M2P named_modules:\n" + text)
    else:
        print(text)


def apply_lora_to_m2p(m2p_model: nn.Module, lora_config_args: Any, logger: Any = None) -> nn.Module:
    LoraConfig, PeftModel, TaskType, get_peft_model = _require_peft()
    adapter_path = cfg_get(lora_config_args, "adapter_path")
    adapter_name = cfg_get(lora_config_args, "adapter_name", "default")
    configured_targets = as_list(cfg_get(lora_config_args, "target_modules"))
    target_modules = configured_targets or find_linear_module_names(m2p_model)
    if not target_modules:
        raise ValueError("Could not find Linear modules in M2P for PEFT LoRA.")

    if logger is not None:
        logger.info(f"M2P PEFT LoRA target_modules: {target_modules}")

    if adapter_path:
        peft_model = PeftModel.from_pretrained(
            m2p_model,
            adapter_path,
            adapter_name=adapter_name,
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=int(cfg_get(lora_config_args, "r", 8)),
            lora_alpha=int(cfg_get(lora_config_args, "lora_alpha", 16)),
            lora_dropout=float(cfg_get(lora_config_args, "lora_dropout", 0.05)),
            bias=cfg_get(lora_config_args, "bias", "none"),
            target_modules=target_modules,
        )
        peft_model = get_peft_model(m2p_model, peft_config, adapter_name=adapter_name)

    peft_model.set_adapter(adapter_name)
    freeze_non_lora_params(peft_model)
    return peft_model


def load_saved_adapter_if_present(model: nn.Module, adapter_dir: str, adapter_name: str = "default") -> nn.Module:
    if not is_peft_model(model) or not os.path.isdir(adapter_dir):
        return model
    if not os.path.isfile(os.path.join(adapter_dir, "adapter_config.json")):
        return model
    if adapter_name in getattr(model, "peft_config", {}) and hasattr(model, "delete_adapter"):
        model.delete_adapter(adapter_name)
    model.load_adapter(adapter_dir, adapter_name=adapter_name, is_trainable=True)
    model.set_adapter(adapter_name)
    freeze_non_lora_params(model)
    return model


def save_peft_adapter_if_present(model: nn.Module, out_dir: str) -> None:
    if is_peft_model(model):
        model.save_pretrained(out_dir, safe_serialization=True)


def cfg_to_plain_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, dict):
        return cfg
    return dict(vars(cfg))
