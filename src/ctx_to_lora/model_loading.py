import logging
import os

import torch
from peft import PeftModel
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

try:
    from transformers import Gemma4ForConditionalGeneration
except ImportError:
    Gemma4ForConditionalGeneration = None

from ctx_to_lora.data.definitions import get_chat_template_candidates, get_model_family

QWEN_DISABLE_THINKING = "{%- set enable_thinking = false if enable_thinking is not defined else enable_thinking %}\n"

logger = logging.getLogger()

GEMMA_VISION_MODELS = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]


def check_is_vision_model(model_name):
    if model_name in GEMMA_VISION_MODELS:
        return True
    if get_model_family(model_name) != "gemma4":
        return False
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return True
    return is_conditional_generation_config(config)


def resolve_chat_template_path(model_name_or_path):
    for template_name in get_chat_template_candidates(model_name_or_path):
        template_path = f"chat_templates/{template_name}.jinja"
        if os.path.exists(template_path):
            return template_path
    return None


def is_conditional_generation_config(config) -> bool:
    architectures = getattr(config, "architectures", None) or []
    if any("ConditionalGeneration" in arch for arch in architectures):
        return True
    return hasattr(config, "text_config") and hasattr(config, "vision_config")


def get_conditional_generation_cls(model_name_or_path, config=None):
    family = get_model_family(model_name_or_path)
    if family == "gemma4":
        if config is not None and not is_conditional_generation_config(config):
            return None
        if Gemma4ForConditionalGeneration is None:
            return None
        return Gemma4ForConditionalGeneration
    if family == "gemma" and model_name_or_path in GEMMA_VISION_MODELS:
        return Gemma3ForConditionalGeneration
    return None


def get_model_and_tokenizer(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model = get_model(
        model_name_or_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        use_q_lora,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_name_or_path, tokenizer_kwargs, peft_config, train)
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def get_tokenizer(
    model_name_or_path, tokenizer_kwargs=None, peft_config=None, train=False
):
    padding_side = "left" if not train else "right"
    truncation_side = "left"

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        add_bos_tokens=False,
        add_eos_tokens=False,
        padding_side=padding_side,
        truncation_side=truncation_side,
        trust_remote_code=True,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = resolve_chat_template_path(model_name_or_path)
    if template_path is None:
        logger.warning(
            f"Chat template not found for {model_name_or_path}. Using default template."
        )
        return tokenizer

    logger.info(f"Using chat template from {template_path}")
    chat_template = open(template_path).read()
    if get_model_family(model_name_or_path) == "qwen3_5":
        chat_template = QWEN_DISABLE_THINKING + chat_template
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template
    return tokenizer


def get_model(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    use_q_lora=False,
    device="cuda",
    dtype=torch.bfloat16,
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=None,
    )
    model_config = None
    if get_model_family(model_name_or_path) == "gemma4":
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    conditional_generation_cls = get_conditional_generation_cls(
        model_name_or_path,
        config=model_config,
    )
    is_vision_model = conditional_generation_cls is not None
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)

    is_bidir_model = (
        "bert" in model_name_or_path.lower() or "gte" in model_name_or_path.lower()
    )

    if use_flash_attn:
        if "gte" not in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "flash_attention_2"
        elif "gte" in model_name_or_path:
            model_init_kwargs["attn_implementation"] = "sdpa"

    if is_vision_model:
        # always use sdpa for vision models
        # model_init_kwargs["attn_implementation"] = "sdpa"
        model_init_kwargs.pop("use_cache")
    elif is_bidir_model:
        model_init_kwargs["torch_dtype"] = torch.float32
        model_init_kwargs.pop("use_cache")

    if use_q_lora:
        # https://huggingface.co/blog/4bit-transformers-bitsandbytes
        # https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing
        # see bitsandbytes for the quantization implementation https://github.com/bitsandbytes-foundation/bitsandbytes
        # see unsloth https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth
        # does work currently bc it modifies the forward pass call of Linear
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_init_kwargs["quantization_config"] = bnb_config

    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    if not is_vision_model:
        if is_bidir_model:
            model = AutoModel.from_pretrained(**model_init_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    else:
        if conditional_generation_cls is None:
            config = model_config or AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True
            )
            raise ImportError(
                f"{model_name_or_path} has config type {type(config).__name__}, "
                "but this transformers install does not expose the matching "
                "conditional generation class. Please upgrade transformers before "
                "loading this Gemma conditional-generation model."
            )
        model = conditional_generation_cls.from_pretrained(**model_init_kwargs)
        model = model.language_model
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    return model


def get_lora_config(model_dir, **kwargs):
    if "target_modules" not in kwargs or kwargs["target_modules"] is None:
        logger.info("No target modules specified for LoRA.")
        return None
    r = kwargs.pop("lora_r", 8)
    peft_conf_kwargs = dict(
        r=r,
        peft_type=PeftType.LORA,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
        lora_dropout=kwargs.get("lora_dropout", 0.0),
        lora_alpha=r ** (3 / 2) * 2,
    )

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    return peft_config
