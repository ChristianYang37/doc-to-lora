import json
import os

IGNORE_INDEX = -100
CTX_MARKER = "<ctx_to_lora_context_marker>"

TRANSFORMED_DATA_DIR = "data/processed_datasets"
RAW_DATA_DIR = "data/raw_datasets/"
SELF_GEN_DATA_DIR = f"{RAW_DATA_DIR}/self_gen/"

# for chunking
QWEN3_5_9B_MODEL_ID = "Qwen/Qwen3.5-9B"
GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"

CTX_AFFIXES = {
    "google/gemma-2-2b-it": {
        "prefix": [2, 106, 1645, 110],  # <bos><start_of_turn>user\n\n\n
        "suffix": [107, 108, 106, 2516, 108],  # <end_of_turn>\n<start_of_turn>model\n
    },
    "mistralai/Mistral-7B-Instruct-v0.2": {
        "prefix": [1, 733, 16289, 28793, 28705, 13, 13],  # `<s> [INST] \n\n`
        "suffix": [733, 28748, 16289, 28793],  # ` [/INST] `
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        # `<|im_start|>system\n<|im_end|>\n<|im_start|>user\n`
        "prefix": [151644, 8948, 198, 151645, 198, 151644, 872, 198],
        # `<|im_end|>\n<|im_start|>assistant\n`
        "suffix": [151645, 198, 151644, 77091, 198],
    },
}

CTX_AFFIX_ALIASES = {
    QWEN3_5_9B_MODEL_ID: "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3.5-9B-Instruct": "Qwen/Qwen3-4B-Instruct-2507",
    GEMMA4_MODEL_ID: "google/gemma-2-2b-it",
    "google/gemma-4-E2B-it": "google/gemma-2-2b-it",
    "google/gemma-4-31B-it": "google/gemma-2-2b-it",
}

CHAT_TEMPLATE_ALIASES = {
    QWEN3_5_9B_MODEL_ID: "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-9B-Instruct": "Qwen/Qwen3.5-9B",
    GEMMA4_MODEL_ID: "google/gemma-4-E4B-it",
    "google/gemma-4-E2B-it": "google/gemma-4-E4B-it",
    "google/gemma-4-31B-it": "google/gemma-4-E4B-it",
}


def get_model_family_from_config(model_name_or_path: str | None) -> str | None:
    if model_name_or_path is None:
        return None
    config_path = os.path.join(os.path.expanduser(model_name_or_path), "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    text_config = config.get("text_config") or {}
    model_type = str(config.get("model_type") or text_config.get("model_type") or "")
    architectures = " ".join(config.get("architectures") or [])
    if not architectures and text_config:
        architectures = " ".join(text_config.get("architectures") or [])
    family_text = f"{model_type} {architectures}".lower()

    if "qwen3_5_moe" in family_text or "qwen3.5moe" in family_text:
        return "qwen3_5_moe"
    if "moe" in family_text and (
        "qwen3_5" in family_text or "qwen3.5" in family_text
    ):
        return "qwen3_5_moe"
    if "qwen3_5" in family_text or "qwen3.5" in family_text:
        return "qwen3_5"
    if "qwen" in family_text:
        return "qwen"
    if "gemma4" in family_text or "gemma-4" in family_text:
        return "gemma4"
    if "gemma" in family_text:
        return "gemma"
    return None


def get_model_family(model_name_or_path: str | None) -> str | None:
    if model_name_or_path is None:
        return None
    config_family = get_model_family_from_config(model_name_or_path)
    if config_family is not None:
        return config_family
    model_name = model_name_or_path.lower()
    compact_name = (
        model_name.replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .replace("/", "")
    )
    if (
        "qwen3.5" in model_name
        or "qwen3_5" in model_name
        or "qwen35" in compact_name
    ):
        if "moe" in compact_name:
            return "qwen3_5_moe"
        return "qwen3_5"
    if "qwen" in model_name:
        return "qwen"
    if "gemma-4" in model_name or "gemma4" in compact_name:
        return "gemma4"
    if "gemma" in model_name:
        return "gemma"
    return None


def get_ctx_affix_key(model_name_or_path: str) -> str:
    if model_name_or_path in CTX_AFFIXES:
        return model_name_or_path
    if model_name_or_path in CTX_AFFIX_ALIASES:
        return CTX_AFFIX_ALIASES[model_name_or_path]

    family = get_model_family(model_name_or_path)
    if family == "qwen3_5_moe":
        raise ValueError(
            f"{model_name_or_path!r} looks like a Qwen3.5 MoE model, "
            "but this adapter is configured for the dense Qwen3.5 9B model."
        )
    if family in {"qwen", "qwen3_5"}:
        return "Qwen/Qwen3-4B-Instruct-2507"
    if family in {"gemma", "gemma4"}:
        return "google/gemma-2-2b-it"

    supported = sorted(set(CTX_AFFIXES) | set(CTX_AFFIX_ALIASES))
    raise KeyError(
        f"No context affix is registered for {model_name_or_path!r}. "
        f"Supported model keys/families include: {supported}"
    )


def get_ctx_affixes(model_name_or_path: str) -> dict[str, list[int]]:
    return CTX_AFFIXES[get_ctx_affix_key(model_name_or_path)]


def find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence:
        return -1
    for i in range(len(sequence) - len(subsequence) + 1):
        if sequence[i : i + len(subsequence)] == subsequence:
            return i
    return -1


def infer_ctx_affixes_from_tokenizer(tokenizer) -> dict[str, list[int]]:
    messages = [
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": CTX_MARKER},
        ]
    ]
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_attention_mask=False,
        padding=False,
        truncation=False,
        add_special_tokens=False,
        return_dict=True,
    )
    input_ids = tokenized["input_ids"][0]
    marker_ids = tokenizer(
        CTX_MARKER,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    start = find_subsequence(input_ids, marker_ids)
    if start >= 0:
        end = start + len(marker_ids)
        return {"prefix": input_ids[:start], "suffix": input_ids[end:]}

    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )
    rendered_text = rendered[0] if isinstance(rendered, list) else rendered
    marker_start = rendered_text.find(CTX_MARKER)
    if marker_start < 0:
        raise ValueError(
            f"Could not infer context affixes with tokenizer {tokenizer.name_or_path!r}"
        )
    marker_end = marker_start + len(CTX_MARKER)
    return {
        "prefix": tokenizer(
            rendered_text[:marker_start],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"],
        "suffix": tokenizer(
            rendered_text[marker_end:],
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"],
    }


def get_ctx_affixes_for_tokenizer(
    tokenizer,
    model_name_or_path: str | None = None,
) -> dict[str, list[int]]:
    model_name_or_path = model_name_or_path or tokenizer.name_or_path
    if tokenizer is None:
        return get_ctx_affixes(model_name_or_path)
    if model_name_or_path in CTX_AFFIXES:
        return get_ctx_affixes(model_name_or_path)
    if tokenizer.chat_template is not None:
        try:
            return infer_ctx_affixes_from_tokenizer(tokenizer)
        except Exception:
            pass
    return get_ctx_affixes(model_name_or_path)


def get_chat_template_candidates(model_name_or_path: str) -> list[str]:
    candidates = [model_name_or_path]

    if model_name_or_path in CHAT_TEMPLATE_ALIASES:
        candidates.append(CHAT_TEMPLATE_ALIASES[model_name_or_path])

    family = get_model_family(model_name_or_path)
    if family == "qwen3_5_moe":
        raise ValueError(
            f"{model_name_or_path!r} looks like a Qwen3.5 MoE model, "
            "but this adapter is configured for the dense Qwen3.5 9B model."
        )
    if family == "qwen3_5":
        candidates.extend(
            [
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3-4B-Instruct-2507",
            ]
        )
    elif family == "qwen":
        candidates.append("Qwen/Qwen3-4B-Instruct-2507")
    elif family == "gemma4":
        candidates.extend(
            [
                "google/gemma-4-E4B-it",
                "google/gemma-2-2b-it",
            ]
        )
    elif family == "gemma":
        candidates.append("google/gemma-2-2b-it")

    return list(dict.fromkeys(candidates))


LONGBENCH_TASKS = [
    "longbench/qasper",
    "longbench/multifieldqa_en",
    "longbench/2wikimqa",
]

LONGBENCH_E_TASKS = [
    "longbench/qasper_e",
    "longbench/multifieldqa_en_e",
    "longbench/2wikimqa_e",
]

DS_KWARGS = {
    "pwc": dict(
        train=dict(path="sggetao/PwC", split="train"),
        validation=dict(path="sggetao/PwC", split="test[:900]"),
        test=dict(path="sggetao/PwC", split="test"),
    ),
    "pwc_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/pwc_compact/train/ds.parquet",
            split="train",
        ),
    ),
    "pwc_compact_tiny": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/pwc_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train[60:200]",
        ),
    ),
    "pwc_tiny": dict(
        train=dict(path="sggetao/PwC", split="train[900:2000]"),
        validation=dict(path="sggetao/PwC", split="train[:900]"),
    ),
    "squad": dict(
        train=dict(path="data/raw_datasets/squad", split="train"),
        validation=dict(path="data/raw_datasets/squad", split="validation[:1000]"),
        test=dict(path="data/raw_datasets/squad", split="validation"),
    ),
    "squad_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/squad_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train[180:]",
        ),
    ),
    "squad_negative": dict(
        test=dict(path="data/raw_datasets/squad", split="validation"),
    ),
    "squad_assistant_ctx": dict(
        test=dict(path="data/raw_datasets/squad", split="validation"),
    ),
    "squad_negative_no_passage": dict(
        test=dict(path="data/raw_datasets/squad", split="validation"),
    ),
    "squad_assistant_ctx_no_passage": dict(
        test=dict(path="data/raw_datasets/squad", split="validation"),
    ),
    "drop": dict(
        train=dict(
            path="ucinlp/drop",
            split="train",
        ),
        validation=dict(
            path="ucinlp/drop",
            split="validation[:900]",
        ),
        test=dict(
            path="ucinlp/drop",
            split="validation",
        ),
    ),
    "drop_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/drop_compact/train/ds.parquet",
            # correspond to skipping to first 900 samples in the original dataset
            split="train",
        )
    ),
    "ropes": dict(
        train=dict(
            path="allenai/ropes",
            split="train",
        ),
        validation=dict(
            path="allenai/ropes",
            split="validation[:900]",
        ),
        test=dict(path="allenai/ropes", split="validation"),
    ),
    "ropes_compact": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/ropes_compact/train/ds.parquet",
            split="train",
        )
    ),
}

# add ctx_numbers
tok_bins = [(64, 128), (128, 256), (256, 512)] + [
    (512 + 256 * i, 512 + 256 * (i + 1)) for i in range(14)
]
tok_bins += [(32, 128), (128, 256), (256, 512), (512, 1024), (32, 1024)] + [
    (1024 * i, 1024 * (i + 1)) for i in range(1, 16)
]
tok_bins += [(2**14 + 2**12 * (i), 2**14 + 2**12 * (i + 1)) for i in range(4)]
tok_bins += [(2**15 + 2**13 * (i), 2**15 + 2**13 * (i + 1)) for i in range(12)]
for toy_ds_name in ["ctx_numbers", "ctx_kv", "ctx_magic_number"]:
    for tok_bin in tok_bins:
        DS_KWARGS[f"{toy_ds_name}_{tok_bin[0]}_{tok_bin[1]}"] = dict(
            train=dict(
                path="json",
                data_files=f"data/raw_datasets/{toy_ds_name}_{tok_bin[0]}_{tok_bin[1]}/train.jsonl",
                split="train",
            ),
            validation=dict(
                path="json",
                data_files=f"data/raw_datasets/{toy_ds_name}_{tok_bin[0]}_{tok_bin[1]}/val.jsonl",
                split="train",
            ),
            test=dict(
                path="json",
                data_files=f"data/raw_datasets/{toy_ds_name}_{tok_bin[0]}_{tok_bin[1]}/test.jsonl",
                split="train",
            ),
        )

# LongBench kwargs
for ds_name in LONGBENCH_TASKS + LONGBENCH_E_TASKS:
    DS_KWARGS[ds_name] = dict(
        test=dict(
            path="THUDM/LongBench",
            name=ds_name.split("/")[-1],
            split="test",
        )
    )


CLOSED_QA_DATASETS = {
    "longbench/qasper",
    "longbench/multifieldqa_en",
    "longbench/2wikimqa",
    "squad",
    "squad_negative",
    "squad_assistant_ctx",
    "squad_negative_no_passage",
    "squad_assistant_ctx_no_passage",
    "ropes",
    "drop",
}

MULTI_ANSWER_DATASETS = {
    "longbench/qasper",
    "longbench/multifieldqa_en",
    "longbench/hotpotqa",
    "longbench/2wikimqa",
    "squad",
    "squad_negative",
    "squad_assistant_ctx",
    "squad_negative_no_passage",
    "squad_assistant_ctx_no_passage",
    "drop",
}


for ds_name in list(CLOSED_QA_DATASETS):
    if ds_name.startswith("longbench/"):
        CLOSED_QA_DATASETS.add(f"{ds_name}_e")


for ds_name in list(MULTI_ANSWER_DATASETS):
    if ds_name.startswith("longbench/"):
        MULTI_ANSWER_DATASETS.add(f"{ds_name}_e")

# for training closed qa datasets, e.g., hotpot_qa, squad, etc.
CLOSED_QA_INTX_TEMPLATES = [
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}",
    "Answer without any explanation.\n\nQuestion: {input}",
    "Based on the provided text, what is the answer to the following question? Provide only the answer.\n\nQuestion: {input}",
    "Extract the answer to the question from the text. Be concise. Do not explain.\n\nQuestion: {input}",
    "What is the answer to this question, based on the context? Respond with the answer only.\n\nQuestion: {input}",
    "Provide a direct answer to the question using the given passages. Do not give any explanation.\n\nQuestion: {input}",
    "Answer the question using only information from the provided text. No extra words.\n\nQuestion: {input}",
    "From the passages, answer the question. Just the answer, please.\n\nQuestion: {input}",
    "Give the answer to the question. Do not include any other text.\n\nQuestion: {input}",
    "The answer to the question is in the text. Find it and state it clearly. No need for explanation.\n\nQuestion: {input}",
    "Concisely answer the question based on the text provided. Don't include any other words. Just the answer.\n\nQuestion: {input}",
    "Read the passages and answer the question with the minimal necessary words.\n\nQuestion: {input}",
    "What is the direct response to the question, according to the text? Avoid explanation.\n\nQuestion: {input}",
    "Please provide only the answer to the question, derived from the text.\n\nQuestion: {input}",
    "Using the provided context, answer the question. Output the answer and nothing else.\n\nQuestion: {input}",
    "Identify the answer in the text and present it without elaboration.\n\nQuestion: {input}",
    "Answer the following question based on the text. Your answer should be brief and to the point. No explanation.\n\nQuestion: {input}",
    "Based on the information given, what is the answer to the question? Only state the answer.\n\nQuestion: {input}",
    "Find the answer to the question in the provided passages and write it down. No explanations.\n\nQuestion: {input}",
    "The question is: {input}. Provide the answer based on the text, and nothing more.",
    "Question: {input}\nAnswer directly based on the text provided. No extra words.",
    "Question: {input}\nPlease provide the answer based on the text. No explanation is needed.",
]


EVAL_INTX_TEMPLATES = {
    # binary (yes/no, a/b) qa given ctx
    "ropes": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    # short-ctx reasoning
    "drop": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    # short-ctx extractive qa
    "squad": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "squad_negative": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "squad_assistant_ctx": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "squad_negative_no_passage": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    "squad_assistant_ctx_no_passage": "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {input}",
    # longbench
    "longbench/qasper": 'Answer the question as concisely as you can, using a single phrase or sentence if possible.\nIf the question cannot be answered based on the information in the article, write "unanswerable".\nIf the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}',
    "longbench/multifieldqa_en": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
    "longbench/2wikimqa": "Answer the following question. Only output the answer and do not output any other words.\n\nQuestion: {input}",
}
for ds_name in LONGBENCH_E_TASKS:
    EVAL_INTX_TEMPLATES[ds_name] = EVAL_INTX_TEMPLATES[ds_name[:-2]]
