import collections
import json
import logging
import math
import os
import random
import re
import string
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from ctx_to_lora.model_loading import get_model
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

logger = logging.getLogger()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cuda.matmul.allow_tf32 = True


SHINE_QWEN_CHAT_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\\n\\n' }}
    {%- endif %}
    {{- "# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\"name\\": <function-name>, \\"arguments\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>\\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if (loop.last or (not loop.last and reasoning_content)) and (enable_thinking is not defined or enable_thinking != false) %}
                {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
    {%- if enable_thinking is not defined or enable_thinking != false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}
{%- endif %}"""


def set_shine_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# SHINE metrics, copied semantically from SHINE/calculate_f1.py and
# SHINE/evaluation/hotpotqa.py.
# ---------------------------------------------------------------------------
def normalize_answer(s):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(a_gold, a_pred):
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def hotpotqa_compute_f1(ground_truth, prediction):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_sample_f1(ground_truth: Any, pred: str, f1_metric) -> float:
    if ground_truth is None:
        golds = [""]
    elif isinstance(ground_truth, str):
        golds = [ground_truth]
    elif isinstance(ground_truth, (list, tuple)):
        golds = []
        for x in ground_truth:
            if isinstance(x, str):
                golds.append(x)
            elif isinstance(x, dict):
                if "text" in x and isinstance(x["text"], str):
                    golds.append(x["text"])
                elif "answer" in x and isinstance(x["answer"], str):
                    golds.append(x["answer"])
                else:
                    golds.append(str(x))
            else:
                golds.append(str(x))
        if not golds:
            golds = [""]
    elif isinstance(ground_truth, dict):
        if "text" in ground_truth:
            text = ground_truth["text"]
            if isinstance(text, str):
                golds = [text]
            else:
                golds = [str(t) for t in text] if text else [""]
        elif "answers" in ground_truth and isinstance(ground_truth["answers"], dict):
            text = ground_truth["answers"].get("text", [])
            golds = [str(t) for t in text] if text else [""]
        else:
            golds = [str(ground_truth)]
    else:
        golds = [str(ground_truth)]

    best = 0.0
    for gold in golds:
        try:
            best = max(best, float(f1_metric(gold, pred or "")))
        except Exception:
            best = max(best, 0.0)
    return best


def exact_prefix_match_ratio(ref: list[int], hyp: list[int]) -> float:
    if len(ref) < len(hyp):
        raise ValueError(f"ref length must be >= hyp length, got {len(ref)} < {len(hyp)}")
    if len(ref) == 0:
        return 1.0
    n = 0
    for x, y in zip(ref, hyp):
        if x != y:
            break
        n += 1
    return n / len(ref)


def extract_think_and_answer_qa(text: str) -> tuple[str, str]:
    lower = text.lower()
    start_tag = "<think>"
    end_tag = "</think>"
    think = ""
    answer = text.strip()
    start = lower.find(start_tag)
    end = lower.find(end_tag)
    if start != -1 and end != -1 and end > start:
        think = text[start + len(start_tag) : end].strip()
        answer = text[end + len(end_tag) :].strip()
    else:
        answer = re.sub(
            r"<think>.*?</think>\s*", "", text, flags=re.IGNORECASE | re.DOTALL
        ).strip()
    answer = re.sub(
        r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE
    ).strip()
    if "\n" in answer:
        for line in answer.splitlines():
            if line.strip():
                answer = line.strip()
                break
    return think, answer


def extract_think_and_answer_strict(text: str) -> tuple[str, str]:
    lower = text.lower()
    has_start = "<think>" in lower
    has_end = "</think>" in lower
    if has_start != has_end:
        if text.startswith("<think>\n") or text.startswith("<think>\n\n"):
            return "", text[len("<think>\n") :].strip()
        return "[error]", text
    if not has_start and not has_end:
        answer = text.strip()
        answer = re.sub(
            r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE
        ).strip()
        if "\n" in answer:
            for line in answer.splitlines():
                if line.strip():
                    answer = line.strip()
                    break
        return "", answer
    start = lower.find("<think>")
    end = lower.find("</think>")
    if end < start:
        return "[error]", text
    think = text[start + len("<think>") : end].strip()
    answer = text[end + len("</think>") :].strip()
    answer = re.sub(
        r"^(final answer|answer)\s*:\s*", "", answer, flags=re.IGNORECASE
    ).strip()
    if "\n" in answer:
        for line in answer.splitlines():
            if line.strip():
                answer = line.strip()
                break
    return think, answer


def load_shine_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="left",
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def resize_model_token_embeddings(model, vocab_size: int):
    target = model.base_model if is_modulated_model(model) else model
    if hasattr(target, "resize_token_embeddings"):
        target.resize_token_embeddings(vocab_size)
    elif hasattr(target, "base_model") and hasattr(target.base_model, "resize_token_embeddings"):
        target.base_model.resize_token_embeddings(vocab_size)
    else:
        raise AttributeError(
            f"Cannot resize token embeddings for model type {type(model).__name__}"
        )


def apply_shine_recon_comp_tokenizer_setup(model, tokenizer):
    tokenizer.add_tokens(["<RECON>", "<COMP>"])
    tokenizer.chat_template = SHINE_QWEN_CHAT_TEMPLATE
    resize_model_token_embeddings(model, len(tokenizer))
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id


# ---------------------------------------------------------------------------
# SHINE datasets.
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": str(self.texts[idx])}


class GroupedSquadDataset(Dataset):
    def __init__(
        self,
        data,
        tokenizer,
        context_len: int | None = None,
        sep: str = "<|endoftext|>",
        name: str = "Test",
        seed: int = 42,
    ):
        self.name = f"[GroupedSquadDataset: {name}]"
        self.tokenizer = tokenizer
        self.sep = sep
        self.context_len = context_len
        self.data = data
        self.seed = seed
        self.shuffle()

    def shuffle(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        text_to_idx = defaultdict(list)
        for i, ex in enumerate(self.data):
            ctx = str(ex["context"]).strip()
            text_to_idx[ctx].append(i)

        all_context_list = deepcopy(list(text_to_idx.keys()))
        self.text_to_idx = text_to_idx
        if self.context_len is None or self.context_len <= 0:
            self.groups = [[ctx] for ctx in all_context_list]
        else:
            num_tokens = len(self.tokenizer(self.sep.join(all_context_list))["input_ids"])
            num_groups = (num_tokens + self.context_len - 1) // self.context_len
            random.shuffle(all_context_list)
            context_list_per_group = np.array_split(all_context_list, num_groups)
            self.groups = [[str(s) for s in arr] for arr in context_list_per_group]

        self.idx_to_groupidx = {}
        for group_idx, ctx_list in enumerate(self.groups):
            for ctx in ctx_list:
                for ex_idx in text_to_idx[ctx]:
                    self.idx_to_groupidx[ex_idx] = group_idx

        self.group_token_num = []
        for group in self.groups:
            token_num = len(self.tokenizer(self.sep.join(group))["input_ids"])
            self.group_token_num.append(token_num)

        print(f"{self.name}: {len(self.groups)} groups created from {len(self.data)} examples.")
        print(
            f"{self.name}: Average context token length: {np.mean(self.group_token_num):.2f}, "
            f"Max context token length: {np.max(self.group_token_num)}, "
            f"Min context token length: {np.min(self.group_token_num)}"
        )
        print(
            f"{self.name}: Top 20 largest context token lengths: "
            f"{sorted(self.group_token_num, reverse=True)[:20]}"
        )
        print(f"{self.name}: Average contexts per group: {len(self.data) / len(self.groups):.2f}")
        for group in self.groups:
            for ctx in group:
                assert not ctx.startswith(" "), f"Context has leading space: '{ctx}'"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        group = self.groups[self.idx_to_groupidx[idx]]
        evidence = self.sep.join(list(random.sample(group, len(group))))
        answer = [str(ans).strip() for ans in self.data[idx]["answers"]["text"]]
        for i in range(len(answer)):
            if answer[i][0].islower():
                answer[i] = answer[i][0].upper() + answer[i][1:]
        return {
            "evidence": str(evidence).strip(),
            "question": str(self.data[idx]["question"]).strip(),
            "answer": answer,
        }


class HotpotqaDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        contest_list = []
        item = self.data[index]
        for sentences in item["context"]["sentences"]:
            contest_list.append("".join(sentences))
        context = "\n\n".join(contest_list)
        return {
            "evidence": context,
            "question": item["question"].strip(),
            "answer": [item["answer"].strip()],
        }


class MusiqueDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        context_list = [p["paragraph_text"] for p in item["paragraphs"]]
        answer_aliases = [t.strip() for t in item["answer_aliases"]]
        return {
            "evidence": "\n\n".join(context_list),
            "question": item["question"].strip(),
            "answer": [item["answer"].strip()] + answer_aliases,
        }


class MsmarcoDataset(Dataset):
    def __init__(self, data):
        data = list(data)
        new_data = []
        for item in data:
            passages = item["passages"]
            if sum(passages["is_selected"]) == 0 or len(item["answers"]) == 0:
                continue
            new_data.append(item)
        self.data = new_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        passages = item["passages"]
        context = "\n\n".join(
            [passages["passage_text"][i] for i in range(len(passages["passage_text"]))]
        )
        return {
            "evidence": context,
            "question": item["query"].strip(),
            "answer": item["answers"],
        }


class MQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        contexts = item["context"]
        conversations = item["conversations"]
        final_conversations = []
        for conv in conversations:
            final_conversations.extend(
                [
                    {"role": "user", "content": conv["question"]},
                    {"role": "assistant", "content": conv["answer"]},
                ]
            )
        questions = []
        answers = []
        for conv in conversations:
            questions.append(conv["question"])
            answers.append(conv["answer"])
        return {
            "evidence": contexts,
            "conversations": final_conversations,
            "questions": questions,
            "answers": answers,
        }


# ---------------------------------------------------------------------------
# SHINE collators.
# ---------------------------------------------------------------------------
@dataclass
class BaseCollator:
    tokenizer: Any
    cfg: Any
    context_max_length: int = 1024
    conversation_max_length: int = 1024

    def __post_init__(self):
        if hasattr(self.cfg, "pretrain"):
            self.completion_freq = self.cfg.pretrain.completion_freq
            self.max_completion_ratio = self.cfg.pretrain.max_completion_ratio
            self.min_completion_ratio = self.cfg.pretrain.min_completion_ratio
        self.thinkend_token_id = self.tokenizer.convert_tokens_to_ids("</think>")
        self.eot = "<|endoftext|>"
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids("assistant")
        self.imstart_token_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.imend_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.SYSTEM_PROMPT = (
            "You are a concise assistant. Output only the final answer, in a few words, "
            "as short as possible. No explanations. Do not output anything else."
        )

    def mask_label(self, labels):
        masks = torch.zeros_like(labels)
        for i, ids in enumerate(labels):
            last_imend = self.conversation_max_length
            for j in range(len(ids) - 1, 0, -1):
                if ids[j].item() == self.imend_token_id:
                    last_imend = j
                elif ids[j].item() == self.assistant_token_id and ids[j - 1] == self.imstart_token_id:
                    masks[i, j + 2 : last_imend + 2] = 1
        labels = labels.masked_fill(masks == 0, -100)
        return labels


@dataclass
class SquadCollator(BaseCollator):
    use_reference: bool = False
    metatrain: bool = False
    only_question: bool = False
    thinkend_token_id: int | None = None

    def __call__(self, batch):
        questions = [ex["question"] for ex in batch]
        evidences = [ex["evidence"] for ex in batch]
        assert isinstance(batch[0]["answer"], list), "Answers should be a list of possible answers."
        answers = [str(random.choice(ex["answer"])) for ex in batch]
        full_answers = [ex["answer"] for ex in batch]

        evidence_enc = self.tokenizer(
            evidences,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_enc = self.tokenizer(
            answers,
            max_length=self.conversation_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )

        if self.metatrain:
            messages = [
                [
                    {"role": "user", "content": f"{question}"},
                    {"role": "assistant", "content": f"{answer}"},
                ]
                for question, answer in zip(questions, answers)
            ]
        elif self.use_reference:
            messages = [
                ([{"role": "system", "content": f"{self.SYSTEM_PROMPT}"}] if self.SYSTEM_PROMPT is not None else [])
                + [
                    {
                        "role": "user",
                        "content": f"Reference:\n{evidence}\n\nBased on the reference, answer this question:\n{question}",
                    },
                ]
                for evidence, question in zip(evidences, questions)
            ]
        elif self.only_question:
            messages = [
                ([{"role": "system", "content": f"{self.SYSTEM_PROMPT}"}] if self.SYSTEM_PROMPT is not None else [])
                + [{"role": "user", "content": f"{question}"}]
                for question in questions
            ]
        else:
            messages = [[{"role": "user", "content": f"{question}"}] for question in questions]

        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True if not self.metatrain else False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length + 4
            if (not self.metatrain and not self.use_reference and not self.only_question)
            else self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        input_ids = input_enc["input_ids"]
        input_attention_mask = input_enc["attention_mask"]
        labels = None
        if self.metatrain:
            labels = input_ids.clone()
            labels = self.mask_label(labels)
        elif not self.use_reference and not self.only_question:
            input_ids = input_ids[:, :-4]
            input_attention_mask = input_attention_mask[:, :-4]

        return {
            "evidence": evidences,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "messages": messages,
            "input_ids": input_ids,
            "labels": labels,
            "input_attention_mask": input_attention_mask,
            "answers": answers,
            "full_answers": full_answers,
            "answer_ids": answer_enc["input_ids"],
            "answer_attention_mask": answer_enc["attention_mask"],
            "questions": questions,
        }


@dataclass
class TestPretrainCollator(BaseCollator):
    metatrain: bool = False
    mode: str = "recon"

    def split_text(self, text):
        t = text.split()
        if len(t) < 2:
            return text, "Nothing to complete."
        ratio = 1.0 - random.uniform(self.min_completion_ratio, self.max_completion_ratio)
        split_index = round(len(t) * ratio)
        left = t[:split_index]
        right = t[split_index:]
        if not right:
            left, right = t[:-1], t[-1:]
        elif not left:
            left, right = t[:1], t[1:]
        return " ".join(left), " ".join(right)

    def __call__(self, batch):
        texts = [ex["text"] for ex in batch]
        if self.mode == "comp":
            splits = [self.split_text(text) for text in texts]
            evidence_texts = [split[0] for split in splits]
            answer_texts = texts
            messages = [[{"role": "user", "content": "<COMP>"}] for _ in answer_texts]
            label_messages = [
                [
                    {"role": "user", "content": "<COMP>"},
                    {"role": "assistant", "content": f"{answer}"},
                ]
                for answer in answer_texts
            ]
        elif self.mode == "recon":
            evidence_texts = texts
            answer_texts = texts
            messages = [[{"role": "user", "content": "<RECON>"}] for _ in answer_texts]
            label_messages = [
                [
                    {"role": "user", "content": "<RECON>"},
                    {"role": "assistant", "content": f"{answer}"},
                ]
                for answer in answer_texts
            ]
        else:
            raise NotImplementedError(f"mode {self.mode} is not implemented in TestPretrainCollator.")

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        answer_enc = self.tokenizer(
            answer_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            max_length=9,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        label_enc = self.tokenizer.apply_chat_template(
            label_messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        labels = self.mask_label(label_enc["input_ids"])
        return {
            "evidence": texts,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": input_enc["input_ids"],
            "input_attention_mask": input_enc["attention_mask"],
            "full_input_ids": label_enc["input_ids"],
            "full_input_attention_mask": label_enc["attention_mask"],
            "labels": labels,
            "answers": texts,
            "answer_ids": answer_enc["input_ids"],
            "answer_attention_mask": answer_enc["attention_mask"],
            "questions": [f"{self.mode}"] * len(texts),
        }


@dataclass
class MQACollator(BaseCollator):
    sys_msg: bool = False
    no_evidence: bool = False

    def __call__(self, batch):
        evidence_texts = [t["evidence"] for t in batch]
        conversation_texts = [t["conversations"] for t in batch]
        if self.sys_msg:
            if self.no_evidence:
                prompt = (
                    "You are a helpful assistant. Answer the question concisely with short words or phrases. "
                    "Answer the question directly and output nothing else. Never say you don't know the answer. "
                    "Never enter think mode."
                )
                messages = [
                    [{"role": "system", "content": f"{prompt}"}] + conversation
                    for conversation in conversation_texts
                ]
                initial_messages = [{"role": "system", "content": f"{prompt}"} for _ in conversation_texts]
            else:
                prompt = (
                    "You are a helpful assistant, answer the questions based on the given context. Each answer "
                    "must be directly extractable from the context (i.e., an exact span or minor paraphrase for "
                    "fluency). Do not invent information. Answer the question directly and output nothing else. "
                    "Never enter think mode.\n\nContext: "
                )
                messages = [
                    [{"role": "system", "content": f"{prompt}{evidence}"}] + conversation
                    for evidence, conversation in zip(evidence_texts, conversation_texts)
                ]
                initial_messages = [
                    {"role": "system", "content": f"{prompt}{evidence}"}
                    for evidence in evidence_texts
                ]
        else:
            messages = [conversation for conversation in conversation_texts]
            initial_messages = [{} for _ in conversation_texts]

        evidence_enc = self.tokenizer(
            evidence_texts,
            max_length=self.context_max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        input_enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors="pt",
            max_length=self.conversation_max_length,
            truncation=True,
            return_dict=True,
            padding="max_length",
            enable_thinking=False,
        )
        labels = self.mask_label(input_enc["input_ids"].clone())
        return {
            "initial_messages": initial_messages,
            "evidence": evidence_texts,
            "evidence_ids": evidence_enc["input_ids"],
            "evidence_attention_mask": evidence_enc["attention_mask"],
            "input_ids": input_enc["input_ids"],
            "labels": labels,
            "input_attention_mask": input_enc["attention_mask"],
            "questions": [b["questions"] for b in batch],
            "answers": [b["answers"] for b in batch],
        }


# ---------------------------------------------------------------------------
# Model adapter: only adapts doc-to-lora model calls to SHINE batch semantics.
# ---------------------------------------------------------------------------
def load_eval_model(
    checkpoint_path: str | None = None,
    model_name_or_path: str | None = None,
    gen_lora_scaling: float = 1.0,
):
    if bool(checkpoint_path) == bool(model_name_or_path):
        raise ValueError("Provide exactly one of checkpoint_path or model_name_or_path.")

    if checkpoint_path:
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "pytorch_model.bin")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
        state_dict = torch.load(checkpoint_path, weights_only=False)
        model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=False,
            use_sequence_packing=False,
            user_defined_scaling=gen_lora_scaling,
        )
        tokenizer = load_shine_tokenizer(model.base_model.name_or_path)
    else:
        model = get_model(
            model_name_or_path,
            train=False,
            requires_grad=False,
        )
        tokenizer = load_shine_tokenizer(model_name_or_path)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "patch_lora_forward"):
        model.patch_lora_forward()
    return model, tokenizer


def model_device(model):
    if hasattr(model, "device"):
        return torch.device(model.device)
    return next(model.parameters()).device


def is_modulated_model(model):
    return isinstance(model, ModulatedPretrainedModel)


def move_batch(batch, device, keys):
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def reset_lora_hooks(model):
    if hasattr(model, "reset"):
        model.reset()
    if hasattr(model, "patch_lora_forward"):
        model.patch_lora_forward()


def generate_with_context(model, input_ids, attention_mask, evidence_ids, evidence_attention_mask, **gen_kwargs):
    if is_modulated_model(model):
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ctx_ids=evidence_ids,
            ctx_attn_mask=evidence_attention_mask,
            n_ctx_chunks=torch.ones(evidence_ids.shape[0], dtype=torch.int32, device=evidence_ids.device),
            **gen_kwargs,
        )
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)


def generate_without_context(model, input_ids, attention_mask, **gen_kwargs):
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs)


def forward_with_context(model, input_ids, attention_mask, labels, evidence_ids, evidence_attention_mask):
    if is_modulated_model(model):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ctx_ids=evidence_ids,
            ctx_attn_mask=evidence_attention_mask,
            n_ctx_chunks=torch.ones(evidence_ids.shape[0], dtype=torch.int32, device=evidence_ids.device),
        )
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# ---------------------------------------------------------------------------
# SHINE/test.py.
# ---------------------------------------------------------------------------
@torch.no_grad()
def test_qa_and_save(
    cfg,
    model,
    tokenizer,
    testloader,
    split_name: str,
    f1_metric,
    output_suffix: str = ".json",
):
    device = model_device(model)
    model.eval()
    out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    final_out_path = os.path.join(out_dir, f"{split_name}{output_suffix}")
    rank_tmp_path = os.path.join(out_dir, f"{split_name}.rank0.jsonl")
    results_out_path = os.path.join(out_dir, f"{split_name}_results.json")
    os.makedirs(out_dir, exist_ok=True)

    start_sample_idx = 0
    if os.path.exists(rank_tmp_path):
        with open(rank_tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "sample_idx" in rec:
                    start_sample_idx = max(start_sample_idx, rec["sample_idx"] + 1)

    tmp_f = open(rank_tmp_path, "a", encoding="utf-8")
    sample_idx = 0
    for batch_idx, batch in enumerate(testloader):
        batch_size = len(batch["questions"])
        if sample_idx + batch_size <= start_sample_idx:
            sample_idx += batch_size
            continue
        print(f"[Rank 0] Processing batch {batch_idx + 1}/{len(testloader)}...")

        tensors = move_batch(
            batch,
            device,
            ["evidence_ids", "evidence_attention_mask", "input_ids", "input_attention_mask"],
        )
        gen_out = generate_with_context(
            model,
            tensors["input_ids"],
            tensors["input_attention_mask"],
            tensors["evidence_ids"],
            tensors["evidence_attention_mask"],
            max_new_tokens=cfg.test.max_new_tokens,
            do_sample=False,
        )

        input_lens = tensors["input_attention_mask"].sum(dim=1).tolist()
        gen_out = gen_out.to("cpu")
        input_ids_cpu = tensors["input_ids"].to("cpu")

        for i in range(gen_out.size(0)):
            if sample_idx < start_sample_idx:
                sample_idx += 1
                continue

            full_text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            input_text = tokenizer.decode(
                input_ids_cpu[i][-input_lens[i] :], skip_special_tokens=True
            )
            if full_text.startswith(input_text):
                answer_text = full_text[len(input_text) :]
            else:
                answer_text = full_text
            think, answer = extract_think_and_answer_qa(answer_text)
            gt = batch["full_answers"][i]
            f1_val = compute_sample_f1(gt, answer, f1_metric)

            record = {
                "sample_idx": sample_idx,
                "evidence": batch["evidence"][i],
                "input": input_text,
                "question": batch["questions"][i],
                "think": think,
                "answer": answer,
                "ground_truth": gt,
                "f1": f1_val,
            }
            tmp_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_f.flush()
            sample_idx += 1

    tmp_f.close()
    local_results = []
    with open(rank_tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                local_results.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    local_results.sort(key=lambda x: x.get("sample_idx", 0))
    f1_vals = [
        float(rec["f1"])
        for rec in local_results
        if "f1" in rec and isinstance(rec["f1"], (int, float))
    ]
    avg_f1 = float(sum(f1_vals) / max(1, len(f1_vals)))
    for rec in local_results:
        rec.pop("sample_idx", None)
    with open(final_out_path, "w", encoding="utf-8") as f:
        json.dump(local_results, f, ensure_ascii=False, indent=2)
    with open(results_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset": split_name, "num_samples": len(local_results), "avg_f1": avg_f1},
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_qa_dataset(source, tokenizer, context_avg_len):
    if source == "squad":
        data = load_dataset(os.path.join("data", "squad"), split="validation")
        data = data.shuffle(seed=42)
        subset = data.select(range(1000))
        return [f"squad_{context_avg_len}"], [GroupedSquadDataset(subset, tokenizer, context_avg_len)], compute_f1
    if source == "hotpotqa":
        data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        data = data.shuffle(seed=42)
        subset = data.select(range(1000))
        return ["hotpotqa"], [HotpotqaDataset(subset)], hotpotqa_compute_f1
    if source == "musique":
        data = load_dataset("dgslibisey/MuSiQue", split="validation")
        data = data.shuffle(seed=42)
        subset = data.select(range(1000))
        return ["musique"], [MusiqueDataset(subset)], compute_f1
    if source == "2wikimultihopqa":
        data = load_dataset("framolfese/2WikiMultihopQA", split="validation")
        data = data.shuffle(seed=42)
        subset = data.select(range(1000))
        return ["2wikimultihopqa"], [HotpotqaDataset(subset)], hotpotqa_compute_f1
    if source == "msmarco_v1":
        data = load_dataset("microsoft/ms_marco", "v1.1", split="test")
        data = data.shuffle(seed=42)
        subset = data.select(range(1000))
        return ["msmarco_v1"], [MsmarcoDataset(subset)], compute_f1
    if source == "msmarco_v2":
        data = load_dataset("microsoft/ms_marco", "v2.1", split="validation")
        data = data.shuffle(seed=42)
        subset = data.select(range(1000))
        return ["msmarco_v2"], [MsmarcoDataset(subset)], compute_f1
    raise ValueError(f"Unknown data source: {source}")


def evaluate_qa(model, tokenizer, source, output_dir, batch_size=4, num_workers=0, context_avg_len=512, context_max_length=1300, conversation_max_length=128, max_new_tokens=128):
    set_shine_seed(42)
    cfg = make_cfg(
        source=source,
        save_path=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        context_avg_len=context_avg_len,
        context_max_length=context_max_length,
        conversation_max_length=conversation_max_length,
        max_new_tokens=max_new_tokens,
    )
    names, datasets, f1_metric = build_qa_dataset(source, tokenizer, context_avg_len)
    collator = SquadCollator(
        tokenizer=tokenizer,
        cfg=cfg,
        context_max_length=context_max_length,
        conversation_max_length=conversation_max_length,
    )
    pin = model_device(model).type == "cuda"
    for name, ds in zip(names, datasets):
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            pin_memory=pin,
            num_workers=num_workers,
            persistent_workers=pin and num_workers > 0,
        )
        test_qa_and_save(cfg, model, tokenizer, loader, name, f1_metric)


# ---------------------------------------------------------------------------
# SHINE/test_pretrain.py.
# ---------------------------------------------------------------------------
@torch.no_grad()
def test_pretrain_and_save(cfg, model, tokenizer, testloader, split_name, output_suffix=".json"):
    device = model_device(model)
    model.eval()
    out_dir = os.path.join(cfg.test.save_path, cfg.test.source)
    final_out_path = os.path.join(out_dir, f"{split_name}{output_suffix}")
    rank_tmp_path = os.path.join(out_dir, f"{split_name}.rank0.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(final_out_path):
        logger.info("[SKIP] Found existing %s", final_out_path)
        return

    start_sample_idx = 0
    if os.path.exists(rank_tmp_path):
        with open(rank_tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "sample_idx" in rec:
                    start_sample_idx = max(start_sample_idx, rec["sample_idx"] + 1)

    tmp_f = open(rank_tmp_path, "a", encoding="utf-8")
    sample_idx = 0
    for batch_idx, batch in enumerate(testloader):
        batch_size = len(batch["questions"])
        if sample_idx + batch_size <= start_sample_idx:
            sample_idx += batch_size
            continue
        print(f"[Rank 0] Processing batch {batch_idx + 1}/{len(testloader)}...")
        tensors = move_batch(
            batch,
            device,
            [
                "evidence_ids",
                "evidence_attention_mask",
                "input_ids",
                "input_attention_mask",
                "labels",
                "full_input_ids",
                "full_input_attention_mask",
            ],
        )
        outputs = forward_with_context(
            model,
            tensors["full_input_ids"],
            tensors["full_input_attention_mask"],
            tensors["labels"],
            tensors["evidence_ids"],
            tensors["evidence_attention_mask"],
        )
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = tensors["labels"][:, 1:].contiguous()
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        loss_per_token = loss_per_token.view(shift_labels.size())
        mask = shift_labels != -100
        loss_per_token = loss_per_token * mask
        token_nums = mask[:, :-2].sum(dim=1)
        loss_per_sample = loss_per_token[:, :-2].sum(dim=1) / token_nums

        gen_out = generate_with_context(
            model,
            tensors["input_ids"],
            tensors["input_attention_mask"],
            tensors["evidence_ids"],
            tensors["evidence_attention_mask"],
            max_new_tokens=cfg.test.max_new_tokens,
            do_sample=False,
        )
        gen_out = gen_out[:, 9:].to("cpu")
        input_lens = tensors["input_attention_mask"].sum(dim=1).tolist()
        input_ids_cpu = tensors["input_ids"].to("cpu")

        for i in range(gen_out.size(0)):
            if sample_idx < start_sample_idx:
                sample_idx += 1
                continue
            full_text = tokenizer.decode(gen_out[i], skip_special_tokens=True)
            input_text = tokenizer.decode(
                input_ids_cpu[i][-input_lens[i] :], skip_special_tokens=True
            )
            think, answer = extract_think_and_answer_qa(full_text)
            t = int(token_nums[i].item())
            ref = batch["answer_ids"][i][-t:].tolist()
            hyp = gen_out[i][:t].tolist()
            em = exact_prefix_match_ratio(ref, hyp)
            record = {
                "sample_idx": sample_idx,
                "statistics": {
                    "length": len(ref),
                    "em": em,
                    "exact_prefix_match": em,
                },
                "loss": loss_per_sample[i].item(),
                "evidence": batch["evidence"][i],
                "input": input_text,
                "question": batch["questions"][i],
                "think": think,
                "answer": answer,
                "answer_ids": gen_out[i].tolist(),
                "ground_truth": batch["answers"][i],
                "ground_truth_ids": batch["answer_ids"][i].tolist(),
            }
            tmp_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            tmp_f.flush()
            sample_idx += 1

    tmp_f.close()
    merged = []
    with open(rank_tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                merged.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    merged.sort(key=lambda x: x.get("sample_idx", 0))
    for rec in merged:
        rec.pop("sample_idx", None)

    def _finite_float(x):
        return isinstance(x, (int, float)) and (not math.isnan(x)) and (not math.isinf(x))

    def _stat_pack(arr):
        if arr is None or arr.size == 0:
            nan = float("nan")
            return {"mean": nan, "std": nan, "median": nan, "p10": nan, "p90": nan}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
        }

    losses = [float(x) for x in [r.get("loss") for r in merged] if _finite_float(x)]
    loss_arr = np.asarray(losses, dtype=np.float64) if losses else None
    loss_stats = _stat_pack(loss_arr)
    ppl_arr = np.exp(loss_arr) if loss_arr is not None and loss_arr.size > 0 else None
    ppl_stats = _stat_pack(ppl_arr)
    exacts = []
    lens = []
    for r in merged:
        st = r.get("statistics", {}) or {}
        if _finite_float(st.get("em", st.get("exact_prefix_match"))):
            exacts.append(float(st.get("em", st.get("exact_prefix_match"))))
        if _finite_float(st.get("length")):
            lens.append(float(st["length"]))
    em_arr = np.asarray(exacts, dtype=np.float64) if exacts else None
    len_arr = np.asarray(lens, dtype=np.float64) if lens else None
    summary = {
        "num_samples": len(merged),
        "mean_loss": loss_stats["mean"],
        "std_loss": loss_stats["std"],
        "loss_statistics": loss_stats,
        "mean_statistics": {
            "length": float(np.mean(len_arr)) if len_arr is not None and len_arr.size > 0 else float("nan"),
            "em": float(np.mean(em_arr)) if em_arr is not None and em_arr.size > 0 else float("nan"),
            "exact_prefix_match": float(np.mean(em_arr)) if em_arr is not None and em_arr.size > 0 else float("nan"),
            "ppl": ppl_stats["mean"],
        },
        "std_statistics": {
            "length": float(np.std(len_arr, ddof=0)) if len_arr is not None and len_arr.size > 0 else float("nan"),
            "em": float(np.std(em_arr, ddof=0)) if em_arr is not None and em_arr.size > 0 else float("nan"),
            "exact_prefix_match": float(np.std(em_arr, ddof=0)) if em_arr is not None and em_arr.size > 0 else float("nan"),
            "ppl": ppl_stats["std"],
        },
        "ppl_statistics": ppl_stats,
    }
    with open(final_out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "predictions": merged}, f, ensure_ascii=False, indent=2)


def is_valid_article(text, min_english_words=5, max_non_ascii_ratio=0.001):
    if not text or not text.strip():
        return False
    english_words = re.findall(r"[A-Za-z]{2,}", text)
    if len(english_words) < min_english_words:
        return False
    total_chars = len(text)
    if total_chars == 0:
        return False
    non_ascii_chars = sum(ord(c) > 127 for c in text)
    return non_ascii_chars / total_chars <= max_non_ascii_ratio


def build_wikitext2_raw_articles(ds):
    title_re = re.compile(r"^\s*(=+)\s*([^=].*?)\s*\1\s*$")
    articles = []
    cur_title = None
    cur_lines = []

    def flush():
        nonlocal cur_title, cur_lines
        if cur_title is None:
            cur_lines = []
            return
        text = "\n".join(cur_lines).strip()
        if text and is_valid_article(text):
            articles.append({"title": cur_title, "text": text})
        cur_lines = []

    for line in ds["text"]:
        m = title_re.match(line)
        if m:
            flush()
            cur_title = m.group(2)
            continue
        cur_lines.append(line)
    flush()
    return articles


def visualize_2x2_icml(lens, out_dir, save_name="results_2x2.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [l * 100 for l in lens]

    def _finite(x):
        return isinstance(x, (int, float)) and (not math.isnan(x)) and (not math.isinf(x))

    def read_stat_pack(path, kind):
        nan = float("nan")
        if not os.path.exists(path):
            return {"mean": nan, "std": nan, "median": nan, "p10": nan, "p90": nan}
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        summary = obj.get("summary", {}) or {}
        if kind == "loss":
            pack = summary.get("loss_statistics")
            if isinstance(pack, dict):
                return {k: float(pack.get(k, nan)) for k in ["mean", "std", "median", "p10", "p90"]}
            return {
                "mean": float(summary.get("mean_loss", nan)),
                "std": float(summary.get("std_loss", nan)),
                "median": nan,
                "p10": nan,
                "p90": nan,
            }
        if kind == "ppl":
            pack = summary.get("ppl_statistics")
            if isinstance(pack, dict):
                return {k: float(pack.get(k, nan)) for k in ["mean", "std", "median", "p10", "p90"]}
            mean_stats = summary.get("mean_statistics", {}) or {}
            std_stats = summary.get("std_statistics", {}) or {}
            return {
                "mean": float(mean_stats.get("ppl", nan)),
                "std": float(std_stats.get("ppl", nan)),
                "median": nan,
                "p10": nan,
                "p90": nan,
            }
        return {"mean": nan, "std": nan, "median": nan, "p10": nan, "p90": nan}

    def collect_series(kind):
        keys = ["mean", "median", "p10", "p90"]
        recon = {k: [] for k in keys}
        comp = {k: [] for k in keys}
        for length in lens:
            recon_pack = read_stat_pack(os.path.join(out_dir, f"{length}_recon.json"), kind)
            comp_pack = read_stat_pack(os.path.join(out_dir, f"{length}_comp.json"), kind)
            for key in keys:
                recon[key].append(recon_pack[key])
                comp[key].append(comp_pack[key])
        return recon, comp

    recon_ppl, comp_ppl = collect_series("ppl")
    recon_loss, comp_loss = collect_series("loss")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 7,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
        }
    )
    fig, axs = plt.subplots(2, 2, figsize=(3.3, 4.2), constrained_layout=True)

    def plot_on_ax(ax, series, title, ylabel, ylim):
        line_1, = ax.plot(xs, series["mean"], marker="o", label="Mean", color="#1f77b4")
        line_2, = ax.plot(
            xs,
            series["median"],
            marker="^",
            label="Median",
            linestyle="--",
            color="#ff7f0e",
        )
        line_3, = ax.plot(xs, series["p10"], marker="", linestyle=":", label="P10", color="gray", alpha=0.6)
        ax.plot(xs, series["p90"], marker="", linestyle=":", label="P90", color="gray", alpha=0.6)
        lower = []
        upper = []
        for lo, hi in zip(series["p10"], series["p90"]):
            if _finite(lo) and _finite(hi):
                lower.append(lo)
                upper.append(hi)
            else:
                lower.append(float("nan"))
                upper.append(float("nan"))
        ax.fill_between(xs, lower, upper, color="gray", alpha=0.15)
        ax.set_title(title, pad=3)
        ax.set_xlabel("Context Length", labelpad=2)
        ax.set_ylabel(ylabel, labelpad=2)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xticks([100, 300, 500, 700, 900, 1100])
        ax.set_xlim(50, 1150)
        ax.tick_params(axis="x", which="major", pad=2)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
        ax.grid(True, linestyle="--", alpha=0.3)
        return [line_1, line_2, line_3]

    handles = plot_on_ax(axs[0, 0], recon_ppl, "Recon PPL", "PPL", (1.0, 3.0))
    plot_on_ax(axs[0, 1], comp_ppl, "Comp PPL", "PPL", (1.0, 3.0))
    plot_on_ax(axs[1, 0], recon_loss, "Recon Loss", "Loss", (0.0, 1.0))
    plot_on_ax(axs[1, 1], comp_loss, "Comp Loss", "Loss", (0.0, 1.0))
    fig.legend(
        handles,
        ["Mean", "Median", "P10/P90"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=3,
        frameon=False,
    )
    plt.savefig(os.path.join(out_dir, save_name), dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_wikitext_recon_comp(
    model,
    tokenizer,
    data_dir,
    output_dir,
    split="train",
    lengths=None,
    idx_dict_path=None,
    max_samples_per_length=-1,
    batch_size=4,
    num_workers=0,
    max_new_tokens=500,
):
    set_shine_seed(42)
    apply_shine_recon_comp_tokenizer_setup(model, tokenizer)
    lengths = lengths or list(range(1, 12))
    cfg = make_cfg(
        source="wikitext",
        save_path=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        context_max_length=1024,
        conversation_max_length=1024,
        max_new_tokens=max_new_tokens,
    )
    ds = load_dataset(data_dir, split=split)
    data = build_wikitext2_raw_articles(ds)
    with open(idx_dict_path or os.path.join(data_dir, "idx_dict.json"), "r", encoding="utf-8") as f:
        idx_dict = json.load(f)
    pin = model_device(model).type == "cuda"
    out_dir = os.path.join(cfg.test.save_path, cfg.test.source)

    for length in lengths:
        texts = [data[i]["text"].strip() for i in idx_dict[str(length)]]
        if max_samples_per_length > 0:
            texts = texts[:max_samples_per_length]
        dataset = TextDataset(texts)
        collators = (
            TestPretrainCollator(
                tokenizer=tokenizer,
                cfg=make_pretrain_cfg(cfg),
                context_max_length=length * 100 + 20,
                conversation_max_length=length * 100 + 31,
                mode="recon",
            ),
            TestPretrainCollator(
                tokenizer=tokenizer,
                cfg=make_pretrain_cfg(cfg),
                context_max_length=length * 100 + 20,
                conversation_max_length=length * 100 + 31,
                mode="comp",
            ),
        )
        for mode, collator in zip(("recon", "comp"), collators):
            final_path = os.path.join(out_dir, f"{length}_{mode}.json")
            if os.path.exists(final_path):
                logger.info("[SKIP] Found existing %s", final_path)
                continue
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collator,
                pin_memory=pin,
                num_workers=num_workers,
                persistent_workers=pin and num_workers > 0,
            )
            test_pretrain_and_save(cfg, model, tokenizer, loader, f"{length}_{mode}")
    visualize_2x2_icml(lengths, out_dir, save_name="results_2x2.png")


# ---------------------------------------------------------------------------
# SHINE/test_pwc.py active path: msmacro-mqa generate_multiturn.
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_multiturn(model, dataloader, tokenizer, max_new_tokens=500, max_conversation_length=3000):
    device = model_device(model)
    model.eval()
    results = []
    assert dataloader.batch_size == 1, "generate_multiturn only supports batch_size=1 for simplicity"

    for batch in dataloader:
        questions = batch["questions"][0]
        ground_truths = batch["answers"][0]
        messages = [batch["initial_messages"][0]] if batch["initial_messages"][0] is not {} else []
        evidence_ids = batch["evidence_ids"].to(device, non_blocking=True)
        evidence_attention_mask = batch["evidence_attention_mask"].to(device, non_blocking=True)
        if is_modulated_model(model):
            reset_lora_hooks(model)
            model._internalize_from_ids(evidence_ids, evidence_attention_mask)

        conversation_log = [{"initial message": deepcopy(messages)}]
        f1_scores = []
        error_count_local = 0

        for q_idx, question in enumerate(questions):
            messages.append({"role": "user", "content": question})
            input_enc = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                max_length=max_conversation_length,
                truncation=True,
                return_dict=True,
                padding="max_length",
                enable_thinking=False,
            )
            input_ids = input_enc["input_ids"].to(device)
            attention_mask = input_enc["attention_mask"].to(device)
            outputs = generate_without_context(
                model,
                input_ids,
                attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            new_tokens = outputs[0, input_ids.shape[1] :]
            think_answer_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            think_text, answer_text = extract_think_and_answer_strict(think_answer_text)
            if think_text == "[error]":
                error_count_local += 1
            messages.append({"role": "assistant", "content": answer_text})
            f1 = compute_f1(ground_truths[q_idx], answer_text)
            f1_scores.append(f1)
            conversation_log.append(
                {
                    "turn": q_idx + 1,
                    "question": question,
                    "think": think_text,
                    "answer": answer_text,
                    "ground_truth": ground_truths[q_idx],
                    "f1": f1,
                }
            )

        conversation_log[0]["avg_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        conversation_log[0]["error_count"] = error_count_local
        results.append(conversation_log)

        if is_modulated_model(model):
            reset_lora_hooks(model)

    total_samples = len(results)
    num_turns_expected = 15
    if total_samples == 0:
        return results, {
            "total_samples": 0,
            "total_turns": 0,
            "error_count": 0,
            "overall_avg_f1": 0.0,
            "per_turn_avg_f1": [0.0] * num_turns_expected,
            "per_turn_error_count": [0] * num_turns_expected,
        }

    turn_f1_sums = [0.0] * num_turns_expected
    turn_error_counts = [0] * num_turns_expected
    total_error_count = 0
    all_f1s = []
    for conv_log in results:
        turns = conv_log[1:]
        if len(turns) != num_turns_expected:
            continue
        for turn_idx in range(num_turns_expected):
            turn = turns[turn_idx]
            all_f1s.append(turn["f1"])
            turn_f1_sums[turn_idx] += turn["f1"]
            if turn["think"] == "[error]":
                turn_error_counts[turn_idx] += 1
                total_error_count += 1
    stats = {
        "total_samples": total_samples,
        "total_turns": len(all_f1s),
        "error_count": total_error_count,
        "overall_avg_f1": sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
        "per_turn_avg_f1": [turn_f1_sums[i] / total_samples for i in range(num_turns_expected)],
        "per_turn_error_count": turn_error_counts,
    }
    return results, stats


def evaluate_mqa_multiturn(
    model,
    tokenizer,
    data_path,
    output_dir,
    max_samples=-1,
    batch_size=1,
    num_workers=0,
    context_max_length=1024,
    conversation_max_length=1024,
    max_new_tokens=500,
    max_conversation_length=3000,
):
    set_shine_seed(42)
    apply_shine_recon_comp_tokenizer_setup(model, tokenizer)
    with open(data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    random.seed(42)
    random.shuffle(data)
    if max_samples > 0:
        data = data[:max_samples]
    dataset = MQADataset(data)
    cfg = make_cfg(
        source="msmacro-mqa",
        save_path=output_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        context_max_length=context_max_length,
        conversation_max_length=conversation_max_length,
        max_new_tokens=max_new_tokens,
    )
    collator = MQACollator(
        tokenizer=tokenizer,
        cfg=cfg,
        context_max_length=context_max_length,
        conversation_max_length=conversation_max_length,
    )
    pin = model_device(model).type == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        pin_memory=pin,
        num_workers=num_workers,
        persistent_workers=pin and num_workers > 0,
    )
    results, stats = generate_multiturn(
        model,
        loader,
        tokenizer,
        max_new_tokens=max_new_tokens,
        max_conversation_length=max_conversation_length,
    )
    out_dir = os.path.join(output_dir, "msmacro-mqa")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "generated_results.jsonl"), "w", encoding="utf-8") as f:
        for res in results:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "generation_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def make_cfg(**kwargs):
    test = SimpleNamespace(**kwargs)
    return SimpleNamespace(
        test=test,
        pretrain=SimpleNamespace(
            completion_freq=0.5,
            max_completion_ratio=0.3,
            min_completion_ratio=0.1,
        ),
    )


def make_pretrain_cfg(cfg):
    return SimpleNamespace(test=cfg.test, pretrain=cfg.pretrain)
