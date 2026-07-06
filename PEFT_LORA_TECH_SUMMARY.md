# SHINE + PEFT LoRA 技术概要

## 目标

本方案是在 SHINE 原有框架上加入 PEFT LoRA 微调能力，而不是重新设计新的 LoRA 融合算法。

核心目标是：

- Qwen/Qwen3 仍作为基础语言模型。
- M2P 仍作为 SHINE 的 hypernetwork / adapter generator。
- Qwen base model 冻结，只训练 Qwen 上的 PEFT LoRA。
- M2P 原始参数冻结，只训练 M2P 上新增的 PEFT LoRA。
- 原 SHINE 的动态 `metalora/loradict` 机制继续保留。

## 整体思路

整体思路是把 SHINE 中真正参与建模的两个核心组件分别做轻量可训练化：一部分是负责语言理解和生成的 Qwen，另一部分是负责根据上下文生成 adapter 参数的 M2P。由于 Qwen 本身已经适合通过 LoRA 继续微调，因此不对 Qwen base model 做全参更新，而是在原模型上挂载或加载 PEFT LoRA，只训练这部分新增低秩参数。

对于 M2P，也不重新训练整个 pretrained M2P，而是在它内部的 Linear 投影层上加入 PEFT LoRA。这样可以让 M2P 继续适配新任务，同时避免破坏原本已经学习到的 hypernetwork 能力。

因此，本方案的核心不是设计新的 LoRA 融合机制，而是把训练范围收敛到“Qwen LoRA + M2P LoRA + 原 SHINE metalora”这几个轻量参数集合上。这样既保留 SHINE 原有的动态 LoRA 生成流程，又能对 SHINE 用到的主要模型组件进行可控的继续微调。

## 整体结构

改造后的训练结构可以理解为：

```text
context / document
        |
        v
      Qwen + Qwen PEFT LoRA
        |
        v
   memory_states
        |
        v
      M2P + M2P PEFT LoRA
        |
        v
   generated loradict
        |
        v
      Qwen forward
        |
        v
      loss
```

其中：

- Qwen PEFT LoRA 用于继续微调语言模型本身。
- M2P PEFT LoRA 用于轻量微调 SHINE 的 adapter 生成模块。
- SHINE 原本生成的 `loradict` 仍然用于动态影响 Qwen 推理/训练过程。

## Qwen 上的 LoRA

Qwen 的处理方式是标准 PEFT LoRA：

- 加载 Qwen base model。
- 如果有已有 adapter，则加载已有 adapter 并继续训练。
- 如果没有已有 adapter，则初始化一个新的 LoRA adapter。
- 冻结 Qwen base model。
- 只开放 LoRA 参数训练。

默认 LoRA target modules：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

这覆盖了 Qwen Transformer 中 attention 和 MLP 的主要线性投影层。

## M2P 上的 LoRA

M2P 是 SHINE 中根据上下文生成 adapter / LoRA 参数的模块。

本方案没有重写 M2P，而是在 pretrained M2P 上加 PEFT LoRA：

- 加载原 M2P。
- 冻结 M2P 原始参数。
- 在 M2P 内部 Linear 层上挂 LoRA。
- 只训练 M2P LoRA 参数。

如果 M2P 是 Transformer 结构，优先考虑：

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

如果 M2P 是普通 MLP / Linear 结构，优先考虑：

```text
linear
fc1
fc2
proj
dense
linear1
linear2
out_proj
```

如果没有显式配置 target modules，则自动扫描 M2P 的 Linear 层。

## 和原 SHINE LoRA 的关系

这里有两套不同的 LoRA 概念：

1. SHINE 原有 `metalora/loradict`
   - 由 M2P 根据 context 动态生成。
   - 用于在 Qwen forward 时临时注入 LoRA 权重。
   - 是 SHINE 原始方法的核心。

2. 新增 PEFT LoRA
   - 挂在 Qwen 上，用于微调 Qwen。
   - 挂在 M2P 上，用于微调 M2P。
   - 是标准 PEFT adapter，可保存为 `adapter_config.json` 和 `adapter_model.safetensors`。

本次改造没有删除或替代 SHINE 原有 `metalora/loradict`，而是在 Qwen 和 M2P 外层增加标准 PEFT LoRA 微调能力。

## 训练时哪些参数会更新

启用 PEFT 后，参数训练范围如下：

| 模块 | 原始参数 | LoRA 参数 |
| --- | --- | --- |
| Qwen | 冻结 | 训练 |
| M2P | 冻结 | 训练 |
| SHINE metalora | 按原逻辑训练或冻结 | 不适用 |

也就是说，本方案允许同时训练：

- Qwen PEFT LoRA
- M2P PEFT LoRA
- 原 SHINE 训练流程中需要训练的 `metalora`

但不会训练 Qwen base model，也不会训练 M2P base 参数。

## 保存格式

原 SHINE checkpoint 继续保留：

```text
metanetwork.pth
metalora.pth
mem_tokens.pt
```

新增 PEFT adapter 保存目录：

```text
qwen_peft/
  adapter_config.json
  adapter_model.safetensors

m2p_peft/
  adapter_config.json
  adapter_model.safetensors
```

这样旧 checkpoint 仍可使用，新 checkpoint 也能单独保存和加载标准 PEFT adapter。

## 使用方式

默认不启用 PEFT，原 SHINE 行为不变。

启用 Qwen 和 M2P LoRA：

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.qwen.enabled=true \
  peft.m2p.enabled=true
```

继续训练已有 Qwen adapter：

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.qwen.enabled=true \
  peft.qwen.adapter_path=path/to/qwen_peft \
  peft.m2p.enabled=true
```

继续训练已有 M2P adapter：

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.qwen.enabled=true \
  peft.m2p.enabled=true \
  peft.m2p.adapter_path=path/to/m2p_peft
```

## 结论

本方案实现的是一个基础、稳定、可继续扩展的 PEFT LoRA 微调版本：

- 不引入复杂 LoRA fusion。
- 不引入 MoE-LoRA、LSH、MagicPIG 或动态 adapter 路由。
- 保留 SHINE 原始动态 LoRA 生成逻辑。
- 额外支持 Qwen 和 M2P 的标准 PEFT LoRA 微调。
