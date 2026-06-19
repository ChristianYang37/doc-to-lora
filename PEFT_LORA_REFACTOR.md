# SHINE PEFT LoRA 改造技术文档

## 1. 改造目标

本次改造的目标不是重新设计 SHINE 的 LoRA fusion 或动态路由算法，而是在 SHINE 原有代码结构上，加入一套可选的 PEFT LoRA 微调能力。

改造后支持两类 LoRA：

1. Qwen/Qwen3 上的 PEFT LoRA
   - 加载 Qwen base model。
   - 可选加载已有 Qwen PEFT adapter。
   - 支持在已有 Qwen adapter 基础上继续训练。
   - 冻结 Qwen base model，只训练 Qwen LoRA 参数。

2. M2P/metanetwork 上的 PEFT LoRA
   - 保留 SHINE 原有 pretrained M2P 结构。
   - 在 M2P 的 Linear 层上挂 PEFT LoRA。
   - 冻结 M2P 原始参数，只训练 M2P LoRA 参数。

默认情况下，PEFT 功能关闭。因此原 SHINE 的 `metalora/loradict` 训练、推理和 checkpoint 逻辑仍然保持兼容。

## 2. 原项目结构分析

### 2.1 Qwen/Qwen3 加载位置

训练入口位于：

- `meta_train_parallel.py`

核心加载逻辑在 `main(cfg)` 中：

```python
MetaModelCls = _import_class(cfg.model.metamodel_class_path)
ConfigCls = _import_class(cfg.model.config_class_path)
config = ConfigCls.from_pretrained(cfg.model.model_from)
metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
```

配置文件中默认使用：

```yaml
model:
  metamodel_class_path: "LoraQwen.LoraQwen3ForCausalLM"
  config_class_path: "LoraQwen.Qwen3Config"
  tokenizer_from: "${paths.model_path}"
  model_from: "${paths.model_path}"
```

也就是说，SHINE 不是直接使用 transformers 原生 `Qwen3ForCausalLM`，而是使用本地文件 `LoraQwen.py` 中定制过的 `LoraQwen3ForCausalLM`。

### 2.2 原 Qwen LoRA 机制

原项目的 LoRA 不是 PEFT 实现，而是自定义的动态 LoRA dictionary 机制。

关键文件：

- `LoraQwen.py`

关键类和函数：

- `LoraLinear`
- `LoraQwen3Attention`
- `LoraQwen3MLP`
- `LoraQwen3Model`
- `LoraQwen3ForCausalLM`
- `init_lora_dict`
- `generate_lora_dict`
- `lora_params_numel`

原始逻辑是：

1. `Metanetwork` 根据 evidence/context 得到 `memory_states`。
2. M2P/metanetwork 将 `memory_states` 转换成一段 `plain_output`。
3. Qwen 的 `generate_lora_dict()` 将 `plain_output` 拆成每层的 LoRA A/B 参数。
4. Qwen forward 时通过 `loradict` 临时注入这些 LoRA 权重。

这套逻辑仍然保留。

### 2.3 M2P/hypernetwork 定义位置

M2P 定义在：

- `metanetwork_family.py`

主要类：

```python
class MetanetworkTransformer(nn.Module)
class MetanetworkLinear(nn.Module)
class MetanetworkLinearGate(nn.Module)
class Metanetwork(nn.Module)
```

其中 `Metanetwork` 是总入口，内部根据配置选择具体 M2P 类型：

```python
if cfg.metanetwork.type == "transformer":
    self.metanetwork = MetanetworkTransformer(cfg, self.idx_range)
elif cfg.metanetwork.type == "linear":
    self.metanetwork = MetanetworkLinear(cfg)
elif cfg.metanetwork.type == "lineargate":
    self.metanetwork = MetanetworkLinearGate(cfg)
```

### 2.4 原 checkpoint 逻辑

保存和加载位于：

- `utils/mysaveload.py`

原 checkpoint 文件包括：

```text
mem_tokens.pt
metanetwork.pth
metalora.pth
ift_additional_metalora.pth
trainer_state.json
trainer_state.pt
```

原始项目没有发现标准 PEFT checkpoint：

```text
adapter_config.json
adapter_model.safetensors
adapter_model.bin
```

### 2.5 训练和推理入口

训练入口：

- `meta_train_parallel.py`

测试/推理入口：

- `test.py`
- `test_pretrain.py`
- `test_pwc.py`
- `inference.ipynb`

脚本入口集中在：

- `scripts/Qwen3-8B/*.sh`
- `scripts/Qwen3-1.7B/*.sh`
- `scripts/Qwen3-0.6B/*.sh`

## 3. 新增 PEFT 工具层

新增文件：

- `utils/peft_lora.py`

该文件集中封装 PEFT 相关逻辑，避免把 PEFT 细节散落到训练脚本中。

### 3.1 Qwen PEFT LoRA

新增函数：

```python
def apply_lora_to_qwen(qwen_model, lora_config_args, is_trainable=True):
    ...
    return peft_model
```

功能：

- 如果配置了 `adapter_path`，使用：

```python
PeftModel.from_pretrained(...)
```

- 如果没有配置 `adapter_path`，新建：

```python
LoraConfig(...)
get_peft_model(...)
```

默认 Qwen target modules：

```python
[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

这些模块覆盖 Qwen attention 和 MLP 中常见的 LoRA 注入点。

### 3.2 M2P PEFT LoRA

新增函数：

```python
def apply_lora_to_m2p(m2p_model, lora_config_args, logger=None):
    ...
    return peft_model
```

功能：

- 支持加载已有 M2P PEFT adapter。
- 支持给 M2P 新建 PEFT LoRA。
- 自动扫描 M2P 的 `named_modules()`。
- 自动寻找适合挂 LoRA 的 `nn.Linear` 层。

优先匹配的 target modules：

```python
[
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
```

如果匹配不到这些名字，则退化为使用实际扫描到的 Linear 模块完整名称。

### 3.3 参数冻结

新增函数：

```python
def freeze_non_lora_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
```

作用：

- 冻结 base model。
- 只训练 PEFT LoRA 参数。

## 4. 训练入口改造

修改文件：

- `meta_train_parallel.py`

### 4.1 Qwen 加载后应用 PEFT

原逻辑：

```python
metamodel = MetaModelCls.from_pretrained(cfg.model.model_from, config=config)
metamodel.reset_mem_tokens()
metamodel.resize_token_embeddings(len(tokenizer))
```

新增：

```python
use_qwen_peft = peft_section_enabled(cfg, "qwen")
use_m2p_peft = peft_section_enabled(cfg, "m2p")

if use_qwen_peft:
    metamodel = apply_lora_to_qwen(metamodel, cfg.peft.qwen, is_trainable=True)
```

这样 Qwen base model 先正常加载，再由 PEFT 包装成带 LoRA 的模型。

### 4.2 M2P 构造后应用 PEFT

原逻辑：

```python
metanetwork = Metanetwork(metamodel, cfg, metamodel.lora_params_numel(cfg.model.lora_r))
```

新增：

```python
if use_m2p_peft:
    print_named_modules(metanetwork.metanetwork, logger=logger)
    metanetwork.metanetwork = apply_lora_to_m2p(
        metanetwork.metanetwork,
        cfg.peft.m2p,
        logger=logger,
    )
```

注意这里 PEFT 包装的是：

```python
metanetwork.metanetwork
```

不是整个 `Metanetwork`。这样可以只对真正的 M2P/hypernetwork 部分加 LoRA，而不影响外层的 SHINE 调度逻辑。

### 4.3 冻结逻辑调整

原逻辑会直接冻结 Qwen：

```python
freeze(metamodel)
```

现在改成：

```python
if not use_qwen_peft:
    freeze(metamodel)
```

原因是：启用 Qwen PEFT 时，`apply_lora_to_qwen()` 内部已经完成了参数冻结，只保留 LoRA 参数可训练。如果再调用旧的 `freeze()`，可能会误伤 PEFT wrapper。

## 5. Optimizer 改造

修改位置：

- `meta_train_parallel.py`

原 optimizer 逻辑主要排除了 `metamodel` 参数，只训练 M2P 和 `metalora`。

现在新增参数过滤函数：

```python
def is_trainable_named_param(name, param):
    if not param.requires_grad:
        return False
    if not is_metamodel_param(name):
        return True
    return use_qwen_peft and "lora_" in name
```

效果：

- 非 Qwen 部分只要 `requires_grad=True` 就进入 optimizer。
- Qwen base model 不进入 optimizer。
- Qwen PEFT LoRA 参数可以进入 optimizer。
- M2P 原始参数在启用 PEFT 后被冻结。
- M2P PEFT LoRA 参数可以进入 optimizer。
- 原有 `metalora` 仍按原逻辑加入 optimizer。

## 6. Checkpoint 保存和加载改造

修改文件：

- `utils/mysaveload.py`

### 6.1 保存逻辑

原始 checkpoint 仍然保存：

```text
mem_tokens.pt
metanetwork.pth
metalora.pth
ift_additional_metalora.pth
trainer_state.json
trainer_state.pt
```

新增 PEFT adapter 保存：

```text
qwen_peft/
  adapter_config.json
  adapter_model.safetensors

m2p_peft/
  adapter_config.json
  adapter_model.safetensors
```

对应逻辑：

```python
save_peft_adapter_if_present(metanetwork.metamodel, os.path.join(out_dir, "qwen_peft"))
save_peft_adapter_if_present(metanetwork.metanetwork, os.path.join(out_dir, "m2p_peft"))
```

### 6.2 加载逻辑

加载时先按原 SHINE 方式加载：

```python
metanetwork.pth
metalora.pth
mem_tokens.pt
```

然后检查是否存在：

```text
qwen_peft/adapter_config.json
m2p_peft/adapter_config.json
```

如果存在，则调用：

```python
model.load_adapter(...)
model.set_adapter(...)
```

这样新旧 checkpoint 可以共存：

- 旧 checkpoint 没有 PEFT adapter 时，仍然可加载。
- 新 checkpoint 有 PEFT adapter 时，会额外恢复 Qwen/M2P LoRA。

## 7. 测试入口改造

修改文件：

- `test.py`
- `test_pretrain.py`
- `test_pwc.py`

修改目的：

训练时如果启用了 PEFT，测试时也必须先构造相同的 PEFT wrapper，否则 checkpoint 中的 adapter 无法正确加载。

因此三个测试入口都加入了与训练入口一致的逻辑：

```python
use_qwen_peft = peft_section_enabled(cfg, "qwen")
use_m2p_peft = peft_section_enabled(cfg, "m2p")

if use_qwen_peft:
    metamodel = apply_lora_to_qwen(metamodel, cfg.peft.qwen, is_trainable=True)

metanetwork = Metanetwork(...)

if use_m2p_peft:
    metanetwork.metanetwork = apply_lora_to_m2p(metanetwork.metanetwork, cfg.peft.m2p)
```

## 8. 配置文件改造

修改文件：

- `configs/Qwen3-8B.yaml`
- `configs/Qwen3-1.7B.yaml`
- `configs/Qwen3-0.6B.yaml`

新增配置块：

```yaml
peft:
  qwen:
    enabled: false
    adapter_path: null
    adapter_name: default
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  m2p:
    enabled: false
    adapter_path: null
    adapter_name: default
    r: 8
    lora_alpha: 16
    lora_dropout: 0.05
    bias: none
    target_modules: []
```

说明：

- `enabled: false` 保证默认行为不变。
- `adapter_path: null` 表示从头初始化一个新的 PEFT LoRA。
- 如果传入已有 adapter 路径，则继续训练已有 adapter。
- `m2p.target_modules: []` 表示自动根据 M2P 的 `named_modules()` 推断 Linear target。

## 9. 使用示例

### 9.1 原始 SHINE 训练

不需要改命令。因为 PEFT 默认关闭。

```bash
python meta_train_parallel.py --config-name Qwen3-8B
```

### 9.2 同时训练 Qwen PEFT LoRA 和 M2P PEFT LoRA

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.qwen.enabled=true \
  peft.m2p.enabled=true
```

### 9.3 在已有 Qwen adapter 基础上继续训练

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.qwen.enabled=true \
  peft.qwen.adapter_path=path/to/qwen_peft \
  peft.m2p.enabled=true
```

### 9.4 在已有 M2P adapter 基础上继续训练

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.qwen.enabled=true \
  peft.m2p.enabled=true \
  peft.m2p.adapter_path=path/to/m2p_peft
```

### 9.5 指定 M2P target modules

如果自动推断不符合预期，可以手动指定：

```bash
python meta_train_parallel.py --config-name Qwen3-8B \
  peft.m2p.enabled=true \
  peft.m2p.target_modules='[linear1,linear2,out_proj]'
```

## 10. 改造后的训练参数范围

启用 PEFT 后，训练参数范围如下：

| 组件 | 原始参数 | PEFT LoRA 参数 |
| --- | --- | --- |
| Qwen base model | 冻结 | 可训练 |
| M2P/metanetwork | 冻结 | 可训练 |
| 原 SHINE metalora | 按原逻辑训练或冻结 | 不适用 |
| mem_tokens | 按原逻辑处理 | 不适用 |

这符合当前目标：不做复杂 LoRA 融合，只在 Qwen 和 M2P 上分别挂一个 PEFT LoRA adapter。

## 11. 验证情况

已执行静态编译检查：

```bash
python -m py_compile \
  utils/peft_lora.py \
  utils/mysaveload.py \
  utils/myfreeze.py \
  meta_train_parallel.py \
  test.py \
  test_pretrain.py \
  test_pwc.py
```

结果：通过。

完整训练没有执行，因为当前仓库内没有确认本地 Qwen 模型目录和数据目录是否齐全。

## 12. 注意事项

1. 需要安装 PEFT：

```bash
pip install peft
```

2. 启用 PEFT 时，测试阶段也要使用同样的 `peft.*.enabled` 配置，否则模型结构和 checkpoint 中的 adapter 可能不匹配。

3. 原项目的 `metalora.pth` 仍然存在，它是 SHINE 自定义动态 LoRA 机制的一部分；新增的 PEFT adapter 是额外保存的标准 PEFT 权重。

4. 当前没有实现多 adapter 融合、MoE-LoRA、LSH、MagicPIG 或动态 adapter 路由。这些都不属于本次改造范围。
