# DINOv3 编码器数据流详解（ViT-Small）

以 **DINOv3 ViT-Small, `input_size=256`, `patch_size=16`** 为例，逐步追踪数据流。

---

## 第 0 步：模型构建时的参数解析

```python
# ddcolor_dinov3_arch.py:47-51
self.encoder = DinoVisionTransformerWrapper.from_model_name(
    model_name="vit_small",          # → dinov3-vits16-pretrain-lvd1689m
    backbone_out_layers="FOUR_EVEN_INTERVALS",
    ...
)
```

初始化时在 [dino_vit_wrapper.py:138-153](basicsr/archs/dino_vit_arch_utils/dino_vit_wrapper.py#L138-L153) 完成以下计算：

```
n_blocks = config.num_hidden_layers = 12        # ViT-Small 有 12 层 Transformer
hidden_size = config.hidden_size = 384           # 每层的隐藏维度
patch_size = config.patch_size = 16

# 层选择 (FOUR_EVEN_INTERVALS, n_blocks=12 → else 分支公式)
#   out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
#               = [1*3-1, 2*3-1, 3*3-1, 4*3-1]
backbone_out_indices = [2, 5, 8, 11]             # 0-indexed，均匀分布在 12 层中

# 每层的嵌入维度（DINOv3 所有层维度相同）
embed_dims = [384, 384, 384, 384]
```

`_get_backbone_out_indices` 逻辑（[dino_vit_wrapper.py:30-45](basicsr/archs/dino_vit_arch_utils/dino_vit_wrapper.py#L30-L45)）：

```python
n_blocks = config.num_hidden_layers   # = 12
if n_blocks == 24:
    out_indices = [4, 11, 17, 23]     # 硬编码，仅 ViT-Large 走这个分支
else:
    out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    # n_blocks=12 → 12//4=3 → [2, 5, 8, 11]
```

选层策略的含义：

| 索引 | 对应 Block | 特征层级 |
|---|---|---|
| `2` | 第 3 个 block `hidden_states[3]` | 浅层特征（边缘、纹理） |
| `5` | 第 6 个 block `hidden_states[6]` | 中浅层特征 |
| `8` | 第 9 个 block `hidden_states[9]` | 中深层特征 |
| `11` | 第 12 个 block `hidden_states[12]` | 深层语义特征 |

---

## 第 1 步：forward 入口

```python
# ddcolor_dinov3_arch.py:74-91
def forward(self, x):
    # x: (B, 3, 256, 256)  灰度图转 RGB，值域 [0, 1]
    H, W = x.shape[2:]      # H=256, W=256

    if self.do_normalize:
        x = self.normalize(x)   # ImageNet 标准化
    # x: (B, 3, 256, 256)

    features = self.encoder(x)   # 进入 DINOv3 ViT
```

**ImageNet 归一化** ([ddcolor_dinov3_arch.py:68-72](basicsr/archs/ddcolor_dinov3_arch.py#L68-L72))：

```python
# mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
x = (x - mean) / std
```

输入是灰度 RGB（三个通道值完全相同），归一化后每个通道被分别处理。形状不变：`(B, 3, 256, 256)`。

---

## 第 2 步：CenterPadding

```python
# dino_vit_wrapper.py:197
x = self.patch_size_adapter(x)
```

`CenterPadding` ([center_padding.py](basicsr/archs/dino_vit_arch_utils/center_padding.py)) 确保 H 和 W 能被 `patch_size=16` 整除：

```python
# center_padding.py:12-18
def _get_pad(self, size):
    new_size = math.ceil(size / 16) * 16     # ceil(256/16)*16 = 256, 无需 pad
    pad_size = new_size - size               # 0
    pad_size_left = pad_size // 2            # 0
    pad_size_right = pad_size - pad_size_left # 0
    return pad_size_left, pad_size_right

def forward(self, x):
    # 对 H 和 W 分别计算 pad 值，用 F.pad 做对称填充
    pads = [pad_w_left, pad_w_right, pad_h_left, pad_h_right]
    return F.pad(x, pads)
```

| 输入 H,W | 是否整除 16 | 实际效果 |
|---|---|---|
| 256 | ✅ | 无操作，形状不变 `(B, 3, 256, 256)` |
| 512 | ✅ | 无操作，形状不变 `(B, 3, 512, 512)` |
| 300 | ❌ | pad → `(B, 3, 304, 304)` |
| 513 | ❌ | pad → `(B, 3, 528, 528)` |

---

## 第 3 步：ViT Patch Embedding（HuggingFace 内部）

```python
# dino_vit_wrapper.py:95
outputs = backbone(x, output_hidden_states=True, return_dict=True)
```

这一步在 HuggingFace `AutoModel` 内部自动完成，不是本仓库的代码，但对理解形状至关重要：

```
原始图像: (B, 3, 256, 256)
    │
    ▼  Conv2d(3, 384, kernel=16, stride=16)  — patchify
    │
Patch 序列: (B, 256, 384)     # 256 = (256/16) × (256/16) = 16×16 patches
    │
    ▼  拼接 CLS token + 4 Register tokens + 位置编码
    │
Embedding 输出 (hidden_states[0]): (B, 261, 384)
                                    │  │    └─ hidden_size
                                    │  └─ 1 CLS + 4 register + 256 patches = 261 tokens
                                    └─ batch
```

Token 排列结构：

```
位置:  0      1  2  3  4      5 ───────────── 260
      [CLS] [R1 R2 R3 R4] [patch_0, patch_1, ..., patch_255]
      ──────  ───────────  ──────────────────────────────
      1 个     4 个 register tokens              256 个 patch tokens
      CLS      (DINOv3 特有, 稳定注意力)
```

---

## 第 4 步：通过 12 层 Transformer Blocks

ViT 内部，`(B, 261, 384)` 依次经过 12 个 Transformer block（Self-Attention + FFN），`output_hidden_states=True` 让 HuggingFace 返回所有中间层的 hidden states。共计 13 个 hidden_states（1 个 embedding + 12 个 block 输出）：

```
hidden_states[0]  = Embedding 输出         (B, 261, 384)
hidden_states[1]  = Block 0 输出           (B, 261, 384)
hidden_states[2]  = Block 1 输出           (B, 261, 384)
hidden_states[3]  = Block 2 输出  ← 提取    (B, 261, 384)
hidden_states[4]  = Block 3 输出           (B, 261, 384)
hidden_states[5]  = Block 4 输出           (B, 261, 384)
hidden_states[6]  = Block 5 输出  ← 提取   (B, 261, 384)
hidden_states[7]  = Block 6 输出           (B, 261, 384)
hidden_states[8]  = Block 7 输出           (B, 261, 384)
hidden_states[9]  = Block 8 输出  ← 提取   (B, 261, 384)
hidden_states[10] = Block 9 输出           (B, 261, 384)
hidden_states[11] = Block 10 输出          (B, 261, 384)
hidden_states[12] = Block 11 输出 ← 提取   (B, 261, 384)
```

---

## 第 5 步：Token 拆分（关键步骤）

```python
# dino_vit_wrapper.py:98-111
for block_idx in [2, 5, 8, 11]:
    hidden = outputs.hidden_states[block_idx + 1]  # (B, 261, 384)

    cls_token = hidden[:, 0, :]                    # (B, 384)
    # DINOv3 has 4 register tokens after CLS; patch tokens follow
    patch_tokens = hidden[:, 5:, :]                 # (B, 256, 384)

    B, N, C = patch_tokens.shape                   # N=256, C=384
    h = w = int(math.sqrt(N))                      # h=16, w=16
    spatial = patch_tokens.transpose(1, 2).reshape(B, C, h, w)
    # spatial: (B, 384, 16, 16)

    features.append((spatial, cls_token))
```

逐行解释：

```
hidden[:, 0, :]   → 取位置 0       = CLS token         → (B, 384)
hidden[:, 5:, :]  → 取位置 5~260   = patch tokens      → (B, 256, 384)

位置 1~4 (4 个 register tokens) 被丢弃 —— 它们是 DINOv3 的内部机制，
用于稳定 Self-Attention 的 attention map，不包含空间信息。
```

**为什么保存 CLS token？** CLS token 经过 12 层 Self-Attention 后聚合了全图的全局语义信息（场景类别、整体色调倾向等），在后续的 `WeightedFusion` 中会作为全局颜色线索注入到每个空间位置。

从 token 到空间特征图的转换：

```
patch_tokens: (B, 256, 384)
    │  transpose(1, 2) → (B, 384, 256)
    │  reshape(B, 384, 16, 16)
    ▼
spatial: (B, 384, 16, 16)

其空间对应关系：
    patch_tokens[:, 0, :]   → spatial[:, :, 0, 0]   (左上角 patch)
    patch_tokens[:, 15, :]  → spatial[:, :, 0, 15]  (第一行最右)
    patch_tokens[:, 255, :] → spatial[:, :, 15, 15] (右下角 patch)
```

---

## 第 6 步：最终返回值

```python
# dino_vit_wrapper.py:202
return features
# features = [
#     (spatial_l2,  cls_l2 ),   # (B,384,16,16), (B,384)   ← Block 2
#     (spatial_l5,  cls_l5 ),   # (B,384,16,16), (B,384)   ← Block 5
#     (spatial_l8,  cls_l8 ),   # (B,384,16,16), (B,384)   ← Block 8
#     (spatial_l11, cls_l11),   # (B,384,16,16), (B,384)   ← Block 11
# ]
```

4 个 tuple，每个包含同一层的空间特征和全局语义。随后传入 `SDTColorizationHead` 的 `WeightedFusion` 进行融合。

---

## 完整形状变化总览

以 `vit_small`, `B=1`, `input_size=256` 为例：

```
阶段                        形状                          说明
─────────────────────────────────────────────────────────────────────
forward 输入             (1, 3, 256, 256)              灰度 RGB
normalize                (1, 3, 256, 256)              ImageNet 标准化
CenterPadding            (1, 3, 256, 256)              256 可被 16 整除, 无操作
ViT patch embedding      (1, 261, 384)                 1 CLS + 4 reg + 256 patches
ViT blocks 0~11          (1, 261, 384) × 13            每层形状相同（1 embedding + 12 blocks）
提取 4 层 + token 拆分:
  hidden[:, 0, :]        (1, 384) × 4                 CLS token (全局语义)
  hidden[:, 5:, :]       (1, 256, 384) × 4            patch tokens
  reshape → spatial      (1, 384, 16, 16) × 4         空间特征图

最终编码器输出: 4 × [(B, 384, 16, 16), (B, 384)]
```

三种规格的维度对比：

| 规格 | hidden_size | num_hidden_layers | backbone_out_indices | embed_dims | patch 数 (256×256 输入) |
|---|---|---|---|---|---|
| ViT-Small | 384 | 12 | `[2, 5, 8, 11]` | `[384,384,384,384]` | 256 |
| ViT-Base | 768 | 12 | `[2, 5, 8, 11]` | `[768,768,768,768]` | 256 |
| ViT-Large | 1024 | 24 | `[4, 11, 17, 23]` | `[1024,1024,1024,1024]` | 256 |

---

## 关键设计决策

**为什么丢弃 register tokens？** DINOv3 在 CLS 和 patch tokens 之间插入了 4 个可学习的 register tokens，用于吸收 attention 中的异常值，使 attention map 更平滑。它们不编码任何空间或语义信息，所以在提取特征时需要排除（`[:, 5:, :]`）。

**为什么保留 CLS token？** CLS 经过 12 层全局自注意力，包含了整图的场景级语义——这对颜色化至关重要。例如"这是一张草地照片"意味着绿色主导，"这是室内暖光场景"意味着橙黄色调。`WeightedFusion` 会通过 CLS readout 将这种全局颜色上下文注入到每个 patch 位置。

**为什么选 [2, 5, 8, 11] 这四层？** 均匀采样 ViT 的浅→中层→深层，捕获从局部纹理到全局语义的多粒度特征。公式为 `[i * (n_blocks // 4) - 1 for i in range(1, 5)]`，对于 12 层模型在总层数的 1/4、2/4、3/4、4/4 位置各取一层。浅层保留更多空间细节（利于边缘保持），深层提供更强的语义理解（利于颜色推理）。
