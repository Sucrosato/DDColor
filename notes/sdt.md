# SDT 解码器数据流详解

以 **ViT-Small 编码器, `input_size=256`, `fusion_channels=256`** 为例，从 DINOv3 编码器输出开始，逐步追踪 SDT 解码器的完整数据流。

> 编码器输入: `(B, 3, 256, 256)` → 输出: 4 × `[(B, 384, 16, 16), (B, 384)]`

---

## 解码器整体数据流

```
编码器输出: 4 × [(B,384,16,16), (B,384)]    ← spatial + CLS token
    │
    ▼
┌───────────────────────────────────────────────┐
│  Step 1: WeightedFusion                       │
│  融合 4 层 + CLS readout                       │
│  (B,256,256) token 序列                        │
└───────────────────────────────────────────────┘
    │  reshape
    ▼  (B, 256, 16, 16)
┌───────────────────────────────────────────────┐
│  Step 2: SpatialDetailEnhancer                │
│  Depthwise Conv + 残差                         │
│  (B, 256, 16, 16)                             │
└───────────────────────────────────────────────┘
    │
    ▼  (B, 256, 16, 16)
┌───────────────────────────────────────────────┐
│  Step 3: ColorQueryBottleneck (可选)           │
│  100 个 learnable color queries               │
│  + 3 层 Transformer Decoder                    │
│  (B, 256, 16, 16)                             │
└───────────────────────────────────────────────┘
    │
    ▼  (B, 256, 16, 16)
┌───────────────────────────────────────────────┐
│  Step 4: DySample ×2 (第一次 4× 上采样)        │
│  16×16 → 64×64                                │
│  (B, 256, 64, 64)                             │
└───────────────────────────────────────────────┘
    │
    ▼  (B, 256, 64, 64)
┌───────────────────────────────────────────────┐
│  Step 5: refinement_1                         │
│  Conv3×3 + BN + ReLU                          │
│  (B, 256, 64, 64)                             │
└───────────────────────────────────────────────┘
    │
    ▼  (B, 256, 64, 64)
┌───────────────────────────────────────────────┐
│  Step 6: DySample ×2 (第二次 4× 上采样)        │
│  64×64 → 256×256                              │
│  (B, 256, 256, 256)                           │
└───────────────────────────────────────────────┘
    │
    ▼  (B, 256, 256, 256)
┌───────────────────────────────────────────────┐
│  Step 7: refinement_2                         │
│  Conv3×3 + BN + ReLU                          │
│  (B, 256, 256, 256)                           │
└───────────────────────────────────────────────┘
    │
    ▼  (B, 256, 256, 256)
┌───────────────────────────────────────────────┐
│  Step 8: output_conv                          │
│  256→128→32→2                                 │
│  (B, 2, 256, 256)                             │
└───────────────────────────────────────────────┘
    │
    ▼  F.interpolate(bilinear) → target_size
输出 ab 色度: (B, 2, H, W)
```

---

## Step 1: WeightedFusion — 融合 4 层 + CLS Readout

**源码位置**: [sdt_decoder.py:107-157](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L107-L157)

**输入**: 编码器输出的 4 个 tuple，每个包含 `(spatial_tensor, cls_token)`：

```
features = [
    ( (B,384,16,16), (B,384) ),   ← Block 2  浅层
    ( (B,384,16,16), (B,384) ),   ← Block 5  中浅层
    ( (B,384,16,16), (B,384) ),   ← Block 8  中深层
    ( (B,384,16,16), (B,384) ),   ← Block 11 深层
]
```

### 1.1 对每一层独立处理

```python
# sdt_decoder.py:136-143
for i, layer_feature in enumerate(features):       # i = 0,1,2,3
    spatial_tensor, cls_token = layer_feature
```

#### 子步骤 1：空间特征展平为 token 序列

```
spatial_tensor: (B, 384, 16, 16)
    │  flatten(2) → (B, 384, 256)
    │  permute(0, 2, 1) → (B, 256, 384)
    ▼
spatial_tokens: (B, 256, 384)     # 256 个 patch tokens，每个 384 维
```

#### 子步骤 2：CLS token 广播至每个空间位置

```
cls_token: (B, 384)
    │  unsqueeze(1) → (B, 1, 384)
    │  expand_as (spatial_tokens) → (B, 256, 384)
    ▼
cls_token_expanded: (B, 256, 384)  # 每个空间位置都收到相同的全局语义
```

#### 子步骤 3：拼接空间特征与全局语义

```
tokens_with_cls = torch.cat((spatial_tokens, cls_token_expanded), dim=-1)
# (B, 256, 768) = 384 (spatial) + 384 (CLS)
```

每个空间位置现在同时拥有**局部视觉特征**和**全局场景语义**。

#### 子步骤 4：CLS Readout 投影

```python
# readout_projects[i]: Linear(2*in_dim → in_dim) + GELU
# 对于 ViT-Small: in_dim=384, 2*384=768
enhanced_tokens = self.readout_projects[i](tokens_with_cls)
```

```
(B, 256, 768) → Linear(768, 384) + GELU → (B, 256, 384)
```

这一层的作用是**学习如何将全局语义与局部特征融合**——不是简单的拼接，而是通过可学习的线性变换找到两者之间的交互模式。

#### 子步骤 5：投影到统一维度

```python
# projections[i]: Linear(in_dim → 256) + GELU
projected_tokens = self.projections[i](enhanced_tokens)
```

```
(B, 256, 384) → Linear(384, 256) + GELU → (B, 256, 256)
```

### 1.2 加权融合 4 层

```python
# sdt_decoder.py:153-156
layer_weights = F.softmax(self.layer_weights, dim=0)    # 4 维，和为 1
fused_tokens = sum(layer_weights[i] * projected_tokens[i] for i in range(4))
```

4 个可学习标量权重（初始值均为 1.0）经 softmax 归一化后，对各层特征做加权求和：

```
         浅层 (Block 2)  × w₀
       + 中浅层 (Block 5) × w₁
       + 中深层 (Block 8) × w₂
       + 深层 (Block 11) × w₃
       ──────────────────────
fused_tokens: (B, 256, 256)
```

**注意**: 维度从 `384 → 256` 是在投影时完成的。ViT 不同层的 hidden_size 相同（ViT-Small 均为 384），所以投影到统一维度后再相加。

### 1.3 Reshape 回空间格式

```python
# sdt_decoder.py:389-390
fused_spatial = fused_tokens.permute(0, 2, 1).contiguous().reshape(
    B, self.fusion_channels, H_patches, W_patches)
```

```
(B, 256, 256) → permute(0,2,1) → (B, 256, 256) → reshape → (B, 256, 16, 16)
```

**输出**: `(B, 256, 16, 16)`

---

## Step 2: SpatialDetailEnhancer — 空间细节增强

**源码位置**: [sdt_decoder.py:164-176](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L164-L176)

```python
def forward(self, x):
    residual = x                        # (B, 256, 16, 16)
    x = self.dwconv(x)                  # Depthwise Conv 3×3, groups=256
    x = self.norm(x)                    # BatchNorm2d
    x = self.activation(x + residual)   # ReLU + 残差连接
    return x
```

子步骤：

```
输入: (B, 256, 16, 16)
    │
    ├─→ Depthwise Conv2d(256→256, k3, groups=256)  ← 每通道独立卷积，参数极少
    │   (B, 256, 16, 16)
    │
    ├─→ BatchNorm2d(256)  →  (B, 256, 16, 16)
    │
    ├─→ + residual  (残差连接，保护原始特征)
    │
    ├─→ ReLU(inplace=True)
    │
    ▼
输出: (B, 256, 16, 16)
```

**设计意图**: WeightedFusion 中的操作都是 token 级别的（Linear + 加权求和），缺乏对空间邻域关系的显式建模。Depthwise Conv 以极低的参数量（`256×3×3 = 2304` 个参数）在每个通道内做局部的空间平滑，增强边缘和纹理细节的表示质量。残差连接确保新增的空间细节不会被 ReLU 截断。

---

## Step 3: ColorQueryBottleneck（可选）— 全局颜色推理

**源码位置**: [sdt_decoder.py:210-294](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L210-L294)

**触发条件**: `use_color_queries=True`（否则为 `nn.Identity()`，直接透传）

### 3.1 模块结构

```
d_model=256          # 特征维度
num_queries=100      # 可学习 color query 数量
num_layers=3         # Transformer decoder 层数
nheads=8             # 注意力头数 (每头 256/8=32 维)
dim_feedforward=1024 # FFN 隐藏层维度
```

### 3.2 准备 Query 和 Memory

```python
# sdt_decoder.py:266-273
B, C, H, W = spatial_features.shape   # B, 256, 16, 16

# 空间特征作为 cross-attention 的 memory: 格式要求 (L, B, C)
src = spatial_features.flatten(2).permute(2, 0, 1)  # (256, B, 256)
#    N = 16×16 = 256 个空间位置作为 memory

# 初始化 color queries: (L, B, C)
query = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)   # (100, B, 256)
pos   = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (100, B, 256)
```

`query_feat` 和 `query_embed` 是两个独立的 `nn.Embedding(100, 256)`：
- `query_feat`：100 个 color query 的内容（语义）
- `query_embed`：100 个 color query 的位置编码（区分不同 query 的角色）

### 3.3 三层 Transformer Decoder

```python
# sdt_decoder.py:276-279
for sa, ca, ffn in self.layers:     # 共 3 轮迭代
    query = sa(query, query_pos=pos)  # Self-Attention: query 之间互相交流
    query = ca(query, src)            # Cross-Attention: query 看空间特征
    query = ffn(query)                # FFN: 每个 query 独立非线性变换
```

#### 3.3.1 Self-Attention Layer

```
输入 query: (100, B, 256)
              │
              ▼
┌──────────────────────────────────────────┐
│  pos = query + query_pos                  │  ← 加入位置编码
│  attn_out = MultiheadAttention(q, k, v)   │  ← MHA, 8 heads
│  query = LayerNorm(query + dropout(out))  │  ← 残差 + 归一化
│  输出: (100, B, 256)                      │
└──────────────────────────────────────────┘
```

100 个 color query 通过互相自注意力来协调分工——例如有的 query 专门关注天空的颜色，有的关注植被的颜色。

#### 3.3.2 Cross-Attention Layer

```
输入 query: (100, B, 256)
输入 src (memory): (256, B, 256)   ← 16×16 空间特征
              │
              ▼
┌──────────────────────────────────────────┐
│  Q = query + query_pos     (100, B, 256) │  ← query 的投影
│  K = src                   (256, B, 256) │  ← 空间特征作为 key
│  V = src                   (256, B, 256) │  ← 空间特征作为 value
│  attn_out = MHA(Q, K, V)                 │  ← query 检索空间信息
│  query = LayerNorm(query + dropout(out))  │
│  输出: (100, B, 256)                      │
└──────────────────────────────────────────┘
```

**这是核心步骤**: 每个 color query 从 256 个空间位置中检索自己关心的颜色信息。attention weight 矩阵是 `(100, 256)`——100 个 query 各自关注哪些空间位置。

例如某个 query 可能高注意力在"天空区域"的 patch 上，从中提取蓝色的统计信息。

#### 3.3.3 FFN Layer

```
输入: (100, B, 256)
              │
              ▼
┌──────────────────────────────────────────┐
│  Linear(256 → 1024) + ReLU               │  ← 扩展到 4 倍维度
│  Dropout                                 │
│  Linear(1024 → 256)                      │  ← 压缩回原维度
│  query = LayerNorm(query + dropout(out))  │
│  输出: (100, B, 256)                      │
└──────────────────────────────────────────┘
```

### 3.4 生成全局颜色上下文并融合

```python
# sdt_decoder.py:281-294
query = self.decoder_norm(query)                        # LayerNorm → (100, B, 256)

query = query.permute(1, 0, 2)                          # (B, 100, 256)
color_embed = self.color_mlp(query)                      # MLP(256,256,256,3) → (B, 100, 256)
global_color = color_embed.mean(dim=1, keepdim=True)     # (B, 1, 256)  ← mean 池化
global_color = global_color.transpose(1,2).reshape(B, C, 1, 1)  # (B, 256, 1, 1)
global_color = global_color.expand(B, C, H, W)           # (B, 256, 16, 16)

fused = torch.cat([spatial_features, global_color], dim=1)  # (B, 512, 16, 16)
out = self.out_proj(fused)                                   # Conv2d(512→256, 1) → (B, 256, 16, 16)
return out + spatial_features                                # 残差连接 → (B, 256, 16, 16)
```

逐步解释：

```
100 个 query (B, 100, 256)
    │  MLP: 256→256→256→256 (3 层, 中间 ReLU)
    ▼
100 个 color embedding (B, 100, 256)
    │  mean(dim=1)  ← 100 个 query 的信息合并为 1 个全局向量
    ▼
global_color (B, 1, 256)
    │  reshape → (B, 256, 1, 1) → expand → (B, 256, 16, 16)
    ▼
广播到所有空间位置的全局颜色上下文 (B, 256, 16, 16)
    │  concat with original spatial_features → (B, 512, 16, 16)
    │  Conv2d(512, 256, 1)  ← 1×1 卷积融合
    ▲  + spatial_features  ← 残差连接
输出: (B, 256, 16, 16)
```

---

## Step 4 & 6: DySample — 动态上采样

**源码位置**: [sdt_decoder.py:29-101](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L29-L101)

### DySample 原理

DySample（Dynamic Sampling）的核心思想是**不固定上采样位置，而是从特征中学习采样偏移量**，通过 `F.grid_sample` 实现内容自适应的上采样。

```
参数（SDT 解码器中使用）:
  in_channels=256, scale=2, style='lp', groups=4, dyscope=True
```

### `style='lp'` (learned position) 模式

```python
# sdt_decoder.py:82-87
def forward_lp(self, x):
    offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
    return self.sample(x, offset)
```

#### 子步骤 1：预测采样偏移

```
输入 x: (B, 256, H_in, W_in), 例: (B, 256, 16, 16)

self.offset: Conv2d(256, 32, kernel_size=1)
  out_channels = 2 * groups * scale² = 2 × 4 × 4 = 32
  输出: (B, 32, 16, 16)

self.scope: Conv2d(256, 32, kernel_size=1, bias=False)
  输出: (B, 32, 16, 16) → sigmoid() → 每个位置学习一个 0~1 的调制因子

offset = offset * scope * 0.5 + init_pos
  init_pos: 预定义的均匀网格偏移，作为"默认"采样位置
  scope: 动态调整偏移幅度（0=不偏移即等同上采样"最近邻"，1=最大偏移）
  0.5: 缩放因子，限制偏移范围
```

**`init_pos` 的含义**（[sdt_decoder.py:64-66](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L64-L66)）：

当 `scale=2` 时，`init_pos` 为 4 组预定义的 `[-0.25, +0.25]` 网格偏移（共 8 个值，每通道 group 重复）。对于 `groups=4`，就是 `[8, 1, 1]` 个基础偏移。这 8 个值对应尺度为 2 的上采样中，输出像素对应的 4 个亚像素采样点的坐标。

#### 子步骤 2：Grid Sample 采样

```python
# sdt_decoder.py:68-80
def sample(self, x, offset):
    # offset: (B, 32, 16, 16)
    offset = offset.reshape(B, 2, -1, H, W)        # (B, 2, 16, 16, 16)  — xy 偏移

    # 构建规则网格坐标 (0.5, 1.5, ..., 15.5) 对应输入像素中心
    coords = meshgrid(h, w)                         # (1, 1, 2, 16, 16)
    coords = 2 * (coords + offset) / normalizer - 1 # 归一化到 [-1, 1]
    coords = pixel_shuffle → (B, 2, 64, 32, 32)     # 上采样坐标

    # 按预测的偏移坐标从原特征图中采样
    return F.grid_sample(x, coords, mode='bilinear',
                         align_corners=False, padding_mode="border")
```

**`pixel_shuffle` 的作用**: 将 `2×4=8` 个通道的偏移量重排为 `scale×scale=4` 组 `(x, y)` 坐标对，实现空间尺寸翻倍。这本质上是将每个输入像素"扩展"为 2×2 的输出像素，但采样位置由学习的偏移量决定而非固定网格。

**为什么比传统上采样好**: 传统双线性插值对所有位置一视同仁，而 DySample 在边缘处可以学到"向边缘内侧偏移"以避免模糊，在平坦区域学到"均匀分布"以保持平滑。`scope` 因子让模型可以自适应地决定每个位置的偏移程度。

### DySampleUpsamplerWrapper — 串联两次实现 4× 上采样

**源码位置**: [sdt_decoder.py:183-203](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L183-L203)

```python
class DySampleUpsamplerWrapper(nn.Module):
    def __init__(self, feature_dim=256, scale_factor=4, ...):
        # 每个 DySample 做 2× 上采样，后面接 Conv3×3+BN+ReLU 做平滑
        self.dysample1 = Sequential(
            DySample(256, scale=2, style='lp', groups=4, dyscope=True),
            Conv2d(256, 256, 3, padding=1),
            BatchNorm2d(256),
            ReLU()
        )
        self.dysample2 = Sequential(...)  # 同上
```

**第一次 4× 上采样**（Step 4）：

```
输入: (B, 256, 16, 16)
    │  DySample ×2
    ▼  (B, 256, 32, 32)   ← scale=2
    │  Conv3×3 + BN + ReLU (空间平滑)
    │  DySample ×2
    ▼  (B, 256, 64, 64)   ← 再 scale=2，累计 4×
    │  Conv3×3 + BN + ReLU
    ▼
输出: (B, 256, 64, 64)
```

**refinement_1**（Step 5）:

```
(B, 256, 64, 64) → Conv3×3(256,256) + BN + ReLU → (B, 256, 64, 64)
```

**第二次 4× 上采样**（Step 6）：

```
输入: (B, 256, 64, 64)
    │  DySample ×2 → Conv+BN+ReLU
    ▼  (B, 256, 128, 128)
    │  DySample ×2 → Conv+BN+ReLU
    ▼  (B, 256, 256, 256)
输出: (B, 256, 256, 256)
```

**refinement_2**（Step 7）:

```
(B, 256, 256, 256) → Conv3×3(256,256) + BN + ReLU → (B, 256, 256, 256)
```

---

## Step 8: Output Conv — 输出色度通道

**源码位置**: [sdt_decoder.py:360-365](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L360-L365)

```python
self.output_conv = Sequential(
    Conv2d(256, 128, k3, p1), ReLU,     # 256 → 128, 空间不变
    Conv2d(128,  32, k3, p1), ReLU,     # 128 →  32, 空间不变
    Conv2d( 32,   2, k1),               #  32 →   2 (ab 通道)
)
```

```
输入: (B, 256, 256, 256)
    │  Conv3×3 + ReLU
    ▼  (B, 128, 256, 256)
    │  Conv3×3 + ReLU
    ▼  (B,  32, 256, 256)
    │  Conv1×1
    ▼  (B,   2, 256, 256)
输出: ab 色度通道
```

最后一层用 1×1 卷积而非 3×3，因为它只做通道映射（32→2），不需要空间混合。

---

## Step 9: 最终尺寸对齐

**源码位置**: [sdt_decoder.py:405-407](basicsr/archs/dino_vit_arch_utils/sdt_decoder.py#L405-L407)

```python
target_size = original_size if original_size is not None else self.output_size
if target_size is not None and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
    out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
```

当 `input_size=256` 时，经过 16× 上采样后输出恰好是 `(256, 256)`，无需 resize。当输入如 `512` 时，16× 上采样后也是 `(512, 512)`（512 能被 16 整除）。只有输入尺寸不能被 16 整除时才需要这一步做最终对齐。

---

## 完整形状变化总览

以 `B=1`, `ViT-Small`, `input_size=256` 为例：

```
阶段                              形状                        说明
─────────────────────────────────────────────────────────────────────────
编码器输出                    4×[(1,384,16,16),(1,384)]     4 层 spatial + CLS
                                                              
WeightedFusion (每层):                                                
  spatial tokens              (1,256,384)                   flatten + permute
  CLS expand                  (1,256,384)                   unsqueeze + expand
  concat                      (1,256,768)                   384+384
  readout_proj                (1,256,384)                   Linear(768→384)
  projection                  (1,256,256)                   Linear(384→256)
加权融合                      (1,256,256)                   softmax 权重 ×4
reshape → spatial             (1,256,16,16)                 恢复空间格式
                                                              
SpatialDetailEnhancer         (1,256,16,16)                 DWConv + 残差
                                                              
ColorQueryBottleneck (可选):                                       
  src (memory)                (256,1,256)                   flatten 为 memory
  query feat                  (100,1,256)                   100 个 color queries
  SA → CA → FFN ×3            (100,1,256)                   每层形状不变
  color_mlp → mean            (1,1,256)                     全局池化为 1 个向量
  broadcast + concat          (1,512,16,16)                 
  out_proj                    (1,256,16,16)                 Conv1×1 + 残差
                                                              
DySample ×2 (第一次 4×)       (1,256,64,64)                 16 → 64
refinement_1                  (1,256,64,64)                 Conv3×3
DySample ×2 (第二次 4×)       (1,256,256,256)               64 → 256
refinement_2                  (1,256,256,256)               Conv3×3
                                                              
output_conv:                                                   
  256→128                     (1,128,256,256)               Conv3×3 + ReLU
  128→32                      (1, 32,256,256)               Conv3×3 + ReLU
  32→2                        (1,  2,256,256)               Conv1×1

最终输出                      (1, 2, 256, 256)              ab 色度通道
```

---

## 关键设计要点

| 设计 | 说明 |
|---|---|
| **CLS Readout** | CLS token 在解码器第一步就被广播并融入每个空间位置，之后不再单独存在 |
| **加权融合而非简单拼接** | 4 层特征通过可学习的 softmax 权重融合，模型自动学习每层的贡献比例 |
| **Depthwise Conv 做空间增强** | 以极少参数（2304 个）补充 Linear 融合中缺失的局部空间关系 |
| **Color Query 是全局颜色推理器** | 100 个 query 从 256 个空间位置检索颜色信息，通过 cross-attention 实现全局颜色理解，弥补 CNN decoder 感受野有限的短板 |
| **DySample 替代固定上采样** | 从特征学习采样偏移量，边缘保持能力优于双线性插值和转置卷积 |
| **16× 总上采样** | 两次 4× DySample 串联，从 patch 分辨率恢复到原图 |
| **输出 1×1 卷积** | 最后一步仅做通道映射，不引入额外的空间混合 |
