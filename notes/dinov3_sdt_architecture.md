# DINOv3 ViT + SDT 模型架构详解

## 整体数据流

```
输入灰度图 (B, 3, H, W)
    │
    ▼
ImageNet 归一化 (mean/std)
    │
    ▼
CenterPadding (确保 H、W 能被 patch_size=16 整除)
    │
    ▼
┌─────────────────────────────────────────────┐
│  DINOv3 ViT (冻结, 不参与训练)                │
│  - 通过 output_hidden_states=True 提取       │
│    4 个中间层的 hidden states                │
│  - 每个 hidden state 拆分为:                 │
│    · CLS token [B, C]                        │
│    · Register tokens (4个, 丢弃)              │
│    · Patch tokens → reshape 为 [B, C, h, w]  │
│                                              │
│  输出: 4 × [(B, C, h, w), (B, C)]            │
│        (spatial_features, cls_token)         │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  WeightedFusion (融合 4 层特征)               │
│  - 对每层: CLS token 广播拼接到 patch tokens   │
│    然后做 readout projection + linear proj   │
│  - 4 层特征用 learnable softmax 权重加权求和   │
│  - 输出: (B, N, 256) 融合后的 token 序列      │
└─────────────────────────────────────────────┘
    │
    ▼ reshape → (B, 256, h, w)
    │
    ▼
┌─────────────────────────────────────────────┐
│  SpatialDetailEnhancer                       │
│  - Depthwise Conv 3x3 + BN + ReLU            │
│  - 残差连接 (增强空间细节)                    │
│  - 输出: (B, 256, h, w)                       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  ColorQueryBottleneck (可选, 近期新增)        │
│  - 100 个可学习 color queries                 │
│  - 3 层 Transformer Decoder                   │
│    (Self-Attn → Cross-Attn to 空间特征 → FFN) │
│  - queries 通过 MLP → global color embedding  │
│  - 广播回空间并与原特征 concat + 1x1 Conv     │
│  - 残差连接                                   │
│  - 输出: (B, 256, h, w)                       │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  DySample x2 (第一次 4× 上采样)               │
│  - DySample: 从特征学习采样偏移量              │
│  - scale=2 依次调用两次, 各接 Conv+BN+ReLU    │
│  - refinement_1: Conv3x3+BN+ReLU             │
│  - 输出: (B, 256, 4h, 4w)                     │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  DySample x2 (第二次 4× 上采样)               │
│  - 同上结构                                   │
│  - 输出: (B, 256, 16h, 16w)                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Output Conv                                 │
│  - Conv 256→128, ReLU                        │
│  - Conv 128→32, ReLU                         │
│  - Conv 32→2 (ab 通道)                       │
│  - 输出: (B, 2, 16h, 16w)                    │
└─────────────────────────────────────────────┘
    │
    ▼
F.interpolate → target_size (bilinear resize)
    │
    ▼
输出 ab 色度通道 (B, 2, H, W)
```

## 各模块详解

### 1. DINOv3 ViT 编码器 (`basicsr/archs/dino_vit_arch_utils/dino_vit_wrapper.py`)

- **模型来源**: 从 `pretrain/` 目录加载 HuggingFace 格式的本地权重（`dinov3-vits16-pretrain-lvd1689m` / `vitb16` / `vitl16`）
- **冻结**: 编码器完全不参与训练，仅提取特征
- **层选择策略** (`backbone_out_layers`):
  - `FOUR_EVEN_INTERVALS`（默认）: 对于 24 层 ViT，取第 `[4, 11, 17, 23]` 层
  - `FOUR_LAST`: 取最后 4 层
  - `LAST`: 仅取最后一层
- **Token 拆分**: 每层 hidden state 的维度是 `(B, 1+4+N, C)`，其中 1 个 CLS、4 个 register tokens、N 个 patch tokens。代码丢弃 register tokens（`[:, 5:, :]`），保留 CLS + patch tokens
- **Patch 对齐**: `CenterPadding` 模块在输入边缘做对称 padding，确保 H 和 W 能被 `patch_size=16` 整除

### 2. WeightedFusion (`basicsr/archs/dino_vit_arch_utils/sdt_decoder.py:107-157`)

融合 4 层 ViT 特征的核心模块：

- **CLS Readout**: 将 CLS token 广播到每个 patch token 位置并拼接，通过一个 `Linear(2C→C) + GELU` 将全局语义注入到每个空间位置
- **投影**: 每层用 `Linear(C→256) + GELU` 投影到统一维度
- **加权融合**: 4 个可学习的标量权重经过 softmax 后加权求和

### 3. SpatialDetailEnhancer (`basicsr/archs/dino_vit_arch_utils/sdt_decoder.py:164-176`)

轻量的空间细节增强模块：`Depthwise Conv 3×3 → BN → ReLU`，带残差连接。用分组卷积以极低的参数量增强局部空间细节。

### 4. ColorQueryBottleneck (`basicsr/archs/dino_vit_arch_utils/sdt_decoder.py:210-294`)（可选）

将原始 DDColor 的 "learnable color queries" 思想引入 SDT 解码器：

- **100 个可学习 color queries**（`nn.Embedding`），每个是 256 维向量
- **3 层 Transformer Decoder**：Self-Attention → Cross-Attention（以空间特征为 memory）→ FFN
- 更新后的 queries 经过 3 层 MLP 得到 color embedding
- 对所有 query 的 color embedding 做 **mean pooling** 得到全局颜色上下文
- 将全局颜色广播到每个空间位置，与原始特征 concat 后经 1×1 Conv 融合
- 最后通过残差连接加回原始特征

### 5. DySample (`basicsr/archs/dino_vit_arch_utils/sdt_decoder.py:29-101`)

动态上采样模块，核心思想是**从特征本身学习采样偏移量**：

- 一个 `Conv2d(C→2*G*scale²)` 预测每个像素在低分辨率特征图上的采样偏移
- 使用 `F.grid_sample` 按偏移量采样，实现内容自适应的上采样
- `style='lp'`（learned position）: 直接预测偏移量
- `dyscope=True`: 额外预测一个 scope 因子来调制偏移幅度
- 相比传统的双线性/双三次插值或转置卷积，DySample 能更好地保留边缘和纹理细节

### 6. DySampleUpsamplerWrapper (`basicsr/archs/dino_vit_arch_utils/sdt_decoder.py:183-203`)

将两个 `DySample(scale=2)` 串联，每个后面接 `Conv3×3 + BN + ReLU`，实现 4× 上采样。整个 decoder 中有两个这样的模块串联，实现总共 **16× 上采样**（从 ViT 的 patch 分辨率恢复到原图分辨率）。

---

## 推理流程 (`ddcolor/pipeline.py`)

1. **输入**: OpenCV 读取的 BGR `uint8` 图像
2. **提取 L 通道**: 将原图缩放到 `input_size × input_size`（默认 512），转为 LAB，取 L 通道；ab 通道置为 0 作为 "灰度" 输入
3. **构造输入**: LAB(灰度) → RGB，得到灰度 RGB 张量 `(1, 3, 512, 512)`
4. **模型前向**: 输入经过归一化 → DINOv3 ViT → SDT 解码器 → 输出 ab 通道 `(1, 2, 512, 512)`
5. **后处理**: 将输出 ab resize 回原图尺寸，与原始 L 通道拼接为 LAB → 转 BGR → 输出 `uint8` 彩色图

**ColorQuery 自动检测**: `build_ddcolor_model` 会自动检查 checkpoint 的 state_dict 中是否包含 `color_query_bottleneck` 相关的 key，从而自动决定是否启用 ColorQueryBottleneck。

### `input_size` 参数的作用

1. **控制模型推理分辨率**: 输入图像被缩放到 `input_size × input_size`
2. **作为解码器最终输出的目标尺寸**: 解码器最后一步用 `F.interpolate` 确保输出精确匹配目标尺寸

`input_size` 越大，模型看到的细节越多，但计算量也越大。训练时通常用 256（省显存），推理时用 512（更好的细节）。

---

## 关键设计要点

| 设计 | 说明 |
|---|---|
| **编码器冻结** | DINOv3 ViT 完全冻结，仅充当特征提取器，不参与梯度更新 |
| **CLS Token 利用** | 通过 CLS readout 将全局语义注入每个空间位置，而非简单丢弃 |
| **16× 上采样** | 两个 DySample×4 串联，从 patch 分辨率恢复全分辨率 |
| **动态上采样** | DySample 学习采样偏移，比固定插值更好地保留边缘 |
| **Color Query（可选）** | 一组可学习的 color tokens 通过交叉注意力对全局颜色进行推理，弥补纯 CNN decoder 缺乏全局颜色理解的问题 |
| **ImageNet 归一化** | 由于 DINOv3 在 ImageNet 上预训练，输入需要做 ImageNet 标准化 |

---

## 相关源文件

| 文件 | 说明 |
|---|---|
| `basicsr/archs/ddcolor_dinov3_arch.py` | 顶层模型 `DDColor_DinoV3_SDT` |
| `basicsr/archs/dino_vit_arch_utils/dino_vit_wrapper.py` | DINOv3 ViT 编码器封装 |
| `basicsr/archs/dino_vit_arch_utils/sdt_decoder.py` | SDT 解码器全部模块 |
| `basicsr/archs/dino_vit_arch_utils/center_padding.py` | Patch 对齐 padding |
| `ddcolor/pipeline.py` | 推理管线 |
| `scripts/infer.py` | 推理入口脚本 |
