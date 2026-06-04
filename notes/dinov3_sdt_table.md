# DINOv3 SDT 架构细节表

| 模块 | 子层 | 参数 | 输入形状 | 输出形状 |
|---|---|---|---|---|
| **输入** | — | — | — | $(B, 3, 256, 256)$ |
| **CenterPadding** | — | multiple=16 | $(B, 3, H, W)$ | $(B, 3, \lceil H/16\rceil \cdot 16, \lceil W/16\rceil \cdot 16)$ |
| **Patch Embedding** | Conv2d $3\to 384$, $k=16$, $s=16$ | — | $(B, 3, 256, 256)$ | $(B, 261, 384)$ |
| **DINOv3 ViT-S** | $\times$12 Transformer Blocks | $d=384$, $h=6$, FFN=1536 | $(B, 261, 384)$ | $(B, 261, 384) \times 13$ |
| **Token 拆分** | 提取层索引 $[2,5,8,11]$, 丢弃 Register | — | $(B, 261, 384)$ | $4\times$ {$(B,384,16,16),~(B,384)$} |
| **WeightedFusion** | CLS Readout: Linear $768\!\to\!384$ | $\times$4 | $(B,256,384)+(B,256,384)$ | $(B, 256, 384)$ |
| | Projection: Linear $384\!\to\!256$ | $\times$4 | $(B, 256, 384)$ | $(B, 256, 256)$ |
| | Softmax 加权求和 | 4 个可学习权重 | $4\times$ $(B,256,256)$ | $(B, 256, 256)$ |
| **Reshape** | permute + reshape → 空间格式 | — | $(B, 256, 256)$ | $(B, 256, 16, 16)$ |
| **SpatialDetailEnhancer** | DWConv $3\!\times\!3$, groups=256, 残差 | — | $(B, 256, 16, 16)$ | $(B, 256, 16, 16)$ |
| **ColorQueryBottleneck** | 100 个可学习 Color Queries | $Q=100$, $d=256$ | — | — |
| | Cross-Attn (Q→空间特征) | $h=8$ | $(100,B,256)$ + $(256,B,256)$ | $(100, B, 256)$ |
| | Self-Attn (Q→Q) | $h=8$ | $(100, B, 256)$ | $(100, B, 256)$ |
| | FFN $256\!\to\!1024\!\to\!256$ | — | $(100, B, 256)$ | $(100, B, 256)$ |
| | $\times$3 层 (SA→CA→FFN) | 共 3 轮 | $(100, B, 256)$ | $(100, B, 256)$ |
| | Mean Pool → 全局颜色上下文 | — | $(B, 100, 256)$ | $(B, 256, 1, 1)$ |
| | Concat + Conv1$\times$1(512→256) + 残差 | — | $(B, 512, 16, 16)$ | $(B, 256, 16, 16)$ |
| **DySample 阶段 1** | DySample $\times$2 (lp, groups=4, scope) | $s=2$ | $(B,256,16,16)$ | $(B,256,32,32)$ |
| | Conv3$\times$3 + BN + ReLU | — | $(B,256,32,32)$ | $(B,256,32,32)$ |
| | DySample $\times$2 (lp, groups=4, scope) | $s=2$ | $(B,256,32,32)$ | $(B,256,64,64)$ |
| | Conv3$\times$3 + BN + ReLU | — | $(B,256,64,64)$ | $(B,256,64,64)$ |
| | Conv3$\times$3 + BN + ReLU (refinement) | — | $(B,256,64,64)$ | $(B,256,64,64)$ |
| **DySample 阶段 2** | DySample $\times$2 (lp, groups=4, scope) | $s=2$ | $(B,256,64,64)$ | $(B,256,128,128)$ |
| | Conv3$\times$3 + BN + ReLU | — | $(B,256,128,128)$ | $(B,256,128,128)$ |
| | DySample $\times$2 (lp, groups=4, scope) | $s=2$ | $(B,256,128,128)$ | $(B,256,256,256)$ |
| | Conv3$\times$3 + BN + ReLU | — | $(B,256,256,256)$ | $(B,256,256,256)$ |
| | Conv3$\times$3 + BN + ReLU (refinement) | — | $(B,256,256,256)$ | $(B,256,256,256)$ |
| **Output Conv** | Conv3$\times$3 256→128, ReLU | — | $(B,256,256,256)$ | $(B,128,256,256)$ |
| | Conv3$\times$3 128→32, ReLU | — | $(B,128,256,256)$ | $(B,32,256,256)$ |
| | Conv1$\times$1 32→2 | — | $(B,32,256,256)$ | $(B, 2, 256, 256)$ |
| **输出** | ab 色度通道 | — | $(B, 2, 256, 256)$ | — |

> **注**：ColorQueryBottleneck 为可选模块（`use_color_queries=True`），若不启用则为 $\text{Identity}$ 直接透传。B 为 batch size，输入形状以 256×256 为例。
