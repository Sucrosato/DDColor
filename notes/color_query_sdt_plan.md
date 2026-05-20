# DINOv3+SDT 加入 Color Queries 方案

## 动机

DDColor 的 color queries 通过可学习的颜色 token + Transformer cross-attention
从多尺度特征中蒸馏颜色信息。SDT 的 WeightedFusion 仅做加权融合，
缺少显式的颜色推理。将两者结合可让 SDT 也拥有"调色板"能力。

## 插入位置

在 WeightedFusion 之后、DySample 上采样之前插入 Color Query 模块：

```
DINOv3 ViT (冻结)
  │  4 × [(B, C, 16, 16), (B, C)]
  ▼
WeightedFusion + CLS Readout
  │  (B, 256, 16, 16)
  ▼
┌─────────────────────────────────┐
│  Color Query Bottleneck (新增)  │
│                                 │
│  query_feat + query_embed       │  ← 100 个可学习颜色 token
│       ↓                         │
│  Transformer Decoder × N        │  ← Self-Attn + Cross-Attn + FFN
│    cross-attn ← fused features  │
│       ↓                         │
│  MLP → color_embed (256)        │
│       ↓                         │
│  F.interpolate → (B, 256, 16, 16)│ ← 广播回空间
│  Conv1x1 → (B, 256, 16, 16)     │
│  残差连接 + fused features       │
└────────────┬────────────────────┘
             │  (B, 256, 16, 16) 增强特征
             ▼
       SpatialDetailEnhancer
             ▼
       DySample ×4 → DySample ×4
             ▼
       output_conv → (B, 2, 256, 256)
```

## 实现

### 新建模块

在 `sdt_decoder.py` 中新增 `ColorQueryBottleneck`：

```python
class ColorQueryBottleneck(nn.Module):
    def __init__(self, d_model=256, num_queries=100, num_layers=6,
                 nheads=8, dim_feedforward=1024):
        super().__init__()
        # 可学习颜色查询
        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer 解码器层
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nheads, dim_feedforward)
            for _ in range(num_layers)
        ])

        # 输出投影
        self.color_mlp = MLP(d_model, d_model, d_model, 3)
        self.out_proj = nn.Conv2d(d_model * 2, d_model, 1)

    def forward(self, spatial_features):
        # spatial_features: (B, 256, H, W)
        B, C, H, W = spatial_features.shape

        # 空间特征展平为 key/value
        src = spatial_features.flatten(2).permute(0, 2, 1)  # (B, N, 256)

        # 初始化查询
        query = self.query_feat.weight.unsqueeze(0).repeat(B, 1, 1)
        pos = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)

        # Transformer 迭代
        for layer in self.layers:
            query = layer(query, src, query_pos=pos)

        # query → 颜色嵌入
        color_embed = self.color_mlp(query)  # (B, 100, 256)

        # 广播颜色信息到空间: 全局平均池化 query → 空间广播
        global_color = color_embed.mean(dim=1, keepdim=True)  # (B, 1, 256)
        global_color = global_color.transpose(1, 2).reshape(B, C, 1, 1)
        global_color = global_color.expand(B, C, H, W)

        # 与原始特征拼接后投影
        enhanced = torch.cat([spatial_features, global_color], dim=1)
        out = self.out_proj(enhanced)  # (B, 256, H, W)

        return out + spatial_features  # 残差连接
```

### 集成到 SDTColorizationHead

在 `sdt_decoder.py` 的 `SDTColorizationHead.__init__` 中添加可选参数：

```python
class SDTColorizationHead(nn.Module):
    def __init__(self, ..., use_color_queries=False, num_queries=100,
                 query_layers=6):
        ...
        if use_color_queries:
            self.color_query_bottleneck = ColorQueryBottleneck(
                d_model=fusion_channels, num_queries=num_queries,
                num_layers=query_layers)
        else:
            self.color_query_bottleneck = nn.Identity()
```

forward 中：
```python
def forward(self, features, original_size=None):
    fused_tokens = self.weighted_fusion(features)
    fused_spatial = fused_tokens.permute(0, 2, 1).reshape(B, C, H, W)
    enhanced = self.detail_enhancer(fused_spatial)

    # ← 插入 Color Query Bottleneck
    enhanced = self.color_query_bottleneck(enhanced)

    x = self.upsample_1(enhanced)
    ...
```

## 参数量影响

| 模块 | 新增参数 |
|---|---|
| Color Queries (100×256×2) | ~51K |
| 6 层 Transformer (SA+CA+FFN) | ~9.5M |
| MLP + output proj | ~0.5M |
| **合计** | ~**10M** |

总模型：~31M → ~41M（仍远小于原始 DDColor 的 55M）

## 配置文件

```yaml
network_g:
  type: DDColor_DinoV3_SDT
  model_name: vit_small
  use_color_queries: true    # ← 新增
  num_queries: 100           # ← 新增
  query_layers: 6            # ← 新增
  ...
```

## 预期效果

- Color queries 为 SDT 提供显式的全局颜色推理能力
- 相比纯 SDT，颜色一致性和饱和度可能提升
- 残差连接确保最差情况退化到原始 SDT 行为
