# DINOv3 ViT + SDT → DDColor 图像着色移植方案

## 架构概览

将 AnyDepth 的 DINOv3 ViT 编码器 + SDT 解码器移植到 DDColor，替换 ConvNeXt + ColorDecoder。

```
(B, 3, 256, 256) 灰度 RGB
     │
     ▼  CenterPadding → (B, 3, 266, 266)
┌────────────────────────────┐
│ DINOv3 ViT Encoder (冻结)  │  86.6M params (vit_base)
│ 4×[(B,768,19,19),(B,768)] │
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│ WeightedFusion (可训练)    │  softmax 加权融合 4 层 + CLS readout
│ → (B, 256, 19, 19)        │
│ SpatialDetailEnhancer      │  dw-conv3x3 + 残差
└────────────┬───────────────┘
             ▼
┌────────────────────────────┐
│ DySample×2 (4×)           │  动态上采样: 19→38→76
│ refinement_1               │
├────────────────────────────┤
│ DySample×2 (4×)           │  76→152→304
│ refinement_2               │
├────────────────────────────┤
│ output_conv                │  256→128→32→2(ch) ab 色度
│ → (B, 2, 304, 304)        │
└────────────┬───────────────┘
             ▼
  F.interpolate → (B, 2, 256, 256)
```

## 创建的文件

| 文件 | 说明 |
|------|------|
| `basicsr/archs/dino_vit_arch_utils/__init__.py` | 包初始化 |
| `basicsr/archs/dino_vit_arch_utils/center_padding.py` | CenterPadding（pad 到 patch_size 倍数） |
| `basicsr/archs/dino_vit_arch_utils/dino_vit_wrapper.py` | DinoVisionTransformerWrapper + from_model_name 工厂 |
| `basicsr/archs/dino_vit_arch_utils/sdt_decoder.py` | DySample + WeightedFusion + SDTColorizationHead |
| `basicsr/archs/ddcolor_dinov3_arch.py` | DDColor_DinoV3_SDT（注册于 ARCH_REGISTRY） |
| `options/train/train_ddcolor_dinov3_sdt.yml` | 训练配置 |

## 修改的文件

| 文件 | 改动 |
|------|------|
| `ddcolor/pipeline.py` | build_ddcolor_model() 支持 dinov3_small/base/large |

## 模型规格

| 变体 | ViT | Encoder 参数 | Decoder 参数 | 总可训练 |
|------|-----|-------------|-------------|---------|
| vit_small | DINOv2 ViT-S/14 | ~22M | ~9.5M | ~9.5M |
| vit_base | DINOv2 ViT-B/14 | ~87M | ~9.5M | ~9.5M |
| vit_large | DINOv2 ViT-L/14 | ~304M | ~9.5M | ~9.5M |

## 关键设计决策

1. **归一化**：`do_normalize=True`，ViT 需要 ImageNet 归一化
2. **分辨率**：256→266(pad)→304(SDT)→256(resize)
3. **Encoder 冻结**：始终 `requires_grad_(False) + eval()`
4. **ColorModel 兼容**：`forward(B,3,H,W) → (B,2,H,W)`，训练循环无需修改
5. **权重加载**：首次运行通过 `torch.hub.load('facebookresearch/dinov2', ...)` 自动下载

## 训练启动

```bash
python basicsr/train.py -opt options/train/train_ddcolor_dinov3_sdt.yml
```

## 推理

```bash
python scripts/infer.py \
    --model_path experiments/ddcolor_dinov3_sdt/models/net_g_latest.pth \
    --model_size dinov3_base \
    --input ./test_images \
    --output ./results
```
