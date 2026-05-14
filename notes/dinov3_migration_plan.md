# DINOv2 → DINOv3 迁移方案

## 背景

当前 DDColor 通过 `torch.hub.load('facebookresearch/dinov2', ...)` 加载 DINOv2（patch_size=14），
现需切换为手动下载的 DINOv3 HuggingFace 模型（patch_size=16），位于：
```
E:\work\Code\models\facebook\
├── dinov3-vits16-pretrain-lvd1689m   (ViT-S/16, embed_dim=384)
├── dinov3-vitb16-pretrain-lvd1689m   (ViT-B/16, embed_dim=768)
└── dinov3-vitl16-pretrain-lvd1689m   (ViT-L/16, embed_dim=1024)
```

## DINOv2 vs DINOv3 关键差异

| 特性 | DINOv2 (当前) | DINOv3 (目标) |
|------|--------------|--------------|
| 加载方式 | `torch.hub.load` | `transformers.AutoModel.from_pretrained` |
| 权重格式 | torch.hub 内置 | `config.json` + `model.safetensors` |
| patch_size | **14** | **16** |
| num_hidden_layers | 12 | 12 |
| 中间层提取 | `get_intermediate_layers(n=..., reshape=True, return_class_token=True)` | 无此方法，需手动实现 |
| CenterPadding | 256→266 (ceil(256/14)*14) | **256→256 (无需填充, 256/16=16)** |
| FOUR_EVEN_INTERVALS | n_blocks=12 → [2,5,8,11] | 同左 [2,5,8,11] |
| CLS token | `return_class_token=True` 返回 tuple | 需手动从 hidden_states 分离 |
| register tokens | 无 | **4 个 register tokens**（需处理） |

## 修改清单

### 1. `basicsr/archs/dino_vit_arch_utils/dino_vit_wrapper.py` — 核心修改

#### 1.1 替换 `_HUB_MAP` 和 `_load_dinov2_backbone()`

```python
# 删除
_HUB_MAP = {...}
def _load_dinov2_backbone(model_name): ...

# 新增
_MODEL_PATHS = {
    "vit_small": "E:/work/Code/models/facebook/dinov3-vits16-pretrain-lvd1689m",
    "vit_base":  "E:/work/Code/models/facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vit_large": "E:/work/Code/models/facebook/dinov3-vitl16-pretrain-lvd1689m",
}

def _load_dinov3_backbone(model_name: str):
    from transformers import AutoModel
    model_path = _MODEL_PATHS[model_name]
    backbone = AutoModel.from_pretrained(model_path)
    return backbone
```

#### 1.2 新增中间层提取方法（替代 `get_intermediate_layers`）

DINOv3 HuggingFace 模型没有 `get_intermediate_layers` API，
需要注册 forward hooks 或使用 `output_hidden_states=True`：

```python
def _extract_intermediate_layers(backbone, x, out_indices, reshape=True):
    """Extract intermediate ViT layer features.
    
    DINOv3 HF model 不提供 get_intermediate_layers，
    改用 output_hidden_states=True + 手动提取。
    """
    outputs = backbone(
        x,
        output_hidden_states=True,
        return_dict=True,
    )
    # hidden_states: tuple of (B, N+1+4, C) for each layer
    # N+1 = patch_tokens + CLS, +4 = register_tokens
    features = []
    for idx in out_indices:
        hidden = outputs.hidden_states[idx]  # (B, N+1+4, C)
        cls_token = hidden[:, 0, :]          # CLS token
        patch_tokens = hidden[:, 1:, :]       # patch + register tokens

        if reshape:
            B = patch_tokens.shape[0]
            C = patch_tokens.shape[2]
            h = w = int(math.sqrt(patch_tokens.shape[1]))
            patch_tokens = patch_tokens.transpose(1, 2).reshape(B, C, h, w)

        features.append((patch_tokens, cls_token))
    return features
```

#### 1.3 修改 `DinoVisionTransformerWrapper.forward()`

```python
def forward(self, x):
    x = self.patch_size_adapter(x)
    outputs = _extract_intermediate_layers(
        self.backbone, x,
        out_indices=self.backbone_out_indices,
        reshape=True,
    )
    return outputs  # 4 × [(B, C, h, w), (B, C)]
```

### 2. `basicsr/archs/ddcolor_dinov3_arch.py` — 无需改动

接口不变：`model_name='vit_base'` → 内部调用 `from_model_name` → 自动路由到新的加载函数。

### 3. `ddcolor/pipeline.py` — 无需改动

`build_ddcolor_model` 的 DINOv3 分支参数不变。

### 4. 训练配置 `options/train/*.yml` — 无需改动

`model_name: vit_small/vit_base/vit_large` 保持不变，
`adapt_to_patch_size: center_padding` → 256 可被 16 整除，填充量为 0。

### 5. 推理脚本 `scripts/infer.py` — 已在上一轮修复，无需额外改动

## 数据流变化

```
之前 (DINOv2, patch_size=14):
  (B,3,256,256) → CenterPadding(14) → (B,3,266,266) → ViT → 19×19 patches

之后 (DINOv3, patch_size=16):
  (B,3,256,256) → CenterPadding(16) → (B,3,256,256) [无填充] → ViT → 16×16 patches
```

## 风险点

1. **DINOv3 的 HuggingFace 模型定义可能不在标准 transformers 中**，
   需要确认 `transformers.AutoModel.from_pretrained(path)` 能否直接加载。
   备选方案：使用 `dinov3` pip 包加载权重再包装。

2. **get_intermediate_layers 返回格式差异**：
   DINOv2 torch.hub 返回 `list[(patch_tokens[B,C,h,w], cls_token[B,C])]`，
   需确保 `_extract_intermediate_layers` 完全对齐此协议，
   因为 `WeightedFusion` 依赖此格式。

3. **register tokens 处理**：DINOv3 有 4 个 register tokens，
   它们与 patch tokens 混在一起（hidden[:, 1:, :]），
   代码中 `math.sqrt` 取整时需确保 token 数量是平方数。
   16×16=256, +1(CLS)+4(register)=261。`math.sqrt(260)` 不会得到整数。
   **需要显式去除 register tokens**。

4. **hidden_states 的层索引**：
   HuggingFace `hidden_states` 包含 embedding 层输出作为第 0 层，
   因此中间层索引可能需要 +1 偏移。需验证。

## 执行结果（2026-05-14）

### ✅ 步骤 1：验证 transformers 加载成功
- `transformers.AutoModel.from_pretrained(path, trust_remote_code=True)` 可加载 DINOv3ViTModel
- 确认 hidden_states: 13 层（1 embedding + 12 blocks），261 tokens（1 CLS + 4 register + 256 patches）

### ✅ 步骤 2：dino_vit_wrapper.py 已修改
- 删除 `_HUB_MAP` 和 `_load_dinov2_backbone`
- 新增 `_MODEL_PATHS` 指向本地路径
- 新增 `_load_dinov3_backbone()` 通过 transformers 加载
- 新增 `_extract_intermediate_layers()` 替代 `get_intermediate_layers`，切除 4 个 register tokens
- `DinoVisionTransformerWrapper.__init__` 直接从 `backbone.config` 获取 patch_size / hidden_size / num_layers

### ✅ 步骤 3：前向传播验证通过
- 输出 `(2, 2, 256, 256)` 与预期一致
- 参数量：Encoder 85.7M（冻结）+ Decoder 9.5M（可训练）= 95.1M

### ✅ 步骤 4：推理管线测试通过
- 输入 1210×915 → `ColorizationPipeline.process()` → 输出 1210×915 ✓
- 注意：当前无 DINOv3 checkpoint，推理使用随机初始化权重，结果无意义
- 需先训练 DINOv3 模型才能获得有效推理结果

### ⏳ 步骤 5：训练测试
- 未执行。确认训练前需要 DINOv3 checkpoint 或从头训练
