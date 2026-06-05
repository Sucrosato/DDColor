import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


# ---------------------------------------------------------------------------
# DySample — Dynamic upsampling via learned offsets
# ---------------------------------------------------------------------------

class DySample(nn.Module):
    """Dynamic upsampling: predicts per-pixel sampling offsets from features.

    Args:
        in_channels: Input feature channels.
        scale: Upsampling factor (default 2).
        style: 'lp' (learned position) or 'pl' (pixel shuffle).
        groups: Group count for offset prediction.
        dyscope: Whether to use dynamic scope modulation.
    """

    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.reshape(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).reshape(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)


# ---------------------------------------------------------------------------
# WeightedFusion — fuse 4 ViT layers with learned weights + CLS readout
# ---------------------------------------------------------------------------

class WeightedFusion(nn.Module):
    """Fuses 4 ViT intermediate layers via learned softmax weights.

    Each layer's patch tokens are concatenated with the expanded CLS token
    (readout), projected to a common dimension, then weighted-summed.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_channels, bias=False), nn.GELU())
            for in_dim in in_channels
        ])

        self.readout_projects = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * in_dim, in_dim), nn.GELU())
            for in_dim in in_channels
        ])

        self.layer_weights = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        assert len(features) == len(self.projections)
        projected_layer_tokens = []

        for i, layer_feature in enumerate(features):
            if isinstance(layer_feature, tuple):
                spatial_tensor, cls_token = layer_feature
                B, C, H, W = spatial_tensor.shape
                spatial_tokens = spatial_tensor.flatten(2).permute(0, 2, 1).contiguous()
                cls_token_expanded = cls_token.unsqueeze(1).expand_as(spatial_tokens)
                tokens_with_cls = torch.cat((spatial_tokens, cls_token_expanded), dim=-1)
                enhanced_tokens = self.readout_projects[i](tokens_with_cls)
                projected_tokens = self.projections[i](enhanced_tokens)
            else:
                if layer_feature.dim() == 4:
                    B, C, H, W = layer_feature.shape
                    layer_tokens = layer_feature.flatten(2).permute(0, 2, 1).contiguous()
                else:
                    layer_tokens = layer_feature
                projected_tokens = self.projections[i](layer_tokens)
            projected_layer_tokens.append(projected_tokens)

        layer_weights = F.softmax(self.layer_weights, dim=0)
        fused_tokens = torch.zeros_like(projected_layer_tokens[0])
        for i, projected_tokens in enumerate(projected_layer_tokens):
            fused_tokens = fused_tokens + layer_weights[i] * projected_tokens
        return fused_tokens


# ---------------------------------------------------------------------------
# SpatialDetailEnhancer — depthwise conv + residual for spatial details
# ---------------------------------------------------------------------------

class SpatialDetailEnhancer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.activation(x + residual)
        return x


# ---------------------------------------------------------------------------
# DySampleUpsamplerWrapper — two DySample(scale=2) chained for 4x upsampling
# ---------------------------------------------------------------------------

class DySampleUpsamplerWrapper(nn.Module):
    def __init__(self, feature_dim: int, scale_factor: int = 4, style: str = 'lp',
                 groups: int = 4, dyscope: bool = False):
        super().__init__()
        self.scale_factor = scale_factor
        self.feature_dim = feature_dim
        self.dysample1 = nn.Sequential(
            DySample(feature_dim, scale=2, style=style, groups=groups, dyscope=dyscope),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True))
        self.dysample2 = nn.Sequential(
            DySample(feature_dim, scale=2, style=style, groups=groups, dyscope=dyscope),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True))

    def forward(self, features: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        x = self.dysample1(features)
        x = self.dysample2(x)
        return x


# ---------------------------------------------------------------------------
# ColorQueryBottleneck — learnable color queries + Transformer for global color reasoning
# ---------------------------------------------------------------------------

class ColorQueryBottleneck(nn.Module):
    """Injects learnable color queries via Transformer cross-attention.

    Between WeightedFusion and DySample upsampling, a set of learnable color
    queries attend to the fused spatial features through cross-attention,
    producing a global color context that is broadcast back to enhance the
    per-pixel features before upsampling.

    Args:
        d_model: Feature dimension (default 256).
        num_queries: Number of learnable color queries (default 100).
        num_layers: Transformer decoder layers (default 3).
        nheads: Attention heads (default 8).
        dim_feedforward: FFN hidden dim (default 1024).
    """

    def __init__(self, d_model=256, num_queries=100, num_layers=3,
                 nheads=8, dim_feedforward=1024):
        super().__init__()
        from basicsr.archs.ddcolor_arch_utils.transformer_utils import (
            SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP,
        )

        self.d_model = d_model
        self.num_queries = num_queries

        # learnable color queries
        self.query_feat = nn.Embedding(num_queries, d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer decoder layers (SA → CA → FFN)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                SelfAttentionLayer(d_model, nheads),
                CrossAttentionLayer(d_model, nheads),
                FFNLayer(d_model, dim_feedforward),
            ]))

        self.decoder_norm = nn.LayerNorm(d_model)

        # Query → color embedding
        self.color_mlp = MLP(d_model, d_model, d_model, 3)

        # Fuse global color context back into spatial features
        self.out_proj = nn.Conv2d(d_model * 2, d_model, 1)

    def forward(self, spatial_features):
        """Forward pass.

        Args:
            spatial_features: (B, C, H, W) fused features from WeightedFusion.

        Returns:
            (B, C, H, W) enhanced features with global color context.
        """
        B, C, H, W = spatial_features.shape

        # Flatten spatial features as cross-attention memory: (L, B, C) for MHA
        src = spatial_features.flatten(2).permute(2, 0, 1)  # (N, B, C)

        # Initialize queries: (L, B, C) for MHA
        query = self.query_feat.weight.unsqueeze(1).repeat(1, B, 1)   # (Q, B, C)
        pos = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)

        # Transformer decoder iterations
        for sa, ca, ffn in self.layers:
            query = sa(query, query_pos=pos)
            query = ca(query, src)
            query = ffn(query)

        query = self.decoder_norm(query)

        # Query → color embedding → global pool → broadcast to spatial
        query = query.permute(1, 0, 2)                          # (B, Q, C)
        color_embed = self.color_mlp(query)                     # (B, Q, C)
        global_color = color_embed.mean(dim=1, keepdim=True)    # (B, 1, C)
        global_color = global_color.transpose(1, 2).reshape(B, C, 1, 1)
        global_color = global_color.expand(B, C, H, W)          # (B, C, H, W)

        # Fuse with original features (residual)
        fused = torch.cat([spatial_features, global_color], dim=1)  # (B, 2C, H, W)
        out = self.out_proj(fused)                                   # (B, C, H, W)

        return out + spatial_features  # residual connection


# ---------------------------------------------------------------------------
# SDTColorizationHead — SDT decoder adapted for 2-channel ab colorization
# ---------------------------------------------------------------------------

class SDTColorizationHead(nn.Module):
    """SDT decoder head for image colorization.

    Takes 4 ViT intermediate layer features (with optional CLS tokens),
    fuses them via WeightedFusion, then upsamples via two DySample stages
    (total 16x upsampling) and produces 2-channel ab chrominance output.

    Args:
        in_channels: Per-layer embed dims from ViT (e.g. [768,768,768,768]).
        fusion_channels: Dimension after WeightedFusion (default 256).
        n_output_channels: Output channels (2 for ab).
        use_cls_token: Whether input features include CLS tokens as tuples.
        output_size: Target (H, W) for final bilinear resize.
    """

    def __init__(
        self,
        in_channels: List[int],
        fusion_channels: int = 256,
        n_output_channels: int = 2,
        use_cls_token: bool = True,
        output_size: tuple = (256, 256),
        use_color_queries: bool = False,
        num_queries: int = 100,
        query_layers: int = 3,
        upsample_mode: str = '4x2',  # '4x2' (2 stages of 4x) or '2x4' (4 stages of 2x)
        **kwargs
    ):
        super().__init__()
        assert len(in_channels) == 4, f"Expected 4 ViT layer channels, got {len(in_channels)}"
        assert upsample_mode in ('4x2', '2x4'), f"upsample_mode must be '4x2' or '2x4'"

        self.use_cls_token = use_cls_token
        self.fusion_channels = fusion_channels
        self.output_size = output_size
        self.upsample_mode = upsample_mode

        self.weighted_fusion = WeightedFusion(in_channels, fusion_channels)
        self.detail_enhancer = SpatialDetailEnhancer(fusion_channels)

        # Color Query Bottleneck
        if use_color_queries:
            self.color_query_bottleneck = ColorQueryBottleneck(
                d_model=fusion_channels, num_queries=num_queries,
                num_layers=query_layers)
        else:
            self.color_query_bottleneck = nn.Identity()

        if upsample_mode == '4x2':
            # 2 stages of 4x DySample (original)
            self.upsample_1 = DySampleUpsamplerWrapper(
                fusion_channels, scale_factor=4, style='lp', groups=4, dyscope=True)
            self.refinement_1 = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True))

            self.upsample_2 = DySampleUpsamplerWrapper(
                fusion_channels, scale_factor=4, style='lp', groups=4, dyscope=True)
            self.refinement_2 = nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True))
        else:  # '2x4'
            # 4 stages of 2x DySample
            block_2x = lambda: [
                DySample(fusion_channels, scale=2, style='lp', groups=4, dyscope=True),
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True),
            ]
            refinement = lambda: nn.Sequential(
                nn.Conv2d(fusion_channels, fusion_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(fusion_channels),
                nn.ReLU(inplace=True))

            self.upsample_1 = nn.Sequential(*block_2x())
            self.upsample_2 = nn.Sequential(*block_2x())
            self.upsample_3 = nn.Sequential(*block_2x())
            self.upsample_4 = nn.Sequential(*block_2x())
            self.refinement_1 = refinement()
            self.refinement_2 = refinement()
            self.refinement_3 = refinement()
            self.refinement_4 = refinement()

        self.output_conv = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_output_channels, kernel_size=1))

    def forward(self, features, original_size=None):
        """Forward pass.

        Args:
            features: List of 4 tensors or tuples from ViT encoder.
            original_size: (H, W) to resize output to (optional, falls back to self.output_size).

        Returns:
            Tensor (B, n_output_channels, H, W).
        """
        if isinstance(features[0], tuple):
            features_with_cls = features
        else:
            features_with_cls = features

        spatial_tensors = [f[0] if isinstance(f, tuple) else f for f in features]
        B = spatial_tensors[0].shape[0]
        H_patches = spatial_tensors[0].shape[2]
        W_patches = spatial_tensors[0].shape[3]

        # Fuse + enhance
        fused_tokens = self.weighted_fusion(features_with_cls)
        fused_spatial = fused_tokens.permute(0, 2, 1).contiguous().reshape(
            B, self.fusion_channels, H_patches, W_patches)
        enhanced = self.detail_enhancer(fused_spatial)

        # Color Query Bottleneck (skip for nn.Identity)
        enhanced = self.color_query_bottleneck(enhanced)

        # Upsample 16x
        if self.upsample_mode == '4x2':
            x = self.upsample_1(enhanced)
            x = self.refinement_1(x)
            x = self.upsample_2(x)
            x = self.refinement_2(x)
        else:  # '2x4'
            x = self.upsample_1(enhanced)
            x = self.refinement_1(x)
            x = self.upsample_2(x)
            x = self.refinement_2(x)
            x = self.upsample_3(x)
            x = self.refinement_3(x)
            x = self.upsample_4(x)
            x = self.refinement_4(x)

        out = self.output_conv(x)

        # Resize to target resolution
        target_size = original_size if original_size is not None else self.output_size
        if target_size is not None and (out.shape[2] != target_size[0] or out.shape[3] != target_size[1]):
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)

        return out
