import logging
import math
from enum import Enum
from typing import Sequence, Union

import torch
from torch import Tensor, nn

from .center_padding import CenterPadding

logger = logging.getLogger("basicsr")


class BackboneLayersSet(Enum):
    LAST = "LAST"
    FOUR_LAST = "FOUR_LAST"
    FOUR_EVEN_INTERVALS = "FOUR_EVEN_INTERVALS"


class PatchSizeAdaptationStrategy(Enum):
    CENTER_PADDING = "center_padding"
    NO_ADAPTATION = "never"


def _get_backbone_out_indices(
    model: nn.Module,
    backbone_out_layers: Union[list[int], BackboneLayersSet] = BackboneLayersSet.FOUR_EVEN_INTERVALS,
):
    """Get indices for output layers of the ViT backbone."""
    n_blocks = getattr(model, "n_blocks", 1)
    if isinstance(backbone_out_layers, list):
        out_indices = backbone_out_layers
    elif backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = list(range(n_blocks - 4, n_blocks))
    elif backbone_out_layers == BackboneLayersSet.FOUR_EVEN_INTERVALS:
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    else:
        raise ValueError(f"Unknown backbone_out_layers: {backbone_out_layers}")
    assert all(out_index < n_blocks for out_index in out_indices)
    return out_indices


# DINOv3 model directory names (resolved relative to project root pretrain/)
_MODEL_DIRS = {
    "vit_small": "dinov3-vits16-pretrain-lvd1689m",
    "vit_base":  "dinov3-vitb16-pretrain-lvd1689m",
    "vit_large": "dinov3-vitl16-pretrain-lvd1689m",
}


def _resolve_model_path(model_name: str) -> str:
    """Resolve model directory relative to project root pretrain/."""
    import os as _os
    # Search upward from this file to find the project root (where pretrain/ lives)
    _this_dir = _os.path.dirname(_os.path.abspath(__file__))
    _root = _this_dir
    for _ in range(3):  # basicsr/archs/dino_vit_arch_utils/ → project root
        _root = _os.path.dirname(_root)
    return _os.path.join(_root, "pretrain", _MODEL_DIRS[model_name])


def _load_dinov3_backbone(model_name: str) -> nn.Module:
    """Load a DINOv3 ViT backbone from local pretrain/ directory."""
    if model_name not in _MODEL_DIRS:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(_MODEL_DIRS.keys())}")
    from transformers import AutoModel
    model_path = _resolve_model_path(model_name)
    backbone = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    return backbone


def _extract_intermediate_layers(
    backbone: nn.Module,
    x: Tensor,
    out_indices: list[int],
) -> list[tuple[Tensor, Tensor]]:
    """Extract intermediate ViT layer features.

    DINOv3 HuggingFace model does not provide get_intermediate_layers.
    Uses output_hidden_states=True and manually splits CLS / register / patch tokens.

    Args:
        backbone: DINOv3 ViT model.
        x: Input tensor (B, C, H, W).
        out_indices: 0-indexed block indices to extract.

    Returns:
        List of tuples (patch_features [B, C, h, w], cls_token [B, C]).
    """
    outputs = backbone(x, output_hidden_states=True, return_dict=True)

    features = []
    for block_idx in out_indices:
        # hidden_states[0] = embedding output
        # hidden_states[1] = block 0, ..., hidden_states[12] = block 11
        hidden = outputs.hidden_states[block_idx + 1]  # (B, N, C)

        cls_token = hidden[:, 0, :]                    # (B, C)
        # DINOv3 has 4 register tokens after CLS; patch tokens follow
        patch_tokens = hidden[:, 5:, :]                 # (B, 256, C) for 16×16

        B, N, C = patch_tokens.shape
        h = w = int(math.sqrt(N))
        spatial = patch_tokens.transpose(1, 2).reshape(B, C, h, w)

        features.append((spatial, cls_token))

    return features


class DinoVisionTransformerWrapper(nn.Module):
    """Wrapper around a DINOv3 ViT that extracts intermediate layer features.

    Args:
        backbone_model: Pre-loaded DINOv3 ViT model.
        backbone_out_layers: Which intermediate layers to extract.
        use_backbone_norm: Not used for DINOv3 (kept for API compatibility).
        adapt_to_patch_size: How to handle inputs not divisible by patch_size.
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        backbone_out_layers: Union[str, list[int]] = "FOUR_EVEN_INTERVALS",
        use_backbone_norm: bool = False,
        adapt_to_patch_size: Union[str, PatchSizeAdaptationStrategy] = "center_padding",
    ):
        super().__init__()
        self.final_norm = use_backbone_norm
        self.backbone = backbone_model

        # Set n_blocks for _get_backbone_out_indices
        if not hasattr(self.backbone, 'n_blocks'):
            self.backbone.n_blocks = self.backbone.config.num_hidden_layers

        self.backbone_out_indices = _get_backbone_out_indices(
            self.backbone,
            backbone_out_layers=(
                backbone_out_layers if isinstance(backbone_out_layers, list)
                else BackboneLayersSet(backbone_out_layers)
            ),
        )

        # Compute per-block embed dimensions
        embed_dim = self.backbone.config.hidden_size
        n_blocks = self.backbone.config.num_hidden_layers
        self.embed_dims: Sequence[int] = [embed_dim] * n_blocks
        self.embed_dims = [self.embed_dims[idx] for idx in self.backbone_out_indices]

        # Patch size adaptation
        patch_size = self.backbone.config.patch_size

        if isinstance(adapt_to_patch_size, str):
            adapt_to_patch_size = PatchSizeAdaptationStrategy(adapt_to_patch_size)

        if adapt_to_patch_size is PatchSizeAdaptationStrategy.CENTER_PADDING:
            self.patch_size_adapter = CenterPadding(patch_size)
        elif adapt_to_patch_size is PatchSizeAdaptationStrategy.NO_ADAPTATION:
            self.patch_size_adapter = nn.Identity()
        else:
            raise ValueError(f"Unknown {adapt_to_patch_size=}")

        # Freeze backbone
        self.backbone.requires_grad_(False)

    @classmethod
    def from_model_name(
        cls,
        model_name: str = "vit_base",
        backbone_out_layers: Union[str, list[int]] = "FOUR_EVEN_INTERVALS",
        use_backbone_norm: bool = False,
        adapt_to_patch_size: Union[str, PatchSizeAdaptationStrategy] = "center_padding",
    ) -> "DinoVisionTransformerWrapper":
        """Factory method that loads a DINOv3 backbone by name from local paths."""
        backbone = _load_dinov3_backbone(model_name)
        return cls(
            backbone_model=backbone,
            backbone_out_layers=backbone_out_layers,
            use_backbone_norm=use_backbone_norm,
            adapt_to_patch_size=adapt_to_patch_size,
        )

    def forward(self, x: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Extract intermediate layer features.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            List of tuples (patch_features [B, C, h, w], cls_token [B, C]).
        """
        x = self.patch_size_adapter(x)
        outputs = _extract_intermediate_layers(
            self.backbone, x,
            out_indices=self.backbone_out_indices,
        )
        return outputs
