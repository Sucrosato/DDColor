import logging
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


# Model name to torch.hub mapping
_HUB_MAP = {
    "vit_small": ("facebookresearch/dinov2", "dinov2_vits14"),
    "vit_base":  ("facebookresearch/dinov2", "dinov2_vitb14"),
    "vit_large": ("facebookresearch/dinov2", "dinov2_vitl14"),
}


def _load_dinov2_backbone(model_name: str) -> nn.Module:
    """Load a DINOv2 ViT backbone via torch.hub."""
    if model_name not in _HUB_MAP:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(_HUB_MAP.keys())}")
    repo, model = _HUB_MAP[model_name]
    backbone = torch.hub.load(repo, model)
    return backbone


class DinoVisionTransformerWrapper(nn.Module):
    """Wrapper around a DINOv2 ViT that extracts intermediate layer features.

    Args:
        backbone_model: Pre-loaded DINOv2 ViT model.
        backbone_out_layers: Which intermediate layers to extract.
        use_backbone_norm: Whether to apply final norm to extracted features.
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
        self.backbone_out_indices = _get_backbone_out_indices(
            self.backbone,
            backbone_out_layers=(
                backbone_out_layers if isinstance(backbone_out_layers, list)
                else BackboneLayersSet(backbone_out_layers)
            ),
        )

        # Compute per-block embed dimensions
        try:
            embed_dims = self.backbone.embed_dims
        except AttributeError:
            embed_dim = self.backbone.embed_dim
            n_blocks = self.backbone.n_blocks
            logger.warning(
                f"Backbone does not define embed_dims, using {[embed_dim] * n_blocks=} instead"
            )
            embed_dims = [embed_dim] * n_blocks
        self.embed_dims: Sequence[int] = [embed_dims[idx] for idx in self.backbone_out_indices]

        # Patch size adaptation
        try:
            input_pad_size = self.backbone.input_pad_size
        except AttributeError:
            patch_size = self.backbone.patch_size
            logger.warning(
                f"Backbone does not define input_pad_size, using {patch_size=} instead"
            )
            input_pad_size = patch_size

        if isinstance(adapt_to_patch_size, str):
            adapt_to_patch_size = PatchSizeAdaptationStrategy(adapt_to_patch_size)

        if adapt_to_patch_size is PatchSizeAdaptationStrategy.CENTER_PADDING:
            self.patch_size_adapter = CenterPadding(input_pad_size)
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
        """Factory method that loads a DINOv2 backbone by name via torch.hub."""
        backbone = _load_dinov2_backbone(model_name)
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
        outputs = self.backbone.get_intermediate_layers(
            x,
            n=self.backbone_out_indices,
            reshape=True,
            return_class_token=True,
            norm=self.final_norm,
        )
        return outputs
