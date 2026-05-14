import torch
import torch.nn as nn

from basicsr.archs.dino_vit_arch_utils.dino_vit_wrapper import DinoVisionTransformerWrapper
from basicsr.archs.dino_vit_arch_utils.sdt_decoder import SDTColorizationHead
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DDColor_DinoV3_SDT(nn.Module):
    """DDColor variant using DINOv3 ViT encoder + SDT decoder for colorization.

    Replaces the ConvNeXt encoder and UNet/Transformer decoder with:
    - Frozen DINOv3 ViT backbone (multi-layer CLS-aware features)
    - SDTColorizationHead (WeightedFusion + DySample upsampling + CNN head)

    Compatible with the existing ColorModel training loop.
    Outputs 2-channel ab chrominance in LAB space.

    Args:
        model_name: 'vit_small' | 'vit_base' | 'vit_large'.
        input_size: (H, W) of input images.
        fusion_channels: Hidden dim in SDT decoder (default 256).
        num_output_channels: Output channels (2 for ab).
        do_normalize: Apply ImageNet normalization (ViT expects True).
        backbone_out_layers: 'FOUR_EVEN_INTERVALS' | 'FOUR_LAST' | 'LAST'.
        adapt_to_patch_size: 'center_padding' | 'never'.
    """

    def __init__(
        self,
        model_name='vit_base',
        input_size=(256, 256),
        fusion_channels=256,
        num_output_channels=2,
        do_normalize=True,
        backbone_out_layers='FOUR_EVEN_INTERVALS',
        adapt_to_patch_size='center_padding',
        **sdt_kwargs,
    ):
        super().__init__()

        self.input_size = input_size
        self.do_normalize = do_normalize

        # Frozen DINOv3 ViT encoder
        self.encoder = DinoVisionTransformerWrapper.from_model_name(
            model_name=model_name,
            backbone_out_layers=backbone_out_layers,
            use_backbone_norm=False,
            adapt_to_patch_size=adapt_to_patch_size,
        )
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        # SDT colorization decoder
        self.decoder = SDTColorizationHead(
            in_channels=list(self.encoder.embed_dims),
            fusion_channels=fusion_channels,
            n_output_channels=num_output_channels,
            use_cls_token=True,
            output_size=input_size,
            **sdt_kwargs,
        )

        # ImageNet normalization buffers
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) grayscale RGB (from L channel, ab channels set to 0).

        Returns:
            (B, 2, H, W) predicted ab chrominance channels.
        """
        H, W = x.shape[2:]

        if self.do_normalize:
            x = self.normalize(x)

        features = self.encoder(x)  # 4 × [(B, C, h, w), (B, C)]
        out = self.decoder(features, original_size=(H, W))

        return out
