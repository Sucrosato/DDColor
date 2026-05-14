import itertools
import math
import torch


class CenterPadding(torch.nn.Module):
    """Pad input so that H and W are multiples of `multiple` (ViT patch_size)."""

    def __init__(self, multiple: int):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def forward(self, x):
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:-3:-1]))
        return torch.nn.functional.pad(x, pads)

    def __extra_repr__(self) -> str:
        return f"multiple={self.multiple}"
