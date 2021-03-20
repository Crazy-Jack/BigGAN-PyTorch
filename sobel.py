from kornia.filters import sobel
import torch.nn as nn
import torch 

class Sobel(nn.Module):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Args:
        normalized (bool): if True, L1 norm of the kernel is set to 1.
        eps (float): regularization number to avoid NaN during backprop. Default: 1e-6.

    Return:
        torch.Tensor: the sobel edge gradient magnitudes map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = torch.rand(1, 3, 4, 4)
        >>> output = Sobel()(input)  # 1x3x4x4
    """

    def __init__(self,
                 normalized: bool = True, eps: float = 1e-6) -> None:
        super(Sobel, self).__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'normalized=' + str(self.normalized) + ')'

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return sobel(input, self.normalized, self.eps)


if __name__ == '__main__':
    s = Sobel()
    x = torch.rand(10, 3, 4, 4)
    out = s(x)  
    print(out.sum(1))