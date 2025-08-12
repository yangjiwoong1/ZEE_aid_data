import kornia
import torch
import torch.nn.functional as F
from torch import Tensor
from pytorch_msssim import ms_ssim


def tv_loss(x: Tensor) -> Tensor:
    """
    총 변동(Total Variation) 손실 함수
    입력 이미지에서 인접한 픽셀값들 간의 절대 차이의 합을 계산
        - 이미지의 부드러움을 측정하는데 사용되며, 값이 클수록 이미지에 급격한 변화가 많다는 것을 의미

    Params
    ------
    x : Tensor
        (batch_size, channels, height, width) 형태의 텐서

    Returns
    -------
    Tensor
        (batch_size,) 형태의 텐서
    """
    height, width = x.shape[-2:]
    # kornia.losses.total_variation은 이미 모든 차원에 대해 평균을 계산하여 [B] 형태를 반환
    return kornia.losses.total_variation(x) / (height * width)


def ms_ssim_loss(y_hat, y, window_size):
    """
    다중 스케일 구조적 유사도(MS-SSIM) 손실 함수
    이미지의 구조적 유사도를 여러 스케일에서 측정하여 더 정교한 품질 평가가 가능함
    Ref: https://www.cns.nyu.edu/pub/eero/wang03b.pdf

    Params
    ------
    y_hat : Tensor
        모델의 출력값, (batch_size, channels, height, width) 형태의 텐서
    y : Tensor
        정답 레이블, (batch_size, channels, height, width) 형태의 텐서  
    window_size : int
        MS-SSIM 계산에 사용되는 가우시안 커널의 크기

    Returns
    -------
    Tensor
        (batch_size,) 형태의 텐서
    """
    # ssim은 낮을수록 좋기 때문에 1에서 빼줌
    return 1 - ms_ssim(y_hat, y, data_range=1, win_size=window_size, size_average=False)


def ssim_loss(y_hat, y, window_size=5):
    """ Structural Similarity loss.
    See: http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf

    Params
    ------
    y_hat : Tensor
        (batch_size, channels, height, width) 형태의 텐서
    y : Tensor
        (batch_size, channels, height, width) 형태의 텐서
    window_size : int
        SSIM 계산에 사용되는 가우시안 커널의 크기, 기본값 5
        가우시안 필터로 이미지를 약간 블러처리한 후 유사도를 계산 -> 이미지의 세세한 디테일 보다 전반적인 구조를 비교(인간의 시각 시스템 반영)

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return kornia.losses.ssim_loss(
        y_hat, y, window_size=window_size, reduction="none"
    ).mean(
        dim=(-1, -2, -3)
    )  # over C, H, W


def mae_loss(y_hat, y):
    """ Mean Absolute Error (L1) loss.
    Sum of all the absolute differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.l1_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W


def mse_loss(y_hat, y):
    """ Mean Squared Error (L2) loss.
    Sum of all the squared differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.mse_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W


def psnr_loss(y_hat, y):
    """ Peak Signal to Noise Ratio (PSNR) loss.
    The logarithm of base ten of the mean squared error between the label 
    and the output, multiplied by ten.

    In the proper form, there should be a minus sign in front of the equation, 
    but since we want to maximize the PSNR, 
    we minimize the negative PSNR (loss), thus the leading minus sign has been omitted.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 10.0 * torch.log10(mse_loss(y_hat, y))
