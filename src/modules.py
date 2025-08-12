import torch.nn.functional as F
from torch import nn, cat, Tensor

###############################################################################
# Layers
###############################################################################

class DoubleConv2d(nn.Module):
    """ Two-layer 2D convolutional block with a PReLU activation in between. """

    # TODO: padding_mode: zeros, reflect, replicate

    def __init__(self, in_channels, out_channels, kernel_size=3, padding_mode="reflect", use_batchnorm=False):
        """ Initialize the DoubleConv2d layer.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            kernel_size (int, optional): The kernel size, by default 3.
            padding_mode (str, optional): The padding mode, by default "reflect".
            use_batchnorm (bool, optional): Whether to use batch normalization, by default False.
        """
        super().__init__()

        self.doubleconv2d = nn.Sequential(
            # ------- First block -------
            # First convolutional layer.
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode=padding_mode,
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
            # ------- Second block -------
            # Second convolutional layer.
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode=padding_mode,
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DoubleConv2d layer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.doubleconv2d(x)


class ResidualBlock(nn.Module):
    """ Two-layer 2D convolutional block (DoubleConv2d) 
    with a skip-connection to a sum."""

    def __init__(self, in_channels, kernel_size=3, **kws):
        """ Initialize the ResidualBlock layer.

        Args:
            in_channels (int): The number of input channels.
            kernel_size (int, optional): The kernel size, by default 3.
        """
        super().__init__()
        self.residualblock = DoubleConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            **kws,
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the ResidualBlock layer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output tensor of shape (batch_size, in_channels, height, width).
        """
        x = x + self.residualblock(x)
        return x


class DenseBlock(ResidualBlock):
    """ Two-layer 2D convolutional block (DoubleConv2d) with a skip-connection 
    to a concatenation (instead of a sum used in ResidualBlock)."""

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DenseBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, 2 x in_channels, height, width).
        """
        return cat([x, self.residualblock(x)], dim=1)


class ConvTransposeBlock(nn.Module):
    """ Upsampler block with ConvTranspose2d. """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sr_kernel_size,
        zoom_factor,
        use_dropout=False,
        use_batchnorm=False,
    ):
        """ Initialize the ConvTransposeBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        zoom_factor : int
            The zoom factor.
        use_dropout : bool, optional
            Whether to use dropout, by default False.
        use_batchnorm : bool, optional
            Whether to use batchnorm, by default False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sr_kernel_size = sr_kernel_size
        self.zoom_factor = zoom_factor
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.zoom_factor,
                padding=0,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Dropout(p=0.5) if self.use_dropout else nn.Identity(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )

    def forward(self, x):
        """ Forward pass of the ConvTransposeBlock layer.

        Args:
            x (Tensor): The input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.upsample(x)


class PixelShuffleBlock(ConvTransposeBlock):

    """PixelShuffle block with ConvTranspose2d for sub-pixel convolutions. """

    # TODO: conv 레이어 사이에 dropout 레이어 추가 실험

    def __init__(self, **kws):
        super().__init__(**kws)
        assert self.in_channels % self.zoom_factor ** 2 == 0
        self.in_channels = self.in_channels // self.zoom_factor ** 2
        self.upsample = nn.Sequential( # overide ConvTransposeBlock.upsample
            nn.PixelShuffle(self.zoom_factor),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Dropout(p=0.5) if self.use_dropout else nn.Identity(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )


class Resize(nn.Module):
    def __init__(self, size=None, interpolation='bilinear', align_corners=False):
        super().__init__()
        self.size = size
        self.mode = interpolation
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode=self.mode, align_corners=self.align_corners)

###############################################################################
# Models(SRCNN, BicubicUpscaledBaseline)
###############################################################################


class SRCNN(nn.Module):
    """
    Super-resolution CNN.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        kernel_size,
        residual_layers,
        output_size,
        zoom_factor,
        sr_kernel_size,
        padding_mode="reflect",
        use_dropout=False,
        use_batchnorm=False,
        **kws,
    ) -> None:
        """
        Args:
            in_channels (int): The number of input channels.
            hidden_channels (int): The number of hidden channels.
            out_channels (int): The number of output channels.
            kernel_size (int): The kernel size.
            residual_layers (int): The number of residual layers.
            output_size (tuple): The output size.
            zoom_factor (int): The zoom factor.
            sr_kernel_size (int): The kernel size of the SR convolution.
            padding_mode (str): The padding mode.
            use_dropout (bool): Whether to use dropout.
            use_batchnorm (bool): Whether to use batch normalization.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.residual_layers = residual_layers
        self.output_size = output_size
        self.zoom_factor = zoom_factor
        self.sr_kernel_size = sr_kernel_size
        self.padding_mode = padding_mode
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout

        # Image encoder
        self.encoder = DoubleConv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            padding_mode=self.padding_mode,
            use_batchnorm=self.use_batchnorm,
        )

        #### Delete Fusion block ####

        self.body = nn.Sequential(
            *( # generator unpacking
                ResidualBlock(  
                    in_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    padding_mode=self.padding_mode,
                    use_batchnorm=self.use_batchnorm,
                )
                for _ in range(residual_layers)
            )
        )

        ## Super-resolver (upsampler + renderer)
        self.sr = PixelShuffleBlock(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            sr_kernel_size=self.sr_kernel_size,
            zoom_factor=self.zoom_factor,
            use_dropout=self.use_dropout,
            use_batchnorm=self.use_batchnorm,
        )

        self.resize = Resize(
            size=self.output_size,
            interpolation="bilinear",
            align_corners=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Super-resolution CNN.

        Args:
            x (Tensor): The input tensor (low-res images).
            Shape: (batch_size, channels, height, width).

        Returns:
            Tensor: The output tensor (super-resolved images).
            Shape: (batch_size, channels, height, width).
        """

        # Encoded shape: (B, C, H, W)
        x = self.encoder(x)

        x = self.body(x)
        x = self.sr(x)

        # Ensure output size of (B, C, H, W)
        x = self.resize(x)

        return x


class BicubicUpscaledBaseline(nn.Module):
    """ Bicubic upscaled single-image baseline. """

    def __init__(
        self, output_size, interpolation="bicubic", device=None, **kws
    ):
        """
        Initialize the BicubicUpscaledBaseline.

        Args:
            output_size (tuple of int): The output size.
            interpolation (str, optional): The interpolation method, by default 'bicubic'.
                - Available methods: 'nearest', 'bilinear', 'bicubic'.
        """
        super().__init__()
        assert interpolation in ["bilinear", "bicubic", "nearest"]
        self.output_size = output_size
        self.interpolation = interpolation
        self.resize = Resize(self.output_size, interpolation=self.interpolation)

    def forward(self, x: Tensor) -> Tensor:

        return self.resize(x)
