import torch
import torch.nn as nn


####### MedNeXt Block   #######                  
class MedNeXtBlock(nn.Module):
    """
    A Mednext block for 3D convolutions, consisting of three convolutional layers
    with group normalization and GeLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        exp_r (int): Expansion ratio.
        kernel_size (int): Size of the convolution kernel.
        groups (int): Number of groups for group convolution.
    """
    def __init__(self, in_channels,out_channels, exp_r, kernel_size, groups):
        super().__init__()

        # 1st conv
        self.conv1 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 1,
            padding = kernel_size//2,
            groups = groups,
        )

        # Group Normalization
        self.group_norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        
        # 2nd convolution Expansion
        self.conv2 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
        )
        
        # GeLU activations
        self.gelu_activation = nn.GELU(approximate='none')
        
        # 3rd convolution Compression
        self.conv3 = nn.Conv3d(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
        )
 
    def forward(self, input_data):
        x1 = input_data
        x1 = self.conv1(x1)
        x1 = self.group_norm(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu_activation(x1)
        x1 = self.conv3(x1)
        return x1+input_data
    
    
####### Down-Sampling Block   #######                  
class downBlock(nn.Module):
    """
    A down-sampling block using 3D convolutions, consisting of three convolutional layers
    with group normalization, GeLU activation, and an additional down-sampling convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        exp_r (int): Expansion ratio.
        kernel_size (int): Size of the convolution kernel.
        groups (int): Number of groups for group convolution.
    """
    def __init__(self, in_channels, out_channels, exp_r, kernel_size, groups):
        super().__init__()

        ## extra convolutional layer (residual conv)
        self.convext = nn.Conv3d(
                    in_channels = in_channels,
                    out_channels = out_channels,
                    kernel_size = 1,
                    stride = 2,
                )

        # 1st conv
        self.conv1 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = groups,
        )

        # Group Normalization
        self.group_norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        
        # 2nd convolution Expansion
        self.conv2 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
        )
        
        # GeLU activations
        self.gelu_activation = nn.GELU(approximate='none')
        
        # 3rd convolution Compression
        self.conv3 = nn.Conv3d(
            in_channels = exp_r*in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
        )
 
    def forward(self, input_data):
        x1 = input_data
        extra_input = self.convext(x1)
        x1 = self.conv1(x1)
        x1 = self.group_norm(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu_activation(x1)
        x1 = self.conv3(x1)
        x1 = extra_input + x1
        return x1
    

####### Up-Sampling Block   #######                  
class upBlock(nn.Module):
    """
    An up-sampling block using 3D convolutions, consisting of three convolutional layers
    with group normalization, GeLU activation, and an additional up-sampling transposed convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        exp_r (int): Expansion ratio.
        kernel_size (int): Size of the convolution kernel.
        groups (int): Number of groups for group convolution.
    """
    def __init__(self, in_channels, out_channels, exp_r, kernel_size, groups):
        super().__init__()

        ## extra convolutional layer
        self.convext = nn.ConvTranspose3d(
                    in_channels = in_channels,
                    out_channels = out_channels//2,
                    kernel_size = 1,
                    stride = 2,
                )

        # 1st conv
        self.conv1 = nn.ConvTranspose3d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = 2,
            padding = kernel_size//2,
            groups = groups,
       )

        # Group Normalization
        self.group_norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        
        # 2nd convolution Expansion
        self.conv2 = nn.Conv3d(
            in_channels = in_channels,
            out_channels = exp_r*in_channels,
            kernel_size = 1,
            stride = 1,
        )
        
        # GeLU activations
        self.gelu_activation = nn.GELU(approximate='none')
        
        # 3rd convolution Compression
        self.conv3 = nn.Conv3d(
            in_channels = exp_r*in_channels,
            out_channels = out_channels//2,
            kernel_size = 1,
            stride = 1,
        )
 
    def forward(self, input_data):
        x1 = input_data
        extra_input = self.convext(x1)
        x1 = self.conv1(x1)
        x1 = self.group_norm(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu_activation(x1)
        x1 = self.conv3(x1)
        x1 = nn.functional.pad(extra_input, (1,0,1,0,1,0)) + nn.functional.pad(x1, (1,0,1,0,1,0))
        return x1
    
 
####### The Small MedNeXt Architecture  #######          
class MedNeXt(nn.Module):
    """
    The MedNeXt network for 3D medical image processing, consisting of stem, encoder, bottleneck, and decoder blocks.

    Args:
        in_channels (int): Number of input channels.
        n_channels (int): Number of output channels for the first convolutional layer.
        exp_r (list of int): List of expansion ratios for each block.
        kernel_size (int): Size of the convolution kernel.
        groups (int or None): Number of groups for group convolution. If None, defaults to in_channels.
    """
    def __init__(self, in_channels, n_channels, exp_r, kernel_size, groups=None):
        super().__init__()
        if groups is None:
            groups = in_channels

        ## Stem Block
        self.stem_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=n_channels,
            kernel_size=1,
            stride=1,
        )

        ## Encoder blocks
        self.encoder_block_1 = nn.Sequential(*[MedNeXtBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            exp_r=exp_r[0],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])
        self.down_block_1 = downBlock(
            in_channels=n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=kernel_size,
            groups=groups,
        )

        self.encoder_block_2 = nn.Sequential(*[MedNeXtBlock(
            in_channels=2 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[1],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])
        self.down_block_2 = downBlock(
            in_channels=2 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=kernel_size,
            groups=groups,
        )

        self.encoder_block_3 = nn.Sequential(*[MedNeXtBlock(
            in_channels=4 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[2],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])
        self.down_block_3 = downBlock(
            in_channels=4 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=kernel_size,
            groups=groups,
        )

        self.encoder_block_4 = nn.Sequential(*[MedNeXtBlock(
            in_channels=8 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[3],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])
        self.down_block_4 = downBlock(
            in_channels=8 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=kernel_size,
            groups=groups,
        )

        ## Bottleneck block
        self.bottleneck_block = nn.Sequential(*[MedNeXtBlock(
            in_channels=16 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[4],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])

        ## Upward blocks
        self.up_block_1 = upBlock(
            in_channels=16 * n_channels,
            out_channels=16 * n_channels,
            exp_r=exp_r[5],
            kernel_size=kernel_size,
            groups=groups,
        )
        self.decoder_block_1 = nn.Sequential(*[MedNeXtBlock(
            in_channels=8 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[5],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])

        self.up_block_2 = upBlock(
            in_channels=8 * n_channels,
            out_channels=8 * n_channels,
            exp_r=exp_r[6],
            kernel_size=kernel_size,
            groups=groups,
        )
        self.decoder_block_2 = nn.Sequential(*[MedNeXtBlock(
            in_channels=4 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[6],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])

        self.up_block_3 = upBlock(
            in_channels=4 * n_channels,
            out_channels=4 * n_channels,
            exp_r=exp_r[7],
            kernel_size=kernel_size,
            groups=groups,
        )
        self.decoder_block_3 = nn.Sequential(*[MedNeXtBlock(
            in_channels=2 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[7],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])

        self.up_block_4 = upBlock(
            in_channels=2 * n_channels,
            out_channels=2 * n_channels,
            exp_r=exp_r[8],
            kernel_size=kernel_size,
            groups=groups,
        )
        self.decoder_block_4 = nn.Sequential(*[MedNeXtBlock(
            in_channels=n_channels,
            out_channels=n_channels,
            exp_r=exp_r[8],
            kernel_size=kernel_size,
            groups=groups,
        )for i in range(2)])

        ## Out Block
        self.out_conv = nn.ConvTranspose3d(
            in_channels=n_channels,
            out_channels=4,
            kernel_size=1,
        )

    def forward(self, x):          
        # Stem Block
        x1 = self.stem_conv(x)
    
        # Encoder
        x2 = self.encoder_block_1(x1)
        x3 = self.down_block_1(x2)
        x4 = self.encoder_block_2(x3)
        x5 = self.down_block_2(x4)
        x6 = self.encoder_block_3(x5)
        x7 = self.down_block_3(x6)
        x8 = self.encoder_block_4(x7)
        x9 = self.down_block_4(x8)
    
        # Bottleneck
        x10 = self.bottleneck_block(x9)
    
        # Decoder
        x11 = self.up_block_1(x10)
        x12 = self.decoder_block_1(x8+x11)
        x13 = self.up_block_2(x12)
        x14 = self.decoder_block_2(x6+x13)
        x15 = self.up_block_3(x14)
        x16 = self.decoder_block_3(x4+x15)
        x17 = self.up_block_4(x16)
        x18 = self.decoder_block_4(x2+x17)
    
        # Out Block
        x19 = self.out_conv(x18)
        
        return x16
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    in_channels = 4
    exp_r=[2,3,4,4,4,4,4,3,2]
    kernel_size = 3
    n_channels=16
    groups = in_channels
    model = MedNeXt(in_channels=in_channels, n_channels=n_channels, exp_r=exp_r, kernel_size=kernel_size, groups=groups)
    model.to(device)

    input_tensor = torch.randn(1, in_channels, 32, 32, 32)  
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", count_parameters(model))

