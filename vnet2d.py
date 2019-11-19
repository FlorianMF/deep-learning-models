import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_network import BaseNetwork

"""
First we define the blocks of Convolution-Normalization-Activation 
as well as the block for each depth level (DownConvBlockBlock and UpConvBlock).
This way we do not have to define each layer individually and
we can easily adapt the depth of the U-Net.
"""
class ConvNormActiv(nn.Module):
    """  
    Sequence of a convolutional, a normalization and an activation layer.
    Inherits from the torch.nn.Module class.
    """
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, norm="instance", activ="prelu"):
        """
        Parameters
        ----------
        in_size: int or tuple of ints 
                dimensions of the input image batch
        out_size: int or tuple of ints 
                dimensions of the image batch after the forward pass
        kernel_size: int or tuple of ints 
                dimensions of the filter kernel
        stride: int or tuple of ints 
                value(s) for the stride (shift in pixels after each covolutional operation)
        padding: int or tuple of ints 
                number of zero-value pixels added at the image borders 
                (allows for instance to maintain the image dimensions)
        norm: str
                defines the normalization layer type (Instance Norm or Batch Norm)
        activ: str
                defines the activation layer type (PReLu)
        """

        super().__init__()
        
        # define the normalization layer type
        if norm == "instance":
            norm_layer = nn.InstanceNorm2d(out_size)
        elif norm == "batch":
            norm_layer = nn.BatchNorm2d(out_size)
        else:
            raise NotImplementedError("Only InstanceNorm and BatchNorm are implemented momentarily")
        
        # define the activation layer type
        if activ == "prelu":
            activ_layer = nn.PReLU()
        else:
            raise NotImplementedError("Only PReLu ist implemented momentarily")
        
        # combine all three layers
        self.conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                  norm_layer,
                                  activ_layer)

    def forward(self, x):
        """
        Forward pass of the tensor
        """
        return self.conv(x)



class DownConvBlock(nn.Module):
    """
    Image processing block sequencing several ConvNormActiv blocks followed by a down convolution operation.
    Inherits from the torch.nn.Module class.
    """
    def __init__(self, in_size, out_size, kernel_size=2, stride=2, num_layers=1,
                 down=True, norm="instance", activ="prelu"):
        """
        Parameters
        ----------
        in_size: int or tuple of ints 
                dimensions of the input image batch
        out_size: int or tuple of ints 
                dimensions of the image batch after the forward pass
        kernel_size: int or tuple of ints 
                dimensions of the filter kernel
        stride: int or tuple of ints 
                value(s) for the stride (shift in pixels after each covolutional operation)
        num_layers: int
                number of ConvBlocks which shall be sequenced.
        down: boolean
                defines if a ConvBlock should be applied which reduces the width and height of the images
        norm: str
                defines the normalization layer type (Instance Norm or Batch Norm)
        activ: str
                defines the activation layer type (PReLu)
        """
        
        super().__init__()

        #create a torch.nn.ModuleList to work with the layers like a list
        self.layers = nn.ModuleList()

        # Append as many ConvNormActiv blocks as defined in num_layers
        for _ in range(num_layers):
            self.layers.append(ConvNormActiv(in_size, in_size, kernel_size=5, stride=1, padding=2, 
                                             norm=norm, activ=activ))
        
        # if not in the lower most level of the encoder reduce the image's width and height
        self.down = down
        if self.down:
            self.down_conv = ConvNormActiv(in_size, out_size, kernel_size, stride, padding=0, 
                                           norm=norm, activ=activ)

    def forward(self, x):
        """
        Forward pass of the tensor

        Returns
        -------
        x: tensor
            result after the conv-norm-activ layers and the down convolution
        before_down: tensor
            result after the conv-norm-activ layers, but before the down convolution
        """       
        tensor_in = x.copy()
        for mod in self.layers:
            x = mod(x)
        
        # perform the residual addition
        before_down = x + tensor_in

        # if not in the lower most level of the encoder reduce the image's width and height
        if self.down:
            x = self.down_conv(before_down)

        return x, before_down


class UpConv(nn.Module):
    """
    Image processing block sequencing an upconvolution layer followed by two ConvNormActiv blocks.
    Inherits from the torch.nn.Module class.
    """
    def __init__(self, in_size, out_size, kernel_size=2, stride=2, num_layers=1, norm="instance", activ="prelu"):
        """
        Parameters
        ----------
        in_size: int or tuple of ints
                dimensions of the input image batch
        out_size: int or tuple of ints
                dimensions of the image batch after the forward pass
        kernel_size: int or tuple of ints
                dimensions of the filter kernel
        stride: int or tuple of ints
                value(s) for the stride (shift in pixels after each covolution operation)
        num_layers: int
                number of ConvBlocks which shall be sequenced.
        norm: str
                defines the normalization layer type (Instance Norm or Batch Norm)
        activ: str
                defines the activation layer type (PReLu)
        """
        super().__init__()

       # define the normalization layer type
        if norm == "instance":
            norm_layer = nn.InstanceNorm2d(out_size)
        elif norm == "batch":
            norm_layer = nn.BatchNorm2d(out_size)
        else:
            raise NotImplementedError("Only InstanceNorm and BatchNorm are implemented momentarily")
        
        # define the activation layer type
        if activ == "prelu":
            activ_layer = nn.PReLU()
        else:
            raise NotImplementedError("Only PReLu ist implemented momentarily")

        # define the upconvolution operation
        self.up_conv = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel_size, stride),
                                     norm_layer,
                                     activ_layer)
        
        #create a torch.nn.ModuleList to work with the layers like a list
        self.layers = nn.ModuleList()
        # Append as many ConvNormActiv blocks as defined in num_layers
        for i in range(num_layers):
            self.layers.append(ConvNormActiv(out_size*2, out_size*2, 5, padding=2, norm=norm, activ_layer))

    def forward(self, before_down, from_up):
        """ 
        Paramters
        ----------
        from_down: tensor from the encoder pathway
        from_up: upconvoluted tensor from the decoder pathway
        
        Returns
        -------
        x: tensor
            resulting tensor
        """
        # upconvolution
        from_up = self.up_conv(from_up)

        # concatenate the upconvoluted tensor with the one from the skip connection
        from_up = torch.cat([from_up, before_down], dim=1)
        x = from_up.copy()

        # apply the convolution
        for mod in self.layers:
            x = mod(x)

        return x + from_up

# Define the V-Net model via a loop
class VNet_2D(BaseNetwork):
    """
    V-Net with arbitrary depth.
    Inherits from the BaseNetwork.
    This network does not include a final Softmax activation layer. This is done inside the loss functions.
    This makes it easier to adapt the loss functions.
    """
    def __init__(self, num_classes, channels_in=1, depth=4, start_channels=16,
                 norm="instance", activ="prelu", 
                 init_type="xavier_normal", **kwargs):
        super().__init__(depth=depth, start_channels=start_channels,
                         dropout=dropout, norm=norm, init_type=init_type)
        """
        Parameters
        ----------
        num_classes: int
                number of classes which shall be predicted = output tensor's channel dimensions
        channels_in: int
                number of channels of the input tensor (e.g. for one RGB image: 3)
        depth: int
                number of depth levels in the U-Net
        start_channels: int (usually 2^n)
                number of channels after the second convolution
        norm: str
                defines the normalization layer type (Instance Norm or Batch Norm)
        activ: str
                defines the activation layer type (ReLu)
        init_type: str
                defines the initialization method used (xavier, kaiming he, ...)
        """
        # define the very first convolution
        self.conv_start = ConvNormActiv(channels_in, start_channels, kernel_size=1, padding=0, norm=norm)

        # create the DownConvBlocks and UpConvBlocks using a ModuleList
        # This allows to work with them like a standard list
        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()

        outs = 0
        # define the number of layers per depth level according to the paper
        num_layers = [1, 2] + [3]*(depth - 2)

        # create the encoder pathway and add to a list
        for idx in range(depth):
            ins = start_channels if idx == 0 else outs
            outs = start_channels*(2**(idx+1)) if down else outs
            down = True if idx < depth - 1 else False

            self.down_convs.append(DownConvBlock(ins, outs, down=down, kernel_size=2, 
                                    num_layers=num_layers[idx], norm=norm, acitv=activ))

        lower = outs
        # create the decoder pathway and add to a list
        # !! decoding only requires depth-1 blocks !!
        for idx in range(depth - 1):
            ins = lower if idx < 2 else ins // 2
            outs = ins // 2 if idx == 0 else ins // 4

            self.up_convs.append(UpConv(ins, outs, kernel_size=2, 
                                    num_layers=num_layers[idx], norm=norm, activ=activ))

        # final 1x1 convultion to reduce the channel size to the number of classes
        self.conv_final = nn.Conv2d(outs*2, num_classes, kernel_size=1, padding=0)
        
        # initialize the model's parameters (calls the method of the BaseNetwork)
        self.reset_params(init_type)

    def forward(self, x):
        """
        Forward pass through the entire network
        """
        # create a list to memorize the tensors used for the skip connections
        encoder_outs = []

        x = self.conv_start(x)
        # encoder pathway, save outputs for merging
        for module in self.down_convs:
            x, before_down = module(x)
            encoder_outs.append(before_down)

        for idx, module in enumerate(self.up_convs):
            before_down = encoder_outs[-(idx+2)]
            x = module(before_down, x)

        x = self.conv_final(x)
 
        return x

    def closure(self, criterion, optimizer, metrics, inputs, targets, fold=0, **kwargs):
        """
        Handles the predictions made by the model, calcutes the loss and the different metrics and 
        performs the optimizer step.

        Paramters
        ---------
        criterion: class
            Function or class to calculate the loss value
        optimizer: class
            Class which is called to perform an optimization step
        metrics: dict
            Functions or classes to calculate the wished metrics
        fold: int
            Current Fold in cross validation (default: 0)
        kwargs: dict
            additional keyword arguments
        
        Returns
        -------
        outputs: tensor
                class scores
        preds: tensor
                discrete segmentation mask (prediction)
        loss: tensor
                scalar loss value
        metric_vals: dict
                metric values
        """

        # compute the network's prediction
        metric_vals = {}
        outputs = self.forward(inputs)
        
        # calculate the loss
        loss = criterion(outputs, targets)

        # determine the resulting segmentation masks using the highest class score
        # calculate the different metric values
        with torch.no_grad(): # no need to keep track of the gradient when determining the segmentation mask
            preds = outputs.max(dim=1)[1]
            for key, metric_fn in metrics.items():
                    metric_vals[key] = metric_fn(preds.cpu(), targets.cpu())

        # perform an optimizer step, only if we are training the network
        if self.training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return outputs, preds, loss.detach().cpu(), metric_vals


if __name__ == "__main__":
    net = VNet_2D(num_classes=5)
    x = torch.rand([1, 1, 128, 128])

    output = net(x)
    print("out image size: ", output.shape)

    masks = (torch.max(output, dim=1))[1]
