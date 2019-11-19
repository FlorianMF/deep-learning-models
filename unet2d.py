import torch
import torch.nn.functional as F
from torch import nn
from .base_network import BaseNetwork

"""
First we define the blocks of Convolution-Normalization-Activation 
as well as the block for each depth level (DownConvBlock and UpConvBlock).
This way we do not have to define each layer individually and
we can easily adapt the depth of the U-Net.
"""
class ConvNormActiv(nn.Module):
    """  
    Sequence of a convolutional, a normalization and an activation layer.
    Inherits from the torch.nn.Module class.
    """

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, norm="instance", activ="relu"):
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
                defines the activation layer type (ReLu)
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
        if activ == "relu":
            activ_layer = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError("Only ReLu ist implemented momentarily")
        
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
    Image processing block sequencing two ConvNormActiv blocks followed by a Pooling operation.
    Inherits from the torch.nn.Module class.
    """
    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1,
                 pooling=None, norm="instance", activ="relu"):
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
        pooling: str
                defines the pooling operation after the convolution block, 
                per default None so that no pooling is executed
        norm: str
                defines the normalization layer type (Instance Norm or Batch Norm)
        activ: str
                defines the activation layer type (ReLu)
        """
        super().__init__()
        
        # combine two ConvNormActiv blocks
        self.conv = nn.Sequential(ConvNormActiv(in_size,  out_size, kernel_size, stride, 
                                                padding, norm, activ),
                                  ConvNormActiv(out_size, out_size, kernel_size, stride, 
                                                padding, norm, activ))
        # define the pooling layer
        if pooling == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling == "average":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        elif pooling is None:  # if pooling is None do not perform a pooling operation
            self.pool = None
        else: 
            raise NotImplementedError("Only MaxPooling and AveragePooling are implemented.")

    def forward(self, x):
        """
        Forward pass of the tensor

        Returns
        -------
        x: tensor
            result after the conv-norm-activ and the pooling operations
        before_pool: tensor
            result after the conv-norm-activ, but before the pooling operations
        """
        x = self.conv(x)

        # keep the result before the pooling operation for the skipping connection
        before_pool = x  
        
        # apply pooling if not in the lower most level of the U-Net
        if self.pool:    
            x = self.pool(x)
        
        return x, before_pool


class UpConvBlock(nn.Module):
    """
    Image processing block sequencing an upconvolution layer followed by two ConvNormActiv blocks.
    Inherits from the torch.nn.Module class.
    """

    def __init__(self, in_size, out_size, kernel_size=3, stride=1, padding=1, 
                 merge_mode='concat', up_mode='transpose', norm="instance", activ="relu"):
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
        padding: int or tuple of ints
                number of zero-value pixels added at the image borders 
                (allows for instance to maintain the image dimensions)
        merge_mode: str
                defines how the images from the upsampling operation and 
                the skip connections are combined
        up_mode: str
                defines the upconvolution layer type
        activ: str
                defines the activation layer type (ReLu)
        """
        super().__init__()

        # define the upconvolution operation
        if up_mode == 'transpose':
            self.UpConvBlock = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':  # combine a bilinear upconvolution with a ConvNormActiv block
            self.UpConvBlock = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2, align_corners=None),
                                             ConvNormActiv(in_size, out_size, norm=norm, activ=activ))
        else:
            raise NotImplementedError(f"'{up_mode}' is no valid option."
                    "Only 'transpose' and 'upsample' are implemented up_mode types.")
        
        # define the merging operation
        self.merge_mode = merge_mode
        if self.merge_mode == 'concat': # input channel size doubles due to concatenation
            conv1 = ConvNormActiv(2*out_size, out_size, kernel_size, stride, padding, norm, activ)
        elif self.merge_mode == "add":
            conv1 = ConvNormActiv(out_size, out_size, kernel_size, stride, padding, norm, activ)
        else:
            raise NotImplementedError(f"'{merge_mode} is no valid option." 
                    "Only 'concat' and 'add' are implemented merge_mode types.")

        conv2 = ConvNormActiv(out_size, out_size, kernel_size, stride, padding, norm)
        self.conv = nn.Sequential(conv1, conv2)

    def forward(self, from_down, from_up):
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
        from_up = self.UpConvBlock(from_up)

        # combine the upconvoluted tensor with the one from the skip
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), dim=1) # concatenate along the channel dimension
        else:  # merge_mode == "add"
            x = from_up + from_down
        
        # apply the convolutions
        x = self.conv(x)
        return x


# Define the U-Net model via a loop
class UNet_2D(BaseNetwork):
    """
    U-Net with arbitrary depth.
    Inherits from the BaseNetwork.
    This network does not include a final Softmax activation layer. This is done inside the loss functions.
    This makes it easier to adapt the loss functions.
    """
    def __init__(self, num_classes, channels_in=1, depth=4, 
                 start_channels=32, kernel_size=3,
                 merge_mode='concat', up_mode='transpose', pooling="max",
                 norm="instance", activ ="relu", init_type="xavier_normal",  
                 **kwargs):
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
                number of channels after the first convolution
        kernel_size: int or tuple of ints
                dimensions of the filter kernel
        merge_mode: str
                defines how the images from the upsampling operation and 
                the skip connections are combined
        up_mode: str
                defines the upconvolution layer type
        pooling: str
                defines the pooling operation after the convolution block 
        norm: str
                defines the normalization layer type (Instance Norm or Batch Norm)
        activ: str
                defines the activation layer type (ReLu)
        init_type: str
                defines the initialization method used (xavier, kaiming he, ...)

        """
        super().__init__(num_classes=num_classes, channels_in=channels_in, depth=depth, 
                         start_channels=start_channels, kernel_size=kernel_size, 
                         merge_mode=merge_mode, up_mode=up_mode, pooling=pooling,
                         norm=norm, activ=activ, init_type=init_type)

        # Verify that only implemented unconvolution and merging operation types are passed in as arguments
        if up_mode not in ("transpose", "upsample"):
            raise ValueError(f"'{up_mode}' is not a valid option for upsampling."
                            "Only \"transpose\" and \"upsample\" are implemeted.")
        if merge_mode not in ("concat", "add"):
            raise ValueError(f"'{up_mode}' is not a valid option for merging up and down paths." 
                            "Only \"concat\" and \"add\" are implemented.")
        
        # create the DownConvBlocks and UpConvBlocks using a ModuleList
        # This allows to work with them like a standard list
        self.down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        outs = 0  # save the channel size after a ConvBlock to pass it as input to the next one
        
        # create the encoder pathway and add to the list
        for idx in range(depth):
            ins = channels_in if idx == 0 else outs
            outs = start_channels*(2**idx)
            pool = pooling if idx < depth-1 else None

            self.down_convs.append(DownConvBlock(ins, outs, kernel_size, 
                                                pooling=pool, norm=norm, activ=activ))

        # create the decoder pathway and add to the list
        # !! decoding only requires depth-1 blocks !!
        for idx in range(depth-1):
            ins = outs
            outs = ins // 2
            self.up_convs.append(UpConvBlock(ins, outs, kernel_size, 
                                            merge_mode=merge_mode, up_mode=up_mode, 
                                            norm=norm, activ=activ))

        # final 1x1 convultion to reduce the channel size to the number of classes
        self.conv_final = nn.Conv2d(start_channels, num_classes, kernel_size=1, padding=0)

        # initialize the model's parameters (calls the method of the BaseNetwork)
        self.reset_params(init_type)

    def forward(self, x):
        """
        Forward pass through the entire network
        """
        # create a list to memorize the tensors used for the skip connections
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for module in self.down_convs:
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for idx, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(idx+2)]
            x = module(before_pool, x)

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
    net = UNet_2D(num_classes=5)
    x = torch.rand([1, 1, 320, 320])

    output = net(x)
    print("out image size: ", output.shape)

    masks = (torch.max(output, dim=1))[1]
