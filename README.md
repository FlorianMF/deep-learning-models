# deep-learning-models
Little collection of deep learning models which I used for my master thesis on semantic segmentation of bones and cartilage in the human knee in MRI images.

All models should inherit from the "BaseNetwork". This network has abstract methods for the forward pass and the closure. Moreover it has a staticmethod used to initialize the model's parameters.

The UNet2d is directly taken from the paper by Ronneberger et al. 2015. One can define the depth of the model, the channel size after the first convolution, the used normalization and activation layers as well as the kernel size and the number of classes the model should differentiate simply by adapting the arguments. 
![UNet](UNet.png)

The UNet2d_Dilated is a variation of this U-Net using dilation at the end of each depth level in the encoder. The dilation rates can be adapted via the argument "dilation_vals"
![UNet](UNet_Dilated.jpg)

The VNet2d is a created according to the paper. In contrast to the U-Net the number of blocks (convolution + normalization + activation) on each depth level does vary. Furthermore the V-Net makes use of residual connections at which encapsulate an entire depth level. Instead of using pooling layers the dimensional reduction is performed via the standard convolution with zero padding.
![VNet](VNet.png)
