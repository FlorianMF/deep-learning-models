# deep-learning-models
Little collection of deep learning models which I used for my master thesis of semantic segmentation.

All models should inherit from the "BaseNetwork". This network has abstract methods for the forward pass and the closure. Moreover it has a staticmethod used to initialize the model's parameters.

The UNet2d is directly taken from the paper by Ronneberger et al. 2015. One can define the depth of the model, the channel size after the first convolution, the used normalization and activation layers as well as the kernel size and the number of classes the model should differentiate simply by adapting the arguments. 
The UNet2d_Dilated is a variation of this U-Net using dilation at the end of each depth level in the encoder. The dilation rates can be adapted via the argument "dilation_vals"
