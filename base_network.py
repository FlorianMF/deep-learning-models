import torch
from torch import nn
import abc


class BaseNetwork(torch.nn.Module):
    """
    Base class all networks should be derived from
    """

    _init_kwargs = {}

    @abc.abstractmethod
    def __init__(self, **kwargs):
        """
        Init function to register init kwargs (should be called from all
        subclasses)
        """
        super().__init__()

        for key, val in kwargs.items():
            self._init_kwargs[key] = val

    @abc.abstractmethod
    def forward(self, *inputs):
        """
        Forward pass of inputs through model
        Parameters
        ----------
        inputs: list
            inputs of arbitrary type and number
        Returns
        -------
        result:
            model results of arbitrary type and number
        """
        raise NotImplementedError

    @abc.abstractmethod
    def closure(self, criterion, optimizer, metrics, inputs, targets, fold=0, **kwargs):

        """
        Function which handles prediction from batch, logging, loss calculation
        and the optimizer step
        Parameters
        ----------
        model: BaseNetwork or torch.nn.DataParallel
            model to forward data through
        data_dict: dict
            dictionary containing the data
        optimizers: dict
            dictionary containing all optimizers to perform parameter update
        criterions: dict
            Functions or classes to calculate criterions
        metrics: dict
            Functions or classes to calculate other metrics
        fold: int
            Current Fold in Crossvalidation (default: 0)
        kwargs: dict
            additional keyword arguments

        Returns
        -------
        dict: Metric values (with same keys as input dict metrics)
        dict: Loss values (with same keys as input dict criterions)
        list: Arbitrary number of predictions (each of them as torch.Tensor)
        """
        raise NotImplementedError

    @property
    def init_kwargs(self):
        return self._init_kwargs

    @staticmethod
    def weight_init(m, init_type='kaiming_normal', init_gain=1):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
            # print("batchnorm init done")

    def reset_params(self, init_type='xavier', init_gain=1):
        for m in self.modules():
            self.weight_init(m, init_type, init_gain)


if __name__ == "__main__":
    x = torch.rand([2, 1, 320, 320])

    net = BaseNetwork()
    output = net(x)
    print(output)