import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules.module import Module
from torch._jit_internal import weak_module, weak_script_method

# Todo:
# SDR Conv layer

@weak_module
class SDR_Linear(nn.Module):
    '''
    Linear layer trained with SDR

    '''
    def __init__(self, in_dim, out_dim, bias = True, init_fn = init.xavier_normal_):
        super(SDR_Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Gaussian
        self.weight_means = Parameter(torch.Tensor(out_dim, in_dim))
        self.weight_stds = Parameter(torch.Tensor(out_dim, in_dim))

        if bias:
            self.bias_means = Parameter(torch.Tensor(out_dim))
            self.bias_stds = Parameter(torch.Tensor(out_dim))
        else:
            raise NotImplementedError

        self.init_fn = init_fn # for weight_means
        self.init_params()

    def init_params(self):
        self.init_fn(self.weight_means)
        self.init_fn(self.weight_stds) # could be negative...

        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_means)
        bound = 1/math.sqrt(fan_in)
        init.uniform_(self.bias_means, -bound, bound)
        init.uniform_(self.bias_stds, -bound, bound)

    @weak_script_method
    def forward(self, in_data):
        weight = torch.distributions.Normal(self.weight_means, self.weight_stds).rsample() # the reparameterization trick
        bias = torch.distributions.Normal(self.bias_means, self.bias_stds).rsample()
        return F.linear(in_data, weight, bias)

    """
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    """


@weak_module
class SDR_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True,
                    mu_init = init.kaiming_normal_, sigma_init = init.xavier_normal_):
        """Initialize a (vanilla) Conv2d layer with SDR training routine
        @param in_channels:     number of input channels
        @param out_channels:    number of output channels
        @param kernel_size:     size of convolution filter
        @param stride:          stride step size
        @param padding:         padding size
        @param bias:            whether to add bias
        @param mu_init:         initialization function to the mean of parameters
        @param sigma_init:      initiliazation function to the std.dev of parameters. Initialized values are nonnegative
        """
        super(SDR_Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.bias = bias
        self.mu_init = mu_init
        self.sigma_init = sigma_init

        self.weight_mu = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.weight_sigma = Parameter(torch.Tensor(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))

        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_sigma = Parameter(torch.Tensor(out_channels))

        self.init_params()

    def init_params(self):
        self.mu_init(self.weight_mu, mode="fan_out", nonlinearity='relu')
        self.sigma_init(self.weight_sigma)

        if self.bias:
            init.zeros_(self.bias_mu)
            self.sigma_init(self.bias_sigma)

    @weak_script_method
    def forward(self, in_data):
        weight = torch.distributions.Normal(self.weight_mu, self.weight_sigma).rsample()
        if self.bias:
            bias = torch.distributions.Normal(self.bias_mu, self.bias_sigma).rsample()
        else:
            bias = torch.zeros(self.out_channels).cuda()
        return F.conv2d(in_data, weight, bias, self.stride, self.padding)
