import torch
from torch import nn
from torch.nn import functional as F

def safe_log(z):
    return torch.log(z + 1e-7)

class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length, mu=0.0, std=0.01):
        super().__init__()

        self.transforms = nn.Sequential(*(
            PlanarFlow(dim,mu,std) for _ in range(flow_length)
        ))

        self.log_jacobians = nn.Sequential(*(
            PlanarFlowLogDetJacobian(t) for t in self.transforms
        ))

    def forward(self, z):

        log_jacobians = torch.zeros_like(z)

        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians += log_jacobian(z)
            z = transform(z)

        zk = z

        return zk, log_jacobians


class PlanarFlow(nn.Module):

    def __init__(self, dim, mu=0.0, std=0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.tanh = nn.Tanh()

        self.reset_parameters(mu,std)

    def reset_parameters(self,mu,std):
        self.weight.data.normal_(0, std)
        self.scale.data.normal_(0, std)
        self.bias.data.normal_(mu, std)
        # self.weight.data.uniform_(-std, std)
        # self.scale.data.uniform_(-std, std)
        # self.bias.data.uniform_(-std, std)

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh

    def forward(self, z):

        activation = F.linear(z, self.weight, self.bias)
        psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())