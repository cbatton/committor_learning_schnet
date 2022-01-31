from typing import Optional

import os
import warnings
import os.path as osp
from math import pi as PI

import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear, ModuleList
import numpy as np

from torch_scatter import scatter
from torch_geometric.data.makedirs import makedirs
from torch_geometric.data import download_url, extract_zip, Dataset
from torch_geometric.nn import radius_graph, MessagePassing
from torch.nn.modules.loss import _Loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class SchNet(torch.nn.Module):
    r"""The continuous-filter convolutional neural network SchNet from the
    `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling
    Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper that uses
    the interactions blocks of the form

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \odot
        h_{\mathbf{\Theta}} ( \exp(-\gamma(\mathbf{e}_{j,i} - \mathbf{\mu}))),

    here :math:`h_{\mathbf{\Theta}}` denotes an MLP and
    :math:`\mathbf{e}_{j,i}` denotes the interatomic distances between atoms.

    .. note::

        For an example of using a pretrained SchNet variant, see
        `examples/qm9_pretrained_schnet.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        qm9_pretrained_schnet.py>`_.

    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_interactions (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_gaussians (int, optional): The number of gaussians :math:`\mu`.
            (default: :obj:`50`)
        cutoff (float, optional): Cutoff distance for interatomic interactions.
            (default: :obj:`10.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        readout (string, optional): Whether to apply :obj:`"add"` or
            :obj:`"mean"` global aggregation. (default: :obj:`"add"`)
        dipole (bool, optional): If set to :obj:`True`, will use the magnitude
            of the dipole moment to make the final prediction, *e.g.*, for
            target 0 of :class:`torch_geometric.datasets.QM9`.
            (default: :obj:`False`)
        mean (float, optional): The mean of the property to predict.
            (default: :obj:`None`)
        std (float, optional): The standard deviation of the property to
            predict. (default: :obj:`None`)
        atomref (torch.Tensor, optional): The reference of single-atom
            properties.
            Expects a vector of shape :obj:`(max_atomic_number, )`.
    """

    url = 'http://www.quantum-machine.org/datasets/trained_schnet_models.zip'

    def __init__(self, hidden_channels: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, max_num_neighbors: int = 32,
                 readout: str = 'add', dipole: bool = False,
                 mean: Optional[float] = None, std: Optional[float] = None,
                 atomref: Optional[torch.Tensor] = None, boxsize: float = 5.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.num_gaussians = num_gaussians
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.readout = readout
        self.dipole = dipole
        self.readout = 'add' if self.dipole else self.readout
        self.mean = mean
        self.std = std
        self.scale = None
        self.boxsize = boxsize

        atomic_mass = torch.from_numpy(np.array([1,2]))
        self.register_buffer('atomic_mass', atomic_mass)

        self.embedding = Embedding(2, hidden_channels)
        self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)

        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_channels, num_gaussians,
                                     num_filters, cutoff)
            self.interactions.append(block)

        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)

        self.register_buffer('initial_atomref', atomref)
        self.atomref = None
        if atomref is not None:
            self.atomref = Embedding(100, 1)
            self.atomref.weight.data.copy_(atomref)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)
        if self.atomref is not None:
            self.atomref.weight.data.copy_(self.initial_atomref)


    def forward(self, z, pos, batch=None):
        """"""
        assert z.dim() == 1 and z.dtype == torch.long
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)

        edge_index = radius_graph(pos, r=2*self.boxsize, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        dx = pos[row] - pos[col]
        dx = dx-torch.round(dx/self.boxsize)*self.boxsize
        edge_weight = (dx).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        if self.dipole:
            # Get center of mass.
            mass = self.atomic_mass[z].view(-1, 1)
            c = scatter(mass * pos, batch, dim=0) / scatter(mass, batch, dim=0)
            h = h * (pos - c.index_select(0, batch))

        if not self.dipole and self.mean is not None and self.std is not None:
            h = h * self.std + self.mean

        if not self.dipole and self.atomref is not None:
            h = h + self.atomref(z)

        out = scatter(h, batch, dim=0, reduce=self.readout)

        if self.dipole:
            out = torch.norm(out, dim=-1, keepdim=True)

        if self.scale is not None:
            out = self.scale * out

        return torch.sigmoid(out)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_interactions={self.num_interactions}, '
                f'num_gaussians={self.num_gaussians}, '
                f'cutoff={self.cutoff})')



class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = CFConv(hidden_channels, hidden_channels, num_filters,
                           self.mlp, cutoff)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        x = self.conv(x, edge_index, edge_weight, edge_attr)
        x = self.act(x)
        x = self.lin(x)
        return x


class CFConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_filters, nn, cutoff):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, num_filters, bias=False)
        self.lin2 = Linear(num_filters, out_channels)
        self.nn = nn
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        C = 0.5 * (torch.cos(edge_weight * PI / self.cutoff) + 1.0)
        C *= (edge_weight < self.cutoff).float()
        W = self.nn(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        x = self.propagate(edge_index, x=x, W=W)
        x = self.lin2(x)
        return x

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift

# Loss classes for testing
class BKELoss(_Loss):
    r"""Base classs for computing the loss function corresponding to the variational form 
        of the Backward Kolmogorov Equation. This base class includes default implementation 
        for boundary conditions. 

    Args:
        bc_sampler (tpstorch.MLSamplerEXP): the MD/MC sampler used for obtaining configurations in 
            product and reactant basin.  
        
        committor (tpstorch.nn.Module): the committor function, represented as a neural network. 
        
        lambda_A (float): penalty strength for enforcing boundary conditions at the reactant basin. 
        
        lambda_B (float): penalty strength for enforcing boundary conditions at the product basin. 
            If None is given, lambda_B=lambda_A
        
        start_react (torch.Tensor): starting configuration to sample reactant basin. 
        
        start_prod (torch.Tensor): starting configuration to sample product basin.
       
        n_bc_samples (int, optional): total number of samples to collect at both product and 
            reactant basin. 

        bc_period (int, optional): the number of timesteps to collect one configuration during 
            sampling at either product and reactant basin.

        batch_size_bc (float, optional): size of mini-batch for the boundary condition loss during 
            gradient descent, expressed as fraction of n_bc_samples.
    """

    def __init__(self, committor, zl, dimN, num_replicas): 
        super(BKELoss, self).__init__()
        
        self.main_loss = torch.zeros(1, device=device)
        self.committor = committor
        self.zl = zl
        self.dimN = dimN
        self.num_replicas = num_replicas
        
    # Note I have to make this respect windows
    def compute_bkeloss(self, configs, z_configs, inv_normconstants, batch_size):
        """Computes the loss corresponding to the varitional form of the BKE including 
            the EXP reweighting factors. 
            
            Independent computation is first done on individual MPI process. First, we compute 
            the following quantities at every 'l'-th MPI process: 

            .. math::
                L_l = \frac{1}{2} \sum_{x \in M_l} |\grad q(x)|^2/c(x) ,
                
                c_l = \sum_{ x \in M_l} 1/c(x) ,
            
            where :math: $M_l$ is the mini-batch collected by the l-th MPI
            process. We then collect the computation to compute the main loss as

            .. math::
                \ell_{main} = \frac{\sum_{l=1}^{S-1} L_l z_l)}{\sum_{l=1}^{S-1} c_l z_l)}
            where :math: 'S' is the MPI world size. 

            Args:
                gradients (torch.Tensor): mini-batch of \grad q(x). First dimension is the size of
                    the mini-batch while the second is system size (flattened).
                
                inv_normconstants (torch.Tensor): mini-batch of 1/c(x).

            Note that PyTorch does not track arithmetic operations during MPI
            collective calls. Thus, the last sum containing L_l is not reflected 
            in the computational graph tracked by individual MPI process. The 
            final gradients will be collected in each respective optimizer.
        """
        main_loss_num = torch.zeros(1, device=device)
        main_loss_denom = torch.zeros(1, device=device)
        for i in range(self.num_replicas):
            grads_factor = torch.zeros(1,device=device)
            for j in range(batch_size):
                config_ = configs[i][j].clone()
                z_config_ = z_configs[i][j].detach().clone()
                grads = torch.autograd.grad(self.committor(z_config_, config_), config_, create_graph=True)[0].reshape(-1)
                grads_factor = grads_factor+torch.sum(grads*grads)/inv_normconstants[i][j]
                self.committor.zero_grad()
            main_loss_num = main_loss_num+self.zl[i]*grads_factor/batch_size
            main_loss_denom = main_loss_denom+self.zl[i]*torch.mean(1.0/inv_normconstants[i])
                
         
        return main_loss_num/main_loss_denom

    def forward(self, configs, z_configs, inv_normconstants, batch_size):
        self.main_loss = self.compute_bkeloss(configs, z_configs, inv_normconstants, batch_size)
        return self.main_loss

class CommittorLoss2(_Loss):
    r"""Loss function which implements the MSE loss for the committor function. 
        
        This loss function automatically collects the committor values through brute-force simulation.

        Args:
            cl_sampler (tpstorch.MLSampler): the MC/MD sampler to perform unbiased simulations.
            
            committor (tpstorch.nn.Module): the committor function, represented as a neural network. 
            
            lambda_cl (float): the penalty strength of the MSE loss. Defaults to one. 
            
            batch_size_cl (float): size of mini-batch used during training, expressed as the fraction of total batch collected at that point. 
    """
    def __init__(self, committor, cl_configs, cl_z_configs, dimN, cl_configs_values, cl_configs_count, cl_configs_replica, lambda_cl=1.0, batch_size_cl=0.5):
        super(CommittorLoss2, self).__init__()
        
        self.cl_loss = torch.zeros(1, device=device)
        self.committor = committor 

        self.lambda_cl = lambda_cl
        self.batch_size_cl = batch_size_cl
        
        self.cl_configs = cl_configs
        self.cl_z_configs = cl_z_configs
        self.dimN = dimN
        self.cl_configs_values = cl_configs_values
        self.cl_configs_count = cl_configs_count
        self.cl_configs_replica = cl_configs_replica
        self.batch_indices = torch.zeros((cl_configs_count,dimN), dtype=torch.int64, device=device)
        for i in range(cl_configs_count):
            self.batch_indices[i] = i
    
    # Note have to make this respect windows
    def compute_cl(self):
        """Computes the committor loss function 
            TO DO: Complete this docstrings 
        """
        #Initialize loss to zero
        loss_cl = torch.zeros(1, device=device)
        # Compute loss by sub-sampling however many batches we have at the moment
        for i in range(self.cl_configs_replica):
            indices_committor = torch.randperm(self.cl_configs_count)[:int(self.batch_size_cl*self.cl_configs_count)]
            if self.cl_configs_count == 1:
                indices_committor = 0
            committor_penalty = torch.mean((self.committor(self.cl_z_configs[i][indices_committor].view(-1),self.cl_configs[i][indices_committor].view(-1,3), batch=self.batch_indices[:int(self.batch_size_cl*self.cl_configs_count)].view(-1)).view(-1)-self.cl_configs_values[i][indices_committor]))
            loss_cl += committor_penalty**2
        return 0.5*self.lambda_cl*loss_cl
    
    def forward(self):
        self.cl_loss = self.compute_cl()
        return self.cl_loss
