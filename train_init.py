from schnet_committor import SchNet
import numpy as np
import torch
torch.manual_seed(5070)
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)

boxsize = 3.28828276577396
net = SchNet(hidden_channels = 64, num_filters = 64, num_interactions = 3, num_gaussians = 50, cutoff = boxsize, max_num_neighbors = 31, boxsize=boxsize)

# Load configurations
num_folders = 32
dimN = 32
dimD = 3
data_amount = int(20)
configs_all = np.zeros((num_folders,dimN*data_amount,dimD), dtype='f')
for i in range(num_folders):
    configs_all[i] = np.genfromtxt(str(i)+"/config.xyz", max_rows=dimN*data_amount, usecols=(1,2,3), dtype='f')

print(configs_all.shape)
configs_all = configs_all.reshape((num_folders,data_amount,dimN,dimD))
z_configs = np.zeros((num_folders,data_amount,dimN), dtype=int)
z_configs[:,:,0] = 1
z_configs[:,:,1] = 1
configs = torch.from_numpy(configs_all)
configs.requires_grad_(True)
z_configs = torch.from_numpy(z_configs)

# Load committor values
committor_values = np.zeros((num_folders,data_amount), dtype='f')
for i in range(num_folders):
    committor_values[i] = np.genfromtxt(str(i)+"/committor_output.txt", max_rows=data_amount, usecols=(0), dtype='f')

# Now going to precompute all c(x; r) values
# Why? Because I can
bond_values = np.zeros((num_folders,data_amount))
c_values = np.zeros((num_folders,data_amount), dtype='f')
for i in range(num_folders):
    bond_values[i] = np.genfromtxt(str(i)+"/bond_storage.txt", skip_header=1, max_rows=data_amount, dtype='f')

kappa = np.zeros((num_folders,))
bias = np.zeros((num_folders,))
for i in range(num_folders):
    data = np.genfromtxt(str(i)+"/param", skip_header=15, usecols=(1,2))
    kappa[i] = data[0]
    bias[i] = data[1]

# Now compute them all
for i in range(num_folders):
    c_values += np.exp(-0.5*kappa[i]*(bond_values-bias[i])**2)

committor_values = torch.from_numpy(committor_values)
bond_values = torch.from_numpy(bond_values)
c_values = torch.from_numpy(c_values)

# Should probably have some initial training part in order to initialize NN
# Should just try to minimize committor value of single configurations along
# training poitns

# Load free energies, convert into reweighting factors
f_k = np.genfromtxt("free_energy/f_analysis.txt", usecols=(0))
zl = np.exp(-f_k)
zl /= np.sum(zl)
zl = torch.from_numpy(zl)

criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(net.parameters(), lr=5e-5)
batch_size = 32
batch_indices = torch.zeros(batch_size*dimN, dtype=torch.int64)
for i in range(batch_size):
    batch_indices[(dimN*i):(dimN*(i+1))] = i
net = net.to(device)
z_configs = z_configs.to(device)
configs = configs.to(device)
batch_indices = batch_indices.to(device)
committor_values = committor_values.to(device)
c_values = c_values.to(device)
zl = zl.to(device)
loss_io = open("simple_statistic_init.txt",'w')
configs = configs.view(-1,32,3)
z_configs = z_configs.view(-1,32)
committor_values = committor_values.view(-1)

for i in range(10000):
    indices_batch = torch.randperm(num_folders*data_amount, device=device)[:batch_size]
    configs_ = configs[indices_batch].detach().clone()
    z_configs_ = z_configs[indices_batch].detach().clone()
    committor_pred = net(z_configs_.view(-1), configs_.view(-1,3), batch=batch_indices)
    committor_data = committor_values[indices_batch].view(-1,1)
    loss = criterion(committor_pred, committor_data)
    with torch.no_grad():
        if i%1 == 0:
            print(i, loss.item())
            loss_io.write('{:d} {:.5E}\n'.format(i,loss.item()))
            loss_io.flush()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        torch.save(optimizer.state_dict(), "optimizer_params_init")
        torch.save(net.state_dict(), "simple_params_init")
        torch.save(torch.get_rng_state(), "rng_seed_init")
        if i%100 == 0:
            torch.save(optimizer.state_dict(), "optimizer_params_init_{}".format(i))
