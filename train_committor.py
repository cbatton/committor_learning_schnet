from schnet_committor import SchNet, BKELoss, CommittorLoss2
import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(device)
torch.manual_seed(5070)
torch.cuda.manual_seed(5070)

boxsize = 3.28828276577396
net = SchNet(hidden_channels = 64, num_filters = 64, num_interactions = 3, num_gaussians = 50, cutoff = 2.0, max_num_neighbors = 31, boxsize=boxsize)

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

batch_size = 8
batch_indices = torch.zeros(batch_size*dimN, dtype=torch.int64)
for i in range(batch_size):
    batch_indices[(dimN*i):(dimN*(i+1))] = i
# Save all things are PyTorch arrays for faster loading later
torch.save(configs, "configs.pt")
torch.save(z_configs, "z_configs.pt")
torch.save(committor_values, "committor_values.pt")
torch.save(bond_values, "bond_values.pt")
torch.save(c_values, "c_values.pt")
torch.save(zl, "zl.pt")
torch.save(batch_indices, "batch_indices.pt")

net = net.to(device)
z_configs = z_configs.to(device)
configs = configs.to(device)
batch_indices = batch_indices.to(device)
c_values = c_values.to(device)
zl = zl.to(device)
committor_values = committor_values.to(device)
bkeloss = BKELoss(committor=net, zl=zl, dimN=dimN, num_replicas=num_folders)
cmloss = CommittorLoss2(committor=net, cl_configs=configs.detach().clone(), cl_z_configs=z_configs.detach().clone(), dimN=dimN, cl_configs_values=committor_values.detach().clone(), cl_configs_count=data_amount, cl_configs_replica=num_folders, lambda_cl=100.0, batch_size_cl=0.05) 
lambda_cl_end = 10**4
cl_start = 200
cl_end = 10000
cl_stepsize = (lambda_cl_end-cmloss.lambda_cl)/(cl_end-cl_start)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
loss_io = open("simple_statistic.txt",'w')
for i in range(200):
    if (i>cl_start) and (i <= cl_end):
        cmloss.lambda_cl += cl_stepsize
    elif i > cl_end:
        cmloss.lambda_cl = lambda_cl_end
    indices_batch = torch.randperm(data_amount, device=device)[:int(batch_size)]
    configs_ = configs[:,indices_batch].detach().clone()
    configs_.requires_grad_(True)
    z_configs_ = z_configs[:,indices_batch].detach().clone()
    c_values_ = c_values[:,indices_batch].detach().clone()
    cmcost = cmloss()
    bkecost = bkeloss(configs_, z_configs_, c_values_, batch_size)
    cost = bkecost+cmcost
    with torch.no_grad():
        if i%1 == 0:
            main_loss = bkeloss.main_loss
            cm_loss = cmloss.cl_loss
            print(i, main_loss.item(), cm_loss.item())
            loss_io.write('{:d} {:.5E} {:.5E}\n'.format(i,main_loss.item(),cm_loss.item()))
            loss_io.flush()
            np.savetxt("count.txt", np.array((i+1,)))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    net.zero_grad()
    with torch.no_grad():
        torch.save(optimizer.state_dict(), "optimizer_params")
        torch.save(net.state_dict(), "simple_params")
        torch.save(torch.get_rng_state(), "rng_seed")
        torch.save(torch.cuda.get_rng_state(), "cuda_rng_seed")
        torch.save(cmloss.lambda_cl, "lambda_cl.pt")
        if i%100 == 0:
            torch.save(optimizer.state_dict(), "optimizer_params_{}".format(i))
            torch.save(net.state_dict(), "simple_params_{}".format(i))
