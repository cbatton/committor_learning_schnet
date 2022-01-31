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
rng_seed = torch.load("rng_seed")
cuda_rng_seed = torch.load("cuda_rng_seed")
torch.set_rng_state(rng_seed)
torch.cuda.set_rng_state(cuda_rng_seed)

boxsize = 3.28828276577396
net = SchNet(hidden_channels = 64, num_filters = 64, num_interactions = 3, num_gaussians = 50, cutoff = boxsize, max_num_neighbors = 31, boxsize=boxsize)
net.load_state_dict(torch.load("simple_params"))

# Load configurations
num_folders = 32
dimN = 32
dimD = 3
data_amount = int(5000)

batch_size = 8
# Load all the things
configs = torch.load("configs.pt")
z_configs = torch.load("z_configs.pt")
committor_values = torch.load("committor_values.pt")
bond_values = torch.load("bond_values.pt")
c_values = torch.load("c_values.pt")
zl = torch.load("zl.pt")
batch_indices = torch.load("batch_indices.pt")

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
cmloss.lambda_cl = torch.load("lambda_cl.pt")
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
optimizer.load_state_dict(torch.load("optimizer_params"))
loss_io = open("simple_statistic.txt",'a')
count = int(np.genfromtxt("count.txt"))
# Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=200, min_lr=1e-6)
scheduler.load_state_dict(torch.load("scheduler_params"))
for i in range(count,count+200):
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
    scheduler.step(cost)
    with torch.no_grad():
        if i%1 == 0:
            main_loss = bkeloss.main_loss
            cm_loss = cmloss.cl_loss
            print(i, main_loss.item(), cm_loss.item())
            loss_io.write('{:d} {:.5E} {:.5E} {:.5E}\n'.format(i,main_loss.item(),cm_loss.item(),optimizer.param_groups[0]['lr']))
            loss_io.flush()
            np.savetxt("count.txt", np.array((i+1,)))

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    net.zero_grad()
    with torch.no_grad():
        torch.save(optimizer.state_dict(), "optimizer_params")
        torch.save(scheduler.state_dict(), "scheduler_params")
        torch.save(net.state_dict(), "simple_params")
        torch.save(torch.get_rng_state(), "rng_seed")
        torch.save(torch.cuda.get_rng_state(), "cuda_rng_seed")
        torch.save(cmloss.lambda_cl, "lambda_cl.pt")
        if i%100 == 0:
            torch.save(optimizer.state_dict(), "optimizer_params_{}".format(i))
            torch.save(net.state_dict(), "simple_params_{}".format(i))
