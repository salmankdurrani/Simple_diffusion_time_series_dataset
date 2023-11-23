# Package import & Device setting

import torch
import torch.nn as nn
import torchvision
from torch.nn import init
from torch.nn import functional as F
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.interpolation import rotate
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from IPython.display import HTML
from IPython.display import clear_output
import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import partial
import time
import datetime
import cv2 as cv
import glob
import os
#
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Dataset


t = torch.arange(1500, 1500+2048)/(1500+2048)
x1 = t*3.-2
x2 = torch.sin(t*2*np.pi*4)*t
x3 = torch.cos(t*2*np.pi*4)*t
x_ = torch.stack([x1, x2, x3], -1)
x = x_ + torch.randn(len(t), 3)*0.02



def plot_3d(x):
    fig = plt.figure(figsize=(20, 6))
    ax0 = fig.add_subplot(131, projection="3d")
    ax1 = fig.add_subplot(132, projection="3d")
    ax2 = fig.add_subplot(133, projection="3d")


    ax0.scatter(x[:, 0], x[:, 1], x[:, 2], s = 2)
    ax0.set_xlabel('x Label')
    ax0.set_ylabel('y Label')
    ax0.set_zlabel('z Label')
    ax0.view_init(elev=15., azim=45)

    ax1.scatter(x[:, 0], x[:, 1], x[:, 2], s = 2)
    ax1.set_xlabel('x Label')
    ax1.set_ylabel('y Label')
    ax1.set_zlabel('z Label')
    ax1.view_init(elev=30., azim=120)

    ax2.scatter(x[:, 0], x[:, 1], x[:, 2], s = 2)
    ax2.set_xlabel('x Label')
    ax2.set_ylabel('y Label')
    ax2.set_zlabel('z Label')
    ax2.view_init(elev=45., azim=170)

    plt.show()

plot_3d(x_)
plot_3d(x)

#%%
## Dataloader
num_train = 256
rand_idx = torch.randperm(len(x))[:num_train]
train_x = x[rand_idx]

plot_3d(train_x)

#%%
class CustomDataset(Dataset):
    def __init__(self, xs):
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx]


dataset = CustomDataset(train_x)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


#%%

# Model
## Configurations

beta_1 = 1e-4
beta_T = 0.001
T = 500
shape = (3,)
betas = torch.linspace(start = beta_1, end=beta_T, steps=T)
alphas = 1 - betas
device = torch.device('cuda:0')
alpha_bars = torch.cumprod(torch.linspace(start = alphas[0], end = alphas[-1], steps=T), dim = 0).to(device = device)
alpha_prev_bars = torch.cat([torch.Tensor([1]).to(device=device), alpha_bars[:-1]])


#%%
plt.plot(torch.sqrt(alpha_bars.cpu()))
plt.plot(torch.sqrt(1-alpha_bars.cpu()))

#%%
## Network

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=25.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)

class Swish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class MLP(nn.Module):
    def __init__(self, T, h_channels = 64, embed_dim=64):
        super().__init__()
        
        data_dim = 3
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
             nn.Linear(embed_dim, embed_dim))
        self.T = T
        
        self.in_layer = nn.Linear(data_dim, h_channels)
        self.dense1 = Dense(embed_dim, h_channels)
        
        self.h_layer1 = nn.Linear(h_channels, h_channels)
        self.dense2 = Dense(embed_dim, h_channels)
                                   
        self.h_layer2 = nn.Linear(h_channels, h_channels)
        self.dense3 = Dense(embed_dim, h_channels)
                        
        self.out_layer = nn.Linear(h_channels, data_dim)
        
        self.act = Swish()

    def forward(self, x, t):
        
        # residual path
        
        t = t/self.T
        embed = self.act(self.embed(t))
        
        h0 = self.in_layer(x)

        h1 = h0 + self.dense1(embed)
        h1 = self.act(h1)
        h1 = self.h_layer1(h1) + h0
        
        h2 = h1 + self.dense2(embed)
        h2 = self.act(h2)
        h2 = self.h_layer2(h2) + h1
        
        h3 = h2 + self.dense3(embed)
        h3 = self.act(h3)
        h = self.out_layer(h3)
        
        return h


#%%

model = MLP(T)
model.cuda()
optim = torch.optim.Adam(model.parameters(), lr = 0.0001)


#%%
## Functions
### Loss and Prediction
def loss_fn(output, epsilon, used_alpha):
    loss = (output - epsilon).square().mean()
    return loss

def pred_step(x, model, alpha_bars, device, idx=None, is_train=False):
    if is_train: # training
        idx = torch.randint(0, len(alpha_bars), (x.size(0), )).to(device = device)
        used_alpha = alpha_bars[idx][:, None]
        epsilon = torch.randn_like(x)
        x_tilde = torch.sqrt(used_alpha) * x + torch.sqrt(1 - used_alpha) * epsilon
    else: # inference
        idx = torch.Tensor([idx for _ in range(x.size(0))]).to(device = device).long()
        x_tilde = x
    output = model(x_tilde, idx)
    return (output, epsilon, used_alpha) if is_train else output


pred_step_fn = partial(pred_step, 
                       alpha_bars = alpha_bars, 
                       device = device)

#%%
### Diffusion process

def diffusion_process(x, model, alphas, betas, alpha_bars, pred_step_fn, alpha_prev_bars): #generator
    for idx in reversed(range(len(alpha_bars))):
        noise = torch.zeros_like(x) if idx == 0 else torch.randn_like(x)
        sqrt_tilde_beta = torch.sqrt((1 - alpha_prev_bars[idx]) / (1 - alpha_bars[idx]) * betas[idx])
        predict_epsilon = pred_step_fn(x, model, idx = idx)
        mu_theta_xt = torch.sqrt(1 / alphas[idx]) * (x - betas[idx] / torch.sqrt(1 - alpha_bars[idx]) * predict_epsilon)
        x = mu_theta_xt + sqrt_tilde_beta * noise
        yield x

@torch.no_grad()
def sampling(sampling_number, shape, device, diffusion_step, only_final_out = False):
    sample = torch.randn([sampling_number,*shape]).to(device = device)
    sampling_list = []
    final = None
    for sample in diffusion_step(sample):
        final = sample
        if not only_final_out:
            sampling_list.append(final)
    return final if only_final_out else torch.stack(sampling_list)

diffusion_process_fn = partial(diffusion_process, 
                               model = model, 
                               alphas = alphas, 
                               betas = betas, 
                               alpha_bars = alpha_bars, 
                               pred_step_fn = pred_step_fn,
                               alpha_prev_bars = alpha_prev_bars)

sampling_fn = partial(sampling, 
                      shape = shape, 
                      device = device, 
                      diffusion_step = diffusion_process_fn)



### Utils

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self. count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\r' + '\t'.join(entries), end = '')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


#%%
## Training
### Training setting

total_iteration = 400000
current_iteration = 0
display_iteration = 40000

sampling_number = 1024
only_final_out = True

losses = AverageMeter('Loss', ':.4f')
progress = ProgressMeter(total_iteration, [losses], prefix='Iteration ')

#%%

### Result (Before)

sample = sampling_fn(sampling_number, only_final_out = only_final_out)
plot_3d(sample.cpu().detach().numpy())

#%%

### Training

tic = time.time()
torch.randperm(len(train_x))
while current_iteration != total_iteration:
    try:
        data = dataiterator.next()
    except:
        dataiterator = iter(dataloader)
        data = dataiterator.next()
        
    data = data.to(device = device)
    train_output = pred_step_fn(data, model, is_train = True)
    loss = loss_fn(*train_output)
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    losses.update(loss.item())
    progress.display(current_iteration)
    current_iteration += 1
    
    if current_iteration % display_iteration == 0:
        sample = sampling_fn(sampling_number, only_final_out = only_final_out)
        plot_3d(sample.cpu().detach().numpy())
        losses.reset()
toc = time.time()

train_time = toc - tic
print("Training time : ", datetime.timedelta(seconds=train_time))


#%%

### Model save

# torch.save(model.state_dict(), './diff_model_weights_toy')

### Model load
model.load_state_dict(torch.load('./diff_model_weights_toy'))


### Result (After)

def animate_scatters3d(iteration, data, scatters):
    for i in range(3):
        scatters[i]._offsets3d = (data[iteration, :, 0], data[iteration, :, 1], data[iteration, :, 2])
    return scatters


import matplotlib
import seaborn as sns
matplotlib.rcParams['animation.embed_limit'] = 2**32

sampling_number = 2048
tic = time.time()
collect_samples = sampling_fn(sampling_number, only_final_out = False)
toc = time.time()
sampling_time = toc - tic
print(f"Sampling time : (#{sampling_number})", datetime.timedelta(seconds=sampling_time))
show_sample = collect_samples[9::10]
sorted_idx = torch.argsort(show_sample[0, :, 0])
show_sample = show_sample[:, sorted_idx, :]
show_sample = show_sample.cpu().detach().numpy()
iterations = len(show_sample)
# cmap = sns.color_palette("mako",n_colors = show_sample.shape[1], as_cmap=True)
cmap = sns.color_palette("ch:start=.5,rot=-.75",n_colors = show_sample.shape[1], as_cmap=True)

fig, axs = plt.subplots(ncols=3, figsize=(20, 6), subplot_kw={"projection":"3d"})
scatters = []
for ax, angle in zip(axs, [(15, 45), (30, 120), (45, 170)]):
    scatters.append(ax.scatter(x[:, 0], x[:, 1], x[:, 2], s = 2, c = x[:, 0], cmap = cmap))
    ax.set_xlabel('x Label')
    ax.set_ylabel('y Label')
    ax.set_zlabel('z Label')
    ax.view_init(elev=angle[0], azim=angle[1])
        
ani = animation.FuncAnimation(fig, animate_scatters3d, iterations, interval = 100, fargs=(show_sample, scatters), repeat_delay=1000, blit=True)

writergif = animation.PillowWriter(fps=1)
ani.save('ddpm_toy.gif', writer=writergif)
HTML(ani.to_jshtml())

#%%

def animate_scatters3d(iteration, data, scatters, spec_point_idx = 1024):
    for i in range(3):
        scatters[i*2]._offsets3d = (data[iteration, :, 0], data[iteration, :, 1], data[iteration, :, 2])
        scatters[i*2 + 1]._offsets3d = (data[iteration, spec_point_idx-1 : spec_point_idx, 0], data[iteration, spec_point_idx-1 : spec_point_idx, 1], data[iteration, spec_point_idx-1 : spec_point_idx, 2])
    return scatters

#%%%

import matplotlib
import seaborn as sns
matplotlib.rcParams['animation.embed_limit'] = 2**32

sampling_number = 2048
tic = time.time()
collect_samples = sampling_fn(sampling_number, only_final_out = False)
toc = time.time()
sampling_time = toc - tic
print(f"Sampling time : (#{sampling_number})", datetime.timedelta(seconds=sampling_time))
show_sample = collect_samples[9::10]
sorted_idx = torch.argsort(show_sample[0, :, 0])
show_sample = show_sample[:, sorted_idx, :]
show_sample = show_sample.cpu().detach().numpy()
iterations = len(show_sample)
# cmap = sns.color_palette("mako",n_colors = show_sample.shape[1], as_cmap=True)
cmap = sns.color_palette("ch:start=.5,rot=-.75",n_colors = show_sample.shape[1], as_cmap=True)

fig, axs = plt.subplots(ncols=3, figsize=(20, 6), subplot_kw={"projection":"3d"})
scatters = []
for ax, angle in zip(axs, [(15, 45), (30, 120), (45, 170)]):
    scatters.append(ax.scatter(show_sample[0, :, 0], show_sample[0, :, 1], show_sample[0, :, 2], s = 2, alpha = 0.3))
    scatters.append(ax.scatter(show_sample[0, 1023:1024, 0], show_sample[0, 1023:1024, 1], show_sample[0, 1023:1024, 2], s = 20, color = 'red', marker = '^'))
    ax.set_xlabel('x Label')
    ax.set_ylabel('y Label')
    ax.set_zlabel('z Label')
    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(-2, 2)
    ax.view_init(elev=angle[0], azim=angle[1])
        
ani = animation.FuncAnimation(fig, animate_scatters3d, iterations, interval = 100, fa rgs=(show_sample, scatters), repeat_delay=1000, blit=True)

writergif = animation.PillowWriter(fps=1)
ani.save('ddpm_toy3.gif', writer=writergif)
HTML(ani.to_jshtml())
