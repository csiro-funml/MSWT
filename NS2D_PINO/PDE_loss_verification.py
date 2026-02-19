# verify the PDE loss computation
# use the ground truth of the data to see if the boundary condition, and PDE loss is satisfied

import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_utils.datasets import NSLoader2D
from utils.criterion import LpLoss, PINO_loss3d, get_forcing
from argparse import ArgumentParser
import torch


def verify_pde_loss(data, v, t_duration, forcing, device):
    """ 
    data: (N, S, S, T) from NSLoader2D.data, compute PINO_loss to test if the PDE loss is zero
    """
    total_loss_f = 0.0
    total_loss_ic = 0.0
    for i in range(len(data)):
        print(f'Sample {i} shape: {data[i].shape}')
        x = data[i].to(device).unsqueeze(0) # one realization (1, S, S, T)
        u0 = x[..., 0] # initial condition (1, S, S)
        loss_ic, loss_f = PINO_loss3d(x, u0, forcing, v, t_duration)
        print(f'Sample {i} PDE loss: {loss_f.item()}, IC loss: {loss_ic.item()}')
        total_loss_f += loss_f.item()
        total_loss_ic += loss_ic.item()
    print(f'average PDE loss: {total_loss_f / len(data)}')
    print(f'average IC loss: {total_loss_ic / len(data)}')
    return total_loss_f / len(data), total_loss_ic / len(data)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, help='Path to the configuration file')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)


    data_config = config['data']

    full_dataset = NSLoader2D(datapath1=data_config['datapath'],
                                    nx=data_config['nx'], nt=data_config['nt'],
                                    sub=data_config['sub'], sub_t=data_config['sub_t'],
                                    N=data_config['total_num'],
                                    t_interval=data_config['t_duration'],
                                    n_samples=data_config.get('n_sample', data_config.get('n_samples', data_config['total_num'])),
                                    offset=data_config.get('offset', 0))

    data = full_dataset.data
    v = 1.0 / config['data']['Re']
    t_duration = config['data'].get('t_duration', 0.125)
    print("t_duration: ", t_duration)
    print("v: ", v)
    S_forcing = data_config['nx']
    forcing = get_forcing(S_forcing).to(device)
    
    verify_pde_loss(data, v, t_duration, forcing, device)


