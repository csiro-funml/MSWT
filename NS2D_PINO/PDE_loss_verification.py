# verify the PDE loss computation
# use the ground truth of the data to see if the boundary condition, and PDE loss is satisfied

import yaml
from data_utils.datasets import NSLoader2D
from utils.criterion import LpLoss, PINO_loss3d, get_forcing
from argparse import ArgumentParser
import torch


def verify_pde_loss(full_dataset, device):
    for i in range(len(full_dataset)):
        x, y = full_dataset[i]
        x = x.to(device)
        y = y.to(device)
        x_in = torch.cat((x.unsqueeze(-1), grid.unsqueeze(0).expand(B, -1, -1, -1)), dim=-1)
        out = model(x_in)
        loss = lploss(out, y)
        print(f'Sample {i} PDE loss: {loss.item()}')


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
                                    t_interval=data_config['time_interval'],
                                    n_samples=data_config.get('n_sample', data_config.get('n_samples', data_config['total_num'])),
                                    offset=data_config.get('offset', 0))


    verify_pde_loss(full_dataset, device)


