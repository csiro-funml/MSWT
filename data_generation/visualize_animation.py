import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import os


path = '/scratch3/wan410/operator_learning_data/NS_torchcfd/data/Re1000/train'
data_path = os.path.join(path, 'data_0.hdf5')
data = h5py.File(data_path, 'r')['data']
print(data.shape)


def generate_gt_gif(sample_data, log_path=None):
    # keys are (u, vx, vy), shape (T, H, W)
    cmap = 'RdBu_r'
    
    keys = [0, 1]
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    titles = {}
    imgs = {}
    for i, key in enumerate(keys):
        data_c = sample_data[...,key]
        print("data_c.shape", data_c.shape)
        vmax = np.max(np.abs(data_c))
        vmin = -vmax if np.min(data_c) <0 else np.min(data_c)
        imgs[key] = ax[i].imshow(data_c[0], vmin=vmin, vmax=vmax, cmap=cmap)
        ax[i].axis('off')
        titles[key] =ax[i].set_title(str(key) + ' T=0')

    def update(frame_idx):
        print("frame_idx", frame_idx)
        for i, key in enumerate(keys):
            data_c = sample_data[...,key]
            vmax = np.max(np.abs(data_c))
            vmin = -vmax if np.min(data_c) <0 else np.min(data_c)
            imgs[key].set_data(data_c[frame_idx])
            imgs[key].set_clim(vmin=vmin, vmax=vmax)
            titles[key].set_text(str(key) + ' T=' + str(frame_idx))
        # Don't return anything when blit=False
        return []

    anim = FuncAnimation(fig, update, frames=sample_data.shape[-2], interval=200, blit=False)

    nt=30
    sample_id = 0
    gif_path = f'{log_path}/sample_{sample_id}_nt{nt}.gif'
    try:
        anim.save(gif_path, writer=PillowWriter(fps=2))
    except Exception as e:
        print(f'Failed to save GIF due to: {e}')

    plt.close(fig)


log_path = '/home/wan410/multiscale_neural_operators/'
generate_gt_gif(data, log_path)



