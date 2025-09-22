# from https://github.com/scaomath/torch-cfd/
# The MIT License (MIT)
# Copyright © 2024 Shuhao Cao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os

import torch
import torch.fft as fft
import torch.nn.functional as F
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from torch_cfd.grids import Grid
from torch_cfd.initial_conditions import filtered_vorticity_field
from torch_cfd.spectral import *

from fno.data_gen.trajectories import get_trajectory_imex
from data_utils import *

import logging

from fno.pipeline import DATA_PATH, LOG_PATH


def main(args):
    """
    Generate the isotropic turbulence in [1]

    [1]: McWilliams, J. C. (1984). The emergence of isolated coherent vortices in turbulent flow. Journal of Fluid Mechanics, 146, 21-43.

    Training dataset for the SFNO ICLR 2025 paper:
    >>> python data_gen_McWilliams2d.py --num-samples 1152 --batch-size 128 --grid-size 256 --subsample 4 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"

    Testing dataset for plotting the enstrohpy spectrum:
    >>> python data_gen_McWilliams2d.py --num-samples 16 --batch-size 8 --grid-size 256 --subsample 1 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --double

    Training dataset with Re=5k:
    >>> python data_gen_McWilliams2d.py --num-samples 1152 --batch-size 128 --grid-size 512 --subsample 1 --Re 5e3 --dt 5e-4 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi"

    Demo dataset to test if the data generation works:
    >>> python data_gen_McWilliams2d.py --num-samples 4 --batch-size 2 --grid-size 256 --subsample 1 --visc 1e-3 --dt 1e-3 --time 10 --time-warmup 4.5 --num-steps 100 --diam "2*torch.pi" --double --demo

    """
    args = args.parse_args()

    current_time = datetime.now().strftime("%d_%b_%Y_%Hh%Mm")
    log_name = "".join(os.path.basename(__file__).split(".")[:-1])

    log_filename = os.path.join(LOG_PATH, f"{current_time}_{log_name}.log")
    logger = get_logger(log_filename)

    total_samples = args.num_samples
    batch_size = args.batch_size  # 128
    assert batch_size <= total_samples, "batch_size <= num_samples"
    n = args.grid_size  # 256
    viscosity = args.visc if args.Re is None else 1 / args.Re
    Re = 1 / viscosity
    dt = args.dt  # 1e-3
    T = args.time  # 10
    subsample = args.subsample  # 4
    ns = n // subsample
    T_warmup = args.time_warmup  # 4.5
    num_snapshots = args.num_steps  # 100
    random_state = args.seed
    peak_wavenumber = args.peak_wavenumber  # 4
    diam = (
        eval(args.diam) if isinstance(args.diam, str) else args.diam
    )  # "2 * torch.pi"
    force_rerun = args.force_rerun

    logger = logging.getLogger()
    logger.info(f"Generating data for McWilliams2d with {total_samples} samples")

    max_velocity = args.max_velocity  # 5
    dt = stable_time_step(diam / n, dt, max_velocity, viscosity=viscosity)
    logger.info(f"Using dt = {dt}")

    warmup_steps = int(T_warmup / dt)
    total_steps = int((T - T_warmup) / dt)
    record_every_iters = int(total_steps / num_snapshots)

    dtype = torch.float64 if args.double else torch.float32
    cdtype = torch.complex128 if args.double else torch.complex64
    dtype_str = "_fp64" if args.double else ""
    filename = args.filename
    if filename is None:
        # filename = f"McWilliams2d{dtype_str}_{ns}x{ns}_N{total_samples}_v{viscosity:.0e}_T{num_snapshots}.pt".replace(
        #     "e-0", "e-"
        # )
        filename = f"McWilliams2d{dtype_str}_{ns}x{ns}_N{total_samples}_Re{int(Re)}_T{num_snapshots}.pt"
        args.filename = filename
    data_filepath = os.path.join(DATA_PATH, filename)
    progress_filepath = data_filepath.replace('.pt', '_progress.pkl')
    
    # Check for resume capability
    completed_batches = set()
    if os.path.exists(progress_filepath):
        try:
            import pickle
            with open(progress_filepath, 'rb') as f:
                completed_batches = pickle.load(f)
            logger.info(f"Found progress file with {len(completed_batches)} completed batches")
        except Exception as e:
            logger.warning(f"Could not load progress file: {e}")
            completed_batches = set()
    

    cuda = not args.no_cuda and torch.cuda.is_available()
    no_tqdm = args.no_tqdm
    device = torch.device("cuda:0" if cuda else "cpu")

    torch.set_default_dtype(torch.float64)
    logger.info(
        f"Using device: {device} | save dtype: {dtype} | compute dtype: {torch.get_default_dtype()}"
    )

    grid = Grid(shape=(n, n), domain=((0, diam), (0, diam)), device=device)

    ns2d = NavierStokes2DSpectral(
        viscosity=viscosity,
        grid=grid,
        drag=0,
        smooth=True,
        forcing_fn=None,
        step_fn=RK4CrankNicolsonStepper(),
    ).to(device)

    # Calculate batch indices and sizes
    batch_info = []
    for idx in range(0, total_samples, batch_size):
        current_batch_size = min(batch_size, total_samples - idx)
        batch_info.append((idx, current_batch_size))
    
    num_batches = len(batch_info)
    
    # Calculate which batches need to be processed
    remaining_batches = []
    for i, (idx, current_batch_size) in enumerate(batch_info):
        if i not in completed_batches:
            remaining_batches.append((i, idx, current_batch_size))
    
    logger.info(f"Processing {len(remaining_batches)} remaining batches out of {num_batches} total")
    
    for batch_idx, (i, idx, current_batch_size) in enumerate(remaining_batches):
        logger.info(f"Generate trajectory for batch [{i+1}/{num_batches}] (remaining: {len(remaining_batches) - batch_idx})")
        logger.info(
            f"random states: {random_state + idx} to {random_state + idx + current_batch_size-1} (batch size: {current_batch_size})"
        )

        vort_init = torch.stack(
            [
                filtered_vorticity_field(
                    grid, peak_wavenumber, random_state=random_state + idx + k
                ).data
                for k in range(current_batch_size)
            ]
        )
        vort_hat = fft.rfft2(vort_init).to(device)

        with tqdm(total=warmup_steps, disable=no_tqdm) as pbar:
            for j in range(warmup_steps):
                vort_hat, _ = ns2d.step(vort_hat, dt)
                if j % 100 == 0:
                    vort_norm = torch.linalg.norm(fft.irfft2(vort_hat)).item() / n
                    desc = (
                        datetime.now().strftime("%d-%b-%Y %H:%M:%S")
                        + f" - Warmup | vort_hat ell2 norm {vort_norm:.4e}"
                    )
                    pbar.set_description(desc)
                    pbar.update(100)

        result = get_trajectory_imex(
            ns2d,
            vort_hat,
            dt,
            num_steps=total_steps,
            record_every_steps=record_every_iters,
            pbar=not no_tqdm,
            dtype=cdtype,
        )

        for field, value in result.items():
            logger.info(
                f"freq variable:  {field:<12} | shape: {value.shape} | dtype: {value.dtype}"
            )
            value = fft.irfft2(value).real.cpu().to(dtype)
            logger.info(
                f"saved variable: {field:<12} | shape: {value.shape} | dtype: {value.dtype}"
            )
            if subsample > 1:
                # Reshape from (batch, 1, time, height, width) to (batch*time, 1, height, width)
                # for interpolation, then reshape back
                current_batch_size, _, time_steps, height, width = value.shape
                value_reshaped = value.view(current_batch_size * time_steps, 1, height, width)
                value_interp = F.interpolate(value_reshaped, size=(ns, ns), mode="bilinear")
                result[field] = value_interp.view(current_batch_size, 1, time_steps, ns, ns)
            else:
                result[field] = value

        result["random_states"] = torch.tensor(
            [random_state + idx + k for k in range(current_batch_size)], dtype=torch.int32
        )
        if not args.demo:
            save_pickle(result, data_filepath, append=True)
            
            # Update progress tracking
            completed_batches.add(i)
            import pickle
            with open(progress_filepath, 'wb') as f:
                pickle.dump(completed_batches, f)
            logger.info(f"Completed batch {i+1}/{num_batches}, progress saved")
            
            del result

    if not args.demo:
        pickle_to_pt(data_filepath)
        # Clean up progress file on successful completion
        if os.path.exists(progress_filepath):
            os.remove(progress_filepath)
            logger.info("Progress file cleaned up after successful completion")
        logger.info(f"Done saving.")
    else:
        try:
            verify_trajectories(
                result,
                dt=record_every_iters * dt,
                T_warmup=T_warmup,
                n_samples=1,
            )
        except Exception as e:
            logger.error(f"Error in plotting sample trajectories: {e}")
    return 0


if __name__ == "__main__":
    args = get_args_ns2d(
        "Parameters for generating NSE 2d with McWilliams 2d example"
    )
    main(args)
