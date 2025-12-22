For benchmark details refer to the settings in "Physics-Informed Neural Operator for Learning Partial Differential Equations" by Zongyi Li et al:

section 4 Experiments: Navier-Stokes Equation. Chaotic Kolmogorov flow.

### Navier Stokes with Reynolds number 500
- spatial domain: $x\in (0, 2\pi)^2$
- temporal domain: $t \in [0, 0.5]$
- forcing: $-4\cos(4x_2)$
- Reynolds number: 500

Train set: data of shape (N, T, X, Y) where N is the number of instances, T is temporal resolution, X, Y are spatial resolutions. 
only take vorticity as inputs.

1. [NS_fft_Re500_T4000.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_fft_Re500_T4000.npy) : 4000x64x64x65

Test set: data of shape (N, T, X, Y) where N is the number of instances, T is temporal resolution, X, Y are spatial resolutions. 
1. [NS_Re500_s256_T100_test.npy](https://hkzdata.s3.us-west-2.amazonaws.com/PINO/data/NS_Re500_s256_T100_test.npy): 100x129x256x256


Differences:
We compute the mean and var of the benchmark and learn the neural operators on the normalized scale


Baselines:

- FNO ()

- HFS ()

- WNO ()

- SAOT ()

- PDERefiner ()

