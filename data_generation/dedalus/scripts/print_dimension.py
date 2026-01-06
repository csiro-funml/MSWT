

import h5py


def print_dimension(file_path):
    with h5py.File(file_path, 'r') as f:
        for key in f['tasks'].keys():
            print(f"\n{key}")
            dataset = f['tasks'][key]
            shape = dataset.shape
            print(f"  Shape: {shape}")

if __name__ == "__main__":
    # file_path = '/datasets/work/oa-tcch/work/forXuesong/with-forcing/long/realisation_0000/snapshots/snapshots_s1.h5'

    file_path = '/datasets/work/oa-tcch/work/forXuesong/realisation_0000/snapshots/snapshots_s1.h5'
    print_dimension(file_path)