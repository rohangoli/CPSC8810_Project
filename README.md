# Rapids+Dask Environment

### Module on Palmetto Supercomputer

module load anaconda3/2019.10-gcc/8.3.1 cudnn/8.1.0.77-11.2-linux-x64-gcc/8.4.1 cuda/11.2.0-gcc/8.4.1

### Create anaconda environment

conda create -n rapids-22.02 -c rapidsai -c nvidia -c conda-forge rapids=22.02 python=3.8 cudatoolkit=11.2 dask-sql

# Dataset
https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt

