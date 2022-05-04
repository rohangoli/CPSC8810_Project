# Rapids+Dask Environment


## Julia
    TF_IDF_Serial.ipynb - jupyter notebook for serial version of TF-IDF
    TF_IDF_Thread.ipynb - jupyter notebook for parallelizing the looping construct
    TF_IDF_Threads_Spawn.ipynb - jupyter notebook for parallelizing using @spawn macro
    mpi_tf_idf.jl - compute TF-IDf using mpi (mpirun -np <no_of_process> julia mpi_tf_idf.jl
    TF_IDF_distributed.ipynb - jupyter notebook for parallelizing using distributed computing
    
### Module on Palmetto Supercomputer

module load anaconda3/2019.10-gcc/8.3.1 cudnn/8.1.0.77-11.2-linux-x64-gcc/8.4.1 cuda/11.2.0-gcc/8.4.1

### Create anaconda environment

conda create -n rapids-22.02 -c rapidsai -c nvidia -c conda-forge rapids=22.02 python=3.8 cudatoolkit=11.2 dask-sql

# Dataset
### Link
    https://s3.amazonaws.com/amazon-reviews-pds/tsv/index.txt

### Downloaded Dataset available on Palmetto
    /scratch1/rgoli/aws_customer_reviews

### Extracting Dataset
    mkdir -p Data

    cp /scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_PC_v1_00.tsv.gz Data/

    gunzip -k amazon_reviews_us_PC_v1_00.tsv.gz
