# Rapids+Dask Environment
 
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

# Code Directory 
    .
    ├── Julia
    │   ├── TF_IDF_Serial.ipynb
    │   ├── TF_IDF_Serial_python.ipynb
    │   ├── TF_IDF_Thread.ipynb
    │   ├── TF_IDF_Threads_Spawn.ipynb
    │   ├── TF_IDF_distributed.ipynb
    │   ├── mpi_tf_idf.jl
    │   └── sample_us.tsv
    ├── README.md
    ├── Rapids_Dask
    │   ├── GPU2_DataConcat_timings.txt
    │   ├── MultiGPU_DataConcat_timings.txt
    │   ├── rapids_timings.txt
    │   ├── tfidf_rapids_cntvectzr.py
    │   ├── tfidf_rapids_output.txt
    │   ├── tfidf_rapids_pipeline.py
    │   └── tfidf_rapids_pipeline_v2.py
    ├── Spark
    │   ├── TF_IDF_COMBINE_ALL_4Node.ipynb
    │   ├── TF_IDF_COMBINE_ALL_DF_1Node.ipynb
    │   ├── metrics_1node
    │   │   ├── perf_results_counttf_48.csv
    │   │   └── perf_results_hashtf_48.csv
    │   ├── metrics_4node
    │   │   ├── perf_results_counttf_48.csv
    │   │   └── perf_results_hashtf_48.csv
    │   └── source.txt
    ├── create_dataset.sh
    ├── csr_notes.txt
    ├── datasets.txt
    ├── notes.txt
    └── sample_us.tsv

    5 directories, 27 files

### Rapids
    - tfidf_rapids_pipeline.py : Implementation of Rapids TF-IDF Hashing-Vectorizer Pipeline for processing data files individually
    - tfidf_rapids_pipeline_v2.py : Implementation of Rapids TF-IDF Hashing-Vectorizer Pipeline for processing data files by concatenation 
    - tfidf_rapids_cntvectzr.py : Implementation of Rapids TF-IDF Count-Vectorizer Pipeline (Not Working)

# Julia
    TF_IDF_Serial.ipynb - jupyter notebook for serial version of TF-IDF
    TF_IDF_Thread.ipynb - jupyter notebook for parallelizing the looping construct
    TF_IDF_Threads_Spawn.ipynb - jupyter notebook for parallelizing using @spawn macro
    mpi_tf_idf.jl - compute TF-IDf using mpi (mpirun -np <no_of_process> julia mpi_tf_idf.jl)
    TF_IDF_distributed.ipynb - jupyter notebook for parallelizing using distributed computing
    
# Spark
    TF_IDF_COMBINE_ALL_4Node.ipynb - Notebook for running on 4 nodes
    TF_IDF_COMBINE_ALL_DF_1Node.ipynb - Notebook for running on 1 Node
    The core code for computation of TF-IDF is same in the above notebooks. The only difference is with respect to the folder into which the computation time results are dumped. Results of computation on 4 Nodes are dumped into metrics_4node; Results on 1 Node are dumped into metrics_1node. Please ensure bothe these folders are created in the same directory as the jupyter notebook to avoid errors
    
