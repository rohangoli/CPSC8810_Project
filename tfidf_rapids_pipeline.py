import argparse

from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import dask_cudf
import dask
from distributed import wait

import cupy as cp
from cuml.feature_extraction.text import HashingVectorizer
from cuml.dask.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords

STOPWORDS = stopwords.words('english')
PUNCTUATIONS = [x for x in '''!()-[]{};:'"\,<>./?@#$%^&*_~''']

FILES = ['/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Home_Entertainment_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Office_Products_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Kitchen_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Mobile_Apps_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Books_v1_02.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Musical_Instruments_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Music_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Digital_Video_Games_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Grocery_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Digital_Ebook_Purchase_v1_01.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Electronics_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Digital_Ebook_Purchase_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Books_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Automotive_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Toys_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Digital_Software_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Video_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Furniture_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Watches_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Home_Improvement_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Apparel_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Jewelry_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Home_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Video_Games_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Digital_Video_Download_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_PC_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Pet_Products_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Health_Personal_Care_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Camera_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Luggage_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Mobile_Electronics_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Beauty_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Wireless_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Personal_Care_Appliances_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Software_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Books_v1_01.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Baby_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Video_DVD_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Shoes_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Major_Appliances_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Lawn_and_Garden_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Digital_Music_Purchase_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Tools_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Outdoors_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Gift_Card_v1_00.tsv.gz', '/scratch1/rgoli/aws_customer_reviews/amazon_reviews_us_Sports_v1_00.tsv.gz']

def read_data_parquet(parquet_path):
    data = dask_cudf.read_parquet(parquet_path,
                                  columns=["review_body"],
                                 storage_options={'anon': True})
    return data

def read_data_tsv(tsv_path):
    data = dask_cudf.read_csv(tsv_path,sep='\t',columns=["review_body"],
                                 storage_options={'anon': True})
    return data

def text_preprocessor(data, column_name, PUNCTUATIONS,STOPWORDS):
    data = data[data[column_name].notnull()]
    data[column_name] = data[column_name] \
                            .str.lower() \
                            .str.replace_tokens(PUNCTUATIONS, " ") \
                            .str.replace_tokens(STOPWORDS, "") \
                            .str.normalize_spaces() \
                            .str.strip()
    return data

def hashing_vectorizer(data, column_name):
    vectorizer = HashingVectorizer(stop_words=None, preprocessor=None)
    # Meta is an empty dataframe matches the dtypes and columns of the output
    meta = dask.array.from_array(cp.sparse.csr_matrix(cp.zeros(1, dtype=cp.float32)))
    hashing_vectorized = data[column_name].map_partitions(vectorizer.fit_transform, meta=meta).astype(cp.float32)
    hashing_vectorized = hashing_vectorized.persist()
    wait(hashing_vectorized);
    return hashing_vectorized


def tfidf_transformer(data, client):
    multi_gpu_transformer = TfidfTransformer(client=client)
    result = multi_gpu_transformer.fit_transform(data)
    result = result.persist()
    wait(result);
    result.compute_chunk_sizes()
    return result
  
def full_pipeline_parquet(client, parquet_path):
    data = read_data_parquet(parquet_path)
    data = text_preprocessor(data, "review_body", PUNCTUATIONS, STOPWORDS)
    
    result = hashing_vectorizer(data, "review_body")
    result = tfidf_transformer(result, client)
   
    return result

def full_pipeline_tsv(client, data_path):
    data = read_data_tsv(data_path)
    print(data["review_body"][1].compute())
    data = text_preprocessor(data, "review_body", PUNCTUATIONS, STOPWORDS)
    print(data["review_body"][1].compute())
    result = hashing_vectorizer(data, "review_body")
    result = tfidf_transformer(result, client)
   
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-data','--data', required=True, help="Data Directory - Ex. data/")
    parser.add_argument('-ftype','--filetype', help='Input file format - parquest/tsv', required=True)

    args = parser.parse_args()
    
    ## Use 1 GPU as 1 Dask Worker
    cluster = LocalCUDACluster()
    client = Client(cluster)
    
    ## Uses 64 CPU cores as 8 Dask Workers
    #client = Client()
    
    print(PUNCTUATIONS)
    
    if args.filetype == "tsv":
        result = full_pipeline_tsv(client,args.data)
        #print(result.compute())
        print(result[1].compute())
        
    elif args.filetype == "parquet":
        result = full_pipeline_parquet(client,args.data)
        #print(result)
        print(result[1].compute())