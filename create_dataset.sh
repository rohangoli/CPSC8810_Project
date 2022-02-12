#!/bin/bash

exit 0

echo "Downloading Amazon Customer Review Dataset"
echo "=========================================="

mkdir -p /scratch1/$USER/aws_customer_reviews
cd Data
cat ../datasets.txt | xargs -n 1 curl -O