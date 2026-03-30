#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
logs=$base/logs
data=$base/data
tools=$base/tools

dataset="huckleberry"

dropout=0.2

logpath=$logs/dropout$dropout.tsv


mkdir -p $models
mkdir -p $logs

num_threads=6
device=""

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/$dataset \
        --epochs 40 \
        --log-interval 100 \
        --emsize 250 --nhid 250 --dropout $dropout --tied \
        --save $models/model_$dataset\_dropout-$dropout.pt \
        --log-file $logpath
)

echo "time taken:"
echo "$SECONDS seconds"

