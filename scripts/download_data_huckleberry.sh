#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# download Adventures of Huckleberry Finn by Mark Twain

mkdir -p $data/huckleberry

mkdir -p $data/huckleberry/raw

wget https://www.gutenberg.org/files/76/76-0.txt
mv 76-0.txt $data/huckleberry/raw/tales.txt

# preprocess slightly

cat $data/huckleberry/raw/tales.txt | python $base/scripts/preprocess_raw.py > $data/huckleberry/raw/tales.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/huckleberry/raw/tales.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 5000 --tokenize --lang "en" --sent-tokenize > \
    $data/huckleberry/raw/tales.preprocessed.txt

# split into train, valid and test

head -n 440 $data/huckleberry/raw/tales.preprocessed.txt | tail -n 400 > $data/huckleberry/valid.txt
head -n 840 $data/huckleberry/raw/tales.preprocessed.txt | tail -n 400 > $data/huckleberry/test.txt
tail -n 3075 $data/huckleberry/raw/tales.preprocessed.txt | head -n 2955 > $data/huckleberry/train.txt
