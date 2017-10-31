#!/bin/bash

DATA_DIR=$1
OUTPUT_DIR=$2

cd $DATA_DIR

## prepere output files 
all_ann=$OUTPUT_DIR/annotations.csv
train_ann=$OUTPUT_DIR/train_annotations.csv
test_ann=$OUTPUT_DIR/test_annotations.csv
rm -rf $train_ann $test_ann xaa xab

# create a randomized and reproducible annotation file
# FIX: lt is not reproducible
RANDOM=1
for tg_file in $DATA_DIR/*.TextGrid; do
    dump_textgrids.py $tg_file | grep -v filename | \
        sed 's/TextGrid/wav/g'
done | shuf > $all_ann

# split will generate two files xaa with 80% of all data
# and xab with the rest
split -l $[ $(wc -l $all_ann |cut -d" " -f1) * 80 / 100 ] $all_ann

echo "filename,start,end,label" > $train_ann
cat xaa >> $train_ann

echo "filename,start,end,label" > $test_ann
cat xab >> $test_ann

rm -rf xaa xab

sed -i '1s/^/filename,start,end,label\n/' $all_ann

cd -

