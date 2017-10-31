#!/bin/bash


## all output files 
all_ann=annotations.csv
train_ann=train_annotations.csv
test_ann=test_annotations.csv
rm -rf all_ann $train_ann $test_ann xaa xab

# create a randomized and reproducible annotation file
# FIX: lt is not reproducible
RANDOM=1
for tg_file in *.TextGrid; do
    ../src/dump_textgrids.py $tg_file | grep -v filename | \
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

#echo "filename,start,end,label" 

sed -i '1s/^/filename,start,end,label\n/' $all_ann

