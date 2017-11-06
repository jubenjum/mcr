#!/bin/bash

set -e

DATA_DIR=$1
OUTPUT_DIR=$2

# equivalent to $(readlink -f $1) in pure bash (compatible with macos)
function realpath {
    pushd $(dirname $1) > /dev/null
    echo $(pwd -P)
    popd > /dev/null
}
export -f realpath;

# called on script errors
function failure { [ ! -z "$1" ] && echo "Error: $1"; exit 1; }


# go to this script directory, restore current directory at exit
trap "cd $(pwd)" EXIT


# check if variables
[ ! -z "$DATA_DIR" ] || failure "DATA_DIR command line argument not set" 
[ ! -z "$OUTPUT_DIR" ] || failure "OUTPUT_DIR command line argument not set" 

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
    text_name=$(basename $tg_file .TextGrid) # only the file name no directory
    wav_name=$(ls -AF1 ${text_name}.* | grep -v TextGrid) # the other file
    ext_wav_file="${wav_name##*.}"
    curr_dir=$(realpath $tg_file)

    #  
    dump_textgrids.py $tg_file | grep -v filename | \
        sed "s/TextGrid/"$ext_wav_file"/g" | \
        sed "s#"$curr_dir"#"$OUTPUT_DIR/wav"#g" 
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

