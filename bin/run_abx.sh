#!/bin/bash

set -e

EXP_NAME=data
BIN=$PWD/bin

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

OUTPUT_PATH=$(realpath $OUTPUT_DIR)
OUTPUT_NAME=$(basename $OUTPUT_DIR)
OUTPUT_DIR=${OUTPUT_PATH}/${OUTPUT_NAME}


DATA_PATH=$(realpath $DATA_DIR)
DATA_NAME=$(basename $DATA_DIR)
DATA_DIR=${OUTPUT_PATH}/${DATA_NAME}


source activate mcr

#
## remove previous outputs
#
rm -rf $OUTPUT_DIR/*.csv $OUTPUT_DIR/wav $OUTPUT_DIR/${EXP_NAME}.*

#
## preparing data
#

mkdir -p $OUTPUT_DIR/wav
echo "################"
echo "# Preparing data"
echo "################"
cd $DATA_DIR

# extracting annotations
prep_annot.sh $DATA_DIR $OUTPUT_DIR 

# fixing sampling rate, wav files are in same directory than 
# Praat TextGrid file
for original_wav in $DATA_DIR/*.{WAV,wav}; do
    new_wav=$OUTPUT_DIR/wav/$(basename $original_wav);
    echo "$original_wav -> $new_wav" | tee -a $OUTPUT_DIR/${EXP_NAME}.log
    sox -V0 -t wav "$original_wav" -e signed-integer -b 16 -c 1 \
        -r 16000 "$new_wav" 2>&1 | tee -a $OUTPUT_DIR/${EXP_NAME}.log 
done

cd -

#
## training
#
#python segmented_train.py $OUTPUT_DIR/train_annotations.csv $OUTPUT_DIR/segmented.cfg $OUTPUT_DIR/trained.clf
#
##
### predicting
##
#python src/segmented_predict.py data/test_annotations.csv trained.clf predicted.csv
#
##
### evaluating
##
#python src/segmented_eval.py data/test_annotations.csv predicted.csv



#
## create ABX files
#

echo "######################################"
echo "# Preparing item and feature files ..."
echo "######################################"
prepare_abx.py  $OUTPUT_DIR/annotations.csv $OUTPUT_DIR/segmented.cfg $OUTPUT_DIR/$EXP_NAME 

source activate zerospeech
echo "#################"
echo "# Running ABX ..."
echo "#################"
run_abx.py $OUTPUT_DIR/$EXP_NAME
source deactivate 

