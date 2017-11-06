#!/bin/bash

set -e

EXP_NAME=data
DATA_DIR="$1" 
OUTPUT_DIR="$2"


# equivalent to $(readlink -f $1) in pure bash (compatible with macos)   
function realpath {                                                      
    readlink -f "$1"
    #pushd $(dirname "$1") > /dev/null                                      
    #echo $(pwd -P)                                                       
    #popd > /dev/null                                                     
}                                                                        
export -f realpath;                                                      
                                                                         
# called on script errors                                                
function failure { [ ! -z "$1" ] && echo "Error: $1"; exit 1; }          
                                                                         
                                                                         
# check if variables                                                     
[ ! -z "$DATA_DIR" ] || failure "DATA_DIR command line argument not set"   
[ ! -z "$OUTPUT_DIR" ] || failure "OUTPUT_DIR command line argument not set" 

OUTPUT_PATH=$(realpath "$OUTPUT_DIR")
OUTPUT_DIR="${OUTPUT_PATH}"

DATA_PATH=$(realpath "$DATA_DIR")
DATA_DIR="${DATA_PATH}"

source activate mcr

#
## remove previous outputs
#
rm -rf "$OUTPUT_DIR/*.csv" "$OUTPUT_DIR/wav" "$OUTPUT_DIR/${EXP_NAME}.*"

#
## preparing data
#

mkdir -p "$OUTPUT_DIR/wav"
echo "################"
echo "# Preparing data"
echo "################"

# extracting annotations
echo "doing prep_annot"
prep_annot.sh "$DATA_DIR" "$OUTPUT_DIR" 

cd "$DATA_DIR"
# fixing sampling rate, wav files are in same directory than 
# Praat TextGrid file
echo "fixing wav files type"
shopt -s nullglob
for original_wav in "$DATA_DIR"/*.{WAV,wav,aif}; do
    ext_file="${original_wav##*.}" 
    new_wav="$OUTPUT_DIR/wav/"$(basename "$original_wav" .$ext_file | tr ' ' '_');
    echo "$original_wav -> $new_wav" >> "${OUTPUT_DIR}/${EXP_NAME}".log
    sox -V0 "$original_wav" -e signed-integer -b 16 -c 1 \
        -r 16000 "$new_wav".wav 2>&1 >> "$OUTPUT_DIR/${EXP_NAME}".log 
done


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
prepare_abx.py "$OUTPUT_DIR/annotations.csv" "$OUTPUT_DIR/segmented.cfg" "$OUTPUT_DIR/$EXP_NAME"

source activate zerospeech
echo "#################"
echo "# Running ABX ..."
echo "#################"
run_abx.py "$OUTPUT_DIR/$EXP_NAME"
source deactivate 

echo "aca"

