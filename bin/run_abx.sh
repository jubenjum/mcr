#!/bin/bash

set -e

EXP_NAME=data
BIN=$PWD/bin
DATA_DIR=$PWD/data
OUTPUT_DIR=$PWD/output
export PATH=$PATH:$PWD/bin


source activate mcr

#
## remove outputs
#
rm -rf $OUTPUT_DIR/*.csv $OUTPUT_DIR/data.*

#
## preparing data
#

mkdir -p $OUTPUT_DIR
echo "################"
echo "# Preparing data"
echo "################"
cd $DATA_DIR

# extracting annotations
prep_annot.sh $DATA_DIR $OUTPUT_DIR 

# fixing sampling rate
for i in *.WAV; do
    f=$(basename $i .WAV);
    sox -V0 -t wav $i -e signed-integer -b 16 -c 1 \
        -r 16000 ${f}.wav 2>&1 >> $OUTPUT_DIR/experiment.log 
done

cd -

#
## training
#
#python ./src/segmented_train.py data/train_annotations.csv src/segmented.cfg trained.clf
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
python $BIN/prepare_abx.py  $OUTPUT_DIR/annotations.csv $OUTPUT_DIR/segmented.cfg $OUTPUT_DIR/$EXP_NAME 

source activate zerospeech
echo "#################"
echo "# Running ABX ..."
echo "#################"
python $BIN/run_abx.py $OUTPUT_DIR/$EXP_NAME
source deactivate 

