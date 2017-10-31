#!/bin/bash

set -e

source activate mcr

#
## remove outputs
#
rm -rf trained.clf predicted.csv data/train_annotations.csv data/test_annotations.csv
rm -rf data.features data.item data.distance data.score data.abx data.csv

#
## preparing data
#

echo "################"
echo "# Preparing data"
echo "################"
cd data

# extracting annotations
./prep_annot.sh

# fixing sampling rate
for i in *.WAV; do
    f=$(basename $i .WAV);
    sox -t wav $i -e signed-integer -b 16 -c 1 -r 16000 ${f}.wav;
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
python ./src/prepare_abx.py  data/annotations.csv src/segmented.cfg  


source activate zerospeech
echo "#################"
echo "# Running ABX ..."
echo "#################"
python ./src/run_abx.py data
source deactivate 

