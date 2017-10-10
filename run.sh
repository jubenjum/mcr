#!/bin/bash

#
## remove outputs
#
#rm -rf trained.clf predicted.csv data/train_annotations.csv data/test_annotations.csv
rm -rf data.features data.item

#
## preparing data
#

##echo "Preparing data"
##cd data
##
### extracting annotations
##rm *.csv
##./prep_annot.sh
##
### fixing sampling rate
##for i in *.WAV; do
##    f=$(basename $i .WAV);
##    sox -t wav $i -e signed-integer -b 16 -c 1 -r 16000 ${f}.wav;
##done
##
##cd -

#
## create ABX files
#

python ./src/prepare_abx.py  data/train_annotations.csv src/segmented.cfg trained.clf


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




