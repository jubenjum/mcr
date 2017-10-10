#!/bin/bash


# selected train data
data_sets=(dl_5_18_11_59_cb.TextGrid e_3_30_09_43_cdbrws.TextGrid de_6_04_15_32.TextGrid l_6_07_12_31.TextGrid l_4_13_7_54_cbsd.TextGrid dl_5_19_08_42_cm.TextGrid l_5_16_08_23_c.TextGrid e_3_31_8_51_c.TextGrid l_4_03_16_20_cmbds.TextGrid l_04_01_09_29.TextGrid de_4_15_09_26.TextGrid de_4_23_11_46_cdsb.TextGrid l_6_13_09_48.TextGrid de_02_12_09_59.TextGrid de_6_02_11_06.TextGrid de_5_26_8_50.TextGrid e_01_31_11_22.TextGrid)


## create the train annotation file
all_ann=annotations.csv
train_ann=train_annotations.csv
test_ann=test_annotations.csv
rm -rf all_ann $train_ann $test_ann xaa xab

# create a randomized and reproducible annotation file
# FIX: lt is not reproducible
RANDOM=1
for i in "${!data_sets[@]}"; do
    ../src/dump_textgrids.py ${data_sets[$i]} | grep -v filename | \
        sed 's/TextGrid/wav/g'
done | shuf > $all_ann

# split will generate two files xaa with 70% of all data
# and xab with the rest
split -l $[ $(wc -l $all_ann |cut -d" " -f1) * 5 / 100 ] $all_ann

echo "filename,start,end,label" > $train_ann
cat xaa >> $train_ann

echo "filename,start,end,label" > $test_ann
cat xab >> $test_ann

rm -rf xaa xab


