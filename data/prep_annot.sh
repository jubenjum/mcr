#!/bin/bash


# selected train data
train=(dl_5_18_11_59_cb.TextGrid e_3_30_09_43_cdbrws.TextGrid de_6_04_15_32.TextGrid l_6_07_12_31.TextGrid l_4_13_7_54_cbsd.TextGrid dl_5_19_08_42_cm.TextGrid l_5_16_08_23_c.TextGrid e_3_31_8_51_c.TextGrid l_4_03_16_20_cmbds.TextGrid l_04_01_09_29.TextGrid de_4_15_09_26.TextGrid)

# selected test dtaa 
tests=(de_4_23_11_46_cdsb.TextGrid l_6_13_09_48.TextGrid de_02_12_09_59.TextGrid de_6_02_11_06.TextGrid de_5_26_8_50.TextGrid e_01_31_11_22.TextGrid)


## create the train annotation file
train_ann=train_annotations.csv
rm -rf $train_ann
echo "filename,start,end,label" | tee $train_ann 
for i in "${!train[@]}"; do
    ../src/dump_textgrids.py ${train[$i]} | grep -v filename | \
        sed 's/TextGrid/wav/g' | tee -a $train_ann
done


# create the test annotation file
test_ann=test_annotations.csv
rm -rf $test_ann
echo "filename,start,end,label" | tee $test_ann
for i in "${!tests[@]}"; do
    ../src/dump_textgrids.py ${tests[$i]} | grep -v filename | \
        sed 's/TextGrid/wav/g' | tee -a $test_ann
done

