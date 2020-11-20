#!/bin/bash

# convert the images folder to the test.list and train.list file according to
#   the distribution, command will clear the train.list and test.list files first
#   Need to create the test.list and train.list files.
#
#   Args:
#       path: the path to the video folder
#       factor: denominator that split the train and test data. if the number 
#               is 4, then 1/4 of the data will be written to test.list and the
#               rest of the data will be written to train.list
#   Usage:
#       ./convert_images_to_list.sh path/to/video 4
#   Example Usage:
#       ./convert_images_to_list.sh ~/document/videofile 4
#   Example Output(train.list and test.list):
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d1_uncomp 0
#       /Volumes/passport/datasets/action_kth/origin_images/boxing/person01_boxing_d2_uncomp 0
#       ...
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d1_uncomp 1
#       /Volumes/passport/datasets/action_kth/origin_images/handclapping/person01_handclapping_d2_uncomp 1
#       ...

> train.list
> test.list
COUNT=-1
STEP=-1
DIV=0

for folder in $1/*
do 
    echo "Class:" $folder
    temp_idx=0
    shuffle_folder=0
    COUNT=$[$COUNT + 1]
    
    for temp_folder in $folder/*/*
    do
    shuffle_folder[$temp_idx]=$temp_folder
    temp_idx=$[$temp_idx + 1]
    done
    
    #echo ${shuffle_folder[@]}
    shuffle_folder=( $(shuf -e ${shuffle_folder[@]}) ) 
    echo "Total number of videos in class:$COUNT ${#shuffle_folder[@]}" 
     
    
    
    for imagesFolder in "${shuffle_folder[@]}"
    do  
        #echo "Sub folder: "$imagesFolder
        if [[ $imagesFolder == *"."* ]]; then
            rm "$imagesFolder"
            echo "Skipped"
            continue
        fi
        STEP=$[$STEP + 1]
        VAL=$(($STEP % $2))
        if  (($VAL > $DIV)) ; then 
            #echo 'train:'$COUNT
            echo "$imagesFolder" $COUNT >> train.list
        else
            echo "$imagesFolder" $COUNT >> test.list
            #echo 'test:'$COUNT
        fi 
    done
    unset shuffle_folder
done