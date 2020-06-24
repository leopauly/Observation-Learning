#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -l nodes=10,tpp=64,node_type=256thread-112G,mcdram_mode=half
#$ -M cnlp@leeds.ac.uk
#$ -m be

module load ffmpeg

echo "Converting viedos to frames...!"

#!/bin/bash

# convert the avi video to images
#   Usage (sudo for the remove priviledge):
#       sudo ./convert_video_to_images.sh path/to/video fps
#   Example Usage:
#       sudo ./convert_video_to_images.sh ~/document/videofile/ 5
#   Example Output:
#       ~/document/videofile/walk/video1.avi 
#       #=>
#       ~/document/videofile/walk/video1/00001.jpg
#       ~/document/videofile/walk/video1/00002.jpg
#       ~/document/videofile/walk/video1/00003.jpg
#       ~/document/videofile/walk/video1/00004.jpg
#       ~/document/videofile/walk/video1/00005.jpg
#       ...

for folder in $1/*;
do
    echo $folder
    for file in "$folder"/*/*.mp4
    do
        echo "Converting $folder :$file"
        if [[ $file == *"depth"* ]]; then
            rm "$file"
            echo "Skipped"
            continue
        fi
        if [[ ! -d "${file[@]%.mp4}" ]]; then
            mkdir -p "${file[@]%.mp4}"
        fi
        ffmpeg -i "$file" -vf fps=$2 "${file[@]%.mp4}"/%05d.png
        rm "$file"
    done
done

echo "Convertion over!"