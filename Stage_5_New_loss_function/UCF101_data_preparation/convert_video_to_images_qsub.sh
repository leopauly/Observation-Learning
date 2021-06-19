#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -M cnlp@leeds.ac.uk
#$ -m be

module load ffmpeg/3.4

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

fps_value=10
for folder in $1/*;
do
    for file in "$folder"/*.avi
    do
        if [[ ! -d "${file[@]%.avi}" ]]; then
            mkdir -p "${file[@]%.avi}"
        fi
        ffmpeg -i "$file" -vf fps=$2 "${file[@]%.avi}"/%05d.jpg  # use fps=$2, if fps value is loaded from command line
        rm "$file"
    done
done

echo "Convertion over!"