# PNG to MP4 (x264)
ffmpeg -framerate 10 -i reach0/%02drgb.png -c:v libx264 -r 10 reach0.mp4

# MP4 to PNG
ffmpeg -i reach0.mp4 temp/%2d.png
