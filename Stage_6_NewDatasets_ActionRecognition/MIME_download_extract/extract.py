# Description: Extracting .zip files
# Author : @leopauly | www.leopauly.com


import zipfile
import os

file_names=sorted(os.listdir('./'))
for file in file_names:
    if file.endswith(".zip"):
        print("\n","Extracting file:",file,"\n")
        with zipfile.ZipFile(file, "r") as z:
            z.extractall("./MIME_videos/"+file.split('.')[0]+"/")
