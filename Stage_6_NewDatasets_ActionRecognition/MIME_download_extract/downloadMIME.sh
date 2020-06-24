# Script for installing MIME Dataset task by task
# Author : @leopauly | www.leopauly.com
#!/bin/bash
input="mime.txt"
while IFS= read -r line
do
  wget -cO $line
done < "$input"
