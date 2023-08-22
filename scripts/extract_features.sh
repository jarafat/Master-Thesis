#!/bin/bash

video_list="/nfs/home/arafatj/master_project/annotations/all/video_list.txt"
total_files=$(wc -l < "$video_list")
counter=1

while read video_path; do
    basename=$(basename "$video_path")
    echo "Processing file $basename: $counter/$total_files"
    counter=$((counter+1))

    python /nfs/home/arafatj/master_project/src/features.py --video "$video_path" --clip --asr --pos --scenes --noprint #--asr --diarize --places --clip --sentiment --scenes --pos --imgemb --sentemb --sbd --sd --ner --noprint   

    echo
done < "$video_list"