#!/bin/bash

video_list="/nfs/home/arafatj/master_project/annotations/all/video_list.txt"
total_files=$(wc -l < "$video_list")
counter=1

while read video_path; do
    basename=$(basename "$video_path")
    echo "Processing file $basename: $counter/$total_files"
    counter=$((counter+1))

    python /nfs/home/arafatj/master_project/feature_extraction_code/features.py --video "$video_path" --diarize --places --clip --sbd --sd --noprint

    echo
done < "$video_list"