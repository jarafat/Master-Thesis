import os
import json
import argparse
import yaml
from datetime import timedelta

with open('/nfs/home/arafatj/master_project/src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


def get_annotated_videos():
    """
    Iterates over all annotations files and stores the absolute path of their associated video

    A txt with all video files can be found in the resulting video_list.txt
    """
    annotated_videos = []

    for file in sorted(os.listdir(config["annotations"])):
        if not file.endswith(".json"):
            continue

        with open(config['annotations'] + file) as f:
            annotations = json.load(f)
        
        # get the filename of the video associated with the annotations
        video_fn = annotations['video_fn']

        # check news source of video file
        video_path = None
        if os.path.exists(config['BildTV_videos'] + video_fn):
            video_path = config['BildTV_videos'] + video_fn
        elif os.path.exists(config['CompactTV_videos'] + video_fn):
            video_path = config['CompactTV_videos'] + video_fn
        elif os.path.exists(config['Tagesschau_videos'] + video_fn):
            video_path = config['Tagesschau_videos'] + video_fn
        elif os.path.exists(config['Tagesschau_videos2'] + video_fn):
            video_path = config['Tagesschau_videos2'] + video_fn
            
        if video_path:
            annotated_videos.append(video_path)
        else:
            print(f"Could not find video with file name {video_fn} in the json file {file} !")

    # write annotated files in video_list.txt
    with open(config['annotations'] + 'video_list.txt', 'w') as video_list:
        for video_path in annotated_videos:
            video_list.write(video_path + '\n')
        print(f"Successfully stored the video list in {video_list.name}")



def diarization_error_rate():
    """
    Calculates the Diarization Error Rate (DER) of the Whisper speaker diarization on the annotated data set.
    """
    import pickle
    import simpleder
    import numpy as np

    erroneous_files = ["TV-20220104-2020-5700.webl.h264.mp4", "TV-20230104-2024-3400.webl.h264.mp4", "TV-20230119-2021-3600.webl.h264.mp4", "compacttv_2022_01_21_l8TwbvtsSkc.mp4",
                       "compacttv_2022_01_26_wzQkoDUGMSA.mp4"]

    total_duration = 0
    errors = []
    for annotation_fn in os.listdir(config['annotations']):
        annotation_path = os.path.join(config['annotations'], annotation_fn)
        
        if not annotation_fn.endswith(".json"):
            continue

        with open(annotation_path) as f:
            annotations = json.load(f)
        
        video_fn = annotations['video_fn']
        if video_fn in erroneous_files:
            continue

        # load diarization pkl file
        diarization_file = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''), 'speaker_diarization_large-v2.pkl')
        with open(diarization_file, 'rb') as pkl:
            diarization_data = pickle.load(pkl)

        # extract annotation speaker segments
        reference_segments = []
        hypothesis_segments = []
        if 'speaker-gender' in annotations.keys(): # ANNOTATIONS with explicit speaker-diarization annotations
            for speaker_segment in annotations['speaker-gender']['segments']:
                start = speaker_segment['start']
                end = speaker_segment['end']
                speaker_label = speaker_segment['label']
                #reference_segments.append(('test', start, end))
                reference_segments.append((speaker_label.lower(), start, end))
        
            
            # only evaluate annotation parts where corona-news is annotated
            corona_segments = []
            for corona_segment in annotations['asr-de: korona']['segments']:
                if corona_segment['end'] - corona_segment['start'] < 2:
                    continue
                corona_segments.append(corona_segment)

            
            # extract speaker diarization speaker segments
            for speaker_segment in diarization_data:
                overlaps_annotations = False
                for corona_segment in corona_segments:
                    # check if the diarization segment overlaps the annotated parts
                    if (corona_segment['start'] <= speaker_segment['start_time'] <= corona_segment['end']):
                        overlaps_annotations = True
                        # if diarization speaker segment overshoots the annotated part, set the end time to the anntoation end
                        if speaker_segment['end_time'] > corona_segment['end']:
                            speaker_segment['end_time'] = corona_segment['end']
                    elif (corona_segment['start'] <= speaker_segment['end_time'] <= corona_segment['end']):
                        overlaps_annotations = True
                        if speaker_segment['start_time'] < corona_segment['start']:
                            speaker_segment['start_time'] = corona_segment['start']
                
                if not overlaps_annotations:
                    continue

                start = speaker_segment['start_time']
                end = speaker_segment['end_time']
                speaker_label = speaker_segment['speaker']
                #hypothesis_segments.append(('test', start, end))
                hypothesis_segments.append((speaker_label, start, end))
            
        
        else: # ANNOTATIONS without explicit speaker-diarization annotations (only speaker annotations)
            continue
            """
            for speaker_segment in annotations['speaker']['segments']:
                start = speaker_segment['start']
                end = speaker_segment['end']
                speaker_label = speaker_segment['label']
                reference_segments.append(('test', start, end))
                #reference_segments.append((speaker_label.lower(), start, end))

            for speaker_segment in diarization_data:
                start = speaker_segment['start_time']
                end = speaker_segment['end_time']
                speaker_label = speaker_segment['speaker']
                hypothesis_segments.append(('test', start, end))
                #hypothesis_segments.append((speaker_label, start, end))
            """
        
        for seg in reference_segments:
            total_duration += seg[2] - seg[1]

        error = simpleder.DER(reference_segments, hypothesis_segments)
        errors.append(error)

    avg_error = sum(errors) / len(errors)
    print(f"Total evaluation duration: {timedelta(seconds=total_duration)}")
    print("DER: {:.4f}".format(avg_error))
    print("Standard Deviation: {:.4f}".format(np.std(errors)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')
    parser.add_argument('--videolist', action='store_true', help="Extracts a list of videos that are annotated")
    parser.add_argument('--der', action='store_true', help="Calculate Diarization Error Rate")
    args = parser.parse_args()

    if args.videolist:
        get_annotated_videos()

    if args.der:
        diarization_error_rate()