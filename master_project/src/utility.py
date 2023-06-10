import os
import json
import argparse
import yaml

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

    erroneous_files = ["TV-20220104-2020-5700.webl.h264.mp4"]

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

        # extract annotation speaker segments
        reference_segments = []
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

        # load diarization pkl file
        diarization_file = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''), 'speaker_diarization_large-v2.pkl')
        with open(diarization_file, 'rb') as pkl:
            diarization_data = pickle.load(pkl)

        # extract speaker diarization speaker segments
        hypothesis_segments = []
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

        
        error = simpleder.DER(reference_segments, hypothesis_segments)
        errors.append(error)

    avg_error = sum(errors) / len(errors)
    print("DER: {:.3f}".format(avg_error))
    print("Standard Deviation: {:.3f}".format(np.std(errors)))



def scene_detection():
    from scenedetect import detect, ContentDetector
    
    scene_list = detect('/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220106-2021-1700.webl.h264.mp4', ContentDetector())

    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / End %s' % (
            i+1,
            scene[0].get_timecode(),
            scene[1].get_timecode()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')
    parser.add_argument('--videolist', action='store_true', help="Extracts a list of videos that are annotated")
    parser.add_argument('--der', action='store_true', help="Calculate Diarization Error Rate")
    parser.add_argument('--scenes', action='store_true', help="Get a list of scenes")
    args = parser.parse_args()

    if args.videolist:
        get_annotated_videos()

    if args.der:
        diarization_error_rate()

    if args.scenes:
        scene_detection()