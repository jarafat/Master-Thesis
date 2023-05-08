import os
import json
import argparse


config = {
    "annotations": "/nfs/home/arafatj/master_project/annotations/all/",
    "BildTV_videos": "/nfs/data/fakenarratives/BildTV/videos/",
    "CompactTV_videos": "/nfs/data/fakenarratives/CompactTV/videos/",
    "Tagesschau_videos": "/nfs/data/fakenarratives/tagesschau/videos/2022/",
    "clip_queries": "/nfs/home/arafatj/master_project/models/CLIP/CLIP_queries.json",
    "output_dir": "/nfs/home/arafatj/master_project/OUTPUT/",
    "trainingdata_dir": "/nfs/home/arafatj/master_project/trainingdata/",
    "fps": 2,
    "speaker_hierarchy_mapping": {
        "anchor": "anchor",
        "reporter": "reporter",
        "doctor-nurse": "other",
        "expert-medicine": "other",
        "expert-other": "other",
        "layperson": "other",
        "layperson-4-covid": "other",
        "layperson-x-covid": "other",
        "police-frwr": "other",
        "politician-ampel": "other",
        "politician-other": "other",
        "politician-n-de": "other",
        "politician-right": "other",
        "celeb": "other",
                        },
    "groundtruth_numerical": {
        "anchor": 0,
        "reporter": 1,
        "other": 2
    }
}

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



def convert_to_trainable_speaker_data():
    """
    Load all .pkls and create trainable data for Random Forest and XGBoost classifiers.

    The result will be a .pkl file with all training samples. 
    Each sample is represented by feature vector containing features that were aggregated during one speaker turn.
    """
    from sklearn.preprocessing import OneHotEncoder
    import pickle
    import time
    import numpy as np

    def get_onehotencoders_speaker():
        """
        Create a OneHotEncoder for speaker labels
        """
        # add every speaker label that occurs in the speaker hierarchy mapping
        speaker_labels = []
        for key in config['speaker_hierarchy_mapping'].keys():
            if config['speaker_hierarchy_mapping'][key] not in speaker_labels:
                speaker_labels.append(config['speaker_hierarchy_mapping'][key])
        
        onehotencoder = OneHotEncoder(sparse_output=False)
        onehotencoder.fit([[label] for label in speaker_labels])
        return onehotencoder
    
    def get_onehotencoders_clip():
        """
        Create a OneHotEncoder for each CLIP_queries.json by extracting the vocabulary of possible labels for each domain.

        Returns a dictionary where the keys are the CLIP_queries.json domains and the value contains the OneHotEncoder.
        """
        with open(config['clip_queries']) as f:
            clip_queries = json.load(f)

        onehotencoders = {}
        for domain in clip_queries.keys():
            vocabulary = []
            
            # special case for places365, where the dictionary values define the vocabularies instead of the dictionary keys
            if domain == 'places365':
                for key in clip_queries[domain].keys():
                    vocabulary.append(clip_queries[domain][key][0])
            else:
                for label in clip_queries[domain].keys():
                    if domain == 'news roles':
                        label = fix_clip_typos(label)
                        # map news roles to our hierarchy
                        label = config['speaker_hierarchy_mapping'][label]

                    if label not in vocabulary:
                        vocabulary.append(label)

            onehotencoders[domain] = OneHotEncoder(sparse_output=False)
            onehotencoders[domain].fit([[label] for label in vocabulary])
            
        return onehotencoders

    def get_speaker_annotation(speaker_start, annotations):
        """
        Helper to return the speaker label (anchor, reporter, etc.) for a given speaker-gender annotation.
        TODO: if speaker annotation does not start exactly at the same time as speaker-gender annotation this won't return a result.
        """
        for segment in annotations['speaker']['segments']:
            if segment['start'] == speaker_start:
                return segment['label']
        print('No speaker annotation found for the given speaker start time')
   
    def fix_clip_typos(label):
        """
        Fix typos in CLIP_queries.json
        TODO: Remove this after .pkls are fixed 
        """
        if label == 'politcian-right':
            label = 'politician-right'
        elif label == 'politcian-n-de':
            label = 'politician-n-de'
        elif label == 'police-feuerwehr':
            label = 'police-frwr'
        elif label == 'ppl-x-covid':
            label = 'layperson-x-covid'
        elif label == 'ppl-4-covid':
            label = 'layperson-4-covid'
        elif label == 'politician':
            label = 'politician-other'
        
        return label
    def process_clip_data(clip_data, onehotencoders, segment_start, segment_end):
        """
        Helper to process clip data:
        Each domain in the CLIP_queries.json is processed seperately, so each domain will result in one feature.
        To extract features from clip data, we will process the clip outputs for each frame that is within the given segment.
        (Inside a domain) For each frame, we store the label that was assigned the highest probability by the clip model.
        All lables with the highest probabilty will be counted across the frames and the label that has
        the highest occurrence after processing all frames within the segment will be taken as the feature.
        Since features need to be numbers, the resulting label will be transformed to a OneHotVector.
        The return value is an array that contains 1 encoded (OneHotEncoded) label for each domain.
        """
        clip_features = []
        for domain in clip_data.keys():
            # return only the entries (frames) of the array that are within the current speaker segment.
            relevant_frames = []
            for i, data in enumerate(clip_data[domain]):
                # clip and places entry indices refer to their respective frame, so we can get the time of a entry by calculating (entry_index / fps)
                frame_time = i / config['fps']
                if segment_start <= frame_time <= segment_end:
                    relevant_frames.append(data)

            # process clip data
            label_count = {}
            for frame in relevant_frames:
                # get most probable tuple that are in the form of (label, probability)
                max_entry = max(frame, key=lambda item: item[1])
                max_label = max_entry[0]

                # map news roles to defined hierarchy
                if domain == 'news roles':
                    max_label = fix_clip_typos(max_label)
                    max_label = config['speaker_hierarchy_mapping'][max_label]

                # Count label occurences
                if max_label in label_count.keys():
                    label_count[max_label] += 1
                else:
                    label_count[max_label] = 1

            # get label with highest occurrence and return it
            max_label = max(label_count, key=label_count.get)
            
            onehotvector = onehotencoders[domain].transform([[max_label]])
            clip_features.append(onehotvector)
        return clip_features
    
    def process_density_data(sd_data, segment_start, segment_end):
        """
        Helper do process shot density data:
        Calculate the average shot density across the collected shot density data within the given segment.
        The return value is the average shot density of the segment, which can then be used as a feature.
        """
        density_with_time = list(zip(sd_data['time'], sd_data['y']))
        
        # return only the entries (frames) of the array that are within the current speaker segment.
        relevant_frames = []
        for data in density_with_time:
            if segment_start <= data[0] <= segment_end:
                relevant_frames.append(data)
        
        # calculate average shot density
        sum_density = 0
        for data in relevant_frames:
            sum_density += data[1]
        avg_shotdensity = sum_density / len(relevant_frames)

        return avg_shotdensity

    # OneHotEncoder for speaker labels
    onehotencoder_speakers = get_onehotencoders_speaker()
    # dictionary with one OneHotEncoder for each clip domain vocabulary
    onehotencoders_clip = get_onehotencoders_clip()
    
    # the following initializations are made to collect data about the processed dataset
    total_annotated_duration = 0
    total_annotated_speaker_turns = 0
    total_annotations_not_in_vocabulary = 0
    annotation_count_for_speaker = {}
    for speaker_label in config['speaker_hierarchy_mapping'].keys():
        annotation_count_for_speaker[speaker_label] = 0

    feature_vectors = []
    # iterate over all annotations
    for annotation_fn in os.listdir(config['annotations']):
        if not annotation_fn.endswith('.json') or 'com-' in annotation_fn: # TODO: REMOVE COMPACT SKIPPING
            continue

        annotation_path = os.path.join(config['annotations'], annotation_fn)
        # load annotation json
        with open(annotation_path) as f:
            annotations = json.load(f)

        # get video file name and full path to video
        video_fn = annotations['video_fn']
        output_dir = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

        # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
        if not os.path.exists(output_dir):
            print(f'{output_dir} not yet processed')
            continue
        elif len(os.listdir(output_dir)) < 5:
            print(f'{output_dir} still being processed...')
            continue

        # load feature data from .pkls for the current video
        clip_data, places365_data, sbd_data, sd_data, diarization_data = None, None, None, None, None
        for pkl_fn in os.listdir(output_dir):
            pkl_path = os.path.join(output_dir, pkl_fn)

            if "clip" in pkl_fn:
                with open(pkl_path, 'rb') as pkl:
                    clip_data = pickle.load(pkl)
            elif "places365" in pkl_fn:
                with open(pkl_path, 'rb') as pkl:
                    places365_data = pickle.load(pkl)
            elif "shot_boundary_detection" in pkl_fn:
                with open(pkl_path, 'rb') as pkl:
                    sbd_data = pickle.load(pkl)
            elif "shot_density" in pkl_fn:
                with open(pkl_path, 'rb') as pkl:
                    sd_data = pickle.load(pkl)
            elif "speaker_diarization" in pkl_fn:
                with open(pkl_path, 'rb') as pkl:
                    diarization_data = pickle.load(pkl)

        # iterate over every speaker segment in the annotations
        for speaker_segment in annotations['speaker-gender']['segments']:
            speaker_start = speaker_segment['start']
            speaker_end = speaker_segment['end']
            speaker_label = speaker_segment['label']
            speaker_groundtruth = get_speaker_annotation(speaker_start, annotations)

            # skip annotations that are not in our defined speaker label vocabulary
            if speaker_groundtruth not in config['speaker_hierarchy_mapping'].keys():
                total_annotations_not_in_vocabulary += 1
                continue

            # map the speaker ground truth label to our speaker hierarchy (e.g. only "anchor", "reporter" and "other")
            hierarchical_groundtruth = config['speaker_hierarchy_mapping'][speaker_groundtruth]
            # encode ground truth as a number
            groundtruth_numerical = config['groundtruth_numerical'][hierarchical_groundtruth]

            # START of feature vector (sample) generation by processing all pkl outputs and aggregating features
            vector = []

            # CLIP
            clip_features = process_clip_data(clip_data, onehotencoders_clip, speaker_start, speaker_end)
            for onehotvector in clip_features:
                # add every element of the onehotvector to our feature vector (including the 0's)
                vector = vector + onehotvector[0].tolist()
            
            # Shot Density
            avg_shotdensity = process_density_data(sd_data, speaker_start, speaker_end)
            vector.append(avg_shotdensity)
                    

            # add numerical ground truth label at the end of the feature vector
            vector.append(groundtruth_numerical)

            # add sample
            feature_vectors.append(vector)

            # annotation statistics
            total_annotated_duration += speaker_end - speaker_start
            total_annotated_speaker_turns +=1
            annotation_count_for_speaker[speaker_groundtruth] += 1

    # convert feature vectors list to a 2-dim np array with the shape (samples, feature_amount)
    feature_vectors = np.vstack(feature_vectors)

    # store feature samples as pkl
    with open(os.path.join(config['trainingdata_dir'], str(total_annotated_speaker_turns) + 'samples_trainingdata.pkl'), 'wb') as pkl:
        pickle.dump(feature_vectors, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Successfully converted collected data to trainable feature vectors! Result: {pkl.name}")
        
    # store annotation statistics as txt along the trainigndata
    total_annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(total_annotated_duration))
    with open(os.path.join(config['trainingdata_dir'], str(total_annotated_speaker_turns) + 'samples_statistics.txt'), 'w') as txt:
            txt.write(f'{total_annotated_speaker_turns} annotated speaker turns with a total duration of {total_annotated_duration_formatted}\n')
            txt.write(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
            txt.write(f'Annotation counts for each speaker role:\n')
            for speaker in annotation_count_for_speaker.keys():
                txt.write(f'\t{speaker}: {annotation_count_for_speaker[speaker]}\n')
            print(f"Statistics about the annotations can be found in {txt.name}")
              

    # annotation files statistics stdout
    print()
    print(f'{total_annotated_speaker_turns} annotated speaker turns with a total duration of {total_annotated_duration_formatted}')
    print(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')
    parser.add_argument('--videolist', action='store_true', help="Extracts a list of videos that are annotated")
    parser.add_argument('--trainingdata', action='store_true', help="Convert the pkl files to data that can be used of training")
    args = parser.parse_args()

    if args.videolist:
        get_annotated_videos()
    
    if args.trainingdata:
        convert_to_trainable_speaker_data()