import os
import json
from sklearn.preprocessing import OneHotEncoder
import pickle
import time
import numpy as np
import yaml
import argparse

with open('/nfs/home/arafatj/master_project/src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# passed in args
speaker_hierarchy = None
hierarchy_level = None

def get_onehotencoders_speaker():
        """
        Create a OneHotEncoder for speaker labels
        """
        # add every speaker label that occurs in the speaker hierarchy mapping
        speaker_labels = []
        for key in speaker_hierarchy.keys():
            if speaker_hierarchy[key] not in speaker_labels:
                speaker_labels.append(speaker_hierarchy[key])
        
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
                        label = speaker_hierarchy[label]
                        

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


def process_clip_data(clip_data, segment_start, segment_end):
        """
        Helper to process clip data:
        Each domain in the CLIP_queries.json is processed seperately, so each domain will result in one feature vector.
        Each feature vector consits of the average probability over the speaker labels within all frames of the speaker segment.
        """
        
        clip_features = []
        for domain in clip_data.keys():
            # skip places365 for now
            if domain == 'places365':
                continue

            # return only the entries (frames) of the array that are within the current speaker segment.
            relevant_frames = []
            for i, data in enumerate(clip_data[domain]):
                # clip and places entry indices refer to their respective frame, so we can get the time of a entry by calculating (entry_index / fps)
                frame_time = i / config['fps']
                if segment_start <= frame_time <= segment_end:
                    relevant_frames.append(data)

            # sum um label probabilities over all frames
            label_sums = {}
            for frame in relevant_frames:
                for tup in frame:
                    label = tup[0]
                    prob = tup[1]

                    if label not in label_sums:
                        label_sums[label] = prob
                    else:
                        label_sums[label] += prob

            # calculate avg probability and append to feature vector
            feature_vec = []
            for label in label_sums.keys():
                avg_prob = label_sums[label] / len(relevant_frames)
                feature_vec.append(avg_prob)

            clip_features.append(feature_vec)
            
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


def process_sentiment_data(sentiment_data, segment_start, segment_end):
    """
    Helper to process sentiment data:
    Compute the average probabilities for positive, negative, and neutral sentiments across sentences
    within the speaker segment.
    Return an array with the entries [probability_positive, probability_negative, probability_neutral].
    """
    sentiments_sum = {'positive': 0, 'negative': 0, 'neutral': 0}
    i = 0
    for sentence in sentiment_data:
        # only take sentences that have a duration overlap of >= 80% with a speaker duration to be certain that the statement was made by the speaker
        ovp = max(0, min(segment_end, sentence['end']) - max(segment_start, sentence['start'])) / (sentence['end'] - sentence['start'])
        if ovp >= 0.8:
            i += 1
            # sum up sentiment probabilities
            for tup in sentence['sentiment_probs']:
                sentiment = tup[0]
                prob = tup[1]
                sentiments_sum[sentiment] += prob
    
    # no sentences found during the speaker segment
    if i == 0:
        return [0, 0, 0]
    
    # calculate average probabilities
    probs = []
    for sentiment in sentiments_sum:
        avg_prob = sentiments_sum[sentiment] / i
        probs.append(avg_prob)

    return probs


def data_speaker_segments():
    """
    Load all .pkls and create trainable data for Random Forest and XGBoost classifiers.

    The result will be a .pkl file with all training samples. 
    Each sample is represented by feature vector containing features that were aggregated during one speaker turn.
    """

    """
    # OneHotEncoder for speaker labels
    onehotencoder_speakers = get_onehotencoders_speaker()
    # dictionary with one OneHotEncoder for each clip domain vocabulary
    onehotencoders_clip = get_onehotencoders_clip()
    """
    
    # the following initializations are made to collect data about the whole dataset
    total_annotated_duration = 0
    total_annotated_speaker_turns = 0
    total_annotations_not_in_vocabulary = 0
    annotation_count_for_speaker = {}
    for speaker_label in speaker_hierarchy.keys():
        annotation_count_for_speaker[speaker_label] = 0

    # split the annotations according to news sources
    dirlist = os.listdir(config['annotations'])
    for abbrev in config['source_abbreviations']:
        news_source_annotations = [annotation_fn for annotation_fn in dirlist if abbrev in annotation_fn]

        # annotation statistics on news source level
        source_annotated_duration = 0
        source_annotated_speaker_turns = 0
        source_annotations_not_in_vocabulary = 0
        source_annotation_count_for_speaker = {}
        for speaker_label in speaker_hierarchy.keys():
            source_annotation_count_for_speaker[speaker_label] = 0

        feature_vectors = []
        # iterate over all annotations for the current news source
        for annotation_fn in news_source_annotations:
            if not annotation_fn.endswith('.json'):
                continue

            annotation_path = os.path.join(config['annotations'], annotation_fn)
            # load annotation json
            with open(annotation_path) as f:
                annotations = json.load(f)

            # get video file name and full path to the 
            video_fn = annotations['video_fn']
            output_path = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

            # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
            if not os.path.exists(output_path):
                print(f'{output_path} not yet processed')
                continue

            # load feature data from .pkls for the current video
            clip_data, places365_data, sbd_data, sd_data, diarization_data, sentiment_data = None, None, None, None, None, None
            for pkl_fn in os.listdir(output_path):
                pkl_path = os.path.join(output_path, pkl_fn)

                if 'clip.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        clip_data = pickle.load(pkl)
                elif 'places365.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        places365_data = pickle.load(pkl)
                elif 'shot_boundary_detection.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sbd_data = pickle.load(pkl)
                elif 'shot_density.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sd_data = pickle.load(pkl)
                elif 'speaker_diarization_large-v2.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        diarization_data = pickle.load(pkl)
                elif 'sentiment_large-v2.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sentiment_data = pickle.load(pkl)

            # iterate over every speaker segment in the annotations
            for speaker_segment in annotations['speaker']['segments']:
                speaker_start = speaker_segment['start']
                speaker_end = speaker_segment['end']
                speaker_groundtruth = speaker_segment['label']
                #speaker_groundtruth = get_speaker_annotation(speaker_start, annotations)

                # skip annotations that are not in our defined speaker label vocabulary
                if speaker_groundtruth not in speaker_hierarchy.keys():
                    total_annotations_not_in_vocabulary += 1
                    source_annotations_not_in_vocabulary += 1
                    continue

                # map the speaker ground truth label to our speaker hierarchy (e.g. only "anchor", "reporter" and "other")
                hierarchical_groundtruth = speaker_hierarchy[speaker_groundtruth]
                # encode ground truth as a number
                groundtruth_numerical = config['groundtruth_numerical'][hierarchical_groundtruth]


                """ START FEATURES """
                # feature vector (sample) generation by processing all pkl outputs and aggregating features
                vector = []

                # CLIP
                clip_features = process_clip_data(clip_data, speaker_start, speaker_end)
                # concat feature vector of every domain
                for clip_feature_vec in clip_features:
                    vector = vector + clip_feature_vec

                # Shot Density
                avg_shotdensity = process_density_data(sd_data, speaker_start, speaker_end)
                vector.append(avg_shotdensity)
                        
                # Length of speech
                los = speaker_end - speaker_start
                vector.append(los)

                # Sentiment
                avg_sentiments = process_sentiment_data(sentiment_data, speaker_start, speaker_end)
                vector = vector + avg_sentiments
                """ END FEATURES """


                # add numerical ground truth label at the end of the feature vector
                vector.append(groundtruth_numerical)

                # add sample
                feature_vectors.append(vector)

                # annotation statistics
                total_annotated_duration += speaker_end - speaker_start
                source_annotated_duration += speaker_end - speaker_start
                total_annotated_speaker_turns +=1
                source_annotated_speaker_turns += 1
                annotation_count_for_speaker[speaker_groundtruth] += 1
                source_annotation_count_for_speaker[speaker_groundtruth] += 1

        # convert feature vectors list to a 2-dim numpy array with the shape (samples, feature_amount)
        feature_vectors = np.vstack(feature_vectors)

        out_dir = f'{config["trainingdata_dir"]}/speaker/segment_based/hierarchy_{hierarchy_level}'
        os.makedirs(out_dir, exist_ok=True)

        # store feature samples as pkl
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_speaker_turns)}_samples_trainingdata.pkl', 'wb') as pkl:
            pickle.dump(feature_vectors, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        # store annotation statistics about the news source dataset
        annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(source_annotated_duration))
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_speaker_turns)}_samples_statistics.txt', 'w') as txt:
                txt.write(f'{source_annotated_speaker_turns} annotated speaker turns with a total duration of {annotated_duration_formatted}\n')
                txt.write(f'{source_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
                txt.write(f'Annotation counts for each speaker role:\n')
                for speaker in source_annotation_count_for_speaker.keys():
                    txt.write(f'\t{speaker}: {source_annotation_count_for_speaker[speaker]}\n')
        
    # store annotation statistics about the whole dataset as txt along the trainingdata
    total_annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(total_annotated_duration))
    with open(f'{out_dir}/all_{str(total_annotated_speaker_turns)}_samples_statistics.txt', 'w') as txt:
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


def data_speaker_windows():
    """
    Creates training data for news situations based on sliding windows.

    Sliding windows in different lengths (e.g. 5s, 10s, 20s, 40s) are placed over the video from start to 
    end every 2.5s. Features are aggregated window-wise and concatenated afterwards. A prediction is made
    every 2.5s based on all sliding windows.
    """
    # the following initializations are made to collect data about the whole dataset
    total_annotated_duration = 0
    total_annotated_speaker_turns = 0
    total_annotations_not_in_vocabulary = 0
    annotation_count_for_speaker = {}
    for speaker_label in speaker_hierarchy.keys():
        annotation_count_for_speaker[speaker_label] = 0

    # split the annotations according to news sources
    dirlist = os.listdir(config['annotations'])
    for abbrev in config['source_abbreviations']:
        news_source_annotations = [annotation_fn for annotation_fn in dirlist if abbrev in annotation_fn]

        # annotation statistics on news source level
        source_annotated_duration = 0
        source_annotated_speaker_turns = 0
        source_annotations_not_in_vocabulary = 0
        source_annotation_count_for_speaker = {}
        for speaker_label in speaker_hierarchy.keys():
            source_annotation_count_for_speaker[speaker_label] = 0

        feature_vectors = []
        # iterate over all annotations for the current news source
        for annotation_fn in news_source_annotations:
            if not annotation_fn.endswith('.json'):
                continue

            annotation_path = os.path.join(config['annotations'], annotation_fn)
            # load annotation json
            with open(annotation_path) as f:
                annotations = json.load(f)

            # get video file name and full path to the 
            video_fn = annotations['video_fn']
            output_path = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

            # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
            if not os.path.exists(output_path):
                print(f'{output_path} not yet processed')
                continue

            # load feature data from .pkls for the current video
            clip_data, places365_data, sbd_data, sd_data, diarization_data, sentiment_data = None, None, None, None, None, None
            for pkl_fn in os.listdir(output_path):
                pkl_path = os.path.join(output_path, pkl_fn)

                if 'clip.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        clip_data = pickle.load(pkl)
                elif 'places365.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        places365_data = pickle.load(pkl)
                elif 'shot_boundary_detection.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sbd_data = pickle.load(pkl)
                elif 'shot_density.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sd_data = pickle.load(pkl)
                elif 'speaker_diarization_large-v2.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        diarization_data = pickle.load(pkl)
                elif 'sentiment_large-v2.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sentiment_data = pickle.load(pkl)

            # iterate over every speaker segment in the annotations
            for speaker_segment in annotations['speaker']['segments']:
                speaker_start = speaker_segment['start']
                speaker_end = speaker_segment['end']
                speaker_groundtruth = speaker_segment['label']

                # skip annotations that are not in our defined speaker label vocabulary
                if speaker_groundtruth not in speaker_hierarchy.keys():
                    total_annotations_not_in_vocabulary += 1
                    source_annotations_not_in_vocabulary += 1
                    continue

                # map the speaker ground truth label to our speaker hierarchy (e.g. only "anchor", "reporter" and "other")
                hierarchical_groundtruth = speaker_hierarchy[speaker_groundtruth]
                # encode ground truth as a number
                groundtruth_numerical = config['groundtruth_numerical'][hierarchical_groundtruth]

                # feature representation for speaker containing one feature vector for each window size
                segment_features = []

                # create one feature representation for each window length
                for window_length in config['window_lengths']:
                    window_features = []

                    # round to the nearest extracted frame
                    speaker_start_fps = round(speaker_start * config['fps']) / config['fps']
                    speaker_end_fps = round(speaker_end * config['fps']) / config['fps']

                    # store all possible windows within the speaker segment for the given window_length
                    windows = []
                    window_start = speaker_start_fps
                    while window_start <= speaker_end_fps:
                        window_end = window_start + window_length
                        # if window overshoots speaker segment, cut the window short
                        if window_end > speaker_end_fps:
                            window_end = speaker_end_fps
                        windows.append((window_start, window_end))
                        # shift by window size
                        window_start += window_length 


                    window_features = []
                    # aggregate features within a single window
                    for window in windows:
                        window_start = window[0]
                        window_end = window[1]

                        """ START FEATURES """
                        feature_dict = {}

                        # CLIP
                        clip_features = process_clip_data(clip_data, window_start, window_end)
                        feature_dict['clip'] = clip_features

                        # Shot Density
                        avg_shotdensity = process_density_data(sd_data, window_start, window_end)
                        feature_dict['shot_density'] = avg_shotdensity
                                
                        # Length of speech
                        los = speaker_end - speaker_start
                        feature_dict['los'] = los

                        # Sentiment
                        avg_sentiments = process_sentiment_data(sentiment_data, window_start, window_end)
                        feature_dict['sentiments'] = avg_sentiments
                        """ END FEATURES """

                        window_features.append(feature_dict)


                    clip_features = []
                    shotdensities = []
                    sentiments = []
                    # collect same features across all windows of the same size to process (aggregate) them afterwards
                    for vector in window_features:
                        clip_features.append(vector['clip'])
                        shotdensities.append(vector['shot_density'])
                        sentiments.append(vector['sentiments'])
                    
                    # CLIP: avg clip probabilities across all windows of the same size
                    num_clip_domains = len(clip_features[0])
                    num_windows = len(windows)
                    for i in range(num_clip_domains):
                        domain_vectors = []
                        for j in range(num_windows):
                            domain_vectors.append(clip_features[j][i])
                        domain_vectors = np.array(domain_vectors)
                        # element wise avg of clip vectors across all windows of the same size
                        avgs = np.mean(domain_vectors, axis=0)
                        segment_features.extend(avgs.tolist())
                    
                    # SHOT_DENSITY: avg shot density across all windows of the same size
                    avg_shotdensity = sum(shotdensities) / len(shotdensities)
                    segment_features.append(avg_shotdensity)
                    
                    # LOS: length of speech stays the same independently of window size
                    los = window_features[0]['los']
                    segment_features.append(los)

                    # SENTIMENTS: take 80% quantile across all collected sentiment probabilities of the same window size
                    probs_pos, probs_neg, probs_neu = zip(*sentiments)
                    quantile_pos = np.quantile(probs_pos, 0.8)
                    quantile_neg = np.quantile(probs_neg, 0.8)
                    quantile_neu = np.quantile(probs_neu, 0.8)
                    segment_features.extend([quantile_pos, quantile_neg, quantile_neu])

                # add numerical ground truth label at the end of the feature vector
                segment_features.append(groundtruth_numerical)

                # add sample
                feature_vectors.append(segment_features)

                # annotation statistics
                total_annotated_duration += speaker_end - speaker_start
                source_annotated_duration += speaker_end - speaker_start
                total_annotated_speaker_turns +=1
                source_annotated_speaker_turns += 1
                annotation_count_for_speaker[speaker_groundtruth] += 1
                source_annotation_count_for_speaker[speaker_groundtruth] += 1

        # convert feature vectors list to a 2-dim numpy array with the shape (samples, feature_amount)
        feature_vectors = np.vstack(feature_vectors)


        out_dir = f'{config["trainingdata_dir"]}/speaker/window_based/hierarchy_{hierarchy_level}'
        os.makedirs(out_dir, exist_ok=True)
        
        # store feature samples as pkl
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_speaker_turns)}_samples_trainingdata.pkl', 'wb') as pkl:
            pickle.dump(feature_vectors, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        # store annotation statistics about the news source dataset
        annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(source_annotated_duration))
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_speaker_turns)}_samples_statistics.txt', 'w') as txt:
                txt.write(f'{source_annotated_speaker_turns} annotated speaker turns with a total duration of {annotated_duration_formatted}\n')
                txt.write(f'{source_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
                txt.write(f'Annotation counts for each speaker role:\n')
                for speaker in source_annotation_count_for_speaker.keys():
                    txt.write(f'\t{speaker}: {source_annotation_count_for_speaker[speaker]}\n')
        
    # store annotation statistics about the whole dataset as txt along the trainingdata
    total_annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(total_annotated_duration))
    with open(f'{out_dir}/all_{str(total_annotated_speaker_turns)}_samples_statistics.txt', 'w') as txt:
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
    parser.add_argument('--seg', action='store_true', help="Create training data based on speaker segments")
    parser.add_argument('--sw', action='store_true', help="Create training data based on sliding windows")
    parser.add_argument('--hierarchy', action='store', type=int, choices=[0, 1], default=0, help="Speaker mapping hierarchy level")

    args = parser.parse_args()

    hierarchy_level = args.hierarchy
    speaker_hierarchy = config[f'speaker_hierarchy_mapping_{hierarchy_level}']

    if args.seg:
        data_speaker_segments()
    
    if args.sw:
        data_speaker_windows()