import os
import json
from sklearn.preprocessing import OneHotEncoder
import pickle
import time
import numpy as np
import yaml
import argparse
import datetime
import math

import torch
import torch.nn.functional as F

with open('/nfs/home/arafatj/master_project/src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# passed in args
speaker_hierarchy = None
hierarchy_level = None

""" START HELPERS """

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


def get_speaker_annotation(speaker_start, annotations):
        """
        Helper to return the speaker label (anchor, reporter, etc.) for a given speaker-gender annotation.
        TODO: if speaker annotation does not start exactly at the same time as speaker-gender annotation this won't return a result.
        """
        for segment in annotations['speaker']['segments']:
            if segment['start'] == speaker_start:
                return segment['label']
        print('No speaker annotation found for the given speaker start time')


def get_current_situation(speaker_start, speaker_end, annotations):
    """
    Returns the dominant (longest overlapping) news situation that occurs during the speaker segment
    """
    ovps = {}
    for label in config['labels_situations']:
        ovps[label] = 0

    for situation in annotations['captions: talking']['segments']:
        # some situations can be 'None' somehow
        if situation['label'] == None:
            continue

        situation_label = situation['label'].strip()

        # check if news situation label is in the defined vocabulary of news situations
        if situation_label not in config['labels_situations']:
            continue

        ovp = max(0, min(speaker_end, situation['end']) - max(speaker_start, situation['start'])) / (speaker_end - speaker_start)
        # during a speaker segment multiple news situation segments can occur, so we sum up the overlap percentages of each occuring situation
        ovps[situation_label] += ovp 
    
    max_label = max(ovps, key=ovps.get)
    
    # unclear news situation (speaker does multiple news situations in one speaker segment)
    if ovps[max_label] < 0.7:
        return None

    return max_label


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
        if len(relevant_frames) == 0:
            avg_shotdensity = 0
        else:
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


def process_scenes_data(scenes_data, segment_start, segment_end):
    """
    Helper to process scenes data.
    Returns the number of scenes during the given (speaker) segment.
    """
    scene_count = 0
    for scene in scenes_data:
        scene_start = scene[0].get_seconds()
        scene_end = scene[1].get_seconds()
        ovp = max(0, min(segment_end, scene_end) - max(segment_start, scene_start)) / (segment_end - segment_start)
        if ovp > 0:
            scene_count += 1

    return scene_count


def process_pos_data(pos_data, segment_start, segment_end):
    """
    Helper to process POS tagger data.
    POS tags are converterted to their relative amount according to the total amount of all POS tags withing a speaker segment.
    That way we can determine if some speaker roles tend to use certain POS tags more often.
    """

    for pos_seg in pos_data:
        # overlap between actual speaker segment timeframe and the speaker diarization timeframe where pos tags have been extracted from
        ovp = max(0, min(segment_end, pos_seg['end_time']) - max(segment_start, pos_seg['start_time'])) / (pos_seg['end_time'] - pos_seg['start_time'])
        # only assign the speaker diarization segment to the speaker if overlap of timeframes is > 70%
        if ovp > 0.7:
            pos_vector = pos_seg['vector']
            # total amount of POS tags in the segment
            num_tags = np.sum(pos_vector)
            if num_tags == 0:
                continue
            # relative values of POS tags according to total POS tags
            relative_pos_vector = pos_vector / num_tags
            return relative_pos_vector.tolist()
    
    # if no speaker diarization segment matches the given annotation segment return a vector with 0's
    vec_len = len(pos_data[0]['vector'])
    return np.zeros(vec_len).tolist()


def process_ner_data(ner_data, segment_start, segment_end):
    """
    Helper to process NER tagger data.
    If a named entity is recognized, a 1 indicates its presence, independently of how many of that named entity are mentioned.
    That way, we can determine if some speaker roles tend to use certain named entities more often.
    """

    for ner_seg in ner_data:
        # overlap between actual speaker segment timeframe and the speaker diarization timeframe where pos tags have been extracted from
        ovp = max(0, min(segment_end, ner_seg['end_time']) - max(segment_start, ner_seg['start_time'])) / (ner_seg['end_time'] - ner_seg['start_time'])
        if ovp > 0.7:
            ner_vector = ner_seg['vector']
            # remove events, since no events are in the dataset
            ner_vector = np.delete(ner_vector, 4)
            # if a named entity is recognized, a 1 indicates its presence, independently of how many of that named entity are mentioned
            ner_vector[ner_vector > 0] = 1
            return ner_vector
        
    # if no speaker diarization segment matches the given annotation segment return a vector with 0's (substract one because events are being removed)
    vec_len = len(ner_data[0]['vector']) - 1
    return np.zeros(vec_len).tolist()


def cosine_similary(list1, list2):
    """
    Helper to calculate the cosine similarities of all pairs between two lists.
    Returns a matrix with the shape (len(list1), len(list2)).
    """
    similarities = torch.zeros((len(list1), len(list2)))

    for i, emb1 in enumerate(list1):
        for j, emb2 in enumerate(list2):
            if not isinstance(emb1, torch.Tensor):
                emb1 = torch.from_numpy(emb1).unsqueeze(0)
            if not isinstance(emb2, torch.Tensor):
                emb2 = torch.from_numpy(emb2).unsqueeze(0)

            similarity = F.cosine_similarity(emb1, emb2)
            similarities[i, j] = similarity
    
    return similarities


def process_imgemb_data(imgemb_data, segment_start, segment_end, spk_index, speaker_annots):
    """
    Helper to process image embeddings.
    All image embeddings of speaker segments in the context are collected and compared to the reference speaker.
    Comparing a reference speaker to the speaker before and after, the 80% quantile across all image similarities is returned.
    """

    # fps the embeddings have been extracted with
    fps = config['fps']
    # context size describes the amount of speakers that should be included before and after the current speaker
    context_size = config['context_size']

    # get the exact frame numbers where the segment starts and ends
    segment_start_frame = round(segment_start * fps)
    segment_end_frame = round(segment_end * fps)

    # get the image embeddings for the segment
    imgembs_segment = imgemb_data[segment_start_frame:segment_end_frame]

    # number of annotated speakers
    annotated_speakers_count = len(speaker_annots)
    
    imgembs_before = []
    imgembs_after = []
    for i in range(1, context_size+1):
        spk_idx_before = spk_index - i
        spk_idx_after = spk_index + i

        # SPEAKER BEFORE
        # if the current speaker is the first speaker, the speaker before does not exist
        if spk_idx_before < 0:
            imgembs_before.append(None)
        else:
            speaker_segment_before = speaker_annots[spk_idx_before]
            speaker_start_frame = round(speaker_segment_before['start'] * fps)
            speaker_end_frame = round(speaker_segment_before['end'] * fps)
            # store image embeddings of the speaker
            imgembs_before.append(imgemb_data[speaker_start_frame:speaker_end_frame])

        # SPEAKER AFTER
        # if the current speaker is the last speaker in the annotations, the speaker after does not exist
        if spk_idx_after >= annotated_speakers_count:
            imgembs_after.append(None)
        else:
            speaker_segment_after = speaker_annots[spk_idx_after]
            speaker_start_frame = round(speaker_segment_after['start'] * fps)
            speaker_end_frame = round(speaker_segment_after['end'] * fps)
            # store image embeddings of the speaker
            imgembs_after.append(imgemb_data[speaker_start_frame:speaker_end_frame])


    # SPEAKER BEFORE aggregation
    speaker_before_img_similarity = []
    # this loop will be iterated as often as the context size
    for speaker_before_embeddings in imgembs_before:
        if speaker_before_embeddings is None:
            speaker_before_img_similarity.append(0)
            continue
        
        cos_sim = cosine_similary(imgembs_segment, speaker_before_embeddings)
        cos_sim_flat = cos_sim.flatten()
        if len(cos_sim_flat):
            similarity_quantile = np.quantile(cos_sim_flat, 0.8)
        else:
            print('here')
            similarity_quantile = 0
        speaker_before_img_similarity.append(similarity_quantile)
    
    # SPEAKER AFTER aggregation
    speaker_after_img_similarity = []
    for speaker_after_embeddings in imgembs_after:
        if speaker_after_embeddings is None:
            speaker_after_img_similarity.append(0)
            continue

        cos_sim = cosine_similary(imgembs_segment, speaker_after_embeddings)
        cos_sim_flat = cos_sim.flatten()
        if len(cos_sim_flat):
            similarity_quantile = np.quantile(cos_sim_flat, 0.8)
        else:
            print('here')
            similarity_quantile = 0
        speaker_after_img_similarity.append(similarity_quantile)

    # the amount of image similarity features is determined by (2 * context_size). One similarity score for each speaker in the context.
    img_similarity_features = speaker_before_img_similarity + speaker_after_img_similarity

    return img_similarity_features


def process_sentemb_data(sentemb_data, segment_start, segment_end, spk_index, speaker_annots):
    """
    Helper to process sentence embeddings.
    All sentence embeddings of speaker segments in the context are collected and compared to the reference speaker.
    Comparing a reference speaker to the speaker before and after, the 80% quantile across all sentence similarities is returned.
    """

    # fps the embeddings have been extracted with
    fps = config['fps']
    # context size describes the amount of speakers that should be included before and after the current speaker
    context_size = config['context_size']

    sentembs_segment = None
    # find the speaker segment in the sentence embedding data that was segmented by speaker diarization
    for diarization_segment in sentemb_data:
        # overlap between actual speaker segment timeframe and the speaker diarization timeframe where sentence embeddings have been extracted from
        ovp = max(0, min(segment_end, diarization_segment['end_time']) - max(segment_start, diarization_segment['start_time'])) / (diarization_segment['end_time'] - diarization_segment['start_time'])
        # only assign the speaker diarization segment to the speaker if overlap of timeframes is > 70%
        if ovp > 0.7:
            sentembs_segment = diarization_segment['sentence_embeddings']
            break
    
    # if no speaker diarization segment matches the annotated speaker segment, return 0's in the expected feature vector size
    if sentembs_segment is None:
        return [0] * (2 * context_size)

    # number of annotated speakers
    annotated_speakers_count = len(speaker_annots)

    sentembs_before = []
    sentembs_after = []
    for i in range(1, context_size+1):
        spk_idx_before = spk_index - i
        spk_idx_after = spk_index + i

        # SPEAKER BEFORE
        # if the current speaker is the first speaker, the speaker before does not exist
        if spk_idx_before < 0:
            sentembs_before.append(None)
        else:
            speaker_segment_before = speaker_annots[spk_idx_before]
            # toggle variable to check if a diarization segment has been found
            found = False
            # find diariaztion segment for the speaker before
            for diarization_segment in sentemb_data:
                ovp = max(0, min(speaker_segment_before['end'], diarization_segment['end_time']) - max(speaker_segment_before['start'], diarization_segment['start_time'])) / (diarization_segment['end_time'] - diarization_segment['start_time'])
                if ovp > 0.7:
                    found = True
                    sentembs_before.append(diarization_segment['sentence_embeddings'])
                    break
            if not found:
                sentembs_before.append(None)

        # SPEAKER AFTER
        # if the current speaker is the last speaker in the annotations, the speaker after does not exist
        if spk_idx_after >= annotated_speakers_count:
            sentembs_after.append(None)
        else:
            speaker_segment_after = speaker_annots[spk_idx_after]
            # toggle variable to check if a diarization segment has been found
            found = False
            # find diariaztion segment for the speaker after
            for diarization_segment in sentemb_data:
                ovp = max(0, min(speaker_segment_after['end'], diarization_segment['end_time']) - max(speaker_segment_after['start'], diarization_segment['start_time'])) / (diarization_segment['end_time'] - diarization_segment['start_time'])
                if ovp > 0.7:
                    found = True
                    sentembs_after.append(diarization_segment['sentence_embeddings'])
                    break
            if not found:
                sentembs_after.append(None)

    # SPEAKER BEFORE aggregation
    speaker_before_sent_similarity = []
    # this loop will be iterated as often as the context size
    for speaker_before_embeddings in sentembs_before:
        if speaker_before_embeddings is None:
            speaker_before_sent_similarity.append(0)
            continue
        
        cos_sim = cosine_similary(sentembs_segment, speaker_before_embeddings)
        cos_sim_flat = cos_sim.flatten()
        similarity_max = torch.max(cos_sim_flat).item()
        similarity_avg = torch.mean(cos_sim_flat)
        similarity_quantile = np.quantile(cos_sim_flat, 0.8)
        speaker_before_sent_similarity.append(similarity_max)

    # SPEAKER AFTER aggregation
    speaker_after_sent_similarity = []
    # this loop will be iterated as often as the context size
    for speaker_after_embeddings in sentembs_after:
        if speaker_after_embeddings is None:
            speaker_after_sent_similarity.append(0)
            continue
        
        cos_sim = cosine_similary(sentembs_segment, speaker_after_embeddings)
        cos_sim_flat = cos_sim.flatten()
        similarity_max = torch.max(cos_sim_flat).item()
        similarity_avg = torch.mean(cos_sim_flat)
        similarity_quantile = np.quantile(cos_sim_flat, 0.8)
        speaker_after_sent_similarity.append(similarity_max)
    
    # the amount of sentence similarity features is determined by (2 * context_size). One similarity score for each speaker in the context.
    sent_similarity_features = speaker_before_sent_similarity + speaker_after_sent_similarity

    return sent_similarity_features

""" END HELPERS """


""" START TRAINING DATA CREATION """

def data_speaker_segments():
    """
    Load all .pkls and create trainable data for Random Forest and XGBoost classifiers.

    The result will be a .pkl file with all training samples. 
    Each sample is represented by feature vector containing features that were aggregated during one speaker turn.
    """

    out_dir = f'{config["trainingdata_dir"]}/speaker/segment_based/hierarchy_{hierarchy_level}'
    os.makedirs(out_dir, exist_ok=True)
    
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

            # get video file name and full path to the directory that contains pkl files for the video
            video_fn = annotations['video_fn']
            output_path = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

            # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
            if not os.path.exists(output_path):
                print(f'{output_path} not yet processed')
                continue

            # load feature data from .pkls for the current video
            for pkl_fn in os.listdir(output_path):
                pkl_path = os.path.join(output_path, pkl_fn)

                if 'clip_v2.pkl' == pkl_fn:
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
                elif 'scenes.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        scenes_data = pickle.load(pkl)
                elif 'pos_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        pos_data = pickle.load(pkl)
                elif 'image_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        imgemb_data = pickle.load(pkl)
                elif 'sentence_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sentemb_data = pickle.load(pkl)
                elif 'ner_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        ner_data = pickle.load(pkl)

            # iterate over every speaker segment in the annotations
            for spk_idx, speaker_segment in enumerate(annotations['speaker']['segments']):
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
                groundtruth_numerical = config['groundtruth_numerical_speaker'][hierarchical_groundtruth]


                """ START FEATURES """
                # feature vector (sample) generation by processing all pkl outputs and aggregating features
                vector = []

                # CLIP
                clip_features = process_clip_data(clip_data, speaker_start, speaker_end)
                # concat feature vector of every domain
                for clip_feature_vec in clip_features:
                    vector.extend(clip_feature_vec)

                # Shot Density
                avg_shotdensity = process_density_data(sd_data, speaker_start, speaker_end)
                vector.append(avg_shotdensity)
                        
                # Length of speech
                los = speaker_end - speaker_start
                vector.append(los)

                # Sentiment
                avg_sentiments = process_sentiment_data(sentiment_data, speaker_start, speaker_end)
                vector.extend(avg_sentiments)

                # Number of Scenes
                scene_count = process_scenes_data(scenes_data, speaker_start, speaker_end)
                vector.append(scene_count)

                # POS tags
                pos_vector = process_pos_data(pos_data, speaker_start, speaker_end)
                vector.extend(pos_vector)

                # NER tags
                ner_vector = process_ner_data(ner_data, speaker_start, speaker_end)
                vector.extend(ner_vector)

                # Image Similarity from Speaker Context
                visual_similarity = process_imgemb_data(imgemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                vector.extend(visual_similarity)

                # Textual Similartiy from Speaker Context
                textual_similartiy = process_sentemb_data(sentemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                vector.extend(textual_similartiy)
                """ END FEATURES """


                # add numerical ground truth label at the end of the feature vector
                vector.append(groundtruth_numerical)

                """
                # get data samples for a certain class
                if groundtruth_numerical == 4:
                    print(annotation_fn, video_fn)
                    print(speaker_start, speaker_end, speaker_groundtruth)
                    print(vector)
                    print()
                """

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
    print(f'{total_annotated_speaker_turns} annotated speaker turns with a total duration of {total_annotated_duration_formatted}')
    print(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")')


def data_speaker_windows():
    """
    Creates training data for news situations based on sliding windows.

    Sliding windows in different lengths (e.g. 5s, 10s, 20s, 40s) are placed over the video from start to 
    end every 2.5s. Features are aggregated window-wise and concatenated afterwards. A prediction is made
    every 2.5s based on all sliding windows.
    """
    out_dir = f'{config["trainingdata_dir"]}/speaker/window_based/hierarchy_{hierarchy_level}'
    os.makedirs(out_dir, exist_ok=True)

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

            # get video file name and full path to the directory that contains pkl files for the video
            video_fn = annotations['video_fn']
            output_path = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

            # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
            if not os.path.exists(output_path):
                print(f'{output_path} not yet processed')
                continue

            # load feature data from .pkls for the current video
            for pkl_fn in os.listdir(output_path):
                pkl_path = os.path.join(output_path, pkl_fn)

                if 'clip_v2.pkl' == pkl_fn:
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
                elif 'scenes.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        scenes_data = pickle.load(pkl)
                elif 'pos_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        pos_data = pickle.load(pkl)
                elif 'image_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        imgemb_data = pickle.load(pkl)
                elif 'sentence_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sentemb_data = pickle.load(pkl)
                elif 'ner_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        ner_data = pickle.load(pkl)

            # iterate over every speaker segment in the annotations
            for spk_idx, speaker_segment in enumerate(annotations['speaker']['segments']):
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
                groundtruth_numerical = config['groundtruth_numerical_speaker'][hierarchical_groundtruth]

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
                    while window_start < speaker_end_fps:
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

                        # Sentiment
                        avg_sentiments = process_sentiment_data(sentiment_data, window_start, window_end)
                        feature_dict['sentiments'] = avg_sentiments

                        # Number of Scenes
                        scene_count = process_scenes_data(scenes_data, window_start, window_end)
                        feature_dict['scenes'] = scene_count
                        """ END FEATURES """

                        window_features.append(feature_dict)


                    clip_features = []
                    shotdensities = []
                    sentiments = []
                    scene_counts = []
                    # collect same features across all windows of the same size to process (aggregate) them afterwards
                    for vector in window_features:
                        clip_features.append(vector['clip'])
                        shotdensities.append(vector['shot_density'])
                        sentiments.append(vector['sentiments'])
                        scene_counts.append(vector['scenes'])

                    
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

                    # SENTIMENTS: take 80% quantile across all collected sentiment probabilities of the same window size
                    # zip the sentiments so we get one array for each sentiment type
                    probs_pos, probs_neg, probs_neu = zip(*sentiments)
                    quantile_pos = np.quantile(probs_pos, 0.8)
                    quantile_neg = np.quantile(probs_neg, 0.8)
                    quantile_neu = np.quantile(probs_neu, 0.8)
                    segment_features.extend([quantile_pos, quantile_neg, quantile_neu])

                    # SCENES: avg scene count across all windows of the same size
                    avg_scenes = sum(scene_counts) / len(scene_counts)
                    segment_features.append(avg_scenes)

                """ General features independent of window size """
                # Length of speech
                los = speaker_end - speaker_start
                segment_features.append(los)

                # POS tags
                pos_vector = process_pos_data(pos_data, speaker_start, speaker_end)
                segment_features.extend(pos_vector)

                # NER tags
                ner_vector = process_ner_data(ner_data, speaker_start, speaker_end)
                segment_features.extend(ner_vector)

                # Image Similarity from Speaker Context
                visual_similarity = process_imgemb_data(imgemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                segment_features.extend(visual_similarity)

                # Textual Similartiy from Speaker Context
                textual_similartiy = process_sentemb_data(sentemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                segment_features.extend(textual_similartiy)


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
    print(f'{total_annotated_speaker_turns} annotated speaker turns with a total duration of {total_annotated_duration_formatted}')
    print(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")')
                    

def data_situations_segments():
    out_dir = f'{config["trainingdata_dir"]}/situations/segment_based/'
    os.makedirs(out_dir, exist_ok=True)

    # the following initializations are made to collect data about the whole dataset
    total_annotated_duration = 0
    total_annotated_situations = 0
    total_annotations_not_in_vocabulary = 0
    annotation_count_for_situation = {}
    for situation in config['labels_situations']:
        annotation_count_for_situation[situation] = 0

    # split the annotations according to news sources
    dirlist = os.listdir(config['annotations'])
    for abbrev in config['source_abbreviations']:
        news_source_annotations = [annotation_fn for annotation_fn in dirlist if abbrev in annotation_fn]

        # annotation statistics on news source level
        source_annotated_duration = 0
        source_annotated_situations = 0
        source_annotations_not_in_vocabulary = 0
        source_annotation_count_for_situation = {}
        for situation in config['labels_situations']:
            source_annotation_count_for_situation[situation] = 0

        feature_vectors = []
        # iterate over all annotations for the current news source
        for annotation_fn in news_source_annotations:
            if not annotation_fn.endswith('.json'):
                continue

            annotation_path = os.path.join(config['annotations'], annotation_fn)
            # load annotation json
            with open(annotation_path) as f:
                annotations = json.load(f)
            
            # skip if no news situations were annotated
            if 'captions: talking' not in annotations.keys():
                continue

            # get video file name and full path to the directory that contains pkl files for the video
            video_fn = annotations['video_fn']
            output_path = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

            # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
            if not os.path.exists(output_path):
                print(f'{output_path} not yet processed')
                continue

            # load feature data from .pkls for the current video
            for pkl_fn in os.listdir(output_path):
                pkl_path = os.path.join(output_path, pkl_fn)

                if 'clip_v2.pkl' == pkl_fn:
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
                elif 'scenes.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        scenes_data = pickle.load(pkl)
                elif 'pos_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        pos_data = pickle.load(pkl)
                elif 'image_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        imgemb_data = pickle.load(pkl)
                elif 'sentence_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sentemb_data = pickle.load(pkl)
                elif 'ner_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        ner_data = pickle.load(pkl)

            # iterate over every speaker segment in the annotations
            for spk_idx, speaker_segment in enumerate(annotations['speaker']['segments']):
                speaker_start = speaker_segment['start']
                speaker_end = speaker_segment['end']
                speaker_groundtruth = speaker_segment['label']

                # skip annotations that are not in our defined speaker label vocabulary
                if speaker_groundtruth not in speaker_hierarchy.keys():
                    continue

                situation_groundtruth = get_current_situation(speaker_start, speaker_end, annotations)
                
                if situation_groundtruth == None:
                    total_annotations_not_in_vocabulary += 1
                    source_annotations_not_in_vocabulary += 1
                    continue

                # encode ground truth as a number
                groundtruth_numerical = config['groundtruth_numerical_situations'][situation_groundtruth]


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

                # Number of Scenes
                scene_count = process_scenes_data(scenes_data, speaker_start, speaker_end)
                vector.append(scene_count)

                # POS Tags
                pos_vector = process_pos_data(pos_data, speaker_start, speaker_end)
                vector.extend(pos_vector)

                # NER Tags
                ner_vector = process_ner_data(ner_data, speaker_start, speaker_end)
                vector.extend(ner_vector)

                # Image Similarity from Speaker Context
                visual_similarity = process_imgemb_data(imgemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                vector.extend(visual_similarity)

                # Textual Similartiy from Speaker Context
                textual_similartiy = process_sentemb_data(sentemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                vector.extend(textual_similartiy)
                """ END FEATURES """


                # add numerical ground truth label at the end of the feature vector
                vector.append(groundtruth_numerical)

                """
                if groundtruth_numerical == 3:
                    print(annotation_fn, video_fn)
                    print(speaker_start, speaker_end, speaker_groundtruth)
                    print(vector)
                    print()
                """
                

                # add sample
                feature_vectors.append(vector)

                # annotation statistics
                total_annotated_duration += speaker_end - speaker_start
                source_annotated_duration += speaker_end - speaker_start
                total_annotated_situations +=1
                source_annotated_situations += 1
                annotation_count_for_situation[situation_groundtruth] += 1
                source_annotation_count_for_situation[situation_groundtruth] += 1

        # convert feature vectors list to a 2-dim numpy array with the shape (samples, feature_amount)
        feature_vectors = np.vstack(feature_vectors)

        # store feature samples as pkl
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_situations)}_samples_trainingdata.pkl', 'wb') as pkl:
            pickle.dump(feature_vectors, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        # store annotation statistics about the news source dataset
        annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(source_annotated_duration))
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_situations)}_samples_statistics.txt', 'w') as txt:
                txt.write(f'{source_annotated_situations} annotated news situations with a total duration of {annotated_duration_formatted}\n')
                txt.write(f'{source_annotations_not_in_vocabulary} news situations were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
                txt.write(f'Annotation counts for each news situation:\n')
                for speaker in source_annotation_count_for_situation.keys():
                    txt.write(f'\t{speaker}: {source_annotation_count_for_situation[speaker]}\n')
        
    # store annotation statistics about the whole dataset as txt along the trainingdata
    total_annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(total_annotated_duration))
    with open(f'{out_dir}/all_{str(total_annotated_situations)}_samples_statistics.txt', 'w') as txt:
            txt.write(f'{total_annotated_situations} annotated speaker turns with a total duration of {total_annotated_duration_formatted}\n')
            txt.write(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
            txt.write(f'Annotation counts for each speaker role:\n')
            for speaker in annotation_count_for_situation.keys():
                txt.write(f'\t{speaker}: {annotation_count_for_situation[speaker]}\n')
            print(f"Statistics about the annotations can be found in {txt.name}")
              

    # annotation files statistics stdout
    print(f'{total_annotated_situations} annotated speaker turns with a total duration of {total_annotated_duration_formatted}')
    print(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")')


def data_situations_windows():
    out_dir = f'{config["trainingdata_dir"]}/situations/window_based/'
    os.makedirs(out_dir, exist_ok=True)

    # the following initializations are made to collect data about the whole dataset
    total_annotated_duration = 0
    total_annotated_situations = 0
    total_annotations_not_in_vocabulary = 0
    annotation_count_for_situation = {}
    for situation in config['labels_situations']:
        annotation_count_for_situation[situation] = 0

    # split the annotations according to news sources
    dirlist = os.listdir(config['annotations'])
    for abbrev in config['source_abbreviations']:
        news_source_annotations = [annotation_fn for annotation_fn in dirlist if abbrev in annotation_fn]

        # annotation statistics on news source level
        source_annotated_duration = 0
        source_annotated_situations = 0
        source_annotations_not_in_vocabulary = 0
        source_annotation_count_for_situation = {}
        for situation in config['labels_situations']:
            source_annotation_count_for_situation[situation] = 0

        feature_vectors = []
        # iterate over all annotations for the current news source
        for annotation_fn in news_source_annotations:
            if not annotation_fn.endswith('.json'):
                continue

            annotation_path = os.path.join(config['annotations'], annotation_fn)
            # load annotation json
            with open(annotation_path) as f:
                annotations = json.load(f)

            # skip if no news situations were annotated
            if 'captions: talking' not in annotations.keys():
                continue

            # get video file name and full path to the directory that contains pkl files for the video
            video_fn = annotations['video_fn']
            output_path = os.path.join(config['output_dir'], video_fn.replace('.mp4', ''))

            # this is only relevant if the sbatch process is still running and the features have not yet been fully extracted yet
            if not os.path.exists(output_path):
                print(f'{output_path} not yet processed')
                continue

            # load feature data from .pkls for the current video
            for pkl_fn in os.listdir(output_path):
                pkl_path = os.path.join(output_path, pkl_fn)

                if 'clip_v2.pkl' == pkl_fn:
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
                elif 'scenes.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        scenes_data = pickle.load(pkl)
                elif 'pos_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        pos_data = pickle.load(pkl)
                elif 'image_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        imgemb_data = pickle.load(pkl)
                elif 'sentence_embedding.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        sentemb_data = pickle.load(pkl)
                elif 'ner_tags.pkl' == pkl_fn:
                    with open(pkl_path, 'rb') as pkl:
                        ner_data = pickle.load(pkl)

            # iterate over every speaker segment in the annotations
            for spk_idx, speaker_segment in enumerate(annotations['speaker']['segments']):
                speaker_start = speaker_segment['start']
                speaker_end = speaker_segment['end']
                speaker_groundtruth = speaker_segment['label']

                # skip annotations that are not in our defined speaker label vocabulary
                if speaker_groundtruth not in speaker_hierarchy.keys():
                    continue

                situation_groundtruth = get_current_situation(speaker_start, speaker_end, annotations)

                if situation_groundtruth == None:
                    total_annotations_not_in_vocabulary += 1
                    source_annotations_not_in_vocabulary += 1
                    continue

                # encode ground truth as a number
                groundtruth_numerical = config['groundtruth_numerical_situations'][situation_groundtruth]

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
                    while window_start < speaker_end_fps:
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

                        # Sentiment
                        avg_sentiments = process_sentiment_data(sentiment_data, window_start, window_end)
                        feature_dict['sentiments'] = avg_sentiments

                        # Number of Scenes
                        scene_count = process_scenes_data(scenes_data, window_start, window_end)
                        feature_dict['scenes'] = scene_count
                        """ END FEATURES """

                        window_features.append(feature_dict)


                    clip_features = []
                    shotdensities = []
                    sentiments = []
                    scene_counts = []
                    # collect same features across all windows of the same size to process (aggregate) them afterwards
                    for vector in window_features:
                        clip_features.append(vector['clip'])
                        shotdensities.append(vector['shot_density'])
                        sentiments.append(vector['sentiments'])
                        scene_counts.append(vector['scenes'])

                    
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

                    # SENTIMENTS: take 80% quantile across all collected sentiment probabilities of the same window size
                    probs_pos, probs_neg, probs_neu = zip(*sentiments)
                    quantile_pos = np.quantile(probs_pos, 0.8)
                    quantile_neg = np.quantile(probs_neg, 0.8)
                    quantile_neu = np.quantile(probs_neu, 0.8)
                    segment_features.extend([quantile_pos, quantile_neg, quantile_neu])

                    # SCENES: avg scene count across all windows of the same size
                    avg_scenes = sum(scene_counts) / len(scene_counts)
                    segment_features.append(avg_scenes)

                """ General features independent of window size """
                # Length of speech
                los = speaker_end - speaker_start
                segment_features.append(los)

                # POS tags
                pos_vector = process_pos_data(pos_data, speaker_start, speaker_end)
                segment_features.extend(pos_vector)

                # NER tags 
                ner_vector = process_ner_data(ner_data, speaker_start, speaker_end)
                segment_features.extend(ner_vector)

                # Image Similarity from Speaker Context
                visual_similarity = process_imgemb_data(imgemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                segment_features.extend(visual_similarity)

                # Textual Similartiy from Speaker Context
                textual_similartiy = process_sentemb_data(sentemb_data, speaker_start, speaker_end, spk_idx, annotations['speaker']['segments'])
                segment_features.extend(textual_similartiy)
                """     """

                # add numerical ground truth label at the end of the feature vector
                segment_features.append(groundtruth_numerical)

                # add sample
                feature_vectors.append(segment_features)

                # annotation statistics
                total_annotated_duration += speaker_end - speaker_start
                source_annotated_duration += speaker_end - speaker_start
                total_annotated_situations +=1
                source_annotated_situations += 1
                annotation_count_for_situation[situation_groundtruth] += 1
                source_annotation_count_for_situation[situation_groundtruth] += 1

        # convert feature vectors list to a 2-dim numpy array with the shape (samples, feature_amount)
        feature_vectors = np.vstack(feature_vectors)

        
        # store feature samples as pkl
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_situations)}_samples_trainingdata.pkl', 'wb') as pkl:
            pickle.dump(feature_vectors, pkl, protocol=pickle.HIGHEST_PROTOCOL)

        # store annotation statistics about the news source dataset
        annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(source_annotated_duration))
        with open(f'{out_dir}/{abbrev}_{str(source_annotated_situations)}_samples_statistics.txt', 'w') as txt:
                txt.write(f'{source_annotated_situations} annotated news situations with a total duration of {annotated_duration_formatted}\n')
                txt.write(f'{source_annotations_not_in_vocabulary} news situations were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
                txt.write(f'Annotation counts for each news situation:\n')
                for speaker in source_annotation_count_for_situation.keys():
                    txt.write(f'\t{speaker}: {source_annotation_count_for_situation[speaker]}\n')
        
    # store annotation statistics about the whole dataset as txt along the trainingdata
    total_annotated_duration_formatted = time.strftime('%H:%M:%S', time.gmtime(total_annotated_duration))
    with open(f'{out_dir}/all_{str(total_annotated_situations)}_samples_statistics.txt', 'w') as txt:
            txt.write(f'{total_annotated_situations} annotated speaker turns with a total duration of {total_annotated_duration_formatted}\n')
            txt.write(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")\n\n')
            txt.write(f'Annotation counts for each speaker role:\n')
            for speaker in annotation_count_for_situation.keys():
                txt.write(f'\t{speaker}: {annotation_count_for_situation[speaker]}\n')
            print(f"Statistics about the annotations can be found in {txt.name}")
              

    # annotation files statistics stdout
    print(f'{total_annotated_situations} annotated speaker turns with a total duration of {total_annotated_duration_formatted}')
    print(f'{total_annotations_not_in_vocabulary} speaker turns were dismissed because their label is not our defined vocabulary (e.g. "", "expert, medicine")')

""" END TRAINING DATA CREATION """



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility methods')

    parser.add_argument('--speaker', action='store_true', help="Create training data for speaker role recognition")
    parser.add_argument('--situations', action='store_true', help="Create training data for news situation recognition")

    parser.add_argument('--seg', action='store_true', help="Create training data based on speaker segments")
    parser.add_argument('--sw', action='store_true', help="Create training data based on sliding windows")
    parser.add_argument('--hierarchy', action='store', type=int, choices=[0, 1], default=0, help="Speaker mapping hierarchy level")

    args = parser.parse_args()

    start_time = time.time()

    hierarchy_level = args.hierarchy
    speaker_hierarchy = config[f'speaker_hierarchy_mapping_{hierarchy_level}']

    if args.speaker and args.seg:
        data_speaker_segments()
    
    if args.speaker and args.sw:
        data_speaker_windows()

    if args.situations and args.seg:
        data_situations_segments()

    if args.situations and args.sw:
        data_situations_windows()

    # processing time
    end_time = time.time()
    duration = end_time - start_time
    td = datetime.timedelta(seconds=duration)
    duration_formatted = str(td).split('.')[0]
    print(f'Processing time: {duration_formatted}')