# Overall
from video_decoder import VideoDecoder
import os 
from pathlib import Path
import numpy as np
import torch
import argparse
import pickle

# ASR & Speaker Diarization
import whisper
import datetime
import pandas as pd
import time
from sklearn.cluster import AgglomerativeClustering
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from gpuinfo import GPUInfo
import wave
import contextlib
import psutil
from moviepy.editor import VideoFileClip
import tempfile

import scipy.cluster.hierarchy as sc
import matplotlib.pyplot as plt

# Shot Boundary Detection
#from transnetv2 import TransNetV2
from pathlib import Path

# Places
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

# Clip
import clip
import json
import copy


config = {
    "fps": 2
}



def automatic_speech_recognition(video_path, model_type):
    model = whisper.load_model(model_type)
    result = model.transcribe(video_path, language="de")
    print(result['segments'])



def speaker_diarization(video_path, selected_source_lang, whisper_model, output_dir, pkl_dir):
    """
    #
    1. Using Open AI's Whisper model to seperate audio into segments and generate transcripts.
    2. Generating speaker embeddings for each segments.
    3. Applying agglomerative clustering on the embeddings to identify the speaker for each segment.
    
    Speech Recognition is based on models from OpenAI Whisper https://github.com/openai/whisper
    Speaker embedding model and pipeline from pyannote https://github.com/pyannote/pyannote-audio
    """
    # directory where the diarization output will be stored
    save_directory = output_dir + '/SPEAKER_DIARIZATION/'
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(pkl_dir + '/SPEAKER_DIARIZATION', exist_ok=True)

    embedding_model = PretrainedSpeakerEmbedding( 
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def convert_time(secs):
        return datetime.timedelta(seconds=round(secs))

        
    model = whisper.load_model(whisper_model)
    time_start = time.time()

    try:
        # OPTION 1: Convert video to .wav using ffmpeg
        #audio_file = "/nfs/home/arafatj/wavs/" + os.path.basename(video_path.replace('.mp4', '.wav'))
        audio_file = "/tmp/" + os.path.basename(video_path.replace('.mp4', '.wav'))
        os.system(f'ffmpeg -i "{video_path}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}" -loglevel error')
        print(f'conversion to .wav done, file: {audio_file}')
            
        # Get duration
        with contextlib.closing(wave.open(audio_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
        print(f"duration of audio file: {duration}")
        
        """"
        # OPTION 2 : Convert to .wav using VideoFileClip
        audio_file = tempfile.NamedTemporaryFile(suffix=".wav")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_file.name)
        duration = video.duration
        print(f"duration of audio file: {duration}")
        """

        # Transcribe audio
        pkl_file = pkl_dir + '/SPEAKER_DIARIZATION/' + os.path.basename(video_path.replace('.mp4', '.pkl'))
        segments = []
        if os.path.exists(pkl_file):
            print(f"Found pkl: {pkl_file}, skip whisper ASR")
            # read pkl file
            with open(pkl_file, 'rb') as pkl:
                segments = pickle.load(pkl)
        else:
            print('start whisper asr...')
            options = dict(language=selected_source_lang, beam_size=5, best_of=5)
            transcribe_options = dict(task="transcribe", **options)
            result = model.transcribe(audio_file, **transcribe_options)
            segments = result['segments']
            # store asr output in pkl
            with open(pkl_file, 'wb') as pkl:
                pickle.dump(segments, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            print('done with whisper')
    except Exception as e:
        # delete tmp file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        raise RuntimeError("Error converting video to audio: ", e)

    try:
        # Create embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(audio_file, clip)
            return embedding_model(waveform[None])

        print('embedding calculation start...')
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)
        print(f'embedding calculation done, embedding shape: {embeddings.shape}')
        # delete tmp file
        os.remove(audio_file)


        # plot dendrogram (DEBUGGING ONLY)
        dist = sc.distance.pdist(embeddings)
        Z = sc.linkage(dist, method='ward')
        fig = plt.figure(figsize=(10,10))
        dn = sc.dendrogram(Z, labels=list(range(len(embeddings))), leaf_font_size=12)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.axhline(y = 640, color = 'r', linestyle = '-')
        plt.savefig('/nfs/home/arafatj/master_project/dendrograms/' + os.path.basename(video_path.replace('.mp4', '.png')))

        # Assign speaker label
        # TODO: FIND optimal distance_threshold
        #clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=550.0).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        # Make output
        objects = {
            'Start' : [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        
        time_end = time.time()
        time_diff = time_end - time_start
        memory = psutil.virtual_memory()
        gpu_utilization, gpu_memory = GPUInfo.gpu_usage()
        gpu_utilization = gpu_utilization[0] if len(gpu_utilization) > 0 else 0
        gpu_memory = gpu_memory[0] if len(gpu_memory) > 0 else 0
    
        system_info = f"""
        *Memory: {memory.total / (1024 * 1024 * 1024):.2f}GB, used: {memory.percent}%, available: {memory.available / (1024 * 1024 * 1024):.2f}GB.* 
        *Processing time: {time_diff:.5} seconds.*
        *GPU Utilization: {gpu_utilization}%, GPU Memory: {gpu_memory}MiB.*
        """

        save_path = save_directory + os.path.basename(video_path.replace('.mp4', '.csv'))
        df_results = pd.DataFrame(objects)
        df_results.to_csv(save_path)
        print(f"Sucessfully applied speaker diarization! Transcript: {save_path}")
        return df_results, system_info, save_path
    
    except Exception as e:
        # delete tmp file
        if os.path.exists(audio_file):
            os.remove(audio_file)
        raise RuntimeError("Error Running inference with local model", e) 



def shot_boundary_detection(video_path, output_dir, pkl_dir):
    # suppress tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from transnetv2 import TransNetV2

    save_directory = output_dir + '/SHOT_BOUNDARY_DETECTION/'
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(pkl_dir + '/SHOT_BOUNDARY_DETECTION', exist_ok=True)
    pkl_file = pkl_dir + '/SHOT_BOUNDARY_DETECTION/' + os.path.basename(video_path.replace('.mp4', '.pkl'))

    if os.path.exists(pkl_file):
        print(f"Found pkl: {pkl_file}, skip TransNet V2 shot boundary detection")
        return

    model = TransNetV2()
    output_data = None
    video_decoder = VideoDecoder(video_path, fps=config["fps"], max_dimension=[27,48])
    # store all video frames to pass them to the TransNet model
    frames = []
    for frame in video_decoder:
        frames.append(frame.get('frame'))
    frames = np.stack(frames, axis=0)
    # reshaping to TransNet requirements
    video = frames.reshape([-1, 27, 48, 3])
    single_frame_predictions, all_frame_predictions = model.predict_frames(video)
    # returns a list of scenes, each defined as a tuple of (start frame, end frame)
    shot_list = model.predictions_to_scenes(single_frame_predictions)
    # create the output data by converting the shot_list to a list of tuples in the form of (start_frame_time, end_frame_time)
    output_data = [(x[0].item() / video_decoder.fps(), x[1].item() / video_decoder.fps()) for x in shot_list]
    with open(pkl_file, 'wb') as pkl:
            pickle.dump(output_data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Successfully applied TransNet V2! Result: {pkl.name}')

    """ DEPRECATED
    # FIRST VERSION
    video_frames, single_frame_predictions, all_frame_predictions = model.predict_video(video_path)
    # returns a list of scenes, each defined as a tuple of (start frame, end frame)
    scene_predictions = model.predictions_to_scenes(single_frame_predictions)
    # store pkl
    data_dict = {'video_frames': video_frames, 'single_frame_predictions': single_frame_predictions, 'all_frame_predictions': all_frame_predictions, 'scene_predictions': scene_predictions}
    with open(pkl_file, 'wb') as pkl:
            pickle.dump(data_dict, pkl, protocol=pickle.HIGHEST_PROTOCOL)

            
    # SAVE ONE IMAGE FOR EACH SCENE
    # if only certain frame span is wanted we can slice the video_frames array to get only wanted frames in the slicing range (e.g. video_frames[1000:2000])
    img = model.visualize_predictions(video_frames, predictions=(single_frame_predictions, all_frame_predictions))
    img.save(save_directory + Path(video_path).stem + '.png')

    i = 1
    print('extracting scene images...')
    for scene in scene_predictions:
        # use the middle frame between start and end frames as the representative shot for the scene
        middle_frame = int((scene[0] + scene[1]) / 2)
        fname = save_directory + 'SCENE' + str(i) + '.png'
        os.system(f'ffmpeg -i "{video_path}" -vf  "select=eq(n\,{middle_frame})" -vframes 1 "{fname}" -loglevel error')
        i += 1
    print('done!')
    """



def places(video_path, pkl_dir):
    arch = 'resnet50'
    # TODO: Pass path to the model as an argument and store model somewhere else
    model_file = f"/nfs/home/arafatj/master_project/models/Places365/{arch}_places365.pth.tar"
    
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # load the image transformer
    centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    file_name = '/nfs/home/arafatj/master_project/models/Places365/categories_places365.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # use video_decoder to apply places365 model on the whole video with a certain fps
    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    for i, frame in enumerate(video_decoder):
        img = Image.fromarray(np.uint8(frame.get('frame')))
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        print(f"{arch} prediction on {img}. Time: {frame.get('time')}")
        # output the prediction
        for i in range(0, 5):
            print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))



def predict_CLIP_queries(video_path, output_dir, pkl_dir):
    """
    This function will predict probabilities given different classes.
    A .json file is required, which includes different domains and their classes (/nfs/home/arafatj/master_project/models/CLIP/CLIP_queries.json)
    The function stores the result in a .pkl file on the system. 

    The result is a dictionary, containing the domains as keys. Each dictionary entry contains an array which represents the
    probability distribution over the classes inside the given domain at a specific frame. 
    For example, result["news roles"][2] returns the probability distribution for the classes in the "news roles" domain
    at the third frame (which is described by the index [2]). Having the frame index, the specific time the frame occurs at
    can easily be calculated by index/framerate (here we are using framerate=2)
    """

    pkl_file = pkl_dir + '/CLIP/' + os.path.basename(video_path.replace('.mp4', '.pkl'))
    save_directory = output_dir + '/CLIP/'
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(pkl_dir + '/CLIP', exist_ok=True)
    np.set_printoptions(suppress=True) # do not print floats in scientific notation (e^...)

    if os.path.exists(pkl_file):
        print(f"Found pkl: {pkl_file}, skip CLIP modeling")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # clip queries json
    with open('/nfs/home/arafatj/master_project/models/CLIP/CLIP_queries.json') as f:
        queries = json.load(f)

    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    duration = video_decoder.duration()

    # create one dictionary entry for each domain in the CLIP_queries.json; the entry will contain the resulting matrix.
    result_matrices = {}
    for domain in queries.keys():
        result_matrices[domain] = []
    
    for i, frame in enumerate(video_decoder):
        print(f'\rProcessing frames: {i}/{int(duration * config["fps"])}', end="")
        image = preprocess(Image.fromarray(np.uint8(frame.get('frame')))).unsqueeze(0).to(device)
    
        # domains such as 'news roles', 'news situations' and 'places'
        for domain in queries.keys():
            max_queries = []
            # single labels like 'anchor', 'interview' etc.
            for label in queries[domain].keys():
                text = clip.tokenize(queries[domain][label]).to(device)

                with torch.no_grad():
                    image_features = model.encode_image(image)
                    text_features = model.encode_text(text)
                    logits_per_image, logits_per_text = model(image, text)

                # create a list of tuples that contains (query, class/label, similarity_score) for each query defined in the json
                logits_with_labels = list(zip(queries[domain][label], [label] * len(queries[domain][label]), logits_per_image.tolist()[0]))
                # get the most probable query for the current class/label as a tuple in the form of (query, class/label, similarity_score)
                max_query = max(logits_with_labels, key=lambda item:item[2])
                max_queries.append(max_query)
            
            # extract only the similarity scores from the max_queries list of tuples and convert them to a tensor
            similarity_scores = torch.tensor([x[2] for x in max_queries], dtype=torch.float32)
            # apply softmax on the similiratiy scores
            probs = similarity_scores.softmax(dim=-1).cpu().numpy()
            result_matrices[domain].append(probs.tolist())
    print("")

    # store result list as pkl
    with open(pkl_dir + '/CLIP/' + os.path.basename(video_path.replace('.mp4', '.pkl')), 'wb') as pkl:
        pickle.dump(result_matrices, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Successfully applied CLIP! Result: {pkl.name}')

        

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction methods')
    parser.add_argument('--video', action='store', type=os.path.abspath, required=True, help="Path to the video (.mp4) to be processed")
    parser.add_argument('--outputDir', action='store', type=os.path.abspath, help="Path to the directory where results will be stored. \
                        Subdirectories (e.g. speaker_diariazation/) will be added automatically, so only the path to the main OUTPUT/ directory is required",
                        default="../OUTPUT/") #TODO: REMOVE "DEFAULT="" AND SET "REQUIRED=TRUE" for generic usage
    parser.add_argument('--pkls', action='store', type=os.path.abspath, help="Path to the directory where the .pkl files will be stored and/or read",
                        default="../PKLs/") #TODO: REMOVE "DEFAULT="" AND SET "REQUIRED=TRUE" for generic usage
    
    parser.add_argument('--asr', action='store_true', help="Automatic Speech Recognition")
    parser.add_argument('--diarize', action='store_true', help="Speaker Diarization")
    parser.add_argument('--sbd', action='store_true', help="Shot boundary detection")
    parser.add_argument('--places', action='store_true', help="Place Recognition")
    parser.add_argument('--clip', action='store_true', help="CLIP")
    
    args = parser.parse_args()

    print('DEVICE:', "cuda" if torch.cuda.is_available() else "cpu")

    video_path = args.video
    pkl_dir = args.pkls
    output_dir = args.outputDir
    # change current working directory to the scripts directory (Windows)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    if args.asr:
        # Whisper
        automatic_speech_recognition(video_path, "base")

    if args.diarize:
        # Whisper
        if not args.pkls:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--pkls")')
            quit()
        """"
        # speaker diarization with pyannote to identify the amount of distinct speakers
        pipeline = pyannote.audio.Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token="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")
        pyannote_diarization = pipeline(audio_file)
        len(pyannote_diarization.labels())
        """
        speaker_diarization(video_path, "de", "base", output_dir, pkl_dir)

    if args.sbd:
        # TransNet V2
        if not args.pkls:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--pkls")')
            quit()
        shot_boundary_detection(video_path, output_dir, pkl_dir)

    if args.places:
        # Places365
        if not args.pkls:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--pkls")')
            quit()
        places(video_path, pkl_dir)

    if args.clip:
        # CLIP
        predict_CLIP_queries(video_path, output_dir, pkl_dir)