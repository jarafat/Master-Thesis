# Overall
from video_decoder import VideoDecoder
import os
import numpy as np
import torch
import argparse
import pickle
from PIL import Image
import time
import datetime
import yaml

with open('/nfs/home/arafatj/master_project/src/config.yaml', 'r') as file:
    config = yaml.safe_load(file)



def asr(video_path, whisper_model, output_dir):
    """
    ASR with OpenAi's Whisper
    github: https://github.com/openai/whisper
    """
    import whisper

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/asr_large-v2.pkl"

    if os.path.exists(pkl_file):
        print(f"[ASR] Found pkl: {pkl_file} , skip Whisper ASR", flush=True)
        return

    # extract transcript
    model = whisper.load_model(whisper_model)
    model.to(device)
    result = model.transcribe(video_path, language='de')

    # store in pkl
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(result, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[ASR] Successfully applied Whisper ASR! Result: {pkl.name}', flush=True)



def speaker_diarization(video_path, whisper_model, output_dir):
    """
    Speaker diarization with the use of OpenAI's Whisper ASR
    github: https://github.com/MahmoudAshraf97/whisper-diarization
    """
    # ASR & Speaker Diarization
    from moviepy.editor import VideoFileClip
    import tempfile
    import subprocess

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/speaker_diarization_large-v2.pkl"

    if os.path.exists(pkl_file):
        print(f"[Speaker Diarization] Found pkl: {pkl_file} , skip Whisper Speaker Diarization", flush=True)
        return

    # extract audio from video and store in a temporary file
    audio_file = tempfile.NamedTemporaryFile(suffix='.wav', dir=config['tmp_dir'])
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_file.name, verbose=False, logger=None)

    print("[Speaker Diarization] Processing Whisper Speaker Diarization...", flush=True)
    # execute diarization script in its dedicated conda environment (speaker_diarization) and suppress stdout outputs
    subprocess.run(f"conda run -n speaker_diarization \
                   python {config['spk_diarization_script']} -a {audio_file.name} --whisper-model {whisper_model} --no-stem --output {pkl_file}",
                   shell=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    
    # file will only be there if the diarization was successful
    if os.path.exists(pkl_file):
        print(f"[Speaker Diarization] Successfully applied Whisper Speaker Diarization! Result: {pkl_file}", flush=True)
    else:
         print(f"[Speaker Diarization] ERROR: Whisper Speaker Diarization failed!", flush=True)



def shot_boundary_detection(video_path, output_dir):
    """
    Shot Boundary Detection with TransNetV2
    github: https://github.com/soCzech/TransNetV2
    """
    # suppress tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from transnetv2 import TransNetV2

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/shot_boundary_detection.pkl"

    if os.path.exists(pkl_file):
        print(f"[Shot Boundary Detection] Found pkl: {pkl_file} , skip TransNet V2 Shot Boundary Detection", flush=True)
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
        print(f'[Shot Boundary Detection] Successfully applied TransNet V2! Result: {pkl.name}', flush=True)



def shot_density(video_path, output_dir):
    """
    Calculate shot density from previously saved shot boundary detection result .pkl

    Stores the result in a .pkl file
    """
    # Shot Density
    from sklearn.neighbors import KernelDensity
    import math

    # check if shot density was already calculated for the requested file
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/shot_density.pkl"
    if os.path.exists(pkl_file):
        print(f"[Shot Density] Found pkl: {pkl_file} , skip Shot Density calculation", flush=True)
        return

    sbd_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/shot_boundary_detection.pkl"
    if not os.path.exists(sbd_pkl):
        print("[Shot Density] No Shot Boundary Detection .pkl found! Please provide SBD results before applying Shot Density calculation", flush=True)
        return

    with open(sbd_pkl, 'rb') as pkl:
        shots_data = pickle.load(pkl)

    last_shot_end = 0
    shots = []
    # shots_data contains tuples in the form of (shot.start, shot.end)
    for shot in shots_data:
         shots.append(shot[0])

         if shot[1] > last_shot_end:
              last_shot_end = shot[1]

    time = np.linspace(0, last_shot_end, math.ceil(last_shot_end * config["fps"]) + 1)[:, np.newaxis]
    shots = np.asarray(shots).reshape(-1, 1)
    kde = KernelDensity(kernel="gaussian", bandwidth=10.0).fit(shots)
    log_dens = kde.score_samples(time)
    shot_density = np.exp(log_dens)
    shot_density = (shot_density - shot_density.min()) / (shot_density.max() - shot_density.min())

    output_data = {
        "y": shot_density.squeeze(),
        "time": time.squeeze().astype(np.float64),
        "delta_time": 1 / config["fps"]
    }
    
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(output_data, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Shot Density] Successfully calculated Shot Density! Result: {pkl.name}", flush=True)

    """
    # VISUALIZATION
    plt.plot(output_data["time"], output_data["y"])
    plt.savefig(f'/nfs/home/arafatj/master_project/graphics/shot_density/{os.path.basename(pkl_file.replace(".pkl", ".png"))}')
    """



def places(video_path, output_dir, noprint):
    """
    Places365 model
    github: https://github.com/CSAILVision/places365
    """
    # Places
    from torch.autograd import Variable as V
    import torchvision.models as models
    from torchvision import transforms as trn
    from torch.nn import functional as F
    arch = config['places365_architecture']

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/places365.pkl"

    if os.path.exists(pkl_file):
        print(f"[Places365] Found pkl: {pkl_file} , skip Places365 modeling", flush=True)
        return

    model_file = config['places365_model'].format(architecture=arch)
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

    places_categories = config['places365_categories']
    classes = []
    with open(places_categories) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # use video_decoder to apply places365 model on the whole video with a certain fps
    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    duration = video_decoder.duration()

    results = []
    if noprint:
        print('[Places365] Processing video frames...', flush=True)
    for i, frame in enumerate(video_decoder):
        if not noprint:
            print(f'[Places365] Processing video frames: {i}/{int(duration * config["fps"])}', end="\r", flush=True)

        img = Image.fromarray(np.uint8(frame.get('frame')))
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        # store top 5 predictions as tuples in the form of (category, probability)
        top5 = []
        for i in range(0, 5):
            top5.append((classes[idx[i]], probs[i].item()))
        results.append(top5)
    if not noprint:
        # remove carriage return
        print("", flush=True)
    
    # store result list as pkl
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(results, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[Places365] Successfully applied Places365! Result: {pkl.name}', flush=True)
            


def predict_CLIP_queries(video_path, output_dir, noprint):
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
    # Clip
    import clip
    import json

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/clip.pkl"

    np.set_printoptions(suppress=True) # do not print floats in scientific notation (e^...)

    if os.path.exists(pkl_file):
        print(f"[CLIP] Found pkl: {pkl_file} , skip CLIP modeling", flush=True)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    # clip queries json
    with open(config['CLIP_queries']) as f:
        queries = json.load(f)

    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    duration = video_decoder.duration()

    # create one dictionary entry for each domain in the CLIP_queries.json; the entry will contain the resulting matrix.
    result_matrices = {}
    for domain in queries.keys():
        result_matrices[domain] = []
    
    if noprint:
        print('[CLIP] Processing video frames...', flush=True)
    for i, frame in enumerate(video_decoder):
        if not noprint:
            print(f'[CLIP] Processing video frames: {i}/{int(duration * config["fps"])}', end="\r", flush=True)

        image = preprocess(Image.fromarray(np.uint8(frame.get('frame')))).unsqueeze(0).to(device)
    
        # domains such as 'news roles', 'news situations' and 'places'
        for domain in queries.keys():
            max_queries = []

            # special case: 
            # pass all places365 categories into one call to clip instead of calculating similarity score for each category separately
            if domain == 'places365':
                places365 = []
                for label in queries[domain]:
                    places365.append(queries[domain][label][0])
                
                text = clip.tokenize(places365).to(device)
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu()

                places_with_probs = list(zip(places365, probs[0].tolist()))
                # only take top5 place categories
                top_places_with_probs = sorted(places_with_probs, key=lambda item:item[1], reverse=True)[:5]
                result_matrices[domain].append(top_places_with_probs)
                continue

            # single labels like 'anchor', 'interview' etc.
            for label in queries[domain].keys():
                text = clip.tokenize(queries[domain][label]).to(device)

                with torch.no_grad():
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
            labels_with_probs = list(zip(queries[domain].keys(), probs.tolist()))
            result_matrices[domain].append(labels_with_probs)
    if not noprint:
        # remove carriage return
        print("", flush=True)

    # store result list as pkl
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(result_matrices, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CLIP] Successfully applied CLIP! Result: {pkl.name}", flush=True)



def clip_image_embeddings(video_path, output_dir, noprint):
    """
    Create image embeddings of every frame using CLIPs image encoder.
    These are later used to calculate similarities between frames.
    """
    import clip

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/image_embedding.pkl"

    if os.path.exists(pkl_file):
        """ Cosine similarity between two tensors (frames)
        with open(pkl_file, 'rb') as pkl:
            data = pickle.load(pkl)
            from sklearn.metrics.pairwise import cosine_similarity
            print(cosine_similarity(data[1].cpu().reshape(1, -1), data[2].cpu().reshape(1, -1)))
            print(cosine_similarity(data[0].cpu().reshape(1, -1), data[200].cpu().reshape(1, -1)))
        """
        print(f"[IMAGE EMBEDDINGS] Found pkl: {pkl_file} , skip Image Embedding extraction", flush=True)
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device=device)

    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    duration = video_decoder.duration()
    
    if noprint:
        print('[IMAGE EMBEDDINGS] Processing video frames...', flush=True)
    
    embeddings = []
    for i, frame in enumerate(video_decoder):
        if not noprint:
            print(f'[IMAGE EMBEDDINGS] Processing video frames: {i}/{int(duration * config["fps"])}', end="\r", flush=True)

        image = preprocess(Image.fromarray(np.uint8(frame.get('frame')))).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)

        embeddings.append(image_features)
        
    if not noprint:
        # remove carriage return
        print("", flush=True)

    # store result list as pkl
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(embeddings, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[IMAGE EMBEDDINGS] Successfully extracted Image Embeddings! Result: {pkl.name}", flush=True)



def scene_detection(video_path, output_dir):
    """
    Detect scenes with PySceneDetect and store them in a pkl.
    Scenes are stored with their corresponding start and end time.
    """
    from scenedetect import detect, ContentDetector
    
    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/scenes.pkl"

    if os.path.exists(pkl_file):
        print(f"[SCENES] Found pkl: {pkl_file} , skip Scene Detection", flush=True)
        return

    scene_list = detect(video_path, ContentDetector())
      
    # store result scenes as pkl
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(scene_list, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SCENES] Successfully applied Scene Detection! Result: {pkl.name}", flush=True)



def sentiment_analysis(video_path, output_dir):
    """
    Predict sentiments based on ASR transcript
    github: https://huggingface.co/mdraw/german-news-sentiment-bert
    """
    from germansentiment import SentimentModel

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    sentiment_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/sentiment_large-v2.pkl"
    asr_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/asr_large-v2.pkl"

    if os.path.exists(sentiment_pkl):
        print(f"[SENTIMENT] Found pkl: {sentiment_pkl} , skip Sentiment Analysis", flush=True)
        return
    elif not os.path.exists(asr_pkl):
        print("[SENTIMENT] Please provide a pkl containing an ASR transcript (--asr) before approaching Sentiment Analysis", flush=True)
        return
    
    with open(asr_pkl, 'rb') as pkl:
        asr_data = pickle.load(pkl)
    
    # store all sentences in one array
    texts = []
    for segment in asr_data['segments']:
        texts.append(segment['text'].strip())

    model = SentimentModel('mdraw/german-news-sentiment-bert')
    preds, probs = model.predict_sentiment(texts, output_probabilities=True)
    
    sentiments = []
    for i, segment in enumerate(asr_data['segments']):
        segment_data = {}
        segment_data['start'] = segment['start']
        segment_data['end'] = segment['end']
        segment_data['text'] = segment['text'].strip()
        segment_data['sentiment'] = preds[i]
        segment_data['sentiment_probs'] = probs[i]
        sentiments.append(segment_data)
    
    # store in pkl
    with open(sentiment_pkl, 'wb') as pkl:
        pickle.dump(sentiments, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'[Sentiment] Successfully applied Sentiment Analysis! Result: {pkl.name}', flush=True)

    

def topic_modeling(video_path, output_dir):
    """
    Topic modeling with BERTopic
    github: https://github.com/MaartenGr/BERTopic/

    Topic modeling feature:
    First let topic model collect topic info about all video using the asr scripts.
    The feature will have counts on how many topics a speaker talked about. These counts
    can be calculated by checking if a speaker segment contains a word in the top_n_words
    that was extracted by topic modeling.
    """
    # suppress tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    from bertopic import BERTopic
    
    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    topic_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/topics.pkl"
    asr_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/asr.pkl"
    diarization_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/speaker_diarization_large-v2.pkl"

    if os.path.exists(topic_pkl):
        print(f"[TOPIC] Found pkl: {topic_pkl} , skip Topic Modeling", flush=True)
        return
    elif not os.path.exists(diarization_pkl):
        print("[TOPIC] Please provide a pkl containing an ASR transcript (--asr) before approaching Topic Modeling", flush=True)
        return

    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    topic_model = BERTopic(language="multilingual")
    stopwords = set([word.strip() for word in open(config['stopwords']).readlines()])

    all_texts = []
    for segment in diarization_data:
        text = segment['text']
        text = " ".join([word.strip(".,?!") for word in text.split() if word.strip(".,?!").lower() not in stopwords])
        all_texts.append(text)
    
    topic_model.fit_transform(all_texts)
    print(topic_model.get_topic_freq())



def pos_tagger(video_path, output_dir):
    """
    POS-tagger for speaker segments
    """
    # suppress tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import spacy
    nlp = spacy.load("de_core_news_lg")

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    tags_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/pos_tags.pkl"
    diarization_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/speaker_diarization_large-v2.pkl"

    if os.path.exists(tags_pkl):
        print(f"[POS TAGGER] Found pkl: {tags_pkl} , skip POS Tagging", flush=True)
        return
    elif not os.path.exists(diarization_pkl):
        print("[POS TAGGER] Please provide a pkl containing Speaker Diariazation data (--diarize) before approaching POS Tagging", flush=True)
        return

    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    pos_dict = config['pos_dict']

    # SPEAKER DIARIZATION BASED
    all_pos = []
    for segment in diarization_data:
        seg_pos = []
        pos_vector = np.zeros(14)

        text = nlp(segment['text'])
        for token in text:
            seg_pos.append((token.text, token.pos_, token.tag_))
            if token.pos_ in pos_dict:
                pos_vector[pos_dict[token.pos_]] += 1
            else:
                pos_vector[pos_dict['X']] += 1

        all_pos.append({"start_time": segment['start_time'], "end_time": segment['end_time'], "vector": pos_vector})
    
    # store result as pkl
    with open(tags_pkl, 'wb') as pkl:
        pickle.dump(all_pos, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[POS TAGGER] Successfully applied POS Tagging! Result: {pkl.name}", flush=True)
    


def ner_tagger(video_path, output_dir):
    """
    Named Entity recognition tagger for speaker segments

    (This function has to be executed in the 'speaker_diarization' conda environment, since protobuf requirements to do not match)
    """
    from nlp_feature_code import feat_functions
    import stanza
    nlp = stanza.Pipeline(lang='de', processors='tokenize,ner,pos', download_method=None, logging_level='ERROR')

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    tags_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/ner_tags.pkl"
    asr_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/asr_large-v2.pkl"
    diarization_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/speaker_diarization_large-v2.pkl"

    if os.path.exists(tags_pkl):
        print(f"[NER TAGGER] Found pkl: {tags_pkl} , skip NER Tagging", flush=True)
        return
    elif not os.path.exists(asr_pkl):
        print("[NER TAGGER] Please provide a pkl containing ASR data (--asr) before approaching NER Tagging", flush=True)
        return
    elif not os.path.exists(diarization_pkl):
        print("[NER TAGGER] Please provide a pkl containing Speaker Diariazation data (--diarize) before approaching NER Tagging", flush=True)
        return
    
    with open(asr_pkl, 'rb') as pkl:
        asr_data = pickle.load(pkl)

    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    proc_text = nlp(asr_data["text"])
    proc_segments = [nlp(segment["text"]) for segment in diarization_data]

    ## NER
    ner_dict = {"EPER": 0, "LPER": 1, "LOC": 2, "ORG": 3, "EVENT": 4, "MISC": 5}
    event_set = set()
    with open('/nfs/home/arafatj/master_project/src/nlp_feature_code/eventKG.csv') as fr:
        for line in fr:
            event_set.add(line.strip())

    ## Get NER Tags and Vectors
    sen_nes, seg_nes = feat_functions.get_ner_outputs(proc_text, proc_segments, ner_dict, event_set)

    all_ner = []
    for i, seg in enumerate(seg_nes):
        # get the diarization segment for the current ner_tagged segment
        diarization_segment = diarization_data[i]
        start = diarization_segment['start_time']
        end = diarization_segment['end_time']
        text = diarization_segment['text']
        vector = seg['vector']
        tags = seg['tags']
        all_ner.append({"start_time": start, "end_time": end, "text": text, "vector": vector, "tags": tags})


    # store result as pkl
    with open(tags_pkl, 'wb') as pkl:
        pickle.dump(all_ner, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[NER TAGGER] Successfully applied NER Tagging! Result: {pkl.name}", flush=True)



def sentence_embeddings(video_path, output_dir):
    """
    Create text embeddings of every frame using CLIPs image encoder.
    These are later used to calculate similarities between frames.
    """
    from sentence_transformers import SentenceTransformer
    # suppress tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import spacy
    nlp = spacy.load("de_core_news_lg")

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    embeddings_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/sentence_embedding.pkl"
    diarization_pkl = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/speaker_diarization_large-v2.pkl"

    if os.path.exists(embeddings_pkl):
        print(f"[SENTENCE EMBEDDINGS] Found pkl: {embeddings_pkl} , skip Sentence Encoding", flush=True)
        return
    elif not os.path.exists(diarization_pkl):
        print("[SENTENCE EMBEDDINGS] Please provide a pkl containing Speaker Diariazation data (--diarize) before approaching Sentence Encoding", flush=True)
        return
    
    with open(diarization_pkl, 'rb') as pkl:
        diarization_data = pickle.load(pkl)

    # load multilingual sentence embedding model
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

    embeddings = []
    for segment in diarization_data:
        text = nlp(segment['text'])
        # get sentences from text segment
        sentences = [sent.text for sent in text.sents]
        # convert sentences to their embeddigns (return a list where each entry is the embedding of a sentence)
        sentence_embeddings = model.encode(sentences)
        embeddings.append({"start_time": segment['start_time'], "end_time": segment['end_time'], "sentences": sentences,"sentence_embeddings": sentence_embeddings})
    
    # store result as pkl
    with open(embeddings_pkl, 'wb') as pkl:
        pickle.dump(embeddings, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SENTENCE EMBEDDINGS] Successfully extracted Sentence Embeddings! Result: {pkl.name}", flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction methods')
    parser.add_argument('--video', action='store', type=os.path.abspath, required=True, help="Path to the video (.mp4) to be processed")
    parser.add_argument('--output', action='store', type=os.path.abspath, help="Path to the directory where results will be stored. \
                        A subdirectory for the corresponding video will be added automatically, so only the path to the main OUTPUT/ directory is required",
                        default="/nfs/home/arafatj/master_project/OUTPUT/") #TODO: REMOVE "DEFAULT="" AND SET "REQUIRED=TRUE" for generic usage
    parser.add_argument('--noprint', action='store_true', help="Disables progression printing by video processors such as CLIP. \
                        Mainly relevant for sbatch processing, so the output log won't be flooded caused by sbatch ignoring the carriage return")

    parser.add_argument('--asr', action='store_true', help="ASR")
    parser.add_argument('--diarize', action='store_true', help="Speaker Diarization")
    parser.add_argument('--places', action='store_true', help="Place Recognition")
    parser.add_argument('--clip', action='store_true', help="CLIP")
    parser.add_argument('--sbd', action='store_true', help="Shot Boundary Detection")
    parser.add_argument('--sd', action='store_true', help="Shot Density")
    parser.add_argument('--sentiment', action='store_true', help="Sentiment Analysis")
    parser.add_argument('--topic', action='store_true', help="Topic Modeling")
    parser.add_argument('--scenes', action='store_true', help="Scene Detection")
    parser.add_argument('--pos', action='store_true', help="POS Tagging")
    parser.add_argument('--imgemb', action='store_true', help="Image Embeddings")
    parser.add_argument('--sentemb', action='store_true', help="Sentence Embeddings")
    parser.add_argument('--ner', action='store_true', help="Named Entity Recognition Tagger")



    args = parser.parse_args()

    start_time = time.time()

    print('DEVICE:', "cuda" if torch.cuda.is_available() else "cpu", flush=True)

    video_path = args.video
    output_dir = args.output

    # change current working directory to the scripts directory (Windows)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    if not output_dir:
        print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--output-dir")', flush=True)
        quit()
    
    if args.asr:
        # Whisper ASR
        asr(video_path, "large-v2", output_dir)

    if args.diarize:
        # Whisper Speaker Diarization
        speaker_diarization(video_path, "large-v2", output_dir)

    if args.places:
        # Places365
        places(video_path, output_dir, args.noprint)

    if args.clip:
        # Clip
        predict_CLIP_queries(video_path, output_dir, args.noprint)

    if args.sentiment:
        # Sentiment Analysis
        sentiment_analysis(video_path, output_dir)

    if args.topic:
        # Sentiment Analysis
        topic_modeling(video_path, output_dir)

    if args.scenes:
        # Scene Detection
        scene_detection(video_path, output_dir)

    if args.pos:
        # POS Tagging
        pos_tagger(video_path, output_dir)

    if args.ner:
        # NER Tagging
        ner_tagger(video_path, output_dir)
    
    if args.imgemb:
        # Image Embeddings using CLIP
        clip_image_embeddings(video_path, output_dir, args.noprint)

    if args.sentemb:
        # Sentence Embeddings
        sentence_embeddings(video_path, output_dir)

    if args.sbd:
        # TransNet V2 (Keep this at the end because of heavy memory usage in TransNet code)
        shot_boundary_detection(video_path, output_dir)

    if args.sd:
        # Shot Density
        shot_density(video_path, output_dir)

    # processing time
    end_time = time.time()
    duration = end_time - start_time
    td = datetime.timedelta(seconds=duration)
    duration_formatted = str(td).split('.')[0]
    print(f'Processing time: {duration_formatted}')