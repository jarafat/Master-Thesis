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

# TODO: store fps in .pkl files

config = {
    "fps": 2,
    "tmp_dir": "/nfs/home/arafatj/tmp/",
    "spk_diarization_script" : "/nfs/home/arafatj/master_project/whisper-diarization-main/diarize.py",
    "CLIP_queries": "/nfs/home/arafatj/master_project/models/CLIP/CLIP_queries.json",
    "places365_architecture": "resnet50",
    "places365_model": "/nfs/home/arafatj/master_project/models/Places365/{architecture}_places365.pth.tar",
    "places365_categories": "/nfs/home/arafatj/master_project/models/Places365/categories_places365.txt"
}


def speaker_diarization(video_path, whisper_model, output_dir):
    # ASR & Speaker Diarization
    from moviepy.editor import VideoFileClip
    import tempfile
    import subprocess

    os.makedirs(f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}", exist_ok=True)
    pkl_file = f"{output_dir}/{os.path.basename(video_path.replace('.mp4', ''))}/speaker_diarization.pkl"

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
                   stdout=subprocess.DEVNULL)
    
    # file will only be there if the diarization was successful
    if os.path.exists(pkl_file):
        print(f"[Speaker Diarization] Successfully applied Whisper Speaker Diarization! Result: {pkl_file}", flush=True)



def shot_boundary_detection(video_path, output_dir):
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
    plt.savefig(f'/nfs/home/arafatj/master_project/graphs/shot_density/{os.path.basename(pkl_file.replace(".pkl", ".png"))}')
    """



def places(video_path, output_dir, noprint):
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
            labels_with_probs = list(zip(queries[domain].keys(), probs.tolist()))
            result_matrices[domain].append(labels_with_probs)
    if not noprint:
        # remove carriage return
        print("", flush=True)

    # store result list as pkl
    with open(pkl_file, 'wb') as pkl:
        pickle.dump(result_matrices, pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[CLIP] Successfully applied CLIP! Result: {pkl.name}", flush=True)

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature extraction methods')
    parser.add_argument('--video', action='store', type=os.path.abspath, required=True, help="Path to the video (.mp4) to be processed")
    parser.add_argument('--output', action='store', type=os.path.abspath, help="Path to the directory where results will be stored. \
                        a subdirectory for the corresponding video will be added automatically, so only the path to the main OUTPUT/ directory is required",
                        default="/nfs/home/arafatj/master_project/OUTPUT/") #TODO: REMOVE "DEFAULT="" AND SET "REQUIRED=TRUE" for generic usage
    parser.add_argument('--noprint', action='store_true', help="Disables progression printing by video processors such as CLIP. \
                        Mainly relevant for sbatch processing, so the output log won't be flooded caused by sbatch ignoring the carriage return")

    parser.add_argument('--diarize', action='store_true', help="Speaker Diarization")
    parser.add_argument('--places', action='store_true', help="Place Recognition")
    parser.add_argument('--clip', action='store_true', help="CLIP")
    parser.add_argument('--sbd', action='store_true', help="Shot Boundary Detection")
    parser.add_argument('--sd', action='store_true', help="Shot Density")
    
    args = parser.parse_args()

    start_time = time.time()

    print('DEVICE:', "cuda" if torch.cuda.is_available() else "cpu", flush=True)

    video_path = args.video
    output_dir = args.output

    # change current working directory to the scripts directory (Windows)
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
    if args.diarize:
        # Whisper
        if not output_dir:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--output-dir")', flush=True)
            quit()
        speaker_diarization(video_path, "base", output_dir)

    if args.places:
        # Places365
        if not output_dir:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--output-dir")', flush=True)
            quit()
        places(video_path, output_dir, args.noprint)

    if args.clip:
        # Clip
        if not output_dir:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--output-dir")', flush=True)
            quit()
        predict_CLIP_queries(video_path, output_dir, args.noprint)

    if args.sbd:
        # TransNet V2
        if not output_dir:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--output-dir")', flush=True)
            quit()
        shot_boundary_detection(video_path, output_dir)

    if args.sd:
        # Shot Density
        if not output_dir:
            print('ERROR: Please provide the path to a directory where .pkl files will be stored and/or read ("--output-dir")', flush=True)
            quit()
        shot_density(video_path, output_dir)

    # processing time
    end_time = time.time()
    duration = end_time - start_time
    td = datetime.timedelta(seconds=duration)
    duration_formatted = str(td)
    print(f'Processing time: {duration_formatted}')