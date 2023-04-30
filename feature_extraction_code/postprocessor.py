# Overall
import os
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt

# shot density
from sklearn.neighbors import KernelDensity

config = {
    "fps": 2
}


# Calculate shot density from previously saved shot boundary detection pkls
def shot_density(pkl_file):
    shots_data = None
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as pkl:
            shots_data = pickle.load(pkl)
    else:
        print('Could not find .pkl file! Please check the path')
        return
    
    last_shot_end = 0
    shots = []
    # shot[0] = shot.start - shot [1] = shot.end
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
        "time": time.squeeze(),
        "delta_time": 1 / config["fps"]
    }
    print(output_data)

    # DEBUG
    plt.plot(output_data["time"], output_data["y"])
    plt.savefig(f'/nfs/home/arafatj/master_project/graphs/shot_density/{os.path.basename(pkl_file.replace(".pkl", ".png"))}')


# apply clip on all 365 places categories
def places_clip(video_path):
    import clip
    import torch
    from PIL import Image
    import json
    from video_decoder import VideoDecoder

    with open('/nfs/home/arafatj/master_project/models/Places365/categories_places365.json') as f:
        places365 = json.load(f)

    places = []
    for place in places365.keys():
        places.append(places365[place][0])
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'device: {device}')
    model, preprocess = clip.load("ViT-B/32", device=device)

    video_decoder = VideoDecoder(video_path, fps=config["fps"])
    duration = video_decoder.duration()

    for i, frame in enumerate(video_decoder):
        print(f'\rProcessing frames: {i}/{int(duration * config["fps"])}', end="")

        image = preprocess(Image.fromarray(np.uint8(frame.get('frame')))).unsqueeze(0).to(device)
        text = clip.tokenize(places).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()


if __name__ == '__main__':
    #shot_density('/nfs/home/arafatj/master_project/PKLs/SHOT_BOUNDARY_DETECTION/20220104_Schluss_mit_dem_Corona_Lockdown_1odqvo1zKzI.pkl')

    places_clip('/nfs/data/fakenarratives/tagesschau/videos/2022/TV-20220106-2021-1700.webl.h264.mp4')