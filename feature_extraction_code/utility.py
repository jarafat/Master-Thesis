import os
import random
import json
import whisper

config = {
    "news_sources": ["Tagesschau", "BildTV", "CompactTV"],
    "Tagesschau_videos": "/nfs/data/fakenarratives/tagesschau/videos/2022/",
    "Tagesschau_annotations": "/nfs/home/arafatj/master_project/annotations/Tagesschau/",
    "BildTV_videos": "/nfs/data/fakenarratives/BildTV/videos/",
    "BildTV_annotations": "/nfs/home/arafatj/master_project/annotations/BildTV/",
    "CompactTV_videos": "/nfs/data/fakenarratives/CompactTV/videos/",
    "CompactTV_annotations": "/nfs/home/arafatj/master_project/annotations/CompactTV",
    "tmp_directory": "/nfs/home/arafatj/tmp/",
    "sample_size": 6
}

def extract_audio(annot_path, news_source):
    """
    Slices the news video to the part, where 'speaker-gender' is annotated

    Returns path to the temporary audio file
    """
    with open(annot_path) as json_file:
        annotations = json.load(json_file)
    
    speaker_diarization = annotations['speaker-gender']['segments']
    diarization_begin = speaker_diarization[0]['start']
    diarization_end = speaker_diarization[-1]['end']

    video_path = config[news_source + '_videos'] + annotations['video_fn']
    tmp_output = config['tmp_directory'] + annotations['video_fn'].replace('.mp4', '.wav')
    os.system(f"ffmpeg -i {video_path} -ss {diarization_begin} -to {diarization_end} -ar 16000 -ac 1 -c:a pcm_s16le {tmp_output} -loglevel error")

    return tmp_output


def find_threshold(news_source):
    model = whisper.load_model("base")

    annot_dir = config[news_source + '_annotations']
    # pick 6 random videos for each source
    annot_files = random.sample(os.listdir(annot_dir), config['sample_size'])
    for i, annot_fn in enumerate(annot_files):
        print(f"Processing videos: {i+1}/{config['sample_size']}")

        audio_file = extract_audio(annot_dir + annot_fn, news_source)
        
        #os.remove(audio_file)

if __name__ == '__main__':
    find_threshold("Tagesschau")
    #find_threshold("BildTV")
    #find_threshold("CompactTV")