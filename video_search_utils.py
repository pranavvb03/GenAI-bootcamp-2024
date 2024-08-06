import youtube_dl
import os
import subprocess
import torch
import clip
from PIL import Image
from annoy import AnnoyIndex

# Function to download videos
def download_videos(video_urls, download_path='./videos'):
    ydl_opts = {
        'outtmpl': f'{download_path}/%(title)s.%(ext)s',
        'format': 'best'
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(video_urls)

# Function to extract key frames using ffmpeg
def extract_key_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_filename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_filename)
    output_pattern = os.path.join(output_folder, f"{video_name}_frame_%03d.png")
    command = ['ffmpeg', '-i', video_path, '-vf', 'select=eq(pict_type\\,I)', '-vsync', 'vfr', output_pattern]
    subprocess.run(command)

def process_videos(video_folder, output_folder):
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.mkv', '.avi'))]

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        extract_key_frames(video_path, output_folder)
# Function to get CLIP embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def get_image_embedding(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features

def get_text_embedding(text):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features

# Function to build Annoy index
def build_annoy_index(embeddings, embedding_dim=512):
    annoy_index = AnnoyIndex(embedding_dim, 'angular')
    for i, emb in enumerate(embeddings):
        annoy_index.add_item(i, emb)
    annoy_index.build(10)
    return annoy_index
