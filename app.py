import streamlit as st
from PIL import Image
from video_search_utils import download_videos, extract_key_frames, get_image_embedding, get_text_embedding, build_annoy_index

# Initialize model and other components
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def main():
    st.title("Video Search Application")
    
    # Input URLs to download videos
    video_urls = st.text_area("Enter YouTube URLs (one per line):").splitlines()
    if st.button("Download Videos"):
        download_videos(video_urls)
        st.success("Videos downloaded successfully.")
    
    # Extract key frames from downloaded videos
    if st.button("Extract Key Frames"):
        video_folder = './videos'
        output_folder = './keyframes'
        video_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
        for video in video_files:
            extract_key_frames(video, output_folder)
        st.success("Key frames extracted successfully.")
    
    # Build embeddings and index
    if st.button("Build Embeddings and Index"):
        image_files = [os.path.join('./keyframes', f) for f in os.listdir('./keyframes') if f.endswith('.png')]
        embeddings = [get_image_embedding(img).cpu().numpy().flatten() for img in image_files]
        annoy_index = build_annoy_index(embeddings)
        st.success("Embeddings and index built successfully.")
    
    # Search functionality
    query = st.text_input("Enter search query (text):")
    uploaded_file = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
    
    if st.button("Search"):
        if query:
            query_embedding = get_text_embedding(query).cpu().numpy().flatten()
        elif uploaded_file:
            query_embedding = get_image_embedding(uploaded_file).cpu().numpy().flatten()
        
        index = annoy_index.get_nns_by_vector(query_embedding, 1)[0]
        best_match = image_files[index]
        st.image(best_match, caption="Best match")
    
if __name__ == "__main__":
    main()
