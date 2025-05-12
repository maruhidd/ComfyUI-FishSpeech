from huggingface_hub import snapshot_download

# Download the entire repository
print("Downloading fishaudio/fish-speech-1 repository...")
download_path = snapshot_download(repo_id="fishaudio/fish-speech-1")
print(f"Repository downloaded successfully to: {download_path}")
