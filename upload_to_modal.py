# upload_to_modal.py
import modal

app = modal.App("upload-phase3-model")

# Create or reference your volume
volume = modal.Volume.from_name("my-models-volume", create_if_missing=True)

@app.function(volumes={"/models": volume})
def upload_model():
    import shutil
    import os
    
    # Copy from local to volume
    local_path = "output/best_phase_3"
    volume_path = "/models/best_phase_3"
    
    print(f"Uploading {local_path} to Modal volume...")
    
    # The volume is mounted at /models
    if os.path.exists(volume_path):
        shutil.rmtree(volume_path)
    
    shutil.copytree(local_path, volume_path)
    
    # Commit changes to persist them
    volume.commit()
    
    print(f"✅ Upload complete! Model saved to {volume_path}")
    
    # List files to verify
    files = os.listdir(volume_path)
    print(f"Files in {volume_path}: {files}")

@app.local_entrypoint()
def main():
    upload_model.remote()