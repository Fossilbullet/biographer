import replicate
import requests
import zipfile
import dropbox
import re
import time
import os
from flask import Flask

app = Flask(__name__)


# === CONFIG ===
image_urls = [
    "https://img.freepik.com/premium-photo/adult-man-serene-face-expression-studio-portrait_53876-75419.jpg?semt=ais_hybrid&w=740",
    "https://media.istockphoto.com/id/507995592/photo/pensive-man-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=fVoaIqpHo07YzX0-Pw51VgDBiWPZpLyGEahSxUlai7M=",
]
zip_filename = "aifvguser0000002.zip"
dropbox_dest_path = f"/{zip_filename}"  # Upload location in Dropbox

# === Get Dropbox token from environment variables ===
dropbox_access_token = os.getenv("DROPBOX_ACCESS_TOKEN")
if not dropbox_access_token:
    print("ERROR: Dropbox access token not found in environment variables.")

# === DOWNLOAD IMAGES AND ZIP ===
def create_zip_from_urls(urls, zip_path):
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for i, url in enumerate(urls):
            response = requests.get(url)
            if response.status_code == 200:
                image_name = f"image_{i+1}.{url.split('.')[-1].split('?')[0]}"
                zipf.writestr(image_name, response.content)
                print(f"Added {image_name} to zip")
            else:
                print(f"Failed to download: {url}")

create_zip_from_urls(image_urls, zip_filename)

# === UPLOAD TO DROPBOX ===
def upload_to_dropbox(zip_path, dropbox_path, token):
    dbx = dropbox.Dropbox(token)
    with open(zip_path, "rb") as f:
        dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
    shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
    direct_url = re.sub(r"dl=0", "dl=1", shared_link_metadata.url)
    return direct_url

link = upload_to_dropbox(zip_filename, dropbox_dest_path, dropbox_access_token)
print(f"✅ Uploaded to Dropbox. Direct link: {link}")


training = replicate.trainings.create(
  # You need to create a model on Replicate that will be the destination for the trained version.
  destination="fossilbullet/aifvgusermodel0000001",
  version="ostris/flux-dev-lora-trainer:26dce37af90b9d997eeb970d92e47de3064d46c300504ae376c75bef6a9022d2",
  input={
    "steps": 1000,
    "lora_rank": 16,
    "optimizer": "adamw8bit",
    "batch_size": 1,
    "resolution": "512,768,1024",
    "autocaption": True,
    "input_images": link,
    "trigger_word": "aifvguser0000001",
    "learning_rate": 0.0004,
    "wandb_project": "flux_train_replicate",
    "wandb_save_interval": 100,
    "caption_dropout_rate": 0.05,
    "cache_latents_to_disk": False,
    "wandb_sample_interval": 100,
    "gradient_checkpointing": False
  },
)

while training.status not in ["succeeded", "failed", "cancelled"]:
    print(f"Status: {training.status}... waiting")
    time.sleep(15)
    training = replicate.trainings.get(training.id)
    
if training.status == "succeeded":
    print("View your trained model here:", training.destination)
else:
    print("Something went wrong" + training.status)
    print(training.logs)
