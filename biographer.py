import replicate
import requests
import zipfile
import dropbox
import re
import time

# === CONFIG ===
image_urls = [
    "https://img.freepik.com/premium-photo/adult-man-serene-face-expression-studio-portrait_53876-75419.jpg?semt=ais_hybrid&w=740",
    "https://media.istockphoto.com/id/507995592/photo/pensive-man-looking-at-the-camera.jpg?s=612x612&w=0&k=20&c=fVoaIqpHo07YzX0-Pw51VgDBiWPZpLyGEahSxUlai7M=",
]
zip_filename = "aifvguser0000001.zip"
dropbox_dest_path = f"/{zip_filename}"  # Upload location in Dropbox
dropbox_access_token = "sl.u.AFy0rC4172N32QnZ_70aeqneb_WzP5MgHR5jonayfo8ZVnrh0GQSMrpN6dfwPVWf00SeNwmTx4wnScqJPGObfO9ieYTfIeNqGxatVjC8UARnB8NaNrGQZKYy5kzHoiSpifKL9GOGBzZh4bvwArb-DggEivtA3qJMYeREu2ukIV3STff_Ay-wARA3uHB_16ANjlRoShTAI6PFfu4XN4deitTuI0NVMoVIJaxZ_XvRXivBmbZRHwjQxw33Cj3q8NRZstSxah_KvCnYEGq_gnIRuYUjQIpM09M5SN53V-YXFTrK6vLgmfOQ0tBSDYgrzhoM4qpWG5ILhAh5v6HNdhJoDsbY1mHPCyaUBCBoAVs_QpWodsDSwwJ9U3RJiwY4OjLeecsYNl4xazHhU0TV97rxjNNqSOV-O05iAfJMM26dfWPeu_u6Bsw-2XmtdqE1o8Zrn0zRwsv0lZmO0r6qKAYeGQXZAHShOCCn7lWPwnd0HZCMkn-TRMLN7p0nUGxmSlaSXJei2_P2rum8N1eZx_ZvrCtaRH7Itzw65e00gYKQErcKeyZx2cNzAqmDIxbHPjmjw4Z49GgE0yvtTTzn90X7G-kuv_eu2b9KaYBo6HCTk6DwzIcprvVZkK1Jh1spr9YH_ggP8osx-Fc4rytN-LWfjnj4hmJy6JcG1K4SPfATzW2zJbxlZ6Us6pCYDIvCowjIZomyiwIX9mTM4tbyeGPRfznkgX43QtBXK3aRtG1Q3fDT2NfquAsR-JwiILcL2I0dVcvj-M8P26d6vyuvugl6GWwVlz7aI7YXVnnV57OLsVelmap5DErPt16kP6cfhBIge8r1ny8g81ehjyh7HvyU77EtfEcoKWm-y34beDrGOerb8nJ2urQ3f__mn3HoJIinITb85WwHCrHfUMxJda9HxE403tQKRDOz5Av7gqdYXBCtXeJ1RT2zJHjlrn1_RQ60bC5n7qetw9T0EStAOHtzXYsW2pJWv1AgqnsgvkgPVhwcFYFWfzvxzir8lRkBX2EiltiBnrZihzWnDvk1M72BPrkZ0_5nn7q-3mTY_wlSD7RSnQkqNdzQhsgVYa9RkkXoNlUL8pm8bHGrRomceSLZm0IAV4k7ZjfEwynvJs3lRqnFJbtdziPKvz5iAkqI8SpyayAfmMY0Cj-Wmc6jQZI28Gw0lav-J-JEseBFgyROB00CUesmfjLEtpaLcMZhrwN191PbhMAZzg3BFErN22ooLarAf4HADY-iNeFJdwXbEdn8sy426Xu0hAdzsK9GbvhvmWC9DBNEygiQhLKV75U0g2li9QCTn4BMrfk3iH4tYwxIuSUgsXjtryaQSoUcX5a-17J9CtnnTjNBHms4oeRztvz49qPeaosJMTtMsm6zVh6Ydm1PG0Hk5ZHomUEQ_Ur19aie-_DtU4k8fnBBwbtGL1h8"

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
