
import os
import glob
import shutil
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import SEED

# Dataset URL slug on Kaggle
DATASET_SLUG = "mateuszbuda/lgg-mri-segmentation"
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def download_dataset():
    """Downloads dataset using Kaggle API."""
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    if len(os.listdir(RAW_DIR)) > 0:
        print("✅ Data already found in data/raw. Skipping download.")
        return

    print("⬇️ Attempting to download dataset via Kaggle API...")
    try:
        os.system(f"kaggle datasets download -d {DATASET_SLUG} -p {RAW_DIR}")
        zip_files = glob.glob(f"{RAW_DIR}/*.zip")
        if zip_files:
            print(f"📦 Unzipping {zip_files[0]}...")
            with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            os.remove(zip_files[0])
            print("✅ Download & Unzip Complete!")
        else:
            print("⚠️ No zip file found. Did Kaggle API work?")
    except Exception as e:
        print(f"⚠️ Error: {e}")

def organize_and_split():
    """Splits data by PATIENT ID (Safety against leakage)."""
    mask_files = glob.glob(f"{RAW_DIR}/**/*.tif", recursive=True)
    mask_files = [f for f in mask_files if '_mask' in f]
    
    if not mask_files:
        print("❌ No data found to process. Please check 'data/raw'.")
        return

    data = []
    for mask_path in mask_files:
        img_path = mask_path.replace('_mask', '')
        filename = os.path.basename(img_path)
        patient_id = "_".join(filename.split("_")[:3]) 
        data.append({"patient_id": patient_id, "image_path": img_path, "mask_path": mask_path})
    
    df = pd.DataFrame(data)
    unique_patients = df['patient_id'].unique()
    
    train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=SEED)
    val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=SEED)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(f"{PROCESSED_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{PROCESSED_DIR}/{split}/masks", exist_ok=True)
    
    print(f"📊 Splitting: {len(train_patients)} Train, {len(val_patients)} Val, {len(test_patients)} Test (Patients)")
    
    def move_files(patient_list, split_name):
        for pid in patient_list:
            patient_data = df[df['patient_id'] == pid]
            for _, row in patient_data.iterrows():
                fname = os.path.basename(row['image_path'])
                shutil.copy(row['image_path'], f"{PROCESSED_DIR}/{split_name}/images/{fname}")
                shutil.copy(row['mask_path'], f"{PROCESSED_DIR}/{split_name}/masks/{fname}")

    move_files(train_patients, 'train')
    move_files(val_patients, 'val')
    move_files(test_patients, 'test')
    print("✅ Data Organization Complete!")

if __name__ == "__main__":
    download_dataset()
    organize_and_split()
