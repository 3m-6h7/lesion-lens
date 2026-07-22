import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def create_subset(metadata_path, source_dir, dest_dir, n_total=3800):
    #1. Load metadata
    df = pd.read_csv(metadata_path)
    
    #2. Stratified sampling to maintain class distribution across the 7 diagnostic classes
    df_subset = df.groupby('dx', group_keys=False).apply(
        lambda x: x.sample(int(n_total * len(x) / len(df)), random_state=42)
    )
    
    #3. Create destination folder
    os.makedirs(dest_dir, exist_ok=True)
    
    #4. Copy images from parts 1 and 2 to the subset folder
    print(f"Copying {len(df_subset)} images to {dest_dir}...")
    for _, row in df_subset.iterrows():
        img_id = row['image_id'] + ".jpg"
        src_path1 = os.path.join(source_dir, 'HAM10000_images_part_1', img_id)
        src_path2 = os.path.join(source_dir, 'HAM10000_images_part_2', img_id)
        target = os.path.join(dest_dir, img_id)
        
        if os.path.exists(src_path1):
            shutil.copy(src_path1, target)
        elif os.path.exists(src_path2):
            shutil.copy(src_path2, target)
            
    #Save the subset metadata for reproducible loading
    df_subset.to_csv(os.path.join(dest_dir, 'metadata_subset.csv'), index=False)
    print("Dataset subset creation complete!")

if __name__ == "__main__":
    create_subset(
        metadata_path='data/HAM10000/HAM10000_metadata.csv',
        source_dir='data/HAM10000/',
        dest_dir='data/subset_3800/'
    )   