import os
import shutil
import kagglehub

# 1. Define your exact target folder
target_folder = r"D:\DATA science projects\BOatQl\Dataset"
os.makedirs(target_folder, exist_ok=True)

# 2. Download the dataset via kagglehub cache
cache_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")
print(f"Downloaded temporarily to cache: {cache_path}")

# 3. Move/Copy files as separate items into your target folder
for item in os.listdir(cache_path):
    source_item = os.path.join(cache_path, item)
    destination_item = os.path.join(target_folder, item)
    
    # Copy files (or folders if any exist) over directly
    if os.path.isdir(source_item):
        if not os.path.exists(destination_item):
            shutil.copytree(source_item, destination_item)
    else:
        shutil.copy2(source_item, destination_item)

print(f"\nSuccess! All separate CSV files are now ready in:\n{target_folder}")
