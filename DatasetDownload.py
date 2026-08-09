import os
import shutil
from pathlib import Path
import kagglehub

# Docker supplies DATA_DIR; the default supports local use from the project
# root without any machine-specific paths.
target_folder = Path(os.environ.get("DATA_DIR", "Dataset"))
target_folder.mkdir(parents=True, exist_ok=True)

cache_path = kagglehub.dataset_download("olistbr/brazilian-ecommerce")

for item in os.listdir(cache_path):
    source_item = Path(cache_path) / item
    destination_item = target_folder / item

    if source_item.is_dir():
        if not destination_item.exists():
            shutil.copytree(source_item, destination_item)
    else:
        shutil.copy2(source_item, destination_item)

print(f"Dataset ready at: {target_folder}")