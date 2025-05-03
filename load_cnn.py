import os

def count_images_in_folder(folder_path):
    total = 0
    for root, dirs, files in os.walk(folder_path):
        image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total += len(image_files)
    return total

base_path = 'Dataset'  # change to your dataset root folder

datasets = ['Train', 'Validation', 'Test']
for ds in datasets:
    ds_path = os.path.join(base_path, ds)
    count = count_images_in_folder(ds_path)
    print(f"Total images in {ds}: {count}")

