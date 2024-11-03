import os
import shutil

def merge_files(jpg_folder, txt_folder, destination_folder):
    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)
    
    # Move .jpg files
    for file_name in os.listdir(jpg_folder):
        if file_name.endswith('.jpg'):
            source_path = os.path.join(jpg_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy(source_path, destination_path)
    
    # Move .txt files
    for file_name in os.listdir(txt_folder):
        if file_name.endswith('.txt'):
            source_path = os.path.join(txt_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            shutil.copy(source_path, destination_path)
    
    print(f"Files have been copied to {destination_folder}")

# Example Usage
jpg_folder = "part_B_final\\train_data\images"
txt_folder = "part_B_final\\train_data\\updated"
destination_folder = "DATA_ROOT2\\train\scene01"

merge_files(jpg_folder, txt_folder, destination_folder)