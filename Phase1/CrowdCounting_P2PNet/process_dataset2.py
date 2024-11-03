import os
import shutil

# Path to the folder where both .jpg and .txt files will be saved

# For test folder
# output_folder = "DATA_ROOT2\\test\scene01"
# write_file="DATA_ROOT2\\test.list.txt"

# For train folder
output_folder = "DATA_ROOT2\\train\scene01"
write_file="DATA_ROOT2\\train.list.txt"

jpg_list=[]
txt_list=[]
# Copy all .jpg files from the jpg_folder to the output_folder
for jpg_filename in os.listdir(output_folder):
    if jpg_filename.endswith(".jpg"):
        output_jpg_path = os.path.join(output_folder, jpg_filename)
        jpg_file_path = ('/').join(output_jpg_path.split('\\')[1:])
        jpg_list.append(jpg_file_path)
        # shutil.copy(jpg_file_path, output_jpg_path)

# Copy all .txt files from the txt_folder to the output_folder
for txt_filename in os.listdir(output_folder):
    if txt_filename.endswith(".txt"):
        output_txt_path = os.path.join(output_folder, txt_filename)
        txt_file_path = ('/').join(output_txt_path.split('\\')[1:])
        txt_list.append(txt_file_path)


with open(write_file, 'w') as output_file:
    # Iterate over both lists
    for i in range(len(jpg_list)):
        # Write both paths on one line
        output_file.write(f"{jpg_list[i]} {txt_list[i]}\n")







