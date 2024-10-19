import os

# Path to the folder containing the .txt files
input_folder = "part_B_final\\test_data\images"
# Path to the folder where updated .txt files will be saved
output_folder = "part_B_final\\test_data\\updated"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    # Process only .txt files
    if filename.endswith(".txt"):
        input_file_path = os.path.join(input_folder, filename)
        
        # Read the content of the .txt file
        with open(input_file_path, 'r') as file:
            lines = file.readlines()
        
        # Update the content by removing the last three columns
        updated_lines = []
        for line in lines:
            columns = line.strip().split()  # Split by whitespace
            updated_line = " ".join(columns[:-3])  # Remove last three columns
            updated_lines.append(updated_line)
        
        # Path to save the updated file in the new folder
        output_file_path = os.path.join(output_folder, filename)
        
        # Write the updated content to the new file in the output folder
        with open(output_file_path, 'w') as file:
            file.write("\n".join(updated_lines) + "\n")

print("Successfully updated all .txt files and saved them to the new folder.")
