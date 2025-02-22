import os

# Path to the folder where the files are located
folder_path = "/Users/konalsmac/upesbuddy/UPESBuddy_Chatbot/KnowledgeBaseInfo"  # Change this to your folder path

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    # Check if the file ends with '.txt'
    if filename.endswith(".txt"):
        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)
        
        # Replace the symbols with underscores
        new_name = name.replace('.', '_').replace('-', '_')
        
        # Create the new filename with the same .txt extension
        new_filename = new_name + ext
        
        # Get the full path of the original and new filenames
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)

        print(f'Renamed: {filename} -> {new_filename}')
