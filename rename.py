import os

# Directory containing the files
FOLDER_PATH = "/Users/konalsmac/upesbuddy/UPESBuddy_Chatbot/web_crawler/summOutput"

def rename_files():
    for filename in os.listdir(FOLDER_PATH):
        old_path = os.path.join(FOLDER_PATH, filename)
        new_filename = f"www.upes.ac.in_{filename}"
        new_path = os.path.join(FOLDER_PATH, new_filename)

        if os.path.isfile(old_path):  # Ensure it's a file
            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    rename_files()
