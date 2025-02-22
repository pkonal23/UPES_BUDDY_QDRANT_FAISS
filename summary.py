import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Directories
INPUT_DIR = "/Users/konalsmac/upesbuddy/UPESBuddy_Chatbot/web_crawler/outputPages"
OUTPUT_DIR = "/Users/konalsmac/upesbuddy/UPESBuddy_Chatbot/web_crawler/summOutput"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OpenAI API Call Function
def get_detailed_summary(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the given content in a highly detailed manner, keeping all key information intact. The summary should be coherent and informative. Make sure to include all relevant contact details if mentioned in the text."},
                {"role": "user", "content": text}
            ],
            temperature=0.5,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

# Process a single file
def process_file(filename):
    input_filepath = os.path.join(INPUT_DIR, filename)
    output_filepath = os.path.join(OUTPUT_DIR, filename)

    # Read content
    with open(input_filepath, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        print(f"Skipping empty file: {filename}")
        return

    print(f"Processing: {filename}")
    
    # Generate summary
    summary = get_detailed_summary(content)

    if summary:
        with open(output_filepath, "w", encoding="utf-8") as file:
            file.write(summary)
        print(f"Saved: {output_filepath}")
    else:
        print(f"Failed to summarize: {filename}")

# Parallel processing
def main():
    files = os.listdir(INPUT_DIR)
    num_workers = min(5, len(files))  # Adjust workers based on files available

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_file, filename): filename for filename in files}

        for future in as_completed(futures):
            future.result()  # Ensures exception handling per task

if __name__ == "__main__":
    main()
