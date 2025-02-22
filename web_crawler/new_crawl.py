import os
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# URL of the sitemap
SITEMAP_URL = "https://www.upes.ac.in/sitemap.xml"

# Output directory
OUTPUT_DIR = "outputPages"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum number of threads for parallel execution
MAX_WORKERS = 20  # Adjust based on system capabilities

# Function to get links from the sitemap
def get_links_from_sitemap(sitemap_url):
    response = requests.get(sitemap_url, timeout=10)
    if response.status_code != 200:
        print("Failed to fetch sitemap")
        return []
    
    root = ET.fromstring(response.text)
    links = [elem.text for elem in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    return links

# Function to extract content from <main class="innerPages"> or fallback to <main>
def extract_and_save_content(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        
        # Try to find <main class="innerPages"> first
        main_tag = soup.find("main", class_="innerPages")

        # Fallback to any <main> tag if specific class is not found
        if not main_tag:
            print(f"<main class='innerPages'> not found, looking for <main> in {url}")
            main_tag = soup.find("main")

        if not main_tag:
            print(f"No <main> tag found in {url}")
            return None

        # Extract unique text content
        text_content = "\n".join(set(main_tag.stripped_strings))
        if not text_content.strip():
            print(f"Empty content for {url}")
            return None

        # Create a filename from the URL
        parsed_url = urlparse(url)
        filename = parsed_url.path.strip("/").replace("/", "_") or "homepage"
        filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")

        # Save content to file
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(text_content)
        
        print(f"Saved: {filepath}")
        return filepath

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

# Main script with parallel execution
def main():
    links = get_links_from_sitemap(SITEMAP_URL)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(extract_and_save_content, link): link for link in links}

        for future in as_completed(future_to_url):
            future.result()  # Ensures exceptions are raised if any occur

if __name__ == "__main__":
    main()
