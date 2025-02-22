import requests
from bs4 import BeautifulSoup
import os
import re

def get_links_from_sitemap(sitemap_url):
    response = requests.get(sitemap_url, verify=False)
    soup = BeautifulSoup(response.content, "xml")
    urls = [loc.text for loc in soup.find_all("loc")]
    return urls

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def extract_text_from_url(url):
    print(f"Parsing: {url}")
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text while preserving some structure
    text_blocks = [tag.get_text(separator="\n", strip=True) for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "div"])]
    text = "\n\n".join(text_blocks)

    return text

def save_text_to_file(url, text, folder):
    filename = re.sub(r"[^a-zA-Z0-9]", "_", url[8:]) + ".txt"
    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)

def process_sitemap(sitemap_url, folder="ParsedPages"):
    create_folder(folder)
    urls = get_links_from_sitemap(sitemap_url)

    for url in urls:
        text = extract_text_from_url(url)
        save_text_to_file(url, text, folder)

    print("Processing completed.")

if __name__ == "__main__":
    sitemap_url = "https://www.upes.ac.in/sitemap.xml"
    process_sitemap(sitemap_url)
