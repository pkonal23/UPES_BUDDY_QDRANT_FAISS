import requests
from bs4 import BeautifulSoup
import os
import re
from fpdf import FPDF

def get_links_from_sitemap(sitemap_url):
    response = requests.get(sitemap_url, verify=False)
    xml = BeautifulSoup(response.content, "lxml")
    urls_from_xml = []
    loc_tags = xml.find_all('loc')
    for loc in loc_tags:
        url = loc.get_text()
        corrected_url = correct_url(url)
        #urls_from_xml.append(loc.get_text())
        if corrected_url:
            urls_from_xml.append(corrected_url)
    return urls_from_xml


def correct_url(url):
    # Check if the URL is missing a slash after 'in' and fix it
    if "infaculty" in url:
        corrected_url = url.replace("infaculty", "in/faculty")
        return corrected_url
    return url

# Function to create a folder if it doesn't exist
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Function to crawl URLs and save content as PDF files
def save_urls_as_text(urls):
    # Specify the target directory
    target_dir = "../KnowledgeBase"
    # Create the directory if it doesn't exist
    create_folder(target_dir)

    for url in urls:
        print("Saving:", url)
        filename = url[8:].replace("/", "_") + ".txt"
        filepath = os.path.join(target_dir, filename)

        with open(filepath, "w", encoding="UTF-8") as f:
            # Get the text from the URL using BeautifulSoup
            soup = BeautifulSoup(requests.get(url, verify=False).text, "html.parser")

            # Get the text but remove the tags
            text = soup.get_text()

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")
            else:
                # Write the text to the file in the target directory
                f.write(text)

def crawl_sitemap_url(sitemap_url):
    url_list = get_links_from_sitemap(sitemap_url)
    save_urls_as_text(url_list)
    print("Crawling of",sitemap_url,"completed.")



def remove_blank_lines(folder_path):
    """
  Removes blank lines, \\n, \\n characters, and double spaces from each line in all .txt files in a folder while preserving line breaks.

  Args:
    folder_path: The path to the folder containing the .txt files.
  """
    temp=1
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            lines = []

            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Clean and write lines back to file
            cleaned_lines = []
            for line in lines:
                stripped_line = line.strip()
                clean_line = re.sub(r"\s{2}", " ", stripped_line)
                clean_line = re.sub(r"\\n", "", clean_line)
                clean_line = re.sub(r"\n", "", clean_line)

                if clean_line:
                    cleaned_lines.append(clean_line + "\n")

            cleaned_lines = cleaned_lines[61:-181]

            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(cleaned_lines)


def clean_files(folder_path):
    """
  Reads files from the specified folder, removes blank lines,
  double spaces, and \\n characters, and erases the last 181 lines.

  Args:
    folder_path: The path to the folder containing the .txt files.
  """
    remove_blank_lines(folder_path)
    print("Files cleaned in", folder_path)


def convert_txt_to_pdf(txt_path, pdf_path):
    # Create a new FPDF object
    pdf = FPDF()

    # Open the text file and read its contents
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read()

    text = text.encode('latin-1', 'replace').decode('latin-1')

    # Add a new page to the PDF
    pdf.add_page()

    # Set the font and font size
    pdf.set_font('Arial', size=12)

    # Write the text to the PDF
    pdf.multi_cell(0, 10, text)

    # Save the PDF with the same name as the original .txt file
    pdf.output(pdf_path)


def convert_files(txt_directory):

    # List all .txt files in the directory
    txt_files = [f for f in os.listdir(txt_directory) if f.endswith(".txt")]

    # Convert each .txt file to .pdf and delete the original .txt file
    for txt_file in txt_files:
        txt_path = os.path.join(txt_directory, txt_file)
        pdf_path = os.path.join(txt_directory, os.path.splitext(txt_file)[0] + ".pdf")

        # Convert
        convert_txt_to_pdf(txt_path, pdf_path)

        # Delete the original .txt file
        os.remove(txt_path)

        print(f"Converted and deleted: {txt_file}")


if __name__ == "__main__":
    sitemap_url="https://www.upes.ac.in/sitemap.xml"
    crawl_sitemap_url(sitemap_url)
    folder_path = "../KnowledgeBase"
    clean_files(folder_path)
    #convert_files(folder_path)