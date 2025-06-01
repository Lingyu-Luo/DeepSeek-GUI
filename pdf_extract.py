import os
from pypdf import PdfReader
from pdfminer.high_level import extract_text
import pandas as pd
import re

# Function to check if a PDF contains selectable text
def is_text_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        if page.extract_text():
            return True
    return False  # No selectable text → likely a scanned PDF

# Function to extract text (from normal PDFs)
def extract_text_pypdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

# Function to extract text (preserving layout)
def extract_text_pdfminer(pdf_path):
    return extract_text(pdf_path)

def filter_headers_footers(text):
    """Remove common header/footer patterns using regex"""
    patterns = [
        r'^Page \d+ of \d+$',          # Page numbers
        r'\d{1,2} [A-Za-z]{3} \d{4}',  # Dates like 31 Mar 2025
        r'https?://\S+',                # URLs
        r'All use subject to \S+',      # JSTOR footer
        r'This content downloaded from',# Download notice
        r'^\d+$'                        # Standalone page numbers
    ]
    combined_pattern = re.compile('|'.join(patterns), flags=re.IGNORECASE | re.MULTILINE)
    return '\n'.join([line for line in text.split('\n') if not combined_pattern.search(line)])

# Function to extract metadata
def extract_metadata(pdf_path):
    reader = PdfReader(pdf_path)
    return reader.metadata  # Returns dictionary

# Function to process PDF and extract all data
def process_pdf(pdf_path):
    print(f"\n📄 Processing PDF: {pdf_path}")

    # Step 1: Detect if the PDF contains selectable text
    text_pdf = is_text_pdf(pdf_path)

    # Step 2: Extract text (choose method)
    print("✅ Text detected: Extracting with pypdf...")
    text = extract_text_pdfminer(pdf_path)

    # Step 4: Extract metadata
    print("📝 Extracting metadata...")
    metadata = extract_metadata(pdf_path)

    # Step 5: Save results
    output_folder = "output"
    os.makedirs(output_folder, exist_ok=True)

    with open(f"{output_folder}/{os.path.basename(pdf_path).split('.')[0]}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    #with open(f"{output_folder}/{os.path.basename(pdf_path).split('.')[0]}_metadata.txt", "w", encoding="utf-8") as f:
    #    f.write(str(metadata))

    print(f"\n✅ PDF Processing Complete! Results saved in '{output_folder}' folder.")

# Run script
if __name__ == "__main__":
    """
    directory = ""
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            process_pdf(os.path.join(directory,filename))
    """
    filename = "E:/Downloads/t2.pdf"
    process_pdf(filename)
