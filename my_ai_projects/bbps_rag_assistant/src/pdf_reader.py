from pypdf import PdfReader


pdf_path = "./pdfs/LPG ADVISORY_2026.pdf"


reader = PdfReader(pdf_path)


print("Number of pages:", len(reader.pages))


for page_number, page in enumerate(reader.pages):

    text = page.extract_text()

    print("\n--- PAGE", page_number + 1, "---")
    print(text[:1000])