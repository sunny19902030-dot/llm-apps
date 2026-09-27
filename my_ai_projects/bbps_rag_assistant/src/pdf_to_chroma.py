import os

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Find the project folder
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

CHROMA_DIR = os.path.join(
    PROJECT_DIR,
    "chroma_db"
)


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 3. Connect to ChromaDB
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)


# --------------------------------------------------
# 4. Create/get collection
# --------------------------------------------------

collection = client.get_or_create_collection(
    name="payment_pdf_knowledge"
)


# --------------------------------------------------
# 5. Create text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

# --------------------------------------------------
# 6. Process every PDF
# --------------------------------------------------

print("\nLooking for PDFs in:")
print(DATA_DIR)

for filename in os.listdir(DATA_DIR):

    if not filename.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        DATA_DIR,
        filename
    )

    print("\nProcessing:", filename)

    # --------------------------------------------------
    # 7. Open PDF
    # --------------------------------------------------

    reader = PdfReader(pdf_path)

    print(
        "Number of pages:",
        len(reader.pages)
    )

    # --------------------------------------------------
    # 8. Process every page
    # --------------------------------------------------

    for page_number, page in enumerate(
        reader.pages
    ):

        text = page.extract_text()

        if not text:
            continue

        # --------------------------------------------------
        # 9. Split page into chunks
        # --------------------------------------------------

        chunks = splitter.split_text(text)

        # --------------------------------------------------
        # 10. Create embedding for every chunk
        # --------------------------------------------------

        for chunk_number, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{filename}_"
                f"page_{page_number + 1}_"
                f"chunk_{chunk_number + 1}"
            )

            embedding = model.encode(
                chunk
            ).tolist()

            metadata = {
                "source": filename,
                "page": page_number + 1,
                "chunk": chunk_number + 1
            }

            # --------------------------------------------------
            # 11. Store in ChromaDB
            # --------------------------------------------------

            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata]
            )

            print(
                "Added:",
                chunk_id
            )


print("\nPDF ingestion completed!")