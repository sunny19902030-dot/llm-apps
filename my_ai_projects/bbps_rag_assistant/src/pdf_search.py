import chromadb
import os
from sentence_transformers import SentenceTransformer


PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHROMA_DIR = os.path.join(
    PROJECT_DIR,
    "chroma_db"
)

# 1. Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to our ChromaDB
client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name="payment_pdf_knowledge"
)

# 3. Get the PDF knowledge collection
collection = client.get_collection(
    name="payment_pdf_knowledge"
)


# 4. Ask a question
question = input("\nAsk your BBPS question: ")


# 5. Convert question into an embedding
query_embedding = model.encode(
    [question]
).tolist()


# 6. Search ChromaDB
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


# 7. Extract results
documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


# 8. Display results
print("\nSEARCH RESULTS\n")


for i in range(len(documents)):

    print("----- RESULT", i + 1, "-----")

    print("Distance:", distances[i])

    print("Source:", metadatas[i]["source"])

    print("Page:", metadatas[i]["page"])

    print("Chunk:", metadatas[i]["chunk"])

    print("\nContent:")
    print(documents[i])

    print()