import chromadb
from sentence_transformers import SentenceTransformer


# 1. Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# 2. Connect to our ChromaDB
client = chromadb.PersistentClient(
    path="./chroma_db"
)


# 3. Get the PDF knowledge collection
collection = client.get_collection(
    name="payment_pdf_knowledge"
)


# 4. Ask a question
question = "What is the process for handling a pending BBPS transaction?"


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