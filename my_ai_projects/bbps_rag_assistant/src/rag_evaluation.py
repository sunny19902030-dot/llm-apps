import os

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Find project directory
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
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

collection = client.get_collection(
    name="payment_pdf_knowledge"
)


# --------------------------------------------------
# 4. Evaluation questions
# --------------------------------------------------

questions = [

    # ------------------------------
    # Relevant BBPS questions
    # ------------------------------

    {
        "question": "What are the different biller transaction flows in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is Fetch and Pay in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is Validation and Pay in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "How are BBPS transactions initiated?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is One-time Pay in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is Register and Pay in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is UPMS in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is the role of BBPCU?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is the role of Operating Units in BBPS?",
        "expected": "RELEVANT"
    },

    {
        "question": "What is the objective of BBPS?",
        "expected": "RELEVANT"
    },


    # ------------------------------
    # Clearly unrelated questions
    # ------------------------------

    {
        "question": "How do I reset my Netflix password?",
        "expected": "IRRELEVANT"
    },

    {
        "question": "What is the weather in Mumbai today?",
        "expected": "IRRELEVANT"
    },

    {
        "question": "How do I book a flight to Dubai?",
        "expected": "IRRELEVANT"
    },

    {
        "question": "What is the capital of France?",
        "expected": "IRRELEVANT"
    },

    {
        "question": "How do I make pizza?",
        "expected": "IRRELEVANT"
    }
]


# --------------------------------------------------
# 5. Run evaluation
# --------------------------------------------------

for item in questions:

    question = item["question"]
    expected = item["expected"]

    print("\n========================================")
    print("QUESTION:")
    print(question)

    print("EXPECTED:")
    print(expected)

    print("========================================")

    # Convert question to embedding

    query_embedding = model.encode(
        [question]
    ).tolist()

    # Search ChromaDB

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Display results

    for i in range(len(documents)):

        print(
            f"\nRESULT {i + 1}"
        )

        print(
            "Distance:",
            round(distances[i], 4)
        )

        print(
            "Source:",
            metadatas[i]["source"]
        )

        print(
            "Page:",
            metadatas[i]["page"]
        )

        print(
            "Chunk:",
            metadatas[i]["chunk"]
        )

        print(
            "Content:",
            documents[i][:250].replace(
                "\n",
                " "
            )
        )