import os

import chromadb
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CHROMA_DIR = os.path.join(
    PROJECT_DIR,
    "chroma_db"
)


# ==========================================
# 2. LOAD MODELS
# ==========================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Loading reranker...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==========================================
# 3. CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name="payment_pdf_knowledge"
)


# ==========================================
# 4. GET USER QUESTION
# ==========================================

question = input(
    "\nAsk your BBPS question: "
)


# ==========================================
# 5. CREATE QUESTION EMBEDDING
# ==========================================

query_embedding = embedding_model.encode(
    [question]
).tolist()


# ==========================================
# 6. VECTOR SEARCH
# ==========================================

results = collection.query(
    query_embeddings=query_embedding,
    n_results=10
)

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]


# ==========================================
# 7. DISPLAY VECTOR SEARCH RESULTS
# ==========================================

print("\n====================================")
print("VECTOR SEARCH RESULTS")
print("====================================")

for i in range(len(documents)):

    print(f"\nRank: {i + 1}")

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


# ==========================================
# 8. THRESHOLD FILTER
# ==========================================

THRESHOLD = 1.0

candidate_documents = []
candidate_metadatas = []
candidate_distances = []

for i in range(len(documents)):

    if distances[i] <= THRESHOLD:

        candidate_documents.append(
            documents[i]
        )

        candidate_metadatas.append(
            metadatas[i]
        )

        candidate_distances.append(
            distances[i]
        )


if not candidate_documents:

    print("\n====================================")
    print("NO RELEVANT INFORMATION FOUND")
    print("====================================")

    print(
        "I could not find enough information "
        "in the provided documents."
    )

    exit()


# ==========================================
# 9. CREATE QUESTION-DOCUMENT PAIRS
# ==========================================

pairs = []

for document in candidate_documents:

    pairs.append(
        [
            question,
            document
        ]
    )


# ==========================================
# 10. RERANK
# ==========================================

print("\nRunning reranker...")

reranker_scores = reranker.predict(
    pairs
)


# ==========================================
# 11. COMBINE RESULTS
# ==========================================

ranked_results = []

for i in range(
    len(candidate_documents)
):

    ranked_results.append({

        "reranker_score":
            float(reranker_scores[i]),

        "document":
            candidate_documents[i],

        "metadata":
            candidate_metadatas[i],

        "distance":
            candidate_distances[i]

    })


# ==========================================
# 12. SORT BY RERANKER SCORE
# ==========================================

ranked_results.sort(
    key=lambda x: x["reranker_score"],
    reverse=True
)


# ==========================================
# 13. TAKE BEST 3
# ==========================================

top_results = ranked_results[:3]


# ==========================================
# 14. DISPLAY RERANKED RESULTS
# ==========================================

print("\n====================================")
print("RERANKED RESULTS")
print("====================================")

for i, result in enumerate(
    top_results,
    start=1
):

    print(f"\nRank: {i}")

    print(
        "Reranker Score:",
        round(
            result["reranker_score"],
            4
        )
    )

    print(
        "Original Vector Distance:",
        round(
            result["distance"],
            4
        )
    )

    print(
        "Source:",
        result["metadata"]["source"]
    )

    print(
        "Page:",
        result["metadata"]["page"]
    )

    print(
        "Chunk:",
        result["metadata"]["chunk"]
    )

    print(
        "Content:",
        result["document"][:500]
    )


# ==========================================
# 15. BUILD LLM CONTEXT
# ==========================================

context_parts = []

for result in top_results:

    source = result["metadata"]["source"]
    page = result["metadata"]["page"]

    context_parts.append(
        f"""
Source: {source}
Page: {page}

{result["document"]}
"""
    )

context = "\n".join(
    context_parts
)


# ==========================================
# 16. CREATE LLM PROMPT
# ==========================================

prompt = f"""
You are a BBPS payment support assistant.

Answer the user's question using ONLY
the information contained in the retrieved
document excerpts below.

Retrieved document excerpts:

{context}

User question:

{question}

Rules:

1. Do not use your general knowledge.
2. Do not invent information.
3. If the retrieved excerpts do not contain
   enough information to answer the question,
   say:

"I could not find enough information in the
provided documents."

4. Keep the answer simple and clear.
5. Mention the relevant source page when
   possible.
"""

print("\n====================================")
print("CONTEXT SENT TO LLM")
print("====================================")

print(context)

context = "\n".join(
    context_parts
)

print("\n====================================")
print("CONTEXT SENT TO LLM")
print("====================================")

print(context)

print("\nGenerating answer...")


# ==========================================
# 17. SEND TO LOCAL LLM
# ==========================================

print("\nGenerating answer...")


response = requests.post(

    "http://localhost:11434/api/generate",

    json={

        "model": "llama3.2:3b",

        "prompt": prompt,

        "stream": False

    },

    timeout=120
)


response.raise_for_status()


answer = response.json()["response"]


# ==========================================
# 18. DISPLAY ANSWER
# ==========================================

print("\n====================================")
print("AI ANSWER")
print("====================================")

print(answer)