import os
import json

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
# 2. MODELS
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
# 3. CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name="payment_pdf_knowledge"
)


# ==========================================
# 4. OLLAMA
# ==========================================

OLLAMA_URL = (
    "http://localhost:11434/api/generate"
)

MODEL = "llama3.2:3b"


# ==========================================
# 5. EVALUATION QUESTIONS
# ==========================================

evaluation_data = [

    {
        "question":
            "What is Fetch and Pay in BBPS?"
    },

    {
        "question":
            "What is Validation and Pay in BBPS?"
    },

    {
        "question":
            "What are the different biller transaction flows in BBPS?"
    },

    {
        "question":
            "What is One-time Pay in BBPS?"
    },

    {
        "question":
            "What is Register and Pay in BBPS?"
    }

]


# ==========================================
# 6. METRIC COUNTERS
# ==========================================

correctness_pass = 0
faithfulness_pass = 0
relevance_pass = 0


# ==========================================
# 7. HELPER FUNCTION
# ==========================================

def call_ollama(prompt):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["response"]


# ==========================================
# 8. RUN EVALUATION
# ==========================================

for item in evaluation_data:

    question = item["question"]

    print("\n\n====================================")
    print("QUESTION")
    print("====================================")

    print(question)


    # --------------------------------------
    # VECTOR SEARCH
    # --------------------------------------

    query_embedding = embedding_model.encode(
        [question]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=10
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]


    # --------------------------------------
    # RERANK
    # --------------------------------------

    pairs = []

    for document in documents:

        pairs.append(
            [
                question,
                document
            ]
        )

    scores = reranker.predict(
        pairs
    )


    reranked = []

    for i in range(len(documents)):

        reranked.append(
            {
                "score": float(scores[i]),
                "document": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            }
        )


    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # --------------------------------------
    # TAKE TOP 3
    # --------------------------------------

    top_results = reranked[:3]


    # --------------------------------------
    # BUILD CONTEXT
    # --------------------------------------

    context_parts = []

    for result in top_results:

        metadata = result["metadata"]

        context_parts.append(
            f"""
Source: {metadata["source"]}
Page: {metadata["page"]}
Chunk: {metadata["chunk"]}

{result["document"]}
"""
        )

    context = "\n".join(
        context_parts
    )


    # ======================================
    # GENERATE ANSWER
    # ======================================

    generation_prompt = f"""
You are a customer-facing BBPS payment assistant.

Answer the question using ONLY the
retrieved document excerpts.

RETRIEVED CONTEXT:

{context}

QUESTION:

{question}

Rules:

1. Answer the question directly.

2. Do NOT start the answer with phrases such as:
   - "According to the document"
   - "According to the retrieved document"
   - "According to the provided context"
   - "The document states"
   - "The retrieved context says"

3. Do NOT mention:
   - retrieved documents
   - chunks
   - vector search
   - embeddings
   - reranking
   - context
   - RAG
   - internal system processing

4. Use natural, customer-friendly language.

5. Keep the answer concise.

6. If the retrieved information contains
   examples, include them when useful.

7. Do not invent information.

8. If the retrieved excerpts do not contain
   enough information to answer the question,
   say:

"I couldn't find enough information to answer that."

9. Do not pretend to know information that is
   not present in the retrieved excerpts.

10. Only provide a source citation when explicitly
    requested by the user.

Return ONLY the customer-facing answer.
"""

    answer = call_ollama(
        generation_prompt
    )


    print("\nANSWER:")
    print(answer)


    # ======================================
    # EVALUATE ANSWER
    # ======================================

    evaluation_prompt = f"""
You are evaluating an AI-generated answer
for a retrieval-augmented generation system.

QUESTION:

{question}

RETRIEVED CONTEXT:

{context}

AI ANSWER:

{answer}

Evaluate three dimensions.

CORRECTNESS:
Does the answer correctly answer the question
based on the retrieved context?

FAITHFULNESS:
Are the important claims in the answer
supported by the retrieved context?

RELEVANCE:
Does the answer directly address the question?

Return ONLY valid JSON:

{{
    "correctness": "PASS or FAIL",
    "faithfulness": "PASS or FAIL",
    "relevance": "PASS or FAIL",
    "explanation": "short explanation"
}}
"""

    evaluation_result = call_ollama(
        evaluation_prompt
    )


    # ======================================
    # PARSE EVALUATION
    # ======================================

    try:

        evaluation = json.loads(
            evaluation_result
        )

    except json.JSONDecodeError:

        print("\nCould not parse evaluator output.")

        print(evaluation_result)

        continue


    print("\nEVALUATION:")

    print(
        json.dumps(
            evaluation,
            indent=4
        )
    )


    # ======================================
    # UPDATE METRICS
    # ======================================

    if evaluation["correctness"] == "PASS":

        correctness_pass += 1

    if evaluation["faithfulness"] == "PASS":

        faithfulness_pass += 1

    if evaluation["relevance"] == "PASS":

        relevance_pass += 1


# ==========================================
# 9. FINAL REPORT
# ==========================================

total = len(evaluation_data)


print("\n\n====================================")
print("END-TO-END RAG EVALUATION")
print("====================================")

print(
    "Correctness:",
    f"{correctness_pass}/{total}",
    f"({correctness_pass / total:.1%})"
)

print(
    "Faithfulness:",
    f"{faithfulness_pass}/{total}",
    f"({faithfulness_pass / total:.1%})"
)

print(
    "Relevance:",
    f"{relevance_pass}/{total}",
    f"({relevance_pass / total:.1%})"
)