import os

import chromadb
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
# 4. EVALUATION DATASET
# ==========================================

evaluation_data = [

    {
        "question": "What is Fetch and Pay in BBPS?",
        "expected_evidence": [
            "Fetch and Pay",
            "financial transaction post fetching the bill"
        ]
    },

    {
        "question": "What is Validation and Pay in BBPS?",
        "expected_evidence": [
            "Validation and Pay",
            "financial transaction post validation of their identifier"
        ]
    },

    {
        "question": "What are the different biller transaction flows in BBPS?",
        "expected_evidence": [
            "Fetch and Pay",
            "Validation and Pay",
            "Validation and Pay, basis plan",
            "Direct Pay"
        ]
    },

    {
        "question": "What is One-time Pay in BBPS?",
        "expected_evidence": [
            "One-time pay",
            "unique identifier",
            "proceeds to initiate the bill payment"
        ]
    },

    {
        "question": "What is Register and Pay in BBPS?",
        "expected_evidence": [
            "Register and Pay",
            "standing instructions",
            "recurring payments"
        ]
    }
]


# ==========================================
# 5. EVALUATION VARIABLES
# ==========================================

vector_hits_at_1 = 0
vector_hits_at_3 = 0
vector_hits_at_5 = 0

reranker_hits_at_1 = 0
reranker_hits_at_3 = 0
reranker_hits_at_5 = 0

vector_reciprocal_ranks = []
reranker_reciprocal_ranks = []


# ==========================================
# 6. HELPER FUNCTIONS
# ==========================================

def normalize_text(text):

    return " ".join(
        text.lower().split()
    )


def contains_expected_evidence(
    document,
    expected_evidence
):

    document_normalized = normalize_text(
        document
    )

    for evidence in expected_evidence:

        evidence_normalized = normalize_text(
            evidence
        )

        if evidence_normalized not in document_normalized:

            return False

    return True


# ==========================================
# 7. RUN EVALUATION
# ==========================================

for item in evaluation_data:

    question = item["question"]

    print("\n====================================")
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
    # VECTOR RANK
    # --------------------------------------

    vector_rank = None

    for rank, document in enumerate(
        documents,
        start=1
    ):

        if contains_expected_evidence(
            document,
            item["expected_evidence"]
        ):

            vector_rank = rank

            break


    # --------------------------------------
    # VECTOR METRICS
    # --------------------------------------

    if vector_rank is not None:

        if vector_rank <= 1:

            vector_hits_at_1 += 1

        if vector_rank <= 3:

            vector_hits_at_3 += 1

        if vector_rank <= 5:

            vector_hits_at_5 += 1

        vector_reciprocal_ranks.append(
            1 / vector_rank
        )

    else:

        vector_reciprocal_ranks.append(0)


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


    # --------------------------------------
    # COMBINE
    # --------------------------------------

    reranked = []

    for i in range(
        len(documents)
    ):

        reranked.append(
            {
                "score": float(scores[i]),
                "document": documents[i],
                "metadata": metadatas[i]
            }
        )


    # --------------------------------------
    # SORT
    # --------------------------------------

    reranked.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # --------------------------------------
    # RERANKER RANK
    # --------------------------------------

    reranker_rank = None

    for rank, result in enumerate(
        reranked,
        start=1
    ):

        if contains_expected_evidence(
            result["document"],
            item["expected_evidence"]
        ):

            reranker_rank = rank

            break


    # --------------------------------------
    # RERANKER METRICS
    # --------------------------------------

    if reranker_rank is not None:

        if reranker_rank <= 1:

            reranker_hits_at_1 += 1

        if reranker_rank <= 3:

            reranker_hits_at_3 += 1

        if reranker_rank <= 5:

            reranker_hits_at_5 += 1

        reranker_reciprocal_ranks.append(
            1 / reranker_rank
        )

    else:

        reranker_reciprocal_ranks.append(0)


    # --------------------------------------
    # DISPLAY RESULT
    # --------------------------------------

    print(
        "\nVector Rank:",
        vector_rank
    )

    print(
        "Reranker Rank:",
        reranker_rank
    )


# ==========================================
# 8. CALCULATE METRICS
# ==========================================

total = len(evaluation_data)


vector_hit_1 = (
    vector_hits_at_1 / total
)

vector_hit_3 = (
    vector_hits_at_3 / total
)

vector_hit_5 = (
    vector_hits_at_5 / total
)


reranker_hit_1 = (
    reranker_hits_at_1 / total
)

reranker_hit_3 = (
    reranker_hits_at_3 / total
)

reranker_hit_5 = (
    reranker_hits_at_5 / total
)


vector_mrr = (
    sum(vector_reciprocal_ranks) / total
)

reranker_mrr = (
    sum(reranker_reciprocal_ranks) / total
)


# ==========================================
# 9. FINAL REPORT
# ==========================================

print("\n\n====================================")
print("RAG RETRIEVAL EVALUATION")
print("====================================")


print("\nVECTOR SEARCH")

print(
    "Hit@1:",
    round(vector_hit_1, 3)
)

print(
    "Hit@3:",
    round(vector_hit_3, 3)
)

print(
    "Hit@5:",
    round(vector_hit_5, 3)
)

print(
    "MRR:",
    round(vector_mrr, 3)
)


print("\nRERANKER")

print(
    "Hit@1:",
    round(reranker_hit_1, 3)
)

print(
    "Hit@3:",
    round(reranker_hit_3, 3)
)

print(
    "Hit@5:",
    round(reranker_hit_5, 3)
)

print(
    "MRR:",
    round(reranker_mrr, 3)
)