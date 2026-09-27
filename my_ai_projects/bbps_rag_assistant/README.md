\# AI-Powered BBPS Payment Knowledge Assistant



A domain-specific Retrieval-Augmented Generation (RAG) system for answering questions from Bharat Bill Payment System (BBPS) procedural documentation.



The project combines \*\*PDF ingestion, text chunking, embeddings, ChromaDB vector search, CrossEncoder reranking, and a local LLM\*\* to retrieve relevant payment information and generate answers grounded in the provided BBPS documents.



\---



\## 1. Problem Statement



Payment and fintech platforms often depend on detailed procedural and regulatory documentation.



Finding a specific answer from lengthy BBPS documentation can require manually searching through multiple pages and sections.



This project explores how an AI-powered knowledge assistant can make this information easier to retrieve and explain.



\### Example



A user can ask:



> What is Fetch and Pay in BBPS?



Instead of manually searching the BBPS documentation, the system:



1\. Converts the question into an embedding.

2\. Searches the vector database.

3\. Retrieves relevant document chunks.

4\. Reranks the retrieved chunks using a CrossEncoder.

5\. Provides the most relevant context to a local LLM.

6\. Generates a grounded answer.



\---



\## 2. Business Use Case



The architecture can support potential use cases such as:



\* Payment operations knowledge assistants

\* Customer-support knowledge systems

\* BBPS product documentation assistants

\* Internal fintech knowledge search

\* Payment-process support

\* Regulatory and procedural document search



The objective is to reduce the effort required to locate relevant information in lengthy payment documentation while keeping answers grounded in approved source material.



\---



\## 3. Solution Architecture



```text

&#x20;                BBPS PDF Documents

&#x20;                        |

&#x20;                        v

&#x20;                PDF Text Extraction

&#x20;                        |

&#x20;                        v

&#x20;               Text Chunking + Metadata

&#x20;                        |

&#x20;                        v

&#x20;             Sentence Transformer

&#x20;                 Embeddings

&#x20;                        |

&#x20;                        v

&#x20;                   ChromaDB

&#x20;                        |

&#x20;                        v

&#x20;                 Vector Search

&#x20;                   Top-K = 10

&#x20;                        |

&#x20;                        v

&#x20;                Threshold Filtering

&#x20;                        |

&#x20;                        v

&#x20;             CrossEncoder Reranker

&#x20;                        |

&#x20;                        v

&#x20;                   Top Results

&#x20;                        |

&#x20;                        v

&#x20;               Context Construction

&#x20;                        |

&#x20;                        v

&#x20;                 Local LLM

&#x20;                Llama 3.2 3B

&#x20;                        |

&#x20;                        v

&#x20;               Customer-facing Answer

```



\---



\## 4. RAG Pipeline



\### Retrieval



The user's question is converted into an embedding using:



```text

all-MiniLM-L6-v2

```



The embedding is compared against document embeddings stored in ChromaDB.



The system initially retrieves the top 10 candidate chunks.



\### Filtering



A distance threshold is applied to remove candidates considered insufficiently similar.



\### Reranking



The remaining candidates are passed to:



```text

cross-encoder/ms-marco-MiniLM-L-6-v2

```



The CrossEncoder evaluates the question and document together and produces a relevance score.



The highest-ranked results are then selected for LLM generation.



\### Generation



The selected document excerpts are supplied to a local:



```text

Llama 3.2 3B

```



model through Ollama.



The LLM is instructed to answer using the retrieved information rather than inventing unsupported information.



\---



\## 5. Why Reranking?



Initial vector retrieval is optimized for fast candidate retrieval.



However, the most relevant document may not always appear at rank #1.



The project therefore uses a two-stage retrieval architecture:



```text

Question

&#x20;  |

&#x20;  v

Vector Search

&#x20;  |

Top 10 candidates

&#x20;  |

&#x20;  v

CrossEncoder

&#x20;  |

&#x20;  v

Reranked results

&#x20;  |

Top results

&#x20;  |

&#x20;  v

LLM

```



This allows the system to combine the speed of vector search with the more detailed relevance assessment of a CrossEncoder.



\---



\## 6. Evaluation



The project includes a 5-question evaluation set based on BBPS transaction-flow concepts.



Questions include:



1\. What is Fetch and Pay in BBPS?

2\. What is Validation and Pay in BBPS?

3\. What are the different biller transaction flows in BBPS?

4\. What is One-time Pay in BBPS?

5\. What is Register and Pay in BBPS?



\### Vector Search vs Reranking



| Metric | Vector Search | With Reranker |

| ------ | ------------: | ------------: |

| Hit@1  |           60% |          100% |

| Hit@3  |           80% |          100% |

| Hit@5  |           80% |          100% |

| MRR    |         0.729 |         1.000 |



\*\*Evaluation scope:\*\* 5-question test set.



These results demonstrate that, for this test set, reranking improved the position of the known answer-bearing chunks.



For example:



```text

Fetch and Pay

Vector rank:    2

Reranker rank:  1



Register and Pay

Vector rank:    7

Reranker rank:  1

```



The results should not be interpreted as a universal benchmark because the evaluation set is intentionally small.



\---



\## 7. Answer Evaluation



The project also evaluates generated answers using three dimensions:



\### Correctness



Does the answer correctly answer the user's question?



\### Faithfulness



Are the important claims supported by the retrieved context?



\### Relevance



Does the answer directly address the user's question?



The project also identified an additional product-quality dimension:



\### Customer Experience



A technically correct answer can still produce poor customer experience if it exposes internal RAG terminology.



For example:



```text

Poor CX:



"According to the retrieved document excerpts..."



Better CX:



"Fetch and Pay is a BBPS transaction flow where

the customer makes the payment after fetching the bill."

```



The system is therefore being evolved toward separating:



```text

Internal retrieval / audit information

&#x20;               +

Customer-facing response

```



\---



\## 8. Technology Stack



| Component          | Technology                               |

| ------------------ | ---------------------------------------- |

| Language           | Python                                   |

| PDF extraction     | pypdf                                    |

| Text splitting     | LangChain RecursiveCharacterTextSplitter |

| Embeddings         | Sentence Transformers                    |

| Vector database    | ChromaDB                                 |

| Reranking          | Sentence Transformers CrossEncoder       |

| LLM                | Llama 3.2 3B                             |

| Local LLM runtime  | Ollama                                   |

| HTTP communication | Requests                                 |

| Version control    | Git / GitHub                             |



\---



\## 9. Project Structure



```text

bbps\_rag\_assistant/

|

├── README.md

├── requirements.txt

├── .gitignore

|

├── data/

|   └── PDF documents

|

├── chroma\_db/

|   └── Local vector database

|

└── src/

&#x20;   ├── pdf\_reader.py

&#x20;   ├── pdf\_to\_chroma.py

&#x20;   ├── pdf\_search.py

&#x20;   ├── rag\_payment\_assistant.py

&#x20;   ├── rag\_evaluation.py

&#x20;   ├── reranker\_test.py

&#x20;   ├── evaluate\_reranker.py

&#x20;   └── answer\_evaluator.py

```



PDF documents and the local ChromaDB database are excluded from Git through `.gitignore`.



\---



\## 10. Installation



Clone the repository:



```bash

git clone https://github.com/sunny19902030-dot/llm-apps.git

```



Navigate to the project:



```bash

cd awesome-llm-apps/my\_ai\_projects/bbps\_rag\_assistant

```



Install dependencies:



```bash

pip install -r requirements.txt

```



Make sure Ollama is installed and the required model is available:



```bash

ollama list

```



The project currently uses:



```text

llama3.2:3b

```



\---



\## 11. Build the Vector Database



Place the required PDF documents inside:



```text

data/

```



Run:



```bash

python src/pdf\_to\_chroma.py

```



The script:



1\. Reads the PDFs.

2\. Extracts text.

3\. Splits the text into chunks.

4\. Generates embeddings.

5\. Stores the embeddings and metadata in ChromaDB.



\---



\## 12. Run the Payment Assistant



From the repository root:



```bash

python my\_ai\_projects\\bbps\_rag\_assistant\\src\\rag\_payment\_assistant.py

```



Example question:



```text

What is Fetch and Pay in BBPS?

```



\---



\## 13. Run Reranker Evaluation



From the repository root:



```bash

python my\_ai\_projects\\bbps\_rag\_assistant\\src\\evaluate\_reranker.py

```



This evaluates vector retrieval and reranked retrieval against the project's test questions.



\---



\## 14. Run Answer Evaluation



The answer evaluator evaluates generated responses for:



\* Correctness

\* Faithfulness

\* Relevance



Example evaluation output:



```json

{

&#x20;   "correctness": "PASS",

&#x20;   "faithfulness": "PASS",

&#x20;   "relevance": "PASS"

}

```



\---



\## 15. Current Limitations



This is an experimental RAG system and has several limitations.



\### Small evaluation dataset



The current retrieval evaluation uses only five questions.



Larger evaluation datasets are required before making broader performance claims.



\### Chunking



The current implementation uses fixed-size recursive chunking.



A logical business concept can sometimes span multiple chunks.



Chunking strategy is therefore an active area of experimentation.



\### Source coverage



The quality of answers depends on the documents supplied to the system.



The LLM cannot reliably answer questions when the required information is absent from the retrieved context.



\### Local LLM



The project currently uses a local 3B parameter model through Ollama.



Larger or specialized models may produce different answer quality and latency characteristics.



\---



\## 16. Current Development Roadmap



```text

\[x] PDF ingestion

\[x] Text chunking

\[x] Embedding generation

\[x] ChromaDB vector search

\[x] Top-K retrieval

\[x] Threshold filtering

\[x] CrossEncoder reranking

\[x] Retrieval evaluation

\[x] LLM answer generation

\[x] Answer evaluation

\[x] GitHub project packaging



\[ ] Chunking strategy experiment

\[ ] Customer-experience evaluation

\[ ] Improved citation handling

\[ ] Advanced retrieval

\[ ] Hybrid search

\[ ] Query transformation

\[ ] Agent-based payment assistant

\[ ] MCP integration

\[ ] Multi-agent payment intelligence platform

```



\---



\## 17. Key Learning



This project demonstrates an important principle of production RAG systems:



> Retrieval quality is not determined by the LLM alone.



The complete system depends on:



```text

Document Quality

&#x20;     +

Chunking

&#x20;     +

Embeddings

&#x20;     +

Retrieval

&#x20;     +

Reranking

&#x20;     +

Context Construction

&#x20;     +

LLM Generation

&#x20;     +

Evaluation

&#x20;     =

RAG System Quality

```



The project is therefore being developed iteratively through measurable experiments rather than assuming that a particular model or architecture is automatically better.



\---



\## 18. Portfolio Summary



\*\*AI-Powered BBPS Payment Knowledge Assistant\*\*



Built a domain-specific RAG system for BBPS payment documentation using Python, ChromaDB, Sentence Transformers, CrossEncoder reranking, and a locally hosted Llama model. Implemented PDF ingestion, vector retrieval, relevance filtering, reranking, grounded answer generation, and automated evaluation. On a 5-question test set, reranking improved Hit@1 from 60% to 100% and MRR from 0.729 to 1.000.



