# Local RAG with OpenSearch and Ollama

A retrieval-augmented generation system running entirely on local hardware.
No data leaves the machine. Built from scratch to understand each component.

## Architecture

1. **Extraction** — `pypdf` pulls text from PDF documents
2. **Chunking** — text split into 500-character chunks with 50-character overlap
3. **Embedding** — `all-MiniLM-L6-v2` produces 384-dimensional vectors
4. **Indexing** — chunks and vectors stored in OpenSearch (`knn_vector` field)
5. **Retrieval** — hybrid search combining BM25 and k-NN
6. **Generation** — Ollama (llama3.2) answers using only the retrieved context

## Setup

```bash
docker run -d --name opensearch -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "DISABLE_SECURITY_PLUGIN=true" \
  opensearchproject/opensearch:2.11.0

pip install -r requirements.txt
ollama pull llama3.2:1b
```

Create the index mapping and the hybrid search pipeline (see `setup/` or Dev Tools).

```bash
python index.py    # index a document
python rag.py      # ask a question
```

## Findings

**Hybrid search outperforms pure vector search on exact terms.** Querying
"What is the ROSES framework?" with k-NN alone returned generic passages about
frameworks, with the actual definition ranked third (scores clustered 0.54–0.57).
Adding BM25 at a 0.3/0.7 weighting surfaced all three relevant chunks with clear
score separation (0.90 / 0.30 / 0.14). Embeddings capture meaning but miss literal
acronyms; the lexical component recovers them.

**Grounding holds.** Asked a question outside the corpus, the system answers
"I don't know" rather than falling back on the model's parametric knowledge.

## Limitations

- Character-level chunking splits words mid-token
- Documents indexed one request at a time (no `bulk` API)
- Single document per index run
