import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from utils import extract_text, split_text

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

text = extract_text("prompt_eng.pdf")
chunks = split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    client.index(
        index="documents",
        id=i,
        body={
            "text": chunk,
            "embedding": embedding.tolist(),
            "filename": "prompt_eng.pdf"
        }
    )

client.indices.refresh(index="documents")
print("indexed", len(chunks), "chunks")
