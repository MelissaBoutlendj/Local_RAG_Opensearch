import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import ollama
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
model = SentenceTransformer("all-MiniLM-L6-v2")

question = "What is the capital of Mongolia?"

# --- RETRIEVAL : on retrouve les chunks les plus proches de la question ---
question_vector = model.encode(question).tolist()

response = client.search(
    index="documents",
    body={
        "size": 3,
        "query": {"knn": {"embedding": {"vector": question_vector, "k": 3}}}
    }
)

# On colle les 3 chunks bout a bout pour former le contexte.
context = "\n\n".join(hit["_source"]["text"] for hit in response["hits"]["hits"])

# --- AUGMENTATION : le contexte est injecte dans le prompt ---
# La consigne "only the context" est ce qui empeche le modele d'inventer.
prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""

# --- GENERATION : le LLM repond a partir de ce qu'on lui a fourni ---
answer = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": prompt}]
)

print(answer["message"]["content"])
