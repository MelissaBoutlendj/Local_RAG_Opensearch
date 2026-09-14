import os
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import ollama
import streamlit as st
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

# @st.cache_resource : Streamlit relance tout le script a chaque interaction.
# Sans ce cache, le modele serait recharge a chaque question.
@st.cache_resource
def load():
    client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, model

client, model = load()

st.title("Local RAG")

question = st.text_input("Your question")

if question:
    vector = model.encode(question).tolist()

    response = client.search(
        index="documents",
        params={"search_pipeline": "nlp-search-pipeline"},
        body={
            "size": 3,
            "query": {
                "hybrid": {
                    "queries": [
                        {"match": {"text": question}},
                        {"knn": {"embedding": {"vector": vector, "k": 3}}}
                    ]
                }
            }
        }
    )

    hits = response["hits"]["hits"]
    context = "\n\n".join(h["_source"]["text"] for h in hits)

    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""

    with st.spinner("Thinking..."):
        answer = ollama.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}]
        )

    st.write(answer["message"]["content"])

    st.subheader("Sources")
    for h in hits:
        with st.expander(f"Score {h['_score']:.3f}"):
            st.text(h["_source"]["text"])
