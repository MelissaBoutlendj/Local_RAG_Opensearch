import os
# Le proxy de la fac intercepterait les appels vers localhost : on l'exclut.
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer

client = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

# MEME modele que pour l'indexation : deux modeles differents produisent
# des espaces vectoriels incompatibles et la recherche renvoie n'importe quoi.
model = SentenceTransformer("all-MiniLM-L6-v2")

question = "What is a prompt framework?"

# La question suit exactement le meme chemin que les chunks : elle devient
# un vecteur de 384 nombres, comparable a ceux stockes dans l'index.
question_vector = model.encode(question).tolist()

query = {
    "size": 3,                        # nombre de resultats renvoyes
    "query": {
        "knn": {
            "embedding": {            # nom du champ knn_vector dans le mapping
                "vector": question_vector,
                "k": 3                # nombre de voisins cherches par l'algorithme
            }
        }
    }
}

response = client.search(index="documents", body=query)

# response["hits"]["hits"] est la liste des resultats, tries du plus proche au moins proche.
for hit in response["hits"]["hits"]:
    print("score:", hit["_score"])
    print(hit["_source"]["text"][:300])
    print("---")
