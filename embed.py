from sentence_transformers import SentenceTransformer
from utils import extract_text, split_text

text = extract_text("prompt_eng.pdf")
chunks = split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)
print(embeddings.shape)

