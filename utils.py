from pypdf import PdfReader


def extract_text(pdf_path):
	reader = PdfReader(pdf_path)
	full_text = ""
	for page in reader.pages:
		full_text+= (page.extract_text() or "") + "\n"
	return full_text



def split_text(text, size=500, overlap=50):
	chunks = []
	start = 0
	while start < len(text):
		chunks.append(text[start:start + size])
		start += size - overlap
	return chunks
