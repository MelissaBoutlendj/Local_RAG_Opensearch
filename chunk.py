from pypdf import PdfReader

reader = PdfReader("prompt_eng.pdf")


full_text = ""
for page in reader.pages:
	full_text += (page.extract_text() or "") + "\n"

def split_text(text, size=500, overlap=50):
	chunks = []
	start = 0
	while start < len(text):
		chunks.append(text[start:start + size])
		start += size - overlap
	return chunks


chunks = split_text(full_text)
print(len(chunks))
print(repr(chunks[0]))
print(repr(chunks[1]))

print(chunks[0][-50:] == chunks[1][:50])
