from pypdf import  PdfReader 

reader = PdfReader("prompt_eng.pdf")
print(len(reader.pages))
print(reader.pages[0].extract_text())

# cut it into 500 character
print(reader.pages[0].extract_text()[:500])

for i, page in enumerate(reader.pages):
	t = page.extract_text() or ""
	print(i, len(t))
