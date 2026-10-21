from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
def extract_text(pdf_path):
    #pdf_path="Raghu_Resume.pdf"
    reader=PdfReader(pdf_path)
    text=""
    for page in reader.pages:
        text+=page.extract_text() or ""
    return text

def create_chunks(text):

    sections = [
        "EDUCATION",
        "TECHNICAL SKILLS",
        "INTERNSHIP EXPERIENCE",
        "PROJECTS",
        "CERTIFICATIONS & ACHIEVEMENTS"
    ]

    chunks=[]
    for i,section in enumerate(sections):
        start=text.find(section)

        if start==-1:
            continue
        if i+1<len(sections):
            end=text.find(sections[i+1])

            if end==-1:
                end=len(text)

        else:
            end=len(text)
        
        chunk=text[start:end].strip()
        chunks.append(chunk)
    return chunks

text=extract_text("Raghu_Resume.pdf")

chunks=create_chunks(text)

for i,chunk in enumerate(chunks):
    print(f"\n----chunk{i+1}----")
    print(chunk)

model=SentenceTransformer("all-MiniLM-L6-v2")
embeddings=model.encode(chunks)

question=input("Enter question: ")

question_embedding=model.encode(question)

similarities=[]

for embedding in embeddings:
    similarity=np.dot(question_embedding,embedding)/(
        np.linalg.norm(question_embedding)*
        np.linalg.norm(embedding)
    )
    similarities.append(similarity)

best_index=np.argmax(similarities)

print("\nQuestion")
print(question)

print("\nmost relevant answer")
print(chunks[best_index])

print("\nsimilarity score")
print(similarities[best_index])
