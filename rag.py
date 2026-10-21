import os
import time
import numpy as np

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# 1. LOAD GROQ API
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=groq_api_key)


# ============================================================
# 2. EXTRACT PDF TEXT
# ============================================================

def extract_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


# ============================================================
# 3. CREATE RESUME CHUNKS
# ============================================================

def create_chunks(text):

    sections = [
        "EDUCATION",
        "TECHNICAL SKILLS",
        "INTERNSHIP EXPERIENCE",
        "PROJECTS",
        "CERTIFICATIONS & ACHIEVEMENTS"
    ]

    chunks = []

    for i, section in enumerate(sections):

        start = text.find(section)

        if start == -1:
            continue

        if i + 1 < len(sections):

            end = text.find(sections[i + 1])

            if end == -1:
                end = len(text)

        else:

            end = len(text)

        chunk = text[start:end].strip()

        chunks.append(chunk)

    return chunks


# ============================================================
# 4. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(question, chunks, embeddings, model, top_k=2):

    question_embedding = model.encode(question)

    similarities = []

    for embedding in embeddings:

        similarity = np.dot(
            question_embedding,
            embedding
        ) / (
            np.linalg.norm(question_embedding)
            *
            np.linalg.norm(embedding)
        )

        similarities.append(similarity)

    question_lower = question.lower()


    # --------------------------------------------------------
    # Direct section matching
    # --------------------------------------------------------

    if "project" in question_lower:

        project_index = next(
            i for i, chunk in enumerate(chunks)
            if chunk.startswith("PROJECTS")
        )

        best_indices = [project_index]


    elif "skill" in question_lower:

        skill_index = next(
            i for i, chunk in enumerate(chunks)
            if chunk.startswith("TECHNICAL SKILLS")
        )

        best_indices = [skill_index]


    elif "education" in question_lower:

        education_index = next(
            i for i, chunk in enumerate(chunks)
            if chunk.startswith("EDUCATION")
        )

        best_indices = [education_index]


    elif "internship" in question_lower:

        internship_index = next(
            i for i, chunk in enumerate(chunks)
            if chunk.startswith("INTERNSHIP EXPERIENCE")
        )

        best_indices = [internship_index]


    # --------------------------------------------------------
    # Otherwise use semantic similarity
    # --------------------------------------------------------

    else:

        best_indices = np.argsort(similarities)[-top_k:][::-1]


    # --------------------------------------------------------
    # Similarity threshold
    # --------------------------------------------------------

    best_similarity = similarities[best_indices[0]]

    if best_similarity < 0.35:

        return [], best_similarity


    selected_chunks = [
        chunks[i] for i in best_indices
    ]

    return selected_chunks, best_similarity


# ============================================================
# 5. GENERATE INTERVIEW QUESTION
# ============================================================

def generate_interview_question(context):

    interview_prompt = f"""
You are a technical interviewer.

Use the following resume context:

{context}

Ask ONE technical interview question based on the
candidate's projects, technical skills, internship,
or experience.

The question should be suitable for a college placement
technical interview.

Do not provide the answer.

Do not ask multiple questions.

Question:
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": interview_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# 6. EVALUATE INTERVIEW ANSWER
# ============================================================

def evaluate_answer(question, answer, context):

    evaluation_prompt = f"""
You are an AI technical interview evaluator.

The candidate is answering a technical interview question
based on their resume.

Resume context:

{context}

Interview question:

{question}

Candidate's answer:

{answer}

Evaluate the candidate's answer.

Give the evaluation in exactly this format:

Score: X/10

What you did well:
- Point 1
- Point 2

What could be improved:
- Point 1
- Point 2

Ideal answer:
Write a concise but technically correct ideal answer.

Be honest but helpful.

Do not give a high score unless the answer is technically
correct and sufficiently detailed.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ]
    )

    return response.choices[0].message.content


# ============================================================
# 7. LOAD RESUME
# ============================================================

text = extract_text("Raghu_Resume.pdf")

chunks = create_chunks(text)


# ============================================================
# 8. CREATE EMBEDDINGS
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Resume loaded successfully.")


# ============================================================
# 9. CONVERSATION HISTORY
# ============================================================

conversation_history = []


# ============================================================
# 10. MAIN CHATBOT LOOP
# ============================================================

while True:

    question = input(
        "\nAsk a question about the resume "
        "(or type 'interview'): "
    )


    # --------------------------------------------------------
    # EXIT
    # --------------------------------------------------------

    if question.lower() == "exit":

        print("Chatbot closed.")

        break


    # ========================================================
    # INTERVIEW MODE
    # ========================================================

    if question.lower() == "interview":

        # Use the complete resume as context
        context = "\n\n".join(chunks)

        try:

            # ------------------------------------------------
            # Generate interview question
            # ------------------------------------------------

            interview_question = generate_interview_question(
                context
            )

            print("\nInterview Question:")

            print(interview_question)


            # ------------------------------------------------
            # Get candidate answer
            # ------------------------------------------------

            user_answer = input(
                "\nYour answer: "
            )


            # ------------------------------------------------
            # Evaluate candidate answer
            # ------------------------------------------------

            evaluation = evaluate_answer(
                interview_question,
                user_answer,
                context
            )


            print("\nEvaluation:")

            print(evaluation)


        except Exception as e:

            print(
                "\nError during interview:"
            )

            print(e)


        continue


    # ========================================================
    # NORMAL RAG MODE
    # ========================================================

    relevant_chunks, best_similarity = retrieve_chunks(
        question,
        chunks,
        embeddings,
        model
    )


    # --------------------------------------------------------
    # Information not found
    # --------------------------------------------------------

    if not relevant_chunks:

        print("\nAnswer:")

        print(
            "I couldn't find that information in the resume."
        )

        continue


    # --------------------------------------------------------
    # Combine retrieved chunks
    # --------------------------------------------------------

    context = "\n\n".join(relevant_chunks)


    # --------------------------------------------------------
    # Conversation history
    # --------------------------------------------------------

    conversation_text = ""

    for message in conversation_history:

        conversation_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )


    # --------------------------------------------------------
    # Create RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are a helpful assistant answering questions about
Raghu's resume.

Answer the user's question using ONLY the provided
resume context.

If the answer is not present in the context, say:

"I couldn't find that information in the resume."

Do not invent or assume information.

Previous conversation:

{conversation_text}

Resume context:

{context}

Current question:

{question}

Answer:
"""


    # ========================================================
    # GROQ WITH RETRY
    # ========================================================

    response = None

    for attempt in range(3):

        try:

            response = client.chat.completions.create(

                model="openai/gpt-oss-20b",

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            print("\nGroq response received.")

            break


        except Exception as e:

            print(
                f"\nAttempt {attempt + 1} failed: {e}"
            )


            if attempt < 2:

                wait_time = 5 * (2 ** attempt)

                print(
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)


            else:

                print(
                    "Groq is currently unavailable."
                )


    # ========================================================
    # DISPLAY ANSWER
    # ========================================================

    if response:

        answer = response.choices[0].message.content

        print("\nAnswer:")

        print(answer)


        # ----------------------------------------------------
        # Save conversation
        # ----------------------------------------------------

        conversation_history.append({
            "role": "user",
            "content": question
        })

        conversation_history.append({
            "role": "assistant",
            "content": answer
        })