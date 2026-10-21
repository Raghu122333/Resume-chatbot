
import os
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# 1. Configure the application
# -----------------------------

st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎯",
    layout="centered"
)

st.title("🎯 AI Interview Coach")
st.caption("Ask questions about your resume or practise technical interviews.")

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY was not found. Check your .env file.")
    st.stop()

client = Groq(api_key=api_key)
MODEL_NAME = "openai/gpt-oss-20b"
PDF_PATH = "Raghu_Resume.pdf"


# -----------------------------
# 2. Resume processing
# -----------------------------

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


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

        end = len(text)

        for next_section in sections[i + 1:]:
            next_start = text.find(next_section, start + len(section))
            if next_start != -1:
                end = min(end, next_start)
                break

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


@st.cache_resource
def load_resume():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(
            f"Resume PDF not found: {PDF_PATH}"
        )

    text = extract_text(PDF_PATH)
    chunks = create_chunks(text)

    if not chunks:
        raise ValueError("No resume sections were extracted.")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(chunks)

    return chunks, embedding_model, embeddings


try:
    chunks, embedding_model, embeddings = load_resume()
except Exception as error:
    st.error(f"Could not load the resume: {error}")
    st.stop()


# -----------------------------
# 3. Retrieve relevant chunks
# -----------------------------

def retrieve_chunks(question, top_k=2):
    question_embedding = embedding_model.encode(question)

    similarities = []

    for embedding in embeddings:
        denominator = (
            np.linalg.norm(question_embedding)
            * np.linalg.norm(embedding)
        )

        similarity = (
            np.dot(question_embedding, embedding) / denominator
            if denominator != 0 else 0.0
        )

        similarities.append(float(similarity))

    question_lower = question.lower()

    section_rules = [
        (["project"], "PROJECTS"),
        (["skill"], "TECHNICAL SKILLS"),
        (["education"], "EDUCATION"),
        (["internship"], "INTERNSHIP EXPERIENCE"),
        (["certification"], "CERTIFICATIONS & ACHIEVEMENTS")
    ]

    selected_indices = []

    for keywords, heading in section_rules:
        if any(keyword in question_lower for keyword in keywords):
            selected_indices = [
                i for i, chunk in enumerate(chunks)
                if chunk.startswith(heading)
            ]
            break

    if not selected_indices:
        selected_indices = np.argsort(similarities)[-top_k:][::-1].tolist()

    best_similarity = max(similarities[i] for i in selected_indices)

    if best_similarity < 0.35:
        return [], best_similarity

    return [chunks[i] for i in selected_indices], best_similarity


# -----------------------------
# 4. Groq response helper
# -----------------------------

def ask_groq(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def generate_interview_question(context):
    prompt = f"""
You are a technical interviewer for college placements.

Candidate resume:
{context}

Ask exactly ONE technical interview question based on the
candidate's projects, skills, or internship.

Do not provide the answer.
"""
    return ask_groq(prompt)


def evaluate_answer(question, answer, context):
    prompt = f"""
You are a helpful technical interview evaluator.

Resume context:
{context}

Interview question:
{question}

Candidate's answer:
{answer}

Evaluate the answer honestly. If the candidate admits they
did not implement something, acknowledge their honesty while
assessing how well they addressed the question.

Return:

Score: X/10

What you did well:
- ...

What could be improved:
- ...

Ideal answer:
...
"""
    return ask_groq(prompt)


# -----------------------------
# 5. Interface
# -----------------------------

mode = st.radio(
    "Choose a mode",
    ["Resume Chat", "Interview Practice"],
    horizontal=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "interview_question" not in st.session_state:
    st.session_state.interview_question = None

if "interview_context" not in st.session_state:
    st.session_state.interview_context = None


if mode == "Resume Chat":
    st.subheader("💬 Ask about your resume")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about your resume...")

    if user_question:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        relevant_chunks, score = retrieve_chunks(user_question)

        if not relevant_chunks:
            answer = "I couldn't find sufficiently relevant information in the resume."
        else:
            context = "\n\n".join(relevant_chunks)
            previous_messages = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in st.session_state.chat_history[-6:]
            )

            prompt = f"""
You answer questions about the candidate's resume.

Use only the resume context. Do not invent facts.
If the information is missing, say so.

Resume context:
{context}

Recent conversation:
{previous_messages}

Current question:
{user_question}

Answer:
"""
            try:
                answer = ask_groq(prompt)
            except Exception as error:
                answer = f"Groq request failed: {error}"

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )

        st.rerun()


else:
    st.subheader("🧑‍💻 Technical Interview")

    st.write("Generate a question, answer it, and receive feedback.")

    if st.button("Generate interview question"):
        context = "\n\n".join(chunks)

        try:
            with st.spinner("Generating question..."):
                question = generate_interview_question(context)

            st.session_state.interview_question = question
            st.session_state.interview_context = context

        except Exception as error:
            st.error(f"Could not generate a question: {error}")

    if st.session_state.interview_question:
        st.markdown("### Interview question")
        st.write(st.session_state.interview_question)

        with st.form("answer_form"):
            user_answer = st.text_area(
                "Your answer",
                height=180,
                placeholder="Explain your answer in your own words..."
            )

            submitted = st.form_submit_button("Evaluate my answer")

        if submitted:
            if not user_answer.strip():
                st.warning("Please enter your answer first.")
            else:
                try:
                    with st.spinner("Evaluating your answer..."):
                        evaluation = evaluate_answer(
                            st.session_state.interview_question,
                            user_answer,
                            st.session_state.interview_context
                        )

                    st.markdown("### 📊 Feedback")
                    st.markdown(evaluation)

                except Exception as error:
                    st.error(f"Evaluation failed: {error}")
