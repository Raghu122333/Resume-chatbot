# Resume RAG Chatbot & AI Interview Coach 🤖

An AI-powered application that allows users to interact with their resume, retrieve relevant information, practice interview questions, and evaluate their answers using Generative AI and Retrieval-Augmented Generation (RAG).

## 🚀 Features

### 📄 Resume Chatbot
- Upload or use a resume in PDF format.
- Extract text from the resume using `pypdf`.
- Split resume content into smaller chunks.
- Generate semantic embeddings using Sentence Transformers.
- Retrieve relevant resume sections using cosine similarity.
- Ask questions about skills, education, projects, and experience.

### 🎯 AI Interview Coach
- Generate interview questions based on resume content.
- Practice answering interview questions.
- Evaluate answers using an LLM.
- Receive feedback to improve technical knowledge and interview performance.
- Maintain conversation history during practice.

### 🧠 AI & Retrieval
- **RAG architecture** for context-aware responses.
- `all-MiniLM-L6-v2` for text embeddings.
- NumPy for cosine similarity-based retrieval.
- Groq API with `openai/gpt-oss-20b` for AI-generated responses and feedback.

### 🖥️ User Interface
- Interactive Streamlit frontend.
- Resume Chat and Interview Practice modes.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Frontend and user interface |
| pypdf | PDF text extraction |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Semantic representation of text |
| NumPy | Similarity calculations |
| Groq API | LLM-powered answers and feedback |
| Git & GitHub | Version control and project hosting |

## 🏗️ Project Structure

```text
Resume-RAG-Chatbot/
│
├── app.py                 # Streamlit frontend
├── rag.py                 # RAG and retrieval logic
├── Raghu_Resume.pdf       # Resume PDF (local input)
├── requirements.txt       # Python dependencies
├── .env                   # API key configuration (not committed)
├── .gitignore             # Files excluded from Git
└── README.md              # Project documentation
```

*Adjust the file structure if your actual project differs.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Resume-RAG-Chatbot.git
cd Resume-RAG-Chatbot
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a `requirements.txt` file, install the main dependencies:

```bash
pip install streamlit pypdf sentence-transformers numpy groq python-dotenv
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own API key. Never upload your API key to GitHub.

### 5. Add your resume

Place your resume PDF in the expected location and update the filename or file path in the code if required.

### 6. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 🔄 How It Works

1. **Document ingestion:** Extract text from the resume PDF.
2. **Text chunking:** Divide the extracted text into smaller sections.
3. **Embedding generation:** Convert the text chunks into numerical vectors.
4. **Semantic retrieval:** Compare the query embedding with resume chunks using cosine similarity.
5. **Context construction:** Select relevant sections to provide context to the language model.
6. **AI response generation:** Use the Groq API to generate context-aware answers.
7. **Interview practice:** Generate questions and evaluate user answers with constructive feedback.

## 🔐 Security

- Store API keys in environment variables.
- Exclude `.env`, virtual environments, and Python cache files from Git.
- Avoid publishing personal information contained in your resume.
- Do not commit private credentials or sensitive documents.

## 🎯 Future Enhancements

- Support multiple resume formats.
- Add job-description matching and skill-gap analysis.
- Provide interview performance scores and progress tracking.
- Add difficulty levels for interview questions.
- Improve retrieval with a vector database.
- Deploy the application to a cloud hosting platform.

## 👨‍💻 Author

**Raghu**

AI/ML and Generative AI Project

Built with Python, RAG, semantic search, and Large Language Models.

## 📌 Key Learning Outcomes

- Implementing a basic RAG pipeline.
- Extracting and processing PDF documents.
- Creating embeddings and performing semantic retrieval.
- Integrating an LLM API into a Python application.
- Building an interactive AI application using Streamlit.
- Managing project dependencies and secrets securely with Git.
