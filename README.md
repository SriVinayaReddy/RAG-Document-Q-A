# 📄 RAG Document Q&A using LangChain and Groq

A Retrieval-Augmented Generation (RAG) application built with Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Groq LLM.

## 🚀 Features

- Upload PDF documents
- Extract and split text into chunks
- Generate vector embeddings using HuggingFace
- Store embeddings using FAISS
- Retrieve relevant document chunks
- Answer user questions using Groq LLM
- Simple Streamlit web interface

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- Groq API
- HuggingFace Embeddings
- FAISS
- PyPDFLoader

## 📂 Project Structure

```
RAG-Document-Q-A/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env (not uploaded)
```

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Document-Q-A.git
cd RAG-Document-Q-A
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GROQ_API_KEY=your_api_key
```

Run the application

```bash
streamlit run app.py
```

## 📸 Output

Ask questions about uploaded PDF documents and receive context-aware answers powered by RAG.

## 📚 Learning Outcomes

- Retrieval-Augmented Generation (RAG)
- Vector Databases (FAISS)
- Embedding Models
- Prompt Engineering
- LangChain
- Streamlit Deployment

## 👨‍💻 Author

**Sri Vinaya Reddy**

GitHub: https://github.com/SriVinayaReddy
