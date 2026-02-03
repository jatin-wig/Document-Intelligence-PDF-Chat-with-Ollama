# Document Intelligence — PDF Chat with Ollama

A production-style local Retrieval-Augmented Generation (RAG) system that enables conversational querying and semantic search over PDF documents.
The application processes documents, generates vector embeddings, and retrieves relevant context to produce grounded responses using a locally hosted LLM via Ollama.

## Demo Video  
[![Watch the video](https://img.youtube.com/vi/taKxQnO6HM0/hqdefault.jpg)](https://www.youtube.com/watch?v=taKxQnO6HM0)

## Features
- Fully local inference using Ollama
- Semantic search powered by FAISS
- Context-aware conversational querying
- Retrieval optimized with Maximum Marginal Relevance (MMR)
- Structured document ingestion pipeline
- Streamlit-based interactive interface
- Workspace reset for controlled lifecycle management
- No external LLM APIs required

## Tech Stack
- LLM: Ollama (Llama3)
- Embeddings: HuggingFace Sentence Transformers
- Vector Store: FAISS
- Framework: LangChain
- Frontend: Streamlit

 ## Project Structure

```bash
pdf-chat/
├── app.py
├── requirements.txt
├── utils/
│   ├── loader.py
│   ├── rag_chain.py
│   └── splitter.py
│   └── vectorstore.py
└── README.md
```

# Setup Instructions

## 1) Clone the Repository
```bash

git clone https://github.com/jatin-wig/Document-Intelligence-PDF-Chat-with-Ollama.git
```


## 2) Install Dependencies
```bash
pip install -r requirements.txt
```

## 3) Install Ollama

```bash

install from https://ollama.com/
```

```bash

ollama pull llama3
```

```bash

ollama run llama3
```

## 4) Run the App
```bash
streamlit run app.py
 ```
or 
```bash
python -m streamlit run app.py 
```

# Built by Jatin Wig
### GitHub: https://github.com/jatin-wig
