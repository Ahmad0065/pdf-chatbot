# 📄 PDF Chatbot

An intelligent PDF chatbot that lets you upload one or more PDFs and ask questions about their content — with hybrid retrieval, reranking, conversation memory, and automatic web search fallback for questions outside the document's scope.

## Features

- 📤 **Multi-PDF upload** — chat across multiple documents at once
- 🔍 **Hybrid retrieval** — combines semantic search (FAISS/Qdrant) with keyword search (BM25) for better accuracy
- 🎯 **Reranking** — a cross-encoder reranker refines retrieved chunks before generating an answer
- 💬 **Conversation memory** — ask follow-up questions naturally without repeating context
- 🤖 **Smart query routing** — automatically detects casual chat, in-document questions, and out-of-scope questions
- 🌐 **Web search fallback** — if the document doesn't fully answer a question, relevant info is fetched from the web and clearly labeled as such
- 📚 **Source citations** — every answer shows which document and page it came from
- ☁️ **Cloud vector database** — powered by Qdrant Cloud for persistent, scalable storage

## Tech Stack

| Component | Technology |
|---|---|
| PDF Processing | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | Qdrant Cloud |
| Keyword Search | BM25 |
| Reranker | BAAI/bge-reranker-base |
| LLM | Groq (openai/gpt-oss-20b) |
| Web Search | Tavily |
| UI | Streamlit |

## Architecture

User → Upload PDF → PyMuPDF → Chunking → Embeddings → Qdrant (+ BM25 index)
↓
User Question → Decision Agent (casual / in-scope / out-of-scope)
↓
Hybrid Retrieval (Qdrant + BM25) → Reranker → LLM
↓
Answer sufficient? → No → Web Search → Final Answer

## Project Structure

pdf-chatbot/
├── app/
│ ├── main.py # Streamlit app entrypoint
│ ├── ingestion/ # PDF loading and chunking
│ ├── embeddings/ # Embedding model wrapper
│ ├── retrieval/ # Vector store, BM25, hybrid retrieval, reranker
│ ├── memory/ # Conversation memory
│ ├── agent/ # Decision agent and web search
│ └── llm/ # Groq client
├── data/
│ └── raw_pdfs/ # Uploaded PDFs are stored here
├── requirements.txt
└── README.md

## Setup

1. Clone the repository
```bash
   git clone <your-repo-url>
   cd pdf-chatbot
```

2. Create a virtual environment and install dependencies
```bash
   python -m venv venv
   venv\Scripts\activate       # Windows
   source venv/bin/activate    # macOS/Linux
   pip install -r requirements.txt
```

3. Set up environment variables — create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

4. Run the app
```bash
   streamlit run app/main.py
```

## Usage

1. Upload one or more PDFs from the sidebar
2. Ask questions in the chat box
3. Expand "Sources" under any answer to see which document/page it came from
4. Ask follow-up questions naturally — the chatbot remembers context

## Future Improvements

- Deduplication check for re-uploaded PDFs
- Support for scanned/image-based PDFs via OCR
- User authentication for multi-user deployments