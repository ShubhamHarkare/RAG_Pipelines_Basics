# RAG Pipeline with Hugging Face & Inngest

A production-ready Retrieval-Augmented Generation (RAG) pipeline using **100% open-source models** from Hugging Face, eliminating costly API fees. Built with FastAPI, Inngest workflows, and Qdrant vector database.

## 🌟 Key Features

- **Zero API Costs**: Uses Hugging Face models instead of OpenAI ($0/month vs $20+/month)
- **Event-Driven Architecture**: Powered by Inngest for reliable workflow orchestration
- **Vector Search**: Fast similarity search with Qdrant database
- **Streamlit UI**: User-friendly interface for PDF upload and querying
- **Open Source Stack**: Sentence Transformers + Mistral-7B-Instruct
- **Production Ready**: Error handling, logging, and async processing

## 🏗️ Architecture

```
PDF Upload → Chunk & Embed → Store in Qdrant → Query → Retrieve Context → Generate Answer
     ↓            ↓                ↓                ↓           ↓              ↓
Streamlit    Sentence         Vector DB      User Query   Top-K      Mistral-7B
              Transformers                               Search
```

## 📋 Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- Hugging Face account (free)
- 4GB+ RAM

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/ShubhamHarkare/RAG_Pipelines_Basics.git
cd RAG_Pipelines_Basics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Hugging Face token
# Get token at: https://huggingface.co/settings/tokens
```

Your `.env` file should look like:
```env
HUGGINGFACE_API_TOKEN=hf_your_token_here
```

### 3. Start Qdrant Vector Database

```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant

# Or using Docker Compose (recommended)
docker-compose up -d
```

### 4. Start Inngest Dev Server

In a new terminal:
```bash
npx inngest-cli@latest dev
```

This starts the Inngest dev server at `http://localhost:8288`

### 5. Run the FastAPI Backend

In another terminal:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 6. Launch Streamlit UI

In another terminal:
```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501`

## 📁 Project Structure

```
RAG_Pipelines_Basics/
│
├── custome_types.py        # Pydantic models for type safety
├── data_loader.py          # PDF loading and embedding (HuggingFace)
├── vector_db.py            # Qdrant vector database operations
├── main.py                 # FastAPI backend with Inngest functions
├── streamlit_app.py        # Streamlit UI for user interaction
│
├── .env                    # Environment variables (create from .env.example)
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🎯 How It Works

### 1. Document Ingestion Flow

```python
# Triggered by: rag/ingest_pdf event
1. Load PDF → Extract text
2. Split into chunks (1000 chars, 400 overlap)
3. Generate embeddings using sentence-transformers
4. Store in Qdrant with metadata
```

### 2. Query Flow

```python
# Triggered by: rag/query event
1. User asks question
2. Embed question using same model
3. Search Qdrant for top-K similar chunks
4. Build prompt with retrieved context
5. Generate answer using Mistral-7B-Instruct
6. Return answer with sources
```

## 🔧 Key Components

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension**: 384
- **Speed**: ~1000 sentences/sec on CPU
- **Quality**: Excellent for semantic search

### LLM Model
- **Model**: `mistralai/Mistral-7B-Instruct-v0.2`
- **Parameters**: 7 billion
- **Context**: 8K tokens
- **Quality**: Comparable to GPT-3.5

### Why These Models?

| Aspect | Our Choice | Alternative |
|--------|-----------|-------------|
| **Cost** | $0 (Hugging Face free tier) | OpenAI: $20/month+ |
| **Privacy** | Data stays with you | OpenAI: Data sent externally |
| **Customization** | Full control | Limited customization |
| **Speed** | Fast enough for most use cases | Slightly faster |

## 💡 Usage Examples

### Upload a PDF via Streamlit

1. Go to `http://localhost:8501`
2. Click "Choose a PDF"
3. Upload your document
4. Wait for "Triggered ingestion" message

### Query Your Documents

1. In the query section, type your question
2. Adjust top_k (number of chunks to retrieve)
3. Click "Ask"
4. View answer and sources

### Programmatic Access

```python
import requests

# Trigger PDF ingestion
response = requests.post(
    "http://localhost:8000/e/rag_application",
    json={
        "name": "rag/ingest_pdf",
        "data": {
            "pdf_path": "/path/to/document.pdf",
            "source_id": "my_document"
        }
    }
)

# Query
response = requests.post(
    "http://localhost:8000/e/rag_application",
    json={
        "name": "rag/query",
        "data": {
            "question": "What is the main topic?",
            "top_k": 5
        }
    }
)
```

## ⚙️ Configuration

### Change Embedding Model

Edit `data_loader.py`:
```python
# For better quality (slower, larger)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBED_DIM = 768

# For faster processing (smaller)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
EMBED_DIM = 384
```

### Change LLM Model

Edit `main.py`:
```python
# For faster responses
model="google/flan-t5-large"

# For better quality (requires HF approval)
model="meta-llama/Llama-2-7b-chat-hf"

# For smaller footprint
model="mistralai/Mistral-7B-Instruct-v0.1"
```

### Adjust Chunking Strategy

Edit `data_loader.py`:
```python
splitter = SentenceSplitter(
    chunk_size=500,     # Smaller = more precise, more chunks
    chunk_overlap=100   # Overlap helps maintain context
)
```

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: "Connection refused" to Qdrant
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker restart qdrant
```

### Issue: "Invalid token" error
```bash
# Verify your .env file has the correct token
cat .env

# Get a new token at https://huggingface.co/settings/tokens
```

### Issue: Slow embedding/generation
```python
# Use smaller models in data_loader.py and main.py
# OR use GPU if available by installing:
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory
```python
# In data_loader.py, process in smaller batches
# In main.py, reduce max_new_tokens to 256
```

## 📊 Performance Benchmarks

Tested on: Intel i5, 16GB RAM, CPU only

| Operation | Time | Notes |
|-----------|------|-------|
| PDF ingestion (10 pages) | ~5-10s | Includes chunking + embedding |
| Single query | ~2-5s | Includes search + generation |
| Embedding 100 chunks | ~2s | On CPU |
| Qdrant search | <100ms | Very fast |

## 🚀 Production Deployment

### Docker Deployment

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage

  rag_api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - HUGGINGFACE_API_TOKEN=${HUGGINGFACE_API_TOKEN}
    depends_on:
      - qdrant
```

### Environment Variables for Production

```env
HUGGINGFACE_API_TOKEN=your_token
QDRANT_URL=http://qdrant:6333
INNGEST_API_BASE=https://your-inngest-instance.com
```

## 🔒 Security Best Practices

- Never commit `.env` file
- Use environment variables for secrets
- Validate file uploads (size, type)
- Rate limit API endpoints
- Sanitize user inputs

## 📈 Future Enhancements

- [ ] Multi-modal support (images, tables)
- [ ] Multiple file format support (DOCX, TXT, CSV)
- [ ] User authentication
- [ ] Query history and caching
- [ ] Advanced chunking strategies
- [ ] Evaluation metrics (accuracy, relevance)
- [ ] Chat interface with conversation memory

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **Hugging Face** - Free model hosting and inference
- **Inngest** - Workflow orchestration
- **Qdrant** - Vector database
- **Sentence Transformers** - Embedding models
- **Mistral AI** - Mistral-7B model

## 📧 Contact

**Shubham Harkare**
- GitHub: [@ShubhamHarkare](https://github.com/ShubhamHarkare)
- LinkedIn: [Your LinkedIn]
- Email: your.email@example.com

## 🔗 Resources

- [Hugging Face Models](https://huggingface.co/models)
- [Inngest Docs](https://www.inngest.com/docs)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers](https://www.sbert.net/)

---

**⭐ If this project helped you, please star it on GitHub!**

**🎯 Looking to hire? This project demonstrates:**
- Modern Python development (FastAPI, async/await)
- ML/AI integration (embeddings, LLMs)
- Event-driven architecture (Inngest)
- Vector databases (Qdrant)
- Full-stack development (Backend + Frontend)
- Production considerations (error handling, logging)
