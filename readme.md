# PDF Question Answering System

This project implements a scalable, efficient, and interactive PDF-based Question Answering (QA) system. By leveraging FAISS for vector store indexing and LangChain for pipeline integration, the system can process and query PDF documents with ease. The application is powered by Google's Generative AI for embeddings and chat-based question answering.

## Features
- **PDF Ingestion:** Load and preprocess PDF files into manageable document chunks.
- **Vector Store:** Efficiently store and retrieve document embeddings using FAISS for fast similarity search.
- **Generative AI Integration:** Utilize Google's Generative AI (Gemini 1.5) for advanced question answering.
- **Scalability:** The use of FAISS makes the system highly scalable for large document repositories.
- **Interactive Query Mode:** Engage in real-time Q&A with access to relevant passages and sources.

## Architecture
1. **PDFLoader:** Processes PDF files into smaller, overlapping text chunks for better search and context retention.
2. **EmbeddingGenerator:** Generates text embeddings using Google's text-embedding-004 model.
3. **VectorStoreManager:** Manages document embeddings and performs similarity searches using FAISS.
4. **QuestionAnsweringSystem:** Combines LangChain, embeddings, and FAISS to deliver accurate responses.

## Setup and Installation

### Prerequisites
- Python 3.8 or above
- A valid Google API Key for Generative AI services

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pdf-qa-system.git
   cd pdf-qa-system
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file and add your Google API Key:
   ```plaintext
   GOOGLE_API_KEY=your-google-api-key
   ```

### Running the System
1. **Feed PDFs:**
   To add documents to the vector store:
   ```bash
   python main.py feed path/to/pdf1 path/to/pdf2
   ```

2. **Query:**
   To start querying the system interactively:
   ```bash
   python main.py query
   ```

## Why FAISS and LangChain?
- **FAISS:**
  - Enables high-speed similarity searches across large datasets.
  - Scalable and memory-efficient, suitable for managing vast document repositories.

- **LangChain:**
  - Simplifies the integration of multiple components (retrievers, embeddings, and LLMs).
  - Flexible and modular, allowing seamless adaptation to new AI models and workflows.

Together, FAISS and LangChain create a robust pipeline for handling complex, large-scale document processing tasks.

## Example Workflow
### Feeding PDFs
```bash
python main.py feed example.pdf
```
Output:
```
Successfully processed 5 document chunks.
```

### Querying
```bash
python main.py query
```
Example Interaction:
```
Ask me anything! (Type 'exit' to quit)
Your question: What is the purpose of the document?

Answer: The purpose of the document is to...

Top Relevant Passages:
1. Location: example.pdf
   The document discusses...

2. Location: example.pdf
   Additional details include...

--------------------------------------------------
```

## Requirements
```plaintext
faiss-cpu
langchain
langchain-community
langchain-google-genai
python-dotenv
google-generativeai
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request to improve the system.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
