import os
import sys
import argparse
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

class PDFLoader:
    """
    Handles loading and preprocessing of PDF documents
    """
    @staticmethod
    def load_and_split_pdfs(pdf_paths: List[str]) -> List[str]:
        """
        Load PDFs and split into document chunks
        
        :param pdf_paths: List of PDF file paths
        :return: List of document chunks
        """
        all_docs = []
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF not found - {pdf_path}")
                continue
            
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            split_docs = text_splitter.split_documents(docs)
            
            all_docs.extend(split_docs)
        
        return all_docs

class EmbeddingGenerator:
    """
    Generates embeddings using Google's text-embedding-004
    """
    def __init__(self, api_key: str):
        """
        Initialize embedding generator
        
        :param api_key: Google API key
        """
        genai.configure(api_key=api_key)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )
    
    def get_embeddings(self):
        """
        Return the embeddings object
        
        :return: Embeddings object
        """
        return self.embeddings

class VectorStoreManager:
    """
    Manages vector store operations using FAISS
    """
    def __init__(self, embeddings, persist_directory: str = 'pdf_vectorstore'):
        """
        Initialize vector store manager
        
        :param embeddings: Embedding generator
        :param persist_directory: Directory to store vector store
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
    
    def create_or_update_vectorstore(self, documents):
        """
        Create or update FAISS vector store
        
        :param documents: List of documents to add to vector store
        """
        if not documents:
            print("No documents to process.")
            return
        
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        vectorstore.save_local(self.persist_directory)
        print(f"Successfully processed {len(documents)} document chunks.")
    
    def load_vectorstore(self):
        """
        Load existing vector store
        
        :return: Loaded FAISS vector store or None
        """
        try:
            return FAISS.load_local(
                self.persist_directory, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        except Exception:
            print("No documents exist in the database.")
            return None

class QuestionAnsweringSystem:
    """
    Main Question Answering system integrating all components
    """
    def __init__(self):
        """
        Initialize QA system
        """
        load_dotenv()
        
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError("Google API Key not found. Set GOOGLE_API_KEY in .env file.")
        
        self.embedding_generator = EmbeddingGenerator(google_api_key)
        self.vector_store_manager = VectorStoreManager(
            self.embedding_generator.get_embeddings()
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", 
            temperature=0.3
        )
    
    def feed_pdfs(self, pdf_paths: List[str]):
        """
        Process and add PDFs to the vector store
        
        :param pdf_paths: List of paths to PDF files
        """
        documents = PDFLoader.load_and_split_pdfs(pdf_paths)
        self.vector_store_manager.create_or_update_vectorstore(documents)
    
    def query(self):
        """
        Interactive query interface
        """
        vectorstore = self.vector_store_manager.load_vectorstore()
        
        if not vectorstore:
            return
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}  # Retrieve top 3 relevant documents
            ),
            return_source_documents=True
        )
        
        print("Ask me anything! (Type 'exit' to quit)")
        while True:
            query = input("Your question: ").strip()
            if query.lower() == 'exit':
                break
            
            try:
                result = qa_chain.invoke({"query": query})
                print("\nAnswer:", result['result'])
                
                print("\nTop Relevant Passages:")
                for idx, doc in enumerate(result['source_documents'], start=1):
                    source = doc.metadata.get('source', 'Unknown source')
                    print(f"{idx}. Location: {source}")
                    print(doc.page_content[:500])  # Display the first 500 characters
                    print()
                
                print("\n" + "-"*50 + "\n")
            
            except Exception as e:
                print(f"An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="PDF Question Answering System")
    parser.add_argument('mode', nargs='?', default='query', 
                        choices=['feed', 'query'], 
                        help="Mode of operation: feed PDFs or query existing database (default: query)")
    parser.add_argument('paths', nargs='*', default=[],
                        help="Paths to PDF files (for 'feed' mode)")
    
    args = parser.parse_args()
    qa_system = QuestionAnsweringSystem()
    
    if args.mode == 'feed':
        if not args.paths:
            print("Please provide PDF file paths.")
            sys.exit(1)
        qa_system.feed_pdfs(args.paths)
    elif args.mode == 'query':
        qa_system.query()

if __name__ == "__main__":
    main()
