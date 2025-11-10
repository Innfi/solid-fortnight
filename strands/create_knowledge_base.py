from pathlib import Path
import PyPDF2
import chromadb
from chromadb.config import Settings
import uuid

def create_knowledge_base(pdf_directory: str, db_directory: str) -> None:
  """
  Create a knowledge base from PDF files in the specified directory.

  Args:
      pdf_directory (str): Path to the directory containing PDF files.
      db_directory (str): Path to the directory where the ChromaDB database will be stored.
  """
  # Initialize ChromaDB client
  client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=db_directory))

  # Create a collection for storing documents
  collection = client.create_collection(name="knowledge_base")

  # Iterate over all PDF files in the specified directory
  pdf_files = Path(pdf_directory).glob("*.pdf")
  for pdf_file in pdf_files:
      with open(pdf_file, "rb") as file:
          reader = PyPDF2.PdfFileReader(file)
          text_content = ""
          for page_num in range(reader.numPages):
              page = reader.getPage(page_num)
              text_content += page.extract_text() + "\n"

          # Create a unique ID for the document
          doc_id = str(uuid.uuid4())

          # Add the document to the collection
          collection.add(
              documents=[text_content],
              ids=[doc_id],
              metadatas=[{"source": str(pdf_file)}]
          )

  # Persist the database
  client.persist()