# -*- coding: utf-8 -*-
"""Assignmentgenai.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Bb1Sc79wLXS1QZAjufYGGzAAnyjiySWf
"""

!pip install uvicorn

!pip install ChromaDB
import chromadb

!pip install pdfminer.six

!pip install python-docx

!pip install python-multipart

from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from io import BytesIO
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings
import asyncio

# Initialize FastAPI app
app = FastAPI()

# Initialize the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize ChromaDB client
chroma_client = Client(Settings(persist_directory="./chromadb_storage"))
collection_name = "documents"

# Check if collection exists, otherwise create it
existing_collections = [col.name for col in chroma_client.list_collections()]
if collection_name not in existing_collections:
    chroma_client.create_collection(collection_name)
collection = chroma_client.get_collection(collection_name)

# Helper function to parse different file types
async def extract_text(file: UploadFile) -> str:
    if file.filename.endswith(".txt"):
        return await file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        return extract_pdf_text(BytesIO(await file.read()))
    elif file.filename.endswith(".docx"):
        doc = Document(BytesIO(await file.read()))
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# Ingestion endpoint using the helper function
@app.post("/ingest/")
async def ingest_documents(files: List[UploadFile]):
    for file in files:
        try:
            # Extract text from file
            text = await extract_text(file)


            embedding = model.encode(text).tolist()

            # Store text and embedding in ChromaDB
            document_id = f"{file.filename}_{len(text)}"
            collection.add(
                documents=[{"id": document_id, "text": text, "embedding": embedding}]
            )
            print("Document ingested:", text[:100])  # Log for demonstration

        except ValueError as e:
            return {"error": str(e)}

    return {"status": "Documents ingested successfully"}

# Query endpoint to retrieve similar documents
@app.post("/query/")
async def query_documents(query: str):

    query_embedding = model.encode(query).tolist()

    # Retrieve similar documents from ChromaDB
    results = collection.query(embedding=query_embedding, k=5)

    # Format results
    retrieved_docs = [{"text": doc["text"][:100], "score": doc["score"]} for doc in results["documents"]]
    return {"results": retrieved_docs}