import os
from dotenv import load_dotenv 

from langchain_community.document_loaders import PyPDFLoader

#recursive character text splitter splits large docs into smaller chunks before creating embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

#converts text to vectors
from langchain_community.embeddings import HuggingFaceEmbeddings
#FAISS :Store embeddings (vectors) and quickly find the most similar ones.
from langchain_community.vectorstores import FAISS

load_dotenv()

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

print(f"loaded {len(documents)} pages")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)
print(f"created {len(docs)} chunks")

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

#FAISS stores vectors and performs similarity search
db = FAISS.from_documents(
    docs,
    embeddings
)

print("Faiss db created with embeddings")

#creating llm
llm = ChatGroq (
    model = "llama-3.1-8b-instant",
    api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0
)

while True:
    question = input("Ask a question about the pdf (type exit to quit): ")
    if question.lower() == "exit":
        break
    
    results = db.similarity_search(
        question,
        k=5
    )

    print("retrieved chunks:")
    for i,doc in enumerate(results):
        print(f"chunk {i+1}")
        print(doc.page_content[:500])

    context = "\n".join([
        doc.page_content for doc in results
    ])

    prompt = f"""
    you are answering questions about a PDF document 
    use ONLY the context provided 

    if answer is not clearly present say:
        "The document does not have the relevant information to support an answer"

    context:
    {context}

    question:
    {question}
"""

    response = llm.invoke(prompt)
    print("answer: ")
    print(response.content)