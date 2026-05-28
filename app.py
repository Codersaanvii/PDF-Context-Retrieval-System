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

questions=[
    "what is the main topic of the pdf?",
    "convert the pdf into bullet points",
    "what are the important concepts discussed",
    "summarize the key findings",
    "what challenges are mentioned?",
    "what conclusions are provided by the document?"
]

#creating llm
llm = ChatGroq (
    model = "llama-3.1-8b-instant",
    api_key = os.getenv("GROQ_API_KEY"),
    temperature = 0
)

#similarity search is semantic retrieval
for question in questions:
    #questions become embeddings -> FAISS compares vecs with chunk vecs -> returns matches 
    results = db.similarity_search(
        question,
        #k means: How many nearest results you want returned from the vector database.->Find the TOP 3 most similar chunks
        k=5
    )


    for i, doc in enumerate(results):
        print(f"Top result: {i+1}\n")
        #print first 500 characters of chunk
        print(doc.page_content[:500]) #contains actual chunk text 
 
#takes retrieved chunks (all 5 in this case) and joins them into one str
    context = "\n".join(
        [
            doc.page_content for doc in results
        ]
    )

#context is added to the prompt, along with the questions
#LLM does not see the entire pdf it just sees the chunks retrieved based on vector similarity of questions and chunks
    prompt = f"""
    context:
    {context}

    question:
    {question}

    answer using only the context and answer each question separately not all together
    """

    response = llm.invoke(prompt)

    print(response.content)