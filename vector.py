from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("Data/Context.csv")
df.head()

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./context_vectors_db"
add_document = not os.path.exists(db_location)

if add_document:
    documents = []
    idx = []

    for index, row in df.iterrows():
        document = Document(
            page_content=str(row["Index"]) + " " + str(row["Value"]),
            metadata={
                "meaning": str(row["Meaning"]),
                "suggestion": str(row["Suggestion"])
            }
        )
        documents.append(document)
        idx.append(str(index))

vector_store = Chroma(
    collection_name="Recommendations",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_document:
    vector_store.add_documents(documents=documents, ids=idx)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)
