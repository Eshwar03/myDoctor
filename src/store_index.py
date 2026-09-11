from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pathlib import Path
from langchain_pinecone import PineconeVectorStore
from .helper import loadPdf,filter_minimal_data,chunk_docs,getEmbeddingModel

load_dotenv()


extracted_data=loadPdf(Path("D:/Projects/myDoctor/data"))
filtered_docs=filter_minimal_data(extracted_data)
chunked_docs=chunk_docs(filtered_docs)


pc = Pinecone()
index_name = "my-doctor-db"


if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )


embedding_model=getEmbeddingModel()


docsearch=PineconeVectorStore.from_documents(
    embedding=embedding_model,
    documents=chunked_docs,
    index_name=index_name
)