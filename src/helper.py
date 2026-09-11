from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader

def loadPdf(location):
    loader=PyPDFDirectoryLoader(location)
    return loader.load()


def filter_minimal_data(docs:List[Document])->List[Document]:
    minimal_docs=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(Document(
            page_content=doc.page_content,
            metadata={"source":src}
        ))
    return minimal_docs


def chunk_docs(docs):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return splitter.split_documents(docs)


def getEmbeddingModel():
    embedding_model= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model