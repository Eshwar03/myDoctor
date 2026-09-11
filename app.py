from flask import Flask, render_template, request
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src import helper
from src import prompt


load_dotenv()
app=Flask(__name__)



embedding_model= helper.getEmbeddingModel()

doc_search=PineconeVectorStore.from_existing_index(
    embedding=embedding_model,
    index_name="my-doctor-db"
)

retriver = doc_search.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)



promptToLLM=ChatPromptTemplate.from_messages(
    [
        ("system",prompt.system_prompt),
        ("human","{input}")
    ]
)


def format_docs(docs):
    print(docs)
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain=(
    {"context":retriver|format_docs,"input":RunnablePassthrough()}
    |promptToLLM
    |llm
    |StrOutputParser()
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = rag_chain.invoke(msg)
    print("Response : ", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)