from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def get_relevant_docs(query, k=3):
    embeddings = OpenAIEmbeddings()

    vectorstore = FAISS.load_local(
        "wiki_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=k)
    return docs
