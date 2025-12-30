import os
os.environ["USER_AGENT"] = "wiki-rag-bot/1.0 (contact: dev@example.com)"

from dotenv import load_dotenv
load_dotenv(override=True)


from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://en.wikipedia.org/wiki/India",
    header_template={
        "User-Agent": "wiki-rag-bot/1.0 (contact: youremail@example.com)"
    }
)

document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
documents = text_splitter.split_documents(document)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("wiki_index")

print("Wikipedia article indexed successfully")