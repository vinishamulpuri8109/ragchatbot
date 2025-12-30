import streamlit as st
from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("Wikipedia RAG Chatbot")

# ---------------- SESSION STATE ----------------
if "active" not in st.session_state:
    st.session_state.active = True

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- EXIT BUTTON ----------------
with st.sidebar:
    if st.button("🛑 Exit Chat"):
        st.session_state.active = False
        st.session_state.messages = []
        st.success("Chat ended. Please close or refresh the tab.")
        st.stop()

# ---------------- STOPPED STATE ----------------
if not st.session_state.active:
    st.info("Chat is closed. Refresh the page to start again.")
    st.stop()

# ---------------- LOAD VECTOR STORE ----------------
embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    "wiki_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ---------------- PROMPT ----------------
prompt = ChatPromptTemplate.from_template(
    """
    You are an expert assistant.
    Answer the question ONLY using the context below.
    If the answer is not in the context, say "I don't know".

    <context>
    {context}
    </context>

    Question: {question}
    """
)

# ---------------- CHAT INPUT ----------------
query = st.text_input("Ask a question:")

if query:
    # Retrieve relevant documents
    docs = db.similarity_search(query, k=6)

    context = "\n\n".join(doc.page_content for doc in docs)

    messages = prompt.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(messages)

    st.session_state.messages.append(
        {"question": query, "answer": response.content}
    )

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    st.markdown(f"**You:** {msg['question']}")
    st.markdown(f"**Bot:** {msg['answer']}")

