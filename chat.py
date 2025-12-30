from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_openai import ChatOpenAI
from retriver import get_relevant_docs
from prompt import prompt
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)
print("🧠 Wikipedia RAG Chatbot Ready (type 'exit' to quit)\n")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    docs = get_relevant_docs(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.format_messages(
        context=context,
        question=query
    )
    response = llm.invoke(messages)
    print("\nBot:", response.content, "\n")
