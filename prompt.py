

from langchain_core.prompts import ChatPromptTemplate

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
