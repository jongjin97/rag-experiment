from src.naive_rag.retriever import retrieve_context
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import MODEL_NAME
from langchain_core.output_parsers import StrOutputParser
from src.prompts.prompt import SAMSUNG_PROMPT
from langchain_core.documents import Document

def get_llm():
    return ChatOpenAI(model_name=MODEL_NAME, temperature=0)

def get_prompt():
    return ChatPromptTemplate.from_template(SAMSUNG_PROMPT)
    
def format_docs(docs: list[Document]):
    return "\n\n".join([doc.page_content for doc in docs])

def rag_chain(query: str):
    llm = get_llm()
    # retrieve_context returns a list of Documents (or similar, depending on retriever)
    # The retriever from src/naive_rag/retriever.py invokes an EnsembleRetriever which returns List[Document]
    docs = retrieve_context(query)
    
    # helper to format
    formatted_docs_str = format_docs(docs)
    
    # print(formatted_docs_str) # Optional logging
    
    prompt = get_prompt()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"query": query, "context": formatted_docs_str})
    
    return {
        "result": answer,
        "context": [d.page_content for d in docs]
    }

if __name__ == "__main__":
    answer = rag_chain("DX 부문의 주요 제품은 무엇인가?")
    print(answer)