# %% [markdown]
# # 2.7 병렬 처리를 통한 효율 개선 (feat. 프롬프트 엔지니어링)

# %%
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

# %%
from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str  # 사용자 질문
    tax_rate: str  # (answer) 세율 -> calculate_tax_rate
    tax_base: str  # 과세표준 계산 -> calculate_tax_base
    tax_base_equation: str  # 과세표준 계산 수식 -> get_tax_base_equation
    tax_deduction: str  # 공제액 -> get_tax_deduction
    market_ratio: str  # 공정시장가액비율 -> get_market_ratio

workflow = StateGraph(AgentState)

# %%
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# pdf_file_path = "./documents/real_estate_tax.pdf"
# pdf_loader = PyPDFLoader(file_path=pdf_file_path)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
# documents = pdf_loader.load_and_split(text_splitter=text_splitter)

# %%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

collection_name = "real_estate_tax"
embedding = HuggingFaceEmbeddings(model="BAAI/bge-m3")

# vector_store = Chroma.from_documents(
#     documents=documents,
#     embedding=embedding,
#     collection_name=collection_name,
#     persist_directory="./chroma"
# )

vector_store = Chroma(
    embedding_function=embedding,
    collection_name=collection_name,
    persist_directory="./chroma"
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# %%
from langchain_ollama import ChatOllama

# llm_llama32v = ChatOllama(model="llama3.2-vision")
llm_llama31 = ChatOllama(model="llama3.1")

# %% [markdown]
# - [rlm/rag-prompt](https://smith.langchain.com/hub/rlm/rag-prompt)
# - HUMAN
#     - You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#     - Question: {question} 
#     - Context: {context} 
# - Answer:

# %%
from langchain_classic import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def get_tax_base_equation(state: AgentState) -> AgentState:
    """
    종합부동산세 과세표준을 계산하는 수식을 가져옵니다.
    `node`로 활용되기 때문에 `state`를 인자로 받지만, 고정된 기능을 수행하기 때문에 `state`를 활용하지는 않습니다.
    Args:
        state (AgentState): 현재 에이전트의 상태를 나타내는 객체입니다.
    Returns:
        AgentState: 'tax_base_equation' 키를 포함하는 새로운 `state`를 반환합니다.
    """
    print("get_tax_base_equation")

    rag_prompt = hub.pull("rlm/rag-prompt")
    tax_base_retrieval_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | rag_prompt
        | llm_llama31
        | StrOutputParser()
    )

    tax_base_equation_prompt = ChatPromptTemplate.from_messages([
        ("system", "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 리턴해주세요"),
        ("human", "{tax_base_equation_information}")
    ])
    tax_base_equation_chain = (
        {"tax_base_equation_information": RunnablePassthrough()}
        | tax_base_equation_prompt
        | llm_llama31
        | StrOutputParser()
    )
    
    tax_base_chain = {"tax_base_equation_information": tax_base_retrieval_chain} | tax_base_equation_chain

    tax_base_equation_question = "주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요"
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {"tax_base_equation": tax_base_equation}


# %%
def get_tax_deduction(state: AgentState) -> AgentState:
    """
    종합부동산세 공제금액에 관한 정보를 가져옵니다.
    `node`로 활용되기 때문에 `state`를 인자로 받지만, 고정된 기능을 수행하기 때문에 `state`를 활용하지는 않습니다.
    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.
    Returns:
        AgentState: 'tax_deduction' 키를 포함하는 새로운 state를 반환합니다.
    """
    print("get_tax_deduction")

    rag_prompt = hub.pull("rlm/rag-prompt") 
    tax_deduction_chain = (
        {"question": RunnablePassthrough(), "context": retriever}
        | rag_prompt
        | llm_llama31
        | StrOutputParser()
    )    
    tax_deduction_question = "주택에 대한 종합부동산세 계산시 공제금액을 알려주세요"
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {"tax_deduction": tax_deduction}


# %%
from datetime import date
from langchain_tavily import TavilySearch

def get_market_ratio(state: AgentState) -> AgentState:
    """
    web 검색을 통해 주택 공시가격에 대한 공정시장가액비율을 가져옵니다.
    `node`로 활용되기 때문에 `state`를 인자로 받지만, 고정된 기능을 수행하기 때문에 `state`를 활용하지는 않습니다.
    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.
    Returns:
        AgentState: 'market_ratio' 키를 포함하는 새로운 state를 반환합니다.
    """
    print("get_market_ratio")

    query = f"오늘 날짜: {date.today()}에 해당하는 주택 공시가격 공정시장가액비율은 몇 %인가요?"
    tavily_search = TavilySearch(
        max_results=5,
        topic="finance",  # "general", "news"
        include_answer=True,
        include_raw_content=True,
        include_images=True,
        search_depth="advanced",  # "basic"
    )
    context = tavily_search.invoke(query)

    market_ratio_prompt = ChatPromptTemplate.from_messages([
        ("system", "아래 정보를 기반으로 공정시장 가액비율을 계산해 주세요\n정보: {context}"),
        ("human", "{query}")
    ])
    market_ratio_chain = (
        market_ratio_prompt
        | llm_llama31
        | StrOutputParser()
    )
    market_ratio = market_ratio_chain.invoke({"query": query, "context": context})
    return {"market_ratio": market_ratio}


# %%
def calculate_tax_base(state: AgentState) -> AgentState:
    """
    주어진 state에서 과세표준을 계산합니다.
    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.
    Returns:
        AgentState: 'tax_base' 키를 포함하는 새로운 state를 반환합니다.
    """
    print("calculate_tax_base")

    query = state["query"]
    tax_base_equation = state["tax_base_equation"]
    tax_deduction = state["tax_deduction"]
    market_ratio = state["market_ratio"]

    tax_base_system_prompt = """
        주어진 내용을 기반으로 과세표준을 계산해 주세요
        과세표준 계산 공식: {tax_base_equation}
        공제 금액: {tax_deduction}
        공정시장가액비율: {market_ratio} """
    tax_base_prompt = ChatPromptTemplate.from_messages([
        ("system", tax_base_system_prompt),
        ("human", "사용자 주택 공시가격 정보: {query}")
    ])
    tax_base_chain = (
        tax_base_prompt
        | llm_llama31
        | StrOutputParser()
    )
    tax_base = tax_base_chain.invoke({
        "query": query,
        "tax_base_equation": tax_base_equation,
        "tax_deduction": tax_deduction,
        "market_ratio": market_ratio,
    })
    return {"tax_base": tax_base}

# %%
def calculate_tax_rate(state: AgentState) -> AgentState:
    """
    주어진 state에서 세율을 계산합니다.
    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체입니다.
    Returns:
        dict: 'tax_rate' 키를 포함하는 새로운 state를 반환합니다.
    """
    print("calculate_tax_rate")

    query = state["query"]
    tax_base = state["tax_base"]
    context = retriever.invoke(query)
    
    tax_rate_system_prompt = """
        당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해 주세요
        종합부동산세 세율: {context} """
    tax_rate_human_prompt = """
        과세 표준과 사용자가 보유한 주택의 수가 아래와 같을 때 종합부동산세를 계산해 주세요
        과세 표준: {tax_base}
        주택 수: {query} """
    tax_rate_prompt = ChatPromptTemplate.from_messages([
        ("system", tax_rate_system_prompt),
        ("human", tax_rate_human_prompt),
    ])
    tax_rate_chain = (
        tax_rate_prompt
        | llm_llama31
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({
        "query": query,
        "tax_base": tax_base,
        "context": context,
    })
    return {"tax_rate": tax_rate}

# %%
from langgraph.graph import START, END

workflow.add_node("get_tax_base_equation", get_tax_base_equation)
workflow.add_node("get_tax_deduction", get_tax_deduction)
workflow.add_node("get_market_ratio", get_market_ratio)
workflow.add_node("calculate_tax_base", calculate_tax_base)
workflow.add_node("calculate_tax_rate", calculate_tax_rate)

workflow.add_edge(START, "get_tax_base_equation")
workflow.add_edge(START, "get_tax_deduction")
workflow.add_edge(START, "get_market_ratio")
workflow.add_edge("get_tax_base_equation", "calculate_tax_base")
workflow.add_edge("get_tax_deduction", "calculate_tax_base")
workflow.add_edge("get_market_ratio", "calculate_tax_base")
workflow.add_edge("calculate_tax_base", "calculate_tax_rate")

# %%
graph = workflow.compile()
