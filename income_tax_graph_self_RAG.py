# %% [markdown]
# # 2.4 생성된 답변을 여러번 검증하는 Self-RAG

# %%
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embedding_function = HuggingFaceEmbeddings(model="BAAI/bge-m3")
collection_name = "chroma-income-tax-ollama_embedding"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_function,
    persist_directory="./chroma"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# %%
from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

workflow = StateGraph(AgentState)

# %%
def retrieve(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 벡터 스토어에서 관련 문서를 검색합니다.
    Args:
        state (AgentState): 사용자의 질문을 포함한 에이전트의 현재 state.
    Returns:
        AgentState: 검색된 문서가 추가된 state를 반환합니다.
    """
    print("retrieve")
    query = state["query"]
    docs = retriever.invoke(query)
    return {"context": docs}

# %%
from langchain_ollama import ChatOllama

llm_ollama = ChatOllama(model="llama3.1")

# %% [markdown]
# - [rlm/rag-prompt](https://smith.langchain.com/hub/rlm/rag-prompt)
# - HUMAN
#     - You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
#     - Question: {question} 
#     - Context: {context} 
# - Answer:

# %%
from langchain_classic import hub

generate_prompt = hub.pull("rlm/rag-prompt")
# generate_llm = ChatOllama(model="llama3.1", max_completion_tokens=100)

def generate(state: AgentState) -> AgentState:
    """
    주어진 state를 기반으로 RAG 체인을 사용하여 응답을 생성합니다.
    Args:
        state (AgentState): 사용자의 질문과 문맥을 포함한 에이전트의 현재 state.
    Returns:
        AgentState: 생성된 응답을 포함하는 state를 반환합니다.
    """
    print("generate")
    query = state["query"]
    context = state["context"]
    generate_chain = generate_prompt | llm_ollama  # generate_llm (X)
    response = generate_chain.invoke({"question": query, "context": context})
    return {"answer": response}

# %% [markdown]
# - [langchain-ai/rag-document-relevance](https://smith.langchain.com/hub/langchain-ai/rag-document-relevance)
# - HUMAN
#     - FACTS: {{documents}}
#     - QUESTION: {{question}}
# - Score:
#     - A score of 1 means that the FACT contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant. This is the highest (best) score. 
#     - A score of 0 means that the FACTS are completely unrelated to the QUESTION. This is the lowest possible score you can give.

# %%
from typing import Literal

doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def is_doc_relevant(state: AgentState) -> Literal["relevant", "irrelevant"]:
    """
    """
    print("is_doc_relevant")
    query = state["query"]
    context = state["context"]
    doc_relevance_chain = doc_relevance_prompt | llm_ollama
    response = doc_relevance_chain.invoke({"question": query, "documents": context})
    if response["Score"] == 1:
        print("---> relevant")
        return "relevant"
    print("---> irrelevant")
    return "irrelevant"

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ["사람과 관련된 표현은 '거주자'로 바꿉니다."]
template = f"""
    아래 사전을 참고해서 사용자의 질문을 변경합니다.
    사전: {dictionary}
    질문: {{query}}
"""
rewrite_prompt = PromptTemplate.from_template(template=template)

def rewrite(state: AgentState) -> AgentState:
    """
    """
    print("rewrite")
    query = state["query"]
    rewrite_chain = rewrite_prompt | llm_ollama | StrOutputParser()
    response = rewrite_chain.invoke({"query": query})
    return {"query": response}

# %% [markdown]
# - [langchain-ai/rag-answer-hallucination](https://smith.langchain.com/hub/langchain-ai/rag-answer-hallucination)
# - HUMAN
#     - FACTS: {{documents}} 
#     - STUDENT ANSWER: {{student_answer}}
# - Score:
#     - A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 
#     - A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

# %%
hallucination_prompt = hub.pull("langchain-ai/rag-answer-hallucination")
hallucination_llm = ChatOllama(model="llama3.1", temperature=0)

def is_hallucinated(state: AgentState) -> Literal["hallucinated", "non-hallucinated"]:
    """
    """
    print("is_hallucinated")
    context = state["context"]
    answer = state["answer"]
    hallucination_chain = hallucination_prompt | hallucination_llm
    response = hallucination_chain.invoke({"documents": context, "student_answer": answer})
    if response["Score"] == 1:
        print("---> non-hallucinated")
        return "non-hallucinated"
    print("---> hallucinated")
    return "hallucinated"

# %% [markdown]
# - [langchain-ai/rag-answer-helpfulness](https://smith.langchain.com/hub/langchain-ai/rag-answer-helpfulness)
# - HUMAN
#     - STUDENT ANSWER: {{student_answer}}
#     - QUESTION: {{question}}
# - Score:
#     - A score of 1 means that the student's answer meets all of the criteria. This is the highest (best) score. 
#     - A score of 0 means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

# %%
helpful_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def is_helpful(state: AgentState) -> Literal["helpful", "unhelpful"]:
    """
    """
    print("is_helpful")
    query = state["query"]
    answer = state["answer"]
    helpful_chain = helpful_prompt | llm_ollama
    response = helpful_chain.invoke({"question": query, "student_answer": answer})
    if response["Score"] == 1:
        print("---> helpful")
        return "helpful"
    print("---> unhelpful")
    return "unhelpful"

def check_helpfulness(state: AgentState) -> AgentState:
    """
    사용자의 질문에 기반하여 생성된 답변의 유용성을 평가합니다.
    Args:
        state (AgentState): 사용자의 질문과 생성된 답변을 포함한 에이전트의 현재 state.
    Returns:
        str: 답변이 유용하다고 판단되면 'helpful', 그렇지 않으면 'unhelpful'을 반환합니다.
    """
    print("check_helpfulness")
    return state

# %%
from langgraph.graph import START, END

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite)
workflow.add_node("check_helpfulness", check_helpfulness)

workflow.add_edge(START, "retrieve")
workflow.add_conditional_edges("retrieve", is_doc_relevant, {
    "relevant": "generate",
    "irrelevant": END,
})
workflow.add_conditional_edges("generate", is_hallucinated, {
    "hallucinated": "generate",
    "non-hallucinated": "check_helpfulness",
})
workflow.add_conditional_edges("check_helpfulness", is_helpful, {
    "helpful": END,
    "unhelpful": "rewrite"
})

# %%

graph = workflow.compile()