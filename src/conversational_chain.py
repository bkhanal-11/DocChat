from functools import partial
from operator import itemgetter
from typing import Sequence
import streamlit as st

from langchain.base_language import BaseLanguageModel
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_transformers import LongContextReorder
from langchain.indexes import SQLRecordManager
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import (
    BaseRetriever,
    Document,
    StrOutputParser,
)
from langchain.schema.messages import BaseMessageChunk
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document

CONDENSE_QUESTION_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Questions generally contains different entities, so you should rephrase \
the question according to the entity that is being asked about. \
Do not made up any information. The only information you can \
use to formulate the standalone question is the conversation and the follow up \
question.

Chat History:
###
{chat_history}
###

Follow Up Input: {question}
Standalone Question:"""

SYSTEM_ANSWER_QUESTION_TEMPLATE = """\
You are an expert in synthesizing information, tasked with answering any question \
with high-quality answers and without making anything up.

Generate a comprehensive and informative answer for the \
given question based solely on the provided documents' content. You must \
only use information from the documents/context provided in various formats. Use an unbiased and \
journalistic tone. Combine information from different documents into a coherent answer. Do not \
repeat text. If \
different documents refer to different aspects of the same topic, write separate \
answers for each aspect.

If you are unsure about the content or relevance of a document, write something down \
but make it clear that you are unsure. In addition, include what should be the expected \
outcome or information based on the question.

If there is nothing in the documents relevant to the question at hand, \
just say "I apologize, but based on the documents provided, I'm unable to find relevant \
information to answer your question. Could you please provide more context or try rephrasing your question?". \
Don't try to make up an answer. This is not a suggestion. This is a rule.

Anything between the following `context` html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user.

<context>
    {context}
</context>

REMEMBER: If there is no relevant information within the context, just say \
"I apologize, but based on the documents provided, I'm unable to find relevant \
information to answer your question. Could you please provide more context or try rephrasing your question?". \
Don't try to make up an answer. This is not a suggestion. This is a rule. \
Anything between the preceding 'context' html blocks is retrieved from a knowledge bank in various document formats \
including pdf, xlsx, pptx, txt, md, docx, \
etc., not part of the conversation with the user.

Take a deep breath and relax. You are an expert in synthesizing information. You can do this.
You can cite all the relevant information from the documents. Let's go!"""

def get_retriever(session_id):
    base_path = f'resources/{session_id}'
    vectorstore = Chroma(
        collection_name="docschat",
        embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        persist_directory=f"{base_path}/vectordb",
    )

    record_manager = SQLRecordManager(
        db_url=f"sqlite:///{base_path}/vectordb/index.sqlite",
        namespace="docschat",
    )

    vector_keys = vectorstore.get(
        ids=record_manager.list_keys(), include=["documents", "metadatas"]
    )

    docs_in_vectorstore = [
        Document(page_content=page_content, metadata=metadata)
        for page_content, metadata in zip(
            vector_keys["documents"], vector_keys["metadatas"]
        )
    ]

    keyword_retriever = BM25Retriever.from_documents(docs_in_vectorstore)
    keyword_retriever.k = 6

    semantic_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.3,
        },
    )

    queries_llm = ChatOpenAI(temperature=0, 
                             model=st.session_state.openai_api_model, 
                             api_key=st.session_state.openai_api_key
                             )
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=semantic_retriever,
        llm=queries_llm,
    )

    retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, multi_query_retriever],
        weights=[0.3, 0.7],
        c=0,
    )
    return retriever

def create_retriever_chain(
    llm: BaseLanguageModel[BaseMessageChunk],
    retriever: BaseRetriever,
    use_chat_history: bool,
):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)

    if not use_chat_history:
        initial_chain = (itemgetter("question")) | retriever
        return initial_chain
    else:
        condense_question_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
        )
        conversation_chain = condense_question_chain | retriever
        return conversation_chain


def get_k_or_less_documents(documents: list[Document], k: int):
    if len(documents) <= k:
        return documents
    else:
        return documents[:k]


def reorder_documents(documents: list[Document]):
    reorder = LongContextReorder()

    for i, doc in enumerate(documents):
        doc.metadata["original_index"] = i

    return reorder.transform_documents(documents)


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs: list[str] = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{doc.metadata.get('original_index', i)}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_answer_chain(
    llm: BaseLanguageModel[BaseMessageChunk],
    retriever: BaseRetriever,
    use_chat_history: bool = True,
    k: int = 5,
) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever, use_chat_history)

    _get_k_or_less_documents = partial(get_k_or_less_documents, k=k)

    context = RunnableMap(
        {
            "context": (
                retriever_chain
                | _get_k_or_less_documents
                | reorder_documents
                | format_docs
            ),
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        messages=[
            ("system", SYSTEM_ANSWER_QUESTION_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = prompt | llm
    response_chain = context | response_synthesizer

    return response_chain

