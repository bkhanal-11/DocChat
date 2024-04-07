# Beyond ChatGPT: Building Intelligent Q&A with LangChain and RAG

If you've ever considered building an AI application on top of Large Language Models (LLMs) like ChatGPT, you've likely encountered LangChain, the leading LLM framework. In this article, we'll explore how Retrieval-Augmented Generation (RAG) can elevate your AI's understanding and response capabilities by leveraging both retrieval of information and generative models. We'll walk through a typical RAG application implemented with LangChain, providing insights into its mechanics and potential applications. By the end of this read, you'll not only grasp the concept of RAG but also be equipped to assess its fit for your project's needs.


## What is Retrieval Augmented Generation (RAG)?

Retrieval-Augmented Generation is a powerful technique that merges retrieval and generation to enhance the quality and pertinence of AI-generated content. Within LangChain, RAG signifies the fusion of retrieval mechanisms and language models, such as ChatGPT, to forge a sophisticated question-answering system.

In essence, RAG empowers you to engage in a Q&A dialogue with your data, be it documents, web content, or an SQL database. Let's unpack how it operates.

Before diving into the implementation, ensure you have the necessary packages installed. Run the following commands to install LangChain and its associated libraries, as well as other dependencies required for document processing and vector database:

```bash
pip install langchain langchain-community langchain-openai
pip install sentence-transformer pypdf chromadb
```

### Document Injection

The journey begins with document injection. Various Document loaders are employed, each providing a "load" method to import data as documents from a designated source. Post-loading, it's crucial to segment lengthy documents into smaller fragments compatible with the LLM's context window. LangChain's array of document transformers simplifies this process, enabling you to split, merge, and filter documents while maintaining semantic coherence.

#### Storing for Retrieval

Next, these segmented texts are stored in a database for future retrieval. A common approach involves embedding the text and saving the resultant vectors. When a query is made, the system embeds the query and fetches the most similar document vectors. This is where a vector store shines, managing data storage and vector searches. Indexing further streamlines this process by preventing content duplication, rewriting unchanged content, and bypassing re-embedding for static content.

Here's an example of how a PDF file is ingested using these techniques:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.indexes import SQLRecordManager, index

# For openai key
import os
os.environ["OPENAI_API_KEY"] = "sk-YOUR_OPENAI_API_KEY"

# load a PDF  
loader = PyPDFLoader("/content/qlora_paper.pdf")
documents = loader.load()

# Split text 
text = RecursiveCharacterTextSplitter().split_documents(documents)

# Create vector store
vectorstore = Chroma(
    collection_name="your_collection_name",
    embedding_function=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
    persist_directory="your_persist_directory",
)

record_manager = SQLRecordManager(
    db_url="your_database_url",
    namespace="your_namespace",
)

record_manager.create_schema()

indexing_result = index(
    docs_source=text,
    record_manager=record_manager,
    vector_store=vectorstore,
    batch_size=1000,
    cleanup="incremental",
    source_id_key="source",
)
```

### The Chat Pipeline

With our documents ingested, we can initiate the chat pipeline, which consists of three stages: Condensed Question, Document Retrieval, and Chat.

#### Condensed Question

Upon receiving a user query, we can refine it by adding context. Utilizing chat history, we can generate context-rich questions, enhancing document retrieval effectiveness. This step, while optional, can significantly improve the subsequent processes.

```python
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from operator import itemgetter

def create_retriever_chain(llm, retriever, chat_history):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_TEMPLATE)
    
    if not chat_history:
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
```

An example of `CONDENSE_QUESTION_PROMPT` can be as follow:

```python
CONDENSE_QUESTION_TEMPLATE = """\
Rephrase the follow-up question based on the chat history to make it standalone.

Chat History:
{chat_history}

Follow Up: {question}
Standalone Question:"""
```

#### Document Retrieval

Now, with our refined query, we aim to retrieve pertinent documents from the vector database. Retrievers come into play here, tasked with sourcing documents based on an unstructured query. They can operate independently of a vector store, focusing solely on retrieval. In this section, we will explore various advanced document retrieval techniques within the langchain framework.

The EnsembleRetriever is particularly noteworthy. It combines the outputs of various retrievers and re-ranks them using the [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) algorithm. This method often outperforms individual algorithms by combining sparse retrievers (like BM25 for keyword relevance) with dense retrievers (like MMR for semantic similarity).

A prevalent approach in document retrieval is the "hybrid search," which leverages the complementary strengths of sparse and dense retrievers. The sparse retriever, exemplified by BM25, excels at identifying documents through keyword matching. BM25 is a ranking function widely used in information retrieval to gauge the relevance of documents to a specific search query.

Conversely, dense retrievers like the Maximum Marginal Relevance (MMR) method focus on semantic similarity. MMR retrieval operates by selecting documents that are not only similar to the query but also diverse. It achieves this by prioritizing documents whose embeddings exhibit the highest cosine similarity to the query and then incrementally includes them, adjusting for similarity to documents already chosen. In conjunction with MMR, we integrate a MultiQuery Retriever to enhance the retrieval process.

The retrieval process in a distance-based vector database involves representing queries as points in a high-dimensional space and identifying documents with closely matching embeddings. However, this process can yield varying results due to minor variations in query phrasing or if the embeddings fail to accurately reflect the data's semantics. To mitigate this, the MultiQuery Retriever employs an LLM to craft multiple variations of the query, each offering a different perspective. It then aggregates the unique documents retrieved by each variant to compile a comprehensive set of relevant documents.


```python
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document

def get_retriever(vectorstore, record_manager):
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
```

#### Reordering for Relevance

After retrieval, document reordering is crucial. LLMs can struggle with long contexts, often overlooking vital information, see: https://arxiv.org/abs/2307.03172. To counter this, we employ Long-Context Reorder, ensuring the most relevant documents are prioritized for the LLM's response.

```python
from langchain_community.document_transformers import LongContextReorder

def reorder_documents(documents):
    reorder = LongContextReorder()

    for i, doc in enumerate(documents):
        doc.metadata["original_index"] = i

    return reorder.transform_documents(documents)
```

### Engaging in Chat

With the context set from relevant documents, we can now engage with our LLM. By providing the context and, if available, chat history alongside the user query, we prompt the LLM to deliver a response that's pertinent to the uploaded documents.

To create an answer chain, we can make use of the LangChain Expression Language (LCEL) to define a chain that takes the retrieved documents and the chat history, and returns an answer. An example is as follow:

```python
from langchain.schema.runnable import RunnableMap
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

def create_answer_chain(llm, retriever, chat_history):
    retriever_chain = create_retriever_chain(llm, retriever, chat_history)

    context = RunnableMap(
        {
            "context": (
                retriever_chain
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
```

Now, to make successful query, we can use the following code:

```python
answer_chain = create_answer_chain(llm, retriever, chat_history)
answer = answer_chain.invoke({"question": "What is the summary of the Machine Learning lecture?"})
```


---

Throughout this exploration of RAG with LangChain, we've uncovered the transformative potential it holds for Q&A systems. By integrating retrieval with generative AI, RAG enables a more nuanced and contextually aware conversation with data. This approach is not just about answering questions—it's about understanding the context, retrieving the right information, and generating responses that are both accurate and relevant. As we look towards the future of AI-driven solutions, whether it be in customer support, education, or search engines, RAG stands out as a pivotal step in creating more intelligent and responsive systems. The journey from a simple Q&A to an intelligent, context-aware system is complex, but with the insights provided here, you're now better prepared to embark on this exciting path.