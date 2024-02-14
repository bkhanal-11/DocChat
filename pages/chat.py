import streamlit as st

from langchain.schema import AIMessage, HumanMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationTokenBufferMemory

from src.conversational_chain import create_answer_chain, get_retriever 
from src.utils import create_llm

st.set_page_config(
    page_title="DocChat",
    page_icon="🦜",
    layout="wide",
)

st.title("Chat with your Documents", help="It uses a combination of keyword and semantic search to find answers.")

st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

chat_memory = StreamlitChatMessageHistory()
memory = ConversationTokenBufferMemory(
            llm=create_llm(st.session_state.openai_api_model, st.session_state.openai_api_key, False),
            chat_memory=chat_memory,
            return_messages=True,
            memory_key="chat_history",
        )

# Initialize chat history
if len(chat_memory.messages) == 0 or st.sidebar.button("Clear message history", key="clear_chat"):
    chat_memory.clear()
    chat_memory.add_ai_message("How can I help you?")

# Display chat messages from history on app rerun
avatars = {"human": "user", "ai": "assistant"}
for msg in chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

# Accept user input
if user_query := st.chat_input("Ask something related to documents!"):
    # Add user message to chat history
    chat_memory.add_user_message(user_query)

    # Display user message in chat message container
    st.chat_message("user").write(user_query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Create answer chain
        # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        use_chat_history = len(chat_memory.messages) > 2

        chain = create_answer_chain(llm=create_llm(st.session_state.openai_api_model, st.session_state.openai_api_key, True),
                                    retriever=get_retriever(st.session_state.session_id),
                                    use_chat_history=use_chat_history,
                                    )

        message_placeholder = st.empty()
        full_response = ""
        chain_input = {
            "question": user_query,
            "chat_history": memory.load_memory_variables({})["chat_history"]
            }
        
        for token in chain.stream(input=chain_input):
            full_response += token.content
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    chat_memory.add_ai_message(full_response)

