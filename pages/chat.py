import streamlit as st

from langchain.schema import AIMessage, HumanMessage
from src.conversational_chain import create_answer_chain, get_retriever 
from src.utils import create_llm

def clear_memory():
    st.session_state.messages.clear()
    st.session_state.messages = [{"role": "assistant", "content": 'How can I help you?'}]

st.set_page_config(
    page_title="DocChat",
    page_icon="🦜",
    layout="wide",
)

if st.sidebar.button("Clear message history"):
    clear_memory()


st.title("Chat with your Documents")

st.subheader("It uses a combination of keyword and semantic search to find answers.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages = [{"role": "assistant", "content": 'How can I help you?'}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if input := st.chat_input("Ask something related to documents?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(input)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Create answer chain
        # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")

        use_chat_history = len(st.session_state.messages) > 2

        chat_history = []
        if use_chat_history:
            for message in st.session_state.messages[:-1]:
                if message["role"] == "user":
                    chat_history.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    chat_history.append(AIMessage(content=message["content"]))

        chain = create_answer_chain(llm=create_llm(st.session_state.openai_api_model, st.session_state.openai_api_key, True),
                                    retriever=get_retriever(st.session_state.session_id),
                                    use_chat_history=use_chat_history,
                                    )

        message_placeholder = st.empty()
        full_response = ""
        input = {
            "question": input,
            "chat_history": chat_history
                     }
        for token in chain.stream(input=input):
            full_response += token.content
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

