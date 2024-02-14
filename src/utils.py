from langchain_openai.chat_models import ChatOpenAI

def create_llm(openai_model: str, api_key: str, stream: bool) -> ChatOpenAI:
    return ChatOpenAI(temperature=0, 
                      model=openai_model, 
                      openai_api_key=api_key,
                      streaming=stream)