import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()



import os
from langchain_groq import ChatGroq

def get_groq_llm():
    return ChatGroq(
        temperature=0,
        model_name="deepseek-r1-distill-llama-70b",
        api_key=os.getenv("GROQ_API_KEY"),
    )



# llm = ChatGroq(
#     groq_api_key=os.getenv("GROQ_API_KEY"),
#     model_name="DeepSeek-R1-Distill-Llama-70B",
#     temperature=0
# )





# llm = ChatOpenAI(
#   base_url="https://api.sambanova.ai/v1",  # 🔁 Replace with SambaNova's actual base URL
#    api_key=os.getenv("SAMBA_API_KEY"),
#    model="DeepSeek-R1",  # 🔁 Replace with the correct model name
#    temperature=0,
# )