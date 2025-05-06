import os
from dotenv import load_dotenv
import httpx
from langchain_core.language_models.llms import LLM
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class SambaNovaLLM(LLM):
    api_key: str
    model_name: str
    api_url: str = "https://api.sambanova.ai/v1/completions"

    def __init__(self, api_key: str, model_name: str):
        super().__init__(api_key=api_key, model_name=model_name)

    def _call(self, prompt: str, stop: list = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.7
        }
        try:
            logger.info("Making request to SambaNova API")
            with httpx.Client() as client:
                response = client.post(self.api_url, json=payload, headers=headers, timeout=30.0)
                response.raise_for_status()
                result = response.json()
                return result.get("choices", [{}])[0].get("text", "")
        except Exception as e:
            logger.error(f"SambaNova API call failed: {str(e)}")
            raise

    @property
    def _llm_type(self) -> str:
        return "sambanova"

llm = SambaNovaLLM(
    api_key=os.getenv("SAMBA_API_KEY"),
    model_name=os.getenv("SAMBA_MODEL", "DeepSeek-R1")
)

qa_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant. Use the provided context to answer the question.
If the answer is not in the context, say "Data is not available".

Context: {context}
Question: {question}
Answer:
"""
)