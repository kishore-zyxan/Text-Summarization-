import time
import logging
from langchain_core.documents import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from llm_client import get_groq_llm

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the summarization prompt
map_prompt = """
Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:
"""
map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

def summarize_text(content: str) -> str:
    start_time = time.time()  # Start timing

    # Split the content into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(content)
    docs = [Document(page_content=t) for t in texts]

    # Load model
    llm = get_groq_llm()

    # Run summarization
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )
    summary = chain.run(docs)

    end_time = time.time()  # End timing
    elapsed = end_time - start_time
    logger.info(f"Summarization completed in {elapsed:.2f} seconds.")

    return summary
