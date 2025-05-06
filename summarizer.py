from llm_client import llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document
import re
import logging
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Token counter
def count_tokens(text: str) -> int:
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


map_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize this in 2-3 sentences: {text}"
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="Create a concise summary with a title and 5-10 bullet points based on: {text}"
)


def clean_text(text: str) -> str:
    start_time = time.time()
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'Page \d+', '', text)
    cleaned_text = text.strip()
    duration = time.time() - start_time
    logger.info(f"Text cleaning completed in {duration:.2f} seconds")
    return cleaned_text


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def invoke_chain(chain, texts):
    try:
        logger.info(f"Invoking chain with {len(texts)} documents")
        result = chain.invoke({"input_documents": texts})
        logger.info(f"Chain invoke result: {type(result)}")
        if isinstance(result, dict) and "output_text" in result:
            return result["output_text"]
        elif isinstance(result, str):
            return result
        else:
            raise ValueError(f"Unexpected chain output format: {type(result)}")
    except Exception as e:
        logger.error(f"Chain invocation failed: {str(e)}")
        raise


def summarize_large_doc(text: str) -> str:
    start_total = time.time()

    # Clean text
    cleaned_text = clean_text(text)
    logger.info(f"Text length: {len(cleaned_text)} characters")

    # Split text
    start_split = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
    texts = text_splitter.create_documents([cleaned_text])
    texts = texts[:3]
    logger.info(f"Number of chunks: {len(texts)}")
    split_duration = time.time() - start_split
    logger.info(f"Text splitting completed in {split_duration:.2f} seconds")

    # Summarize
    start_chain = time.time()

    # Truncate or pre-summarize if too large
    max_token_limit = 4000
    total_tokens = count_tokens(cleaned_text)
    logger.info(f"Total tokens: {total_tokens}")
    if total_tokens > max_token_limit:
        logger.info("Text exceeds token limit, pre-summarizing chunks")
        chunk_summaries = []
        for i, chunk in enumerate(texts):
            chunk_text = chunk.page_content
            chunk_tokens = count_tokens(chunk_text)
            logger.info(f"Processing chunk {i + 1}/{len(texts)} with {chunk_tokens} tokens")
            if chunk_tokens > max_token_limit:
                chunk_text = chunk_text[:max_token_limit * 4 // 3]
                logger.info(f"Truncated chunk {i + 1} to {count_tokens(chunk_text)} tokens")
            chain = load_summarize_chain(
                llm,
                chain_type="stuff",
                prompt=map_prompt,
                verbose=False
            )
            summary = invoke_chain(chain, [chunk])
            chunk_summaries.append(summary)

        combined_text = "\n".join(chunk_summaries)
        combined_tokens = count_tokens(combined_text)
        logger.info(f"Combined pre-summaries length: {len(combined_text)} characters, {combined_tokens} tokens")
        if combined_tokens > max_token_limit:
            combined_text = combined_text[:max_token_limit * 4 // 3]
            logger.info(f"Truncated combined text to {count_tokens(combined_text)} tokens")

        combined_doc = [Document(page_content=combined_text)]
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=combine_prompt,
            verbose=False
        )
        result = invoke_chain(chain, combined_doc)
    else:
        logger.info("Using stuff chain")
        chain = load_summarize_chain(
            llm,
            chain_type="stuff",
            prompt=combine_prompt,
            verbose=False
        )
        result = invoke_chain(chain, texts)

    chain_duration = time.time() - start_chain
    logger.info(f"Chain execution completed in {chain_duration:.2f} seconds")

    total_duration = time.time() - start_total
    logger.info(f"Total summarization time: {total_duration:.2f} seconds")

    return result