from langchain_core.prompts import PromptTemplate

# For map step: Initial summarization of each chunk
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are an expert assistant. Read the following content and write a concise, clear, and informative summary of it.

Content:
{text}

Summary:
"""
)

# For reduce step: Final combination and polishing of the summary
combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are a helpful assistant. Based on the following partial summaries, generate a complete, structured summary of the original content.

- Begin with a suitable title for the content.
- Follow with a brief introduction.
- Present the key points as a numbered list.
- Make sure the summary is contextually accurate and concise.

Partial Summaries:
{text}

Final Structured Summary:
"""
)
