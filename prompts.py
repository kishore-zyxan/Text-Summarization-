from langchain_core.prompts import PromptTemplate

map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You will be provided with a section of a document. Please summarize it concisely.

Content:
{text}

Summary:
"""
)

combine_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Below is a collection of summaries. Combine them into a final summary with:
- A catchy title
- A brief introduction
- A list of key points in a numbered format

Summaries:
{text}

Final Summary:
"""
)
