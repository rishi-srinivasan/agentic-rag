from helpers.utils import get_tools
from llama_index.llms.openai import OpenAI

path = './data/NIPS-2017-attention-is-all-you-need-Paper.pdf'
docName = 'transformers'

# Load the document tools
summary_tool, vector_tool = get_tools(path)

# Setup function calling agent
llm = OpenAI(model='gpt-4o', temperature=0)
