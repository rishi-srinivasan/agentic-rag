from llama_index.core.base.response.schema import StreamingResponse, AsyncStreamingResponse, PydanticResponse
from helpers.helper import get_openai_api_key
from llama_index.core import SimpleDirectoryReader, Response
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core.schema import MetadataMode
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
import nest_asyncio


# implement async and retrieve openAI key
nest_asyncio.apply()
OPENAI_API_KEY = get_openai_api_key()


def get_tools(
        file_path: str
) -> tuple[QueryEngineTool, FunctionTool]:
    """
    Loads the document, sets up the indexes, creates query engines, and returns tools
    :param file_path:
    :return: vector query tool and summary query tool
    """

    # Load the documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Split the document into nodes based on chunk size
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=500)
    nodes = splitter.get_nodes_from_documents(documents)
    print(nodes[0].get_content(metadata_mode=MetadataMode.ALL))

    # Define embedding model and llm model in global settings
    Settings.llm = OpenAI(temperature=0, model="gpt-4o")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Define summary index and vector index - Summary index returns all nodes,
    # vector index returns nodes based on embedding + query
    summary_index = SummaryIndex(nodes)
    vector_index = VectorStoreIndex(nodes)

    # Define query engines which is an abstraction tool over indexes
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    def vector_query(query: str, page_numbers: Optional[List[str]] = None) -> (
            Response | StreamingResponse | AsyncStreamingResponse | PydanticResponse):
        """
        Perform a vector search over an index.
        :param query: string query to be embedded
        :param page_numbers: (Optional) Filter by pages. Leave BLANK if we want to perform vector search over all pages.
        :return: response from LLM as string
        """
        page_numbers = page_numbers or []
        metadata_dict = [
            {
                "key": "page_label", "value": p
            } for p in page_numbers
        ]
        vector_query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filter=MetadataFilters.from_dicts(
                metadata_dict,
                condition=FilterCondition.OR
            )
        )
        response_from_llm = vector_query_engine.query(query)
        return response_from_llm

    # Define query tool which is an abstraction tool over query engines
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description='Summarizing the documents'
    )

    vector_tool = FunctionTool.from_defaults(
        name="vector_tool",
        description="Retrieve specific context from the documents",
        fn=vector_query
    )
    return summary_tool, vector_tool
