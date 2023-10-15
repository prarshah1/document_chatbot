import json
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex,
)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.storage.storage_context import StorageContext
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.storage.storage_context import StorageContext
from src.utils.functions import get_llama2_7B_chat_llm
import logging
import os
from llama_index.indices.knowledge_graph import KGTableRetriever
from llama_index.query_engine import RetrieverQueryEngine
import sys
from llama_index import (
    ServiceContext, load_index_from_storage, set_global_service_context, get_response_synthesizer, QueryBundle,
)
from pyvis.network import Network
import networkx as nx

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

with open('src/resources/jsons/nvenc.json') as f:
    data = json.load(f)

nodes = []
edges = []

def parse_data(data, parent_node=None):
    for key, value in data.items():
        if isinstance(value, dict):
            # create a node for the key
            node = key
            nodes.append(node)

            # create an edge from the parent node to this node
            if parent_node is not None:
                edge = (parent_node, node)
                edges.append(edge)

            # recursively parse the nested data
            parse_data(value, parent_node=node)
        else:
            # create a node for the value
            node = value
            nodes.append(node)

            # create an edge from the key to this node
            edge = (key, node)
            edges.append(edge)


# data = parse_data(data)


os.environ["LLAMA_INDEX_CACHE_DIR"] = "/Users/prar_shah/study/PycharmProjects/document_chatbot/src/resources/cache"
llm = get_llama2_7B_chat_llm()

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=512)

graph_store = SimpleGraphStore()
storage_context = StorageContext.from_defaults(graph_store=graph_store)

# NOTE: can take a while!
from llama_index import TreeIndex
index = TreeIndex.build_index_from_nodes(
    data,  # assuming data is your loaded and parsed JSON
    max_triplets_per_chunk=3,
    storage_context=storage_context,
    service_context=service_context,
)


query_engine = index.as_query_engine(include_text=False, response_mode="tree_summarize")
response = query_engine.query("How to create a floating CUDA context?")
# Display the response: Finally, you can display the response from your query. Here's an example:

print(response)