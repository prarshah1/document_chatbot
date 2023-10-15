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

# from IPython.display import Markdown, display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



os.environ["LLAMA_INDEX_CACHE_DIR"] = "/Users/prar_shah/study/PycharmProjects/document_chatbot/src/resources/cache"
llm = get_llama2_7B_chat_llm()

# graph_store = SimpleGraphStore()

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=500, chunk_overlap=150, context_window=2048, num_output=512,)
set_global_service_context(service_context)

# storage_context = StorageContext.from_defaults(graph_store=graph_store)
persist_dir = "src/resources/storage_context1/"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

kg_index = load_index_from_storage(storage_context, index_id="143e309f-ee16-444a-9220-7e11a09fc321")
# noinspection PyTypeChecker
kg_retriever = KGTableRetriever(
    index=kg_index, retriever_mode="keyword", include_text=False
)
question = "How to pass externally allocated input buffers to NVENC?"
query_bundle = QueryBundle(question)

query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
response = query_engine.query(question)
# create graph
print(str(response))
# g = index.get_networkx_graph()
# net = Network(notebook=True, cdn_resources="in_line", directed=True)
# net.from_nx(g)
# net.show("example.html")
# a = 0
#
