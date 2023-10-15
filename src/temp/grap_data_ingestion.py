import logging
import os

from llama_index.indices.knowledge_graph import KGTableRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import KnowledgeGraphRAGRetriever

from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.vector_stores import SimpleVectorStore
from pyvis.network import Network
import sys
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    KnowledgeGraphIndex, load_index_from_storage, set_global_service_context, get_response_synthesizer, QueryBundle,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import OpenAI
from src.utils.functions import get_llama2_7B_chat_llm

# from IPython.display import Markdown, display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



os.environ["LLAMA_INDEX_CACHE_DIR"] = "/Users/prar_shah/study/PycharmProjects/document_chatbot/src/resources/cache"
documents = SimpleDirectoryReader(
    "/src/resources/new_pdfs/temp/"
).load_data()
llm = get_llama2_7B_chat_llm()

# graph_store = SimpleGraphStore()

service_context = ServiceContext.from_defaults(llm=llm, chunk_size=500, chunk_overlap=150, context_window=2048, num_output=512,)
set_global_service_context(service_context)

# storage_context = StorageContext.from_defaults(graph_store=graph_store)
persist_dir = "src/resources/storage_context/"
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
# NOTE: can take a while!
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    max_triplets_per_chunk=10,
    storage_context=storage_context,
    service_context=service_context,
    include_embeddings=True,
)

storage_context.persist(persist_dir.replace("storage_context", "storage_context1"))
# kg_index = load_index_from_storage(storage_context, service_context=service_context)
# noinspection PyTypeChecker
kg_retriever = KGTableRetriever(
    index=kg_index, retriever_mode="keyword", include_text=False
)
question = "How to pass externally allocated input buffers to NVENC ?"
query_bundle = QueryBundle(question)
kg_nodes = kg_retriever.retrieve(query_bundle)
response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    response_mode="tree_summarize",
)
kg_query_engine = RetrieverQueryEngine(
    retriever=kg_retriever,
    response_synthesizer=response_synthesizer,
)

response = kg_query_engine.query(question)

# query_engine = index.as_retriever(
#     include_text=True,
#     response_mode="tree_summarize",
#     embedding_mode="hybrid",
#     similarity_top_k=5,
# )
# response = query_engine.query(question)
## create graph

# g = index.get_networkx_graph()
# net = Network(notebook=True, cdn_resources="in_line", directed=True)
# net.from_nx(g)
# net.show("example.html")
# a = 0

