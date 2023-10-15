import logging
import os
import fitz  # PyMuPDF
from langchain import HuggingFacePipeline, FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import CTransformers
from transformers import pipeline, TextStreamer, AutoTokenizer
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_cpp import llama_cpp

from src.utils.project_config import config
import torch
from langchain import HuggingFacePipeline, PromptTemplate
from transformers import AutoTokenizer, TextStreamer, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_pdf_paths(dir_path):
    paths = []

    for file_path in os.listdir(dir_path):
        if file_path.endswith(".pdf"):
            doc_path = os.path.join(dir_path, file_path)
            paths.append(doc_path)
        else:
            paths.extend(get_pdf_paths(f"{dir_path}/{file_path}"))
    return paths
def read_pdf_to_string(paths, pdfs_start):
    pdf_to_str = []
    doc_names = []
    for doc_path in paths:
        try:
            doc = fitz.open(doc_path)
            text = ""
            for page_num in range(doc.page_count):
                if page_num > pdfs_start[doc_path]:
                    page = doc[page_num]
                    text += page.get_text()
            pdf_to_str.append(text)
            doc_names.append(doc_path)
        except:
            print(f"Empty PDF: {doc_path}")
    return (pdf_to_str, doc_names)

def get_rails():
    import nemoguardrails
    rails_config = nemoguardrails.RailsConfig.from_path("resources/rails_config")
    # colang_content=config["rails_colang_config"],
    # yaml_content=config["rails_yaml_config"])
    # rails_config.config_path = config["rails_config_path"]
    llm=get_llama2_7B_chat_llm()
    rails = nemoguardrails.LLMRails(rails_config, llm=llm)
    rails.runtime.register_action_param("llm", llm)
    return rails

def get_llama2_7B_chat_llm():
    # llama_cpp.llama_load_model_from_file(
    #     path_model="models/7B_chat/ggml-model-q4_0.bin".encode("utf-8"), params=llama_cpp.llama_context_default_params()
    # )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    return LlamaCpp(model_path=config["llama2_model_path"],
                    temperature=0.5,
                    n_ctx=2048,
                    max_tokens=4096,
                    top_p=1,
                    n_gpu_layers=1,
                    f16_kv=True,
                    n_batch=512,
                    n_gqa=8,
                    n_threads=128,
                    callback_manager=callback_manager,
                    verbose=True)

def load_db(_type):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=config['EMBEDDING_MODEL_NAME'],
        model_kwargs={"device": "cpu"},
    )
    return FAISS.load_local(embeddings=embeddings, folder_path=os.path.join(config['vector_store_path'], _type))
