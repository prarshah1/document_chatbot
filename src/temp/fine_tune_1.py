import json
from llama_index.finetuning import (
    generate_qa_embedding_pairs,
    EmbeddingQAFinetuneDataset,
)
from llama_index.finetuning import SentenceTransformersFinetuneEngine

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
from sentence_transformers import SentenceTransformer

import src.utils.functions

TRAIN_FILES = ["src/resources/new_pdfs/temp/NVDEC_VideoDecoder_API_ProgGuide.pdf"]
VAL_FILES = ["src/resources/new_pdfs/temp/NVDEC_VideoDecoder_API_ProgGuide.pdf"]

# TRAIN_CORPUS_FPATH = "src/resources/data_corpus/train_corpus.json"
# VAL_CORPUS_FPATH = "src/resources/data_corpus/val_corpus.json"

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")
    return nodes

train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

llm = src.utils.functions.get_llama2_7B_chat_llm()

prompt_template = """
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/Professor. Your task is to setup 
{num_questions_per_chunk} questions for an upcoming 
quiz/examination. The questions should be diverse in nature 
across the document. Restrict the questions to the 
context information provided.
Every question should be on a new line.
Do not include multiple choice questions.
"""
train_dataset = generate_qa_embedding_pairs(train_nodes, llm=llm, qa_generate_prompt_tmpl=prompt_template, num_questions_per_chunk=10)

train_dataset.save_json("src/resources/finetune_embed_model_data/train_dataset.json")

train_dataset = EmbeddingQAFinetuneDataset.from_json("src/resources/finetune_embed_model_data/train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("src/resources/finetune_embed_model_data/train_dataset.json")

finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="test_model",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()

