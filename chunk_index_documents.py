import logging as logger
import os.path

from langchain import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter, NLTKTextSplitter

from utils.functions import read_pdf_to_string, get_pdf_paths
from utils.project_config import config
import nltk

nltk.download('punkt')
from langchain.embeddings import HuggingFaceInstructEmbeddings


pdfs = get_pdf_paths(config['data_dir_path'])
print(f"PDFs scanned = {len(pdfs)}")

for i in range(0, len(pdfs), 2):
    print(f"i = {i}")
    paths = pdfs[i:(i+2)]
    (documents_string, document_names) = read_pdf_to_string(paths)
    logger.info("Documents scanned = " + str(len(documents_string)))
    # TODO plot token graph
    # model.config.max_sequence_length

    logger.info("======Context aware chunking======")
    # Llama 2 supports up to 4096 tokens
    chunk_size = 600
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(separators=[".", "\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # split text into chunks
    text_content = []
    metadata = []
    for i in range(0, len(documents_string)):
        print(f"Splitting document: {document_names[i]}")
        splits = text_splitter.split_text(documents_string[i])
        text_content.extend(splits)
        metadata.extend([{"file_name": document_names[i]}]*len(splits))

    logger.info("Number of chunks = " + str(len(text_content)))

    logger.info("======Embedding model======")
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=config['EMBEDDING_MODEL_NAME'],
        model_kwargs={"device": "cpu"},
    )

    if not os.path.exists(config['vector_store_path']):
        print("\n\n\n\n Creating embeddings folder")
        os.makedirs(config['vector_store_path'], exist_ok=True)
        print("Creating vector store ")
        vector_store = FAISS.from_texts(
            embedding=embeddings,
            texts=text_content,
            metadatas=metadata
        )
    else:
        vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])
        vector_store.add_texts(text_content)
    print("Saving vector store")
    vector_store.save_local(folder_path=config['vector_store_path'])
