import os

if 'config' not in locals():
  config = {}

config['data_dir_path'] = "new_pdfs"

# Default Instructor Model
config['EMBEDDING_MODEL_NAME'] = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
#### OTHER EMBEDDING MODEL OPTIONS
# EMBEDDING_MODEL_NAME = "hkunlp/instructor-xl" # Uses 5 GB of VRAM (Most Accurate of all models)
# EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2" # Uses 1.5 GB of VRAM (A little less accurate than instructor-large)
# EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2" # Uses 0.5 GB of VRAM (A good model for lower VRAM GPUs)
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Uses 0.2 GB of VRAM (Less accurate but fastest - only requires 150mb of vram)
#### MULTILINGUAL EMBEDDING MODELS
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # Uses 2.5 GB of VRAM
# EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base" # Uses 1.2 GB of VRAM

config["vector_store_path"] = "resources/embeddings"
config["llama2_model_path"] = "/Users/prar_shah/study/PycharmProjects/llama.cpp/models/7B_chat/ggml-model-q4_0.bin"
config["system_message_template"]="""You are an intelligent and excellent at answering questions about NVIDIA technologies.
I will ask questions from the documents and you'll help me try finding the answers from it.
Take a while to think, give the answer using using the document, best of your knowledge.
If possible add code as example. Properly elaborate the answer.
If you are unable to answer say I dont know.
If the context does not provide enough relevant information to determine the answer, just say I don't know. 
If the context is irrelevant to the question, just say I don't know. 
If you did not find a good answer from the context, just say I don't know."""
