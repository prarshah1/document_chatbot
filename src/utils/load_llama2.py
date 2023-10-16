from langchain import LLMChain
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from llama_cpp import llama_cpp

from src.utils.project_config import config

# llm = llama_cpp.llama_load_model_from_file(
#                    path_model="models/7B_chat/ggml-model-q4_0.bin".encode("utf-8"), params=llama_cpp.llama_context_default_params()
#                 )

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
# llama = LlamaCppEmbeddings(model_path=config["llama2_model_path"])
question = """What is NVIDIA's full form?"""

llm = LlamaCpp(model_path=config["llama2_model_path"],
               temperature=0.5,
               max_tokens=2000,
               top_p=1,
               n_gpu_layers=1,
               f16_kv=True,
               n_batch=512,
               n_threads=128,
               callback_manager=callback_manager,
               verbose=True)
print(llm(question))
# llm_chain = LLMChain(prompt=prompt, llm=llm)
# print(llm_chain.run(question))