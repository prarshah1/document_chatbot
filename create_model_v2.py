import asyncio
import logging
from langchain import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
import mlflow
import pandas as pd
import time
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

from src.utils.functions import *
from utils.project_config import config
logging.basicConfig(level=logging.INFO)


# open vector store to access embeddings
embeddings = HuggingFaceInstructEmbeddings(
        model_name=config['EMBEDDING_MODEL_NAME'],
        model_kwargs={"device": "cpu"},
    )
vector_store = FAISS.load_local(embeddings=embeddings, folder_path="/Users/prar_shah/study/PycharmProjects/nvidia_document_bot/resources/embeddings/Video_Codec_SDK")
# vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'] + "/cuda")

# configure document retrieval
n_documents = 30 # number of documents to retrieve
retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism

prompt_template = config['system_message_template']
qa_prompt = PromptTemplate.from_template(prompt_template)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

rails = get_rails()
qa_chain = ConversationalRetrievalChain.from_llm(llm=rails.llm,
                                                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                                                memory=memory,
                                                condense_question_prompt=qa_prompt)
rails.register_action(qa_chain, name="qa_chain")
# for each provided document
question = "What are the different modes to split encoding in ENC?"
docs = retriever.get_relevant_documents(question)
output = rails.generate(
        messages=[{"role": "user", "content": question, "context": ""}])
print(output)
# docs = retriever.get_relevant_documents(question)
# for doc in docs:
#   # get document text
#   print("Generating answer")
#   print("\n\n\n\n")
#   text = doc.page_content
#   print(f"{text}")
