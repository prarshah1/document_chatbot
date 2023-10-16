import sqlite3
import sys
import os

# this is to make src accessible
sys.path.append(os.path.abspath(os.curdir))
from src.utils.functions import *
# __import__('pysqlite3')
import sys

# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
connection = sqlite3.connect('../cache.db', timeout=100)

import streamlit as st
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, FAISS

st.set_page_config(page_title='NVIDIA Documentation Bot')

st.image("/Users/prar_shah/Desktop/nvassist.png")
preloaded_documents = ['Cuda', 'Nsight_compute', 'Nsight_systems', 'Video_Codec_SDK']
LLMDATA = {}
QA_CHAIN = None

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'current_filename' not in st.session_state:
    st.session_state.current_filename = None

if "messages" not in st.session_state:
    st.session_state.messages = {}

if "vector_db" not in st.session_state:
    st.session_state.vector_db = {}

if "index" not in st.session_state:
    st.session_state.index = 0

if "selectbox_options" not in st.session_state:
    st.session_state["selectbox_options"] = ['', *preloaded_documents, 'Upload']

if "document_type" not in st.session_state:
    st.session_state.document_type = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


# with st.write("Loading embeddings:"):
#     LLMDATA = {"cuda": {"db": load_db("cuda")},
#      "Nsight_compute": {"db": load_db("Nsight_compute")},
#      "Nsight_systems": {"db": load_db("Nsight_systems")},
#      "Video_Codec_SDK": {"db": load_db("Video_Codec_SDK")}
#      }

# def create_documents(uploaded_file):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, chunk_overlap=1000, length_function=len)
#     text = []
#     if ("txt" == uploaded_file.name.split(".")[-1]) or ("json" == uploaded_file.name.split(".")[-1]):
#         stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#         text.append(stringio.read())
#     elif "pdf" == uploaded_file.name.split(".")[-1]:
#         pdf_text=""
#         for page_layout in extract_pages(uploaded_file):
#             for element in page_layout:
#                 if isinstance(element, LTTextContainer):
#                     pdf_text+=element.get_text()
#         text.append(pdf_text)
#     else:
#         raise Exception(f"File format {uploaded_file.name.split('.')[-1]} not supported")
#     return text_splitter.create_documents(text)
#
#
# def set_LLM(uploaded_file):
#     global LLMDATA
#     if uploaded_file is not None:
#         filename = uploaded_file.name
#         st.session_state["current_filename"] = filename
#         if filename not in LLMDATA:
#             print("uploaded_file(name)>>>>>>>>>", filename)
#             documents = create_documents(uploaded_file)
#             embeddings = OpenAIEmbeddings()
#             db = Chroma.from_documents(documents, embeddings)
#             LLMDATA[filename] = {
#                 "db": db
#             }
#             st.session_state['LLMDATA'] = LLMDATA
#             print(len(LLMDATA))
#
#
# def generate_response(query_text, filename):
#     if filename in LLMDATA:
#         db = LLMDATA[filename]["db"]
#         system_template = """
#         You are an intelligent bot, excellent at finding answers from the documents.
#         I will ask questions from the documents and you'll help me try finding the answers from it.
#         Give the answer using best of your knowledge, say you dont know if not able to answer.
#         ---------------
#         {context}
#         """
#         qa_prompt = PromptTemplate.from_template(system_template)
#         memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, model_name="gpt-4"), db.as_retriever(),
#                                                    memory=memory, condense_question_prompt=qa_prompt)
#         result = qa({"question": query_text})
#         # return result["answer"]
#         dict = {"question": result["question"], "answer": result["answer"]}
#         st.session_state.QA.append(dict)
#
#
def file_upload_form():
    with st.form('fileform'):
        supported_file_types = ["pdf", "txt", "json"]
        uploaded_file = st.file_uploader("Upload a file", type=(supported_file_types))
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.session_state.uploaded_file = uploaded_file
            if uploaded_file is not None:
                if uploaded_file.name.split(".")[-1] in supported_file_types:
                    # set_LLM(uploaded_file)
                    st.session_state.current_filename = uploaded_file.name
                else:
                    st.write(f"Supported file types are {', '.join(supported_file_types)}")
            else:
                st.write("Please select a file to upload first!")


def load_embeddings(document_type):
    if document_type not in st.session_state.vector_db:
        st.session_state.vector_db[document_type] = load_db(document_type.lower())


def set_retriever(document_type):
    # number of documents to retrieve
    n_documents = 5
    # configure retrieval mechanism
    st.session_state.retriever = st.session_state.vector_db[document_type].as_retriever(
        search_kwargs={'k': n_documents})


def set_qa_chain_preloaded_docs():
    prompt_template = config['system_message_template']
    qa_prompt = PromptTemplate.from_template(template=prompt_template)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    os.environ["DISABLE_NEST_ASYNCIO"] = "true"
    # loop = asyncio.new_event_loop()
    # asyncio.set_event_loop(loop)
    import nemoguardrails

    rails_config = nemoguardrails.RailsConfig.from_path(config["rails_config_path"])
    # colang_content=config["rails_colang_config"],
    # yaml_content=config["rails_yaml_config"])
    # rails_config.config_path = config["rails_config_path"]
    llm = get_llama2_7B_chat_llm()
    rails = nemoguardrails.LLMRails(rails_config, llm=llm)
    rails.runtime.register_action_param("llm", llm)

    qa_chain = ConversationalRetrievalChain.from_llm(llm=rails.llm,
                                                     retriever=st.session_state.retriever,
                                                     memory=memory,
                                                     condense_question_prompt=qa_prompt,
                                                     )
                                                     # combine_docs_chain_kwargs={"prompt": qa_prompt})
    # rails.register_prompt_context("context", st.session_state.retriever.get_relevant_documents)
    rails.register_action(qa_chain, name="qa_chain")
    # st.session_state.qa_chain = rails
    return rails


def display_chat(document_type):
    if st.session_state.document_type in st.session_state.messages:
        for message in st.session_state.messages[document_type]:
            with st.chat_message(message["role"]):
                st.markdown(f"""{message["content"]}""".replace("\n", "\n\n"))
    else:
        st.session_state.messages[document_type] = []

def reset_qa_chain():
    global QA_CHAIN
    QA_CHAIN = None

def start_conversation():
    global preloaded_documents
    global QA_CHAIN
    st.session_state.document_type = st.selectbox(
        label="Select document type",
        placeholder='Select document type',
        options=st.session_state.selectbox_options,
        on_change=reset_qa_chain,
        key="document_type_selectbox",
        index=st.session_state.index)
    if st.session_state.document_type != '':
        document_type = st.session_state.document_type
        st.session_state.index = st.session_state.selectbox_options.index(document_type)
        if document_type in preloaded_documents and QA_CHAIN is None:
            with st.spinner(f"Configuring chatbot for {document_type}"):
                load_embeddings(document_type)
                set_retriever(document_type)
                QA_CHAIN = set_qa_chain_preloaded_docs()
            display_chat(document_type)
        elif document_type == "Upload" and QA_CHAIN is None:
            file_upload_form()

        if prompt := st.chat_input("What is up?"):
            st.session_state.messages[document_type].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(f"""{prompt}""".replace("\n", "\n\n"))

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                docs = st.session_state.retriever.get_relevant_documents(prompt)
                response = QA_CHAIN.generate(  # noqa
                    messages=[{"role": "context", "content": {"context": [doc.page_content for doc in docs], "question": prompt}},
                              {"role": "user", "content": prompt}])
                message_placeholder.markdown(f"""{response['content']}""".replace("\n", "\n\n"))
            st.session_state.messages[document_type].append(response)

start_conversation()
