# import os
# import aiohttp
# import asyncio
# from pathlib import Path
# from typing import List
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema import StrOutputParser
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores.chroma import Chroma
# from langchain.indexes import SQLRecordManager, index
# from langchain.schema import Document
# from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
# from langchain.callbacks.base import BaseCallbackHandler
#
# import chainlit as cl
# from chainlit.types import AskFileResponse
#
# # Define chunk size and overlap for text splitting
# chunk_size = 1024
# chunk_overlap = 50
#
# # Set the embeddings model
# embeddings_model = OpenAIEmbeddings()
#
# # Directory to store PDFs uploaded by the user
# PDF_STORAGE_PATH = "./uploaded_pdfs"
#
# # FastAPI backend URL for processing PDF
# FASTAPI_BACKEND_URL = "http://localhost:8000/assess_o1a/"  # Make sure main.py is running on this URL
#
#
# def process_uploaded_file(file: AskFileResponse):
#     """
#     Save the uploaded file to the local file system and process it using ChromaDB.
#     """
#     if not os.path.exists(PDF_STORAGE_PATH):
#         os.makedirs(PDF_STORAGE_PATH)
#
#     file_path = file.path  # Use the file path from AskFileResponse object
#
#     # Process the saved PDF
#     return process_pdf(file_path)
#
#
# def process_pdf(pdf_path: str):
#     """
#     Process a single PDF file by loading and splitting it into chunks, then index it using ChromaDB.
#     """
#     docs = []  # List to hold split documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#
#     # Load PDF using PyMuPDFLoader
#     loader = PyMuPDFLoader(pdf_path)
#     documents = loader.load()
#     docs += text_splitter.split_documents(documents)
#
#     # Create Chroma vector store for document search
#     doc_search = Chroma.from_documents(docs, embeddings_model)
#
#     # Set up SQL-based record manager for incremental indexing
#     namespace = "chromadb/my_documents"
#     record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
#     record_manager.create_schema()
#
#     # Index documents with Chroma and SQLRecordManager
#     index_result = index(docs, record_manager, doc_search, cleanup="incremental", source_id_key="source")
#     print(f"Indexing stats: {index_result}")
#
#     return doc_search
#
#
# async def call_fastapi_backend(file_path: str):
#     """
#     Function to call FastAPI functionality to process the uploaded PDF.
#     This is run asynchronously in the background while the user does QA.
#     """
#     async with aiohttp.ClientSession() as session:
#         with open(file_path, 'rb') as file_data:
#             files = {'file': file_data}
#             async with session.post(FASTAPI_BACKEND_URL, data=files) as resp:
#                 print(await resp.text())  # Print or log the FastAPI response
#
#
# @cl.on_chat_start
# async def start():
#     """
#     This function handles the start of a Chainlit session, where a user uploads a PDF.
#     """
#     files = None
#     while files is None:
#         files = await cl.AskFileMessage(
#             content="Please upload a PDF file for processing:",
#             accept=["application/pdf"],
#             max_size_mb=20,
#             timeout=180,
#         ).send()
#
#     file = files[0]
#
#     msg = cl.Message(content=f"Processing `{file.name}`...")
#     await msg.send()
#
#     # Process the uploaded PDF and create a document search index
#     doc_search = process_uploaded_file(file)
#
#     # Start the FastAPI process for extraction and evaluation in the background
#     asyncio.create_task(call_fastapi_backend(file.path))
#
#     # Define the prompt template for answering questions based on context
#     template = """Answer the question based only on the following context:
#
#     {context}
#
#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)
#
#     # Helper function to format the documents for context
#     def format_docs(docs):
#         return "\n\n".join([d.page_content for d in docs])
#
#     # Create a runnable pipeline for the document search and LLM response
#     retriever = doc_search.as_retriever()
#     runnable = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()}
#         | prompt
#         | ChatOpenAI(model_name="gpt-4", streaming=True)
#         | StrOutputParser()
#     )
#
#     # Store the runnable chain in the user session
#     cl.user_session.set("runnable", runnable)
#
#     msg.content = f"`{file.name}` has been processed. You can now ask questions!"
#     await msg.update()
#
#
# @cl.on_message
# async def handle_message(message: cl.Message):
#     """
#     Handle incoming user messages to run the question-answering pipeline.
#     """
#     runnable = cl.user_session.get("runnable")  # Retrieve the runnable chain from the session
#     msg = cl.Message(content="")
#
#     # Define a callback handler to display the sources of the retrieved documents
#     class PostMessageHandler(BaseCallbackHandler):
#         def __init__(self, msg: cl.Message):
#             BaseCallbackHandler.__init__(self)
#             self.msg = msg
#             self.sources = set()  # To store unique pairs of document sources and pages
#
#         def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
#             # Store document source and page pairs
#             for d in documents:
#                 source_page_pair = (d.metadata.get('source'), d.metadata.get('page'))
#                 self.sources.add(source_page_pair)
#
#         def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
#             # Display document sources after LLM response
#             if self.sources:
#                 print(self.sources)
#                 sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
#                 self.msg.elements.append(cl.Text(name="Sources", content=sources_text, display="inline"))
#
#     # Run the question-answering pipeline with callback handlers
#     async for chunk in runnable.astream(
#         message.content,
#         config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)]),
#     ):
#         await msg.stream_token(chunk)
#
#     await msg.send()




import os
import aiohttp
import asyncio
import pandas as pd
from pathlib import Path
from typing import List
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain.callbacks.base import BaseCallbackHandler

import chainlit as cl
from chainlit.types import AskFileResponse
import csv
from datetime import datetime

# Define chunk size and overlap for text splitting
chunk_size = 1024
chunk_overlap = 50

# Set the embeddings model
embeddings_model = OpenAIEmbeddings()

# Directory to store PDFs uploaded by the user
PDF_STORAGE_PATH = "./uploaded_pdfs"
RESULTS_DIR = "./results"

# CSV reference file path
REFERENCE_SHEET_PATH = "reference_sheet.csv"
FIRST_PASS_RESULTS_PATH = "first_pass_results.csv"
SECOND_PASS_RESULTS_PATH = "second_pass_results.csv"

# FastAPI backend URL for processing PDF
FASTAPI_BACKEND_URL = "http://localhost:8000/assess_o1a/"  # Make sure main.py is running on this URL


def generate_final_results_txt(evaluations, final_rating):
    """
    Generate a TXT file with the overall results based on the evaluations and final rating.
    """
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    file_path = os.path.join(RESULTS_DIR, f"final_results.txt")

    with open(file_path, 'w') as file:
        file.write("O-1A Visa Assessment Results\n")
        file.write("===================================\n")
        for criterion, evaluation in evaluations.items():
            file.write(f"{criterion}: {evaluation}\n")
        file.write("\nOverall Qualification Rating: " + final_rating)

    return file_path


def determine_final_rating(evaluations):
    """
    Determine the final rating based on the top 3 evidence rankings.
    """
    evidence_ranking = {"no evidence": 0, "weak evidence": 1, "reasonable evidence": 2, "remarkable evidence": 3}

    # Sort the evaluations based on their rankings
    sorted_evaluations = sorted(evaluations.items(), key=lambda x: evidence_ranking[x[1]], reverse=True)

    # Select the top 3 criteria
    top_3 = [evaluation for criterion, evaluation in sorted_evaluations[:3]]

    # Determine the final rating based on the rules provided
    if top_3.count("remarkable evidence") >= 2 and top_3.count("reasonable evidence") >= 1:
        return "[High]"
    if top_3.count("remarkable evidence") == 3 or top_3.count("reasonable evidence") == 3:
        return "[High]"
    if top_3.count("reasonable evidence") == 2 and top_3.count("weak evidence") == 1:
        return "[Medium]"
    return "[Low]"


async def find_sources_for_criteria(criteria, retriever):
    """
    Automatically find sources for each O-1A category using the retriever.
    """
    sources = {}

    for criterion in criteria:
        if criterion in REFERENCE_DATA:
            description, criteria_details = REFERENCE_DATA[criterion]
            query = f"{description}\nCriteria: {criteria_details}"

            # Simulate a message sent by a user to retrieve relevant sources
            user_message = cl.Message(content=query)
            msg = await handle_message(user_message)

            if msg is not None and hasattr(msg, 'elements'):
                # Collect the sources and associate them with the criterion
                sources[criterion] = msg.elements
            else:
                sources[criterion] = ["No sources found"]

    return sources

# Load reference sheet once at the start
def load_reference_sheet():
    """
    Load the reference sheet containing descriptions and criteria for the categories.
    """
    df = pd.read_csv(REFERENCE_SHEET_PATH)
    reference_data = {
        row['Category']: (row['Description'], row['Criteria'])
        for _, row in df.iterrows()
    }
    return reference_data

# Read reference sheet data globally
REFERENCE_DATA = load_reference_sheet()

# Load first-pass CSV results after backend processing
def load_first_pass_results():
    """
    Load the first pass results from a CSV file.
    """
    if os.path.exists(FIRST_PASS_RESULTS_PATH):
        df = pd.read_csv(FIRST_PASS_RESULTS_PATH)
        results = {row['Criterion']: row['Extraction'] for _, row in df.iterrows()}
        return results
    else:
        raise FileNotFoundError(f"File not found: {FIRST_PASS_RESULTS_PATH}")

def load_second_pass_results():
    """
    Load the second pass results from a CSV file.
    """
    if os.path.exists(SECOND_PASS_RESULTS_PATH):
        df = pd.read_csv(SECOND_PASS_RESULTS_PATH)
        results = {row['Criterion']: row['Evaluation'] for _, row in df.iterrows()}
        return results
    else:
        raise FileNotFoundError(f"File not found: {SECOND_PASS_RESULTS_PATH}")

def process_uploaded_file(file: AskFileResponse):
    """
    Save the uploaded file to the local file system and process it using ChromaDB.
    """
    if not os.path.exists(PDF_STORAGE_PATH):
        os.makedirs(PDF_STORAGE_PATH)

    file_path = file.path  # Use the file path from AskFileResponse object

    # Process the saved PDF
    return process_pdf(file_path)


def process_pdf(pdf_path: str):
    """
    Process a single PDF file by loading and splitting it into chunks, then index it using ChromaDB.
    """
    docs = []  # List to hold split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Load PDF using PyMuPDFLoader
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    docs += text_splitter.split_documents(documents)

    # Create Chroma vector store for document search
    doc_search = Chroma.from_documents(docs, embeddings_model)

    # Set up SQL-based record manager for incremental indexing
    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(namespace, db_url="sqlite:///record_manager_cache.sql")
    record_manager.create_schema()

    # Index documents with Chroma and SQLRecordManager
    index_result = index(docs, record_manager, doc_search, cleanup="incremental", source_id_key="source")
    print(f"Indexing stats: {index_result}")

    return doc_search


async def call_fastapi_backend(file_path: str):
    """
    Function to call FastAPI functionality to process the uploaded PDF.
    This is run asynchronously in the background while the user does QA.
    """
    async with aiohttp.ClientSession() as session:
        with open(file_path, 'rb') as file_data:
            files = {'file': file_data}
            async with session.post(FASTAPI_BACKEND_URL, data=files) as resp:
                print(await resp.text())  # Print or log the FastAPI response

@cl.on_chat_start
async def start():
    """
    This function handles the start of a Chainlit session, where a user uploads a PDF.
    """
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file for processing:",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Process the uploaded PDF and create a document search index
    doc_search = process_uploaded_file(file)

    # Start the FastAPI process for extraction and evaluation in the background
    await call_fastapi_backend(file.path)

    # Once backend processing is complete, load the first and second pass results
    try:
        FIRST_PASS_RESULTS = load_first_pass_results()
        SECOND_PASS_RESULTS = load_second_pass_results()
    except FileNotFoundError as e:
        msg.content = str(e)
        await msg.update()
        return

    # Use second pass results for the evaluations
    evaluations = SECOND_PASS_RESULTS

    final_rating = determine_final_rating(evaluations)
    result_file = generate_final_results_txt(evaluations, final_rating)

    # Let the user know that the system is ready with the final assessment
    msg.content = f"`{file.name}` has been processed. You can now ask questions!\nThe final assessment of this O1-A applicant is: {final_rating}.\nYou can also review the detailed results in the file ({result_file})."
    await msg.update()

    # Define the prompt template for answering questions based on context
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Helper function to format the documents for context
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    # Create a runnable pipeline for the document search and LLM response
    retriever = doc_search.as_retriever()
    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | ChatOpenAI(model_name="gpt-4", streaming=True)
        | StrOutputParser()
    )

    # Store the runnable chain in the user session
    cl.user_session.set("runnable", runnable)

    # Find sources for the eight categories and sequentially send the messages
    for category in evaluations.keys():
        source_msg = cl.Message(content=f"Retrieving sources for {category}...")

        # Read content from the CSV first
        csv_content = FIRST_PASS_RESULTS.get(category, "No content found.")
        source_msg.content = f"{category} Details: {csv_content}"

        await source_msg.send()

        # Find sources for the specific category
        sources = await find_sources_for_criteria([category], retriever)

        # Send the sources sequentially
        for source in sources.get(category, ["No sources found"]):
            await cl.Message(content=f"Source for {category}: {source.content}").send()

        # Use handle_message to return CSV content or dynamically generate a response
        await handle_message(cl.Message(content=category), first_pass_results=FIRST_PASS_RESULTS, use_csv_results=True)


@cl.on_message
async def handle_message(message: cl.Message, first_pass_results=None, use_csv_results=False):
    """
    Handle incoming user messages to run the question-answering pipeline.
    If use_csv_results is True, it reads from the first-pass results CSV.
    """
    runnable = cl.user_session.get("runnable")  # Retrieve the runnable chain from the session
    msg = cl.Message(content="")

    # Ensure that first_pass_results is passed in correctly
    if first_pass_results is None:
        first_pass_results = {}  # Fallback if no first-pass results are provided

    # Define a callback handler to display the sources of the retrieved documents
    class PostMessageHandler(BaseCallbackHandler):
        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs of document sources and pages

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            # Store document source and page pairs
            for d in documents:
                source_page_pair = (d.metadata.get('source'), d.metadata.get('page'))
                self.sources.add(source_page_pair)

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            # Display document sources after LLM response
            if self.sources:
                print(self.sources)
                sources_text = "\n".join([f"{source}#page={page}" for source, page in self.sources])
                self.msg.elements.append(cl.Text(name="Sources", content=sources_text, display="inline"))

    # If reading from CSV results, return content from the CSV
    if use_csv_results:
        criterion = message.content.strip()
        if criterion in first_pass_results:
            msg.content = first_pass_results[criterion]
        else:
            msg.content = "No results found for this criterion."
        return msg

    # Run the question-answering pipeline with callback handlers
    else:
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)]),
        ):
            await msg.stream_token(chunk)

        return msg






