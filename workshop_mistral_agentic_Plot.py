import os
import sys
import time
import json
import shutil
from typing import List, Optional
import warnings

# --- Core LangChain and ChromaDB Imports ---
import chromadb
import numpy as np
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# --- Agent Imports ---
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool, render_text_description
from langchain import hub

# --- Visualization Imports ---
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.decomposition import PCA

# ==============================================================================
# VSCode Environment Setup Instructions:
#
# 1. Install Ollama:
#    - Download and run the installer from https://ollama.com/
#
# 2. Pull the required models via your terminal:
#    > ollama pull llama3
#    > ollama pull nomic-embed-text
#    > ollama pull mistral:instruct
#
# 3. Ensure Ollama is running in the background.
#
# 4. Install the required Python packages:
#    > pip install langchain langchain-community langchain-core chromadb pypdf scikit-learn matplotlib
#
# 5. Save this script as (e.g.,) `workshop.py`.
#
# 6. Place your `Souvenir_AIMLAC-2025.pdf` file in the *same directory* as this script.
#
# 7. Run the script from your VSCode terminal:
#    > python workshop.py
#
# ==============================================================================


# --- Global Configuration ---
OLLAMA_EMBED_URL = 'http://localhost:11434' # Base URL for Ollama
LLAMA_MODEL = 'llama3'
EMBED_MODEL = 'nomic-embed-text'
VECTOR_DB_PATH = "./chroma_db" # Directory to store ChromaDB
PDF_PATH = "Souvenir_AIMLAC-2025.pdf"

# --- Global variables for visualization ---
pca_model = None
all_2d_vectors = None
embeddings_model = None # To store the initialized embedding model

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='langsmith')
warnings.filterwarnings("ignore", category=FutureWarning)


# ##############################################################################
# STEP 1: LOAD KNOWLEDGE BASE (The Souvenir)
# ##############################################################################

def plot_vector_distribution(vectors_2d):
    """
    Plots the 2D distribution of all document chunk vectors.
    """
    print("Generating vector distribution plot...")
    plt.figure(figsize=(10, 7))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=10, alpha=0.5)
    plt.title('2D PCA of All Document Chunk Vectors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plot_filename = "vector_distribution_plot.png"
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
    print(f"Vector distribution plot saved to '{plot_filename}'")

def load_knowledge_base():
    """
    Loads the AIMLAC-2025 Souvenir PDF, splits it, and ingests it into ChromaDB.
    Also generates a 2D plot of all chunk vectors.
    """
    print("\n--- STEP 1: LOADING KNOWLEDGE BASE (from PDF) ---")
    
    global pca_model, all_2d_vectors, embeddings_model # To be used by Step 3

    print(f"Running locally. Checking for '{PDF_PATH}' in the current directory.")
    if not os.path.exists(PDF_PATH):
        print(f"\nError: '{PDF_PATH}' not found.")
        print(f"Please download the PDF and place it in the same folder as this script.")
        return None

    # 1. Load the PDF
    print(f"Loading PDF from '{PDF_PATH}'...")
    try:
        loader = PyPDFLoader(PDF_PATH, extract_images=False)
        documents = loader.load()
        if not documents:
            print("Error: Could not load any content from the PDF.")
            return None
        print(f"Loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")
        print("Please ensure 'pypdf' is installed and the file is a valid PDF.")
        return None

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks (using smaller chunk size).")
    
    # 3. Initialize embeddings
    print("Initializing Ollama embeddings...")
    embeddings_model = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_EMBED_URL
    )
    
    # 4. Initialize ChromaDB and ingest documents
    if os.path.exists(VECTOR_DB_PATH):
        print("Deleting old vector store to ensure freshness...")
        shutil.rmtree(VECTOR_DB_PATH)
    
    print("Creating new vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    print("Knowledge base loaded into ChromaDB.")

    # 5. --- NEW: Visualization Step ---
    print("Generating visualization for all document vectors...")
    # Get all embeddings directly from the vector store
    db_data = vectorstore.get(include=['embeddings'])
    all_vectors = np.array(db_data['embeddings'])
    
    if all_vectors.shape[0] > 0:
        # Fit and transform using PCA
        pca_model = PCA(n_components=2, random_state=42)
        all_2d_vectors = pca_model.fit_transform(all_vectors)
        
        # Plot the distribution
        plot_vector_distribution(all_2d_vectors)
    else:
        print("Warning: Could not retrieve vectors for plotting.")

    return vectorstore

# ##############################################################################
# STEP 2: PROMPT ENGINEERING (Zero-shot vs. Role-based)
# ##############################################################################

def demonstrate_prompt_engineering(llm):
    """
    Shows how changing the prompt impacts the LLM's response.
    """
    print("\n--- STEP 2: DEMONSTRATING PROMPT ENGINEERING ---")
    
    question = "What is Kumar Abhinav's talk about?"
    
    # --- Prompt 1: Zero-Shot (Base LLM Knowledge) ---
    print("\n### 2a: Zero-Shot Prompt (Base LLM Knowledge)")
    print(f"PROMPT: {question}")
    
    response = llm.invoke(question)
    print(f"RESPONSE:\n{response.content}\n")
    print("NOTE: The model likely doesn't know the answer as it's not in its training data.")

    # --- Prompt 2: Role-Based Prompt ---
    print("\n### 2b: Role-Based Prompt")
    role_prompt_template = """
    You are a helpful conference assistant for AIMLAC-2025.
    Your knowledge is strictly limited to the official souvenir.
    If you do not know the answer from the souvenir, say 'I do not have that information'.
    
    Answer the following question:
    {question}
    """
    
    role_prompt = ChatPromptTemplate.from_template(role_prompt_template)
    chain = role_prompt | llm | StrOutputParser()
    
    print(f"PROMPT (template):\n{role_prompt_template.format(question=question)}")
    
    response = chain.invoke({"question": question})
    print(f"RESPONSE:\n{response}\n")
    print("NOTE: The model has adopted the persona but still lacks the specific facts.")

# ##############################################################################
# STEP 3: RAG (Retrieval-Augmented Generation)
# ##############################################################################

# These will be our global components for other steps
retriever = None
rag_chain = None

def plot_rag_visualization(query_2d, retrieved_2d):
    """
    Plots the query, its neighbors, and all other vectors.
    """
    print("Generating RAG visualization plot...")
    plt.figure(figsize=(12, 8))
    
    # Plot all document chunks
    plt.scatter(all_2d_vectors[:, 0], all_2d_vectors[:, 1], s=10, alpha=0.3, c='gray', label='All Chunks')
    
    # Plot retrieved document chunks
    plt.scatter(retrieved_2d[:, 0], retrieved_2d[:, 1], s=100, c='blue', label='Retrieved Chunks', edgecolors='k')
    
    # Plot the query vector
    plt.scatter(query_2d[0, 0], query_2d[0, 1], s=250, c='red', marker='*', label='Query Vector', edgecolors='k')
    
    # Find the max distance from query to a retrieved doc to set circle radius
    distances = np.linalg.norm(retrieved_2d - query_2d, axis=1)
    radius = np.max(distances) * 1.1 # Add 10% padding
    
    # Plot a circle
    circle = Circle(query_2d[0], radius, color='red', fill=False, linestyle='--', linewidth=2, label='Retrieval Radius')
    plt.gca().add_patch(circle)
    
    plt.title('2D PCA of RAG Query and Retrieved Chunks')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.axis('equal') # Ensure circle isn't elliptical
    
    plot_filename = "rag_visualization_plot.png"
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
    print(f"RAG visualization plot saved to '{plot_filename}'")

def demonstrate_rag(llm, vectorstore):
    """
    Demonstrates the full RAG pipeline: Retrieve, Augment, Generate.
    """
    print("\n--- STEP 3: DEMONSTRATING RAG (Retrieval-Augmented Generation) ---")
    
    global retriever, rag_chain # Make retriever and chain available globally
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    ) 
    
    question = "What is Kumar Abhinav's talk about, and what is his professional focus?"
    
    # 1. Retrieve: Get relevant docs from ChromaDB
    print(f"\n### 3a: Retrieve (DEBUGGING STEP)")
    print(f"Query: {question}")
    retrieved_docs = retriever.invoke(question)
    print("Retrieved documents (using MMR):")
    if not retrieved_docs:
        print("!!! RETRIEVAL FAILED - NO DOCUMENTS FOUND !!!")
    for i, doc in enumerate(retrieved_docs):
        snippet = doc.page_content.replace('\n', ' ').strip()[:150]
        print(f"--- DOC {i+1} (Page {doc.metadata.get('page', 'N/A')}) ---\n{snippet}...\n---------------")
    
    # 2. --- NEW: RAG Visualization ---
    if pca_model and embeddings_model:
        # Get query vector
        query_vector = embeddings_model.embed_query(question)
        query_2d = pca_model.transform(np.array([query_vector]))
        
        # Get retrieved doc vectors
        retrieved_contents = [doc.page_content for doc in retrieved_docs]
        retrieved_vectors = embeddings_model.embed_documents(retrieved_contents)
        retrieved_2d = pca_model.transform(np.array(retrieved_vectors))
        
        # Plot
        plot_rag_visualization(query_2d, retrieved_2d)
    else:
        print("Warning: Skipping RAG visualization as PCA model or vectors are not available.")

    # 3. Augment & Generate: Create a RAG chain
    print("\n### 3b: Augment and Generate (RAG Chain with a better prompt)")
    
    # General-purpose CoT prompt
    rag_prompt_template = """
    You are a meticulous assistant for the AIMLAC-2025 conference.
    Your task is to answer a question based ONLY on the provided context.
    The context contains multiple text chunks. You must perform the following steps:
    
    1.  Read the QUESTION carefully.
    2.  Scan *all* context chunks to locate the text that is *most relevant* to the question.
    3.  Find the specific details (like names, talk titles, or themes) mentioned in the question.
    4.  Answer the question using ONLY the information from the relevant context chunks.
    5.  **CRITICAL:** Do NOT blend or combine information from different, unrelated chunks. For example, if you find a name in one chunk and a talk title in another, do not assume they are related unless the context explicitly links them.
    
    CONTEXT:
    {context}
    
    QUESTION:
    {question}
    
    Let's think step-by-step:
    1. The question is: "{question}"
    2. I will scan all context chunks to find the most relevant information.
    3. [Model thinks here] I will identify the key entities in the question and find them in the context.
    4. I will formulate an answer based *only* on the directly related information.
    
    ANSWER:
    """
    
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    rag_chain = (
        RunnableParallel(
            # Format docs context
            {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
             "question": RunnablePassthrough()}
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # 4. Invoke the chain
    print("Invoking RAG chain...")
    response = rag_chain.invoke(question)
    
    print(f"QUESTION: {question}")
    print(f"RESPONSE:\n{response}\n")
    print("NOTE: The model should now correctly answer using the new general-purpose CoT prompt.")
    
    return rag_chain

# ##############################################################################
# STEP 4: ONTOLOGY & ENTITY RELATIONSHIP (Structured Extraction)
# ##############################################################################

# Define the "Ontology" or schema using Pydantic
class SpeakerInfo(BaseModel):
    """Information about a speaker at the AIMLAC-2025 conference."""
    name: str = Field(description="The full name of the speaker.")
    affiliation: Optional[str] = Field(description="The speaker's company or university.")
    talk_title: Optional[str] = Field(description="The title of the speaker's talk.")

def demonstrate_structured_extraction(llm):
    """
    Demonstrates extracting structured data (entities and relationships)
    from unstructured text. Uses the globally defined 'retriever'.
    """
    print("\n--- STEP 4: DEMONSTRATING STRUCTURED EXTRACTION (Ontology & Entities) ---")
    
    if retriever is None or rag_chain is None:
        print("Error: RAG chain not initialized. Run Step 3 first.")
        return

    # 1. Instantiate the parser with our Pydantic model
    parser = JsonOutputParser(pydantic_object=SpeakerInfo)

    # 2. Get context from the retriever
    speaker_name = "Dr. Balaji Krishnamurthy"
    context_docs = retriever.invoke(f"Get details for {speaker_name}")
    context_blob = "\n\n".join(doc.page_content for doc in context_docs)
    print(f"Retrieved context for '{speaker_name}' to be used for extraction.")

    # 3. Create a new prompt template that includes the parser's instructions
    extraction_prompt_template = """
    You are an expert at extracting information.
    Extract details for the requested speaker based on the provided context.
    
    CONTEXT:
    {context}
    
    SPEAKER:
    {speaker_name}
    
    FORMAT INSTRUCTIONS:
    {format_instructions}
    """
    
    extraction_prompt = ChatPromptTemplate.from_template(
        extraction_prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 4. Create the new chain by piping to the LLM and then the parser
    extraction_chain = extraction_prompt | llm | parser
    
    print(f"Invoking structured extraction chain for '{speaker_name}'...")
    
    # 5. Invoke the chain
    response_obj = extraction_chain.invoke({
        "context": context_blob,
        "speaker_name": speaker_name
    })
    
    # 6. The 'response_obj' is now a dictionary
    print("\nExtracted Object:")
    print(json.dumps(response_obj, indent=2))
    

    print("\n### 4b: Demonstrating Entity Relationship")
    print("This shows the 'relationship' between 'Dr. B. Vasu' and 'AIMLAC-2025'.")
    
    # We can use our existing RAG chain for this
    relation_question = "What is Dr. B. Vasu's role in the AIMLAC-2025 workshop?"
    response = rag_chain.invoke(relation_question)
    
    print(f"QUESTION: {relation_question}")
    print(f"RESPONSE:\n{response}\n")
    print("NOTE: The LLM has extracted the 'Convener' relationship from the PDF text.")


# ##############################################################################
# STEP 5: AGENTIC AI (Planning and Tool Use)
# ##############################################################################

# --- Agent Tools ---
# The agent will use these functions. They rely on the global 'rag_chain'.

@tool
def get_speaker_details(speaker_name: str) -> str:
    """
    Fetches all available details for a specific speaker from the AIMLAC-2025 souvenir,
    including their talk title and affiliation.
    """
    print(f"\n*** AGENT TOOL USED: get_speaker_details(speaker_name='{speaker_name}') ***")
    if rag_chain is None:
        return "Error: RAG chain is not initialized."
    # Use the RAG chain to answer a question about the speaker
    query = f"What are the details of the talk and affiliation for {speaker_name}?"
    response = rag_chain.invoke(query)
    return response

@tool
def get_talk_title(speaker_name: str) -> str:
    """
    Fetches only the talk title for a specific speaker from the AIMLAC-2025 souvenir.
    """
    print(f"\n*** AGENT TOOL USED: get_talk_title(speaker_name='{speaker_name}') ***")
    if rag_chain is None:
        return "Error: RAG chain is not initialized."
    query = f"What is the title of the talk by {speaker_name}?"
    response = rag_chain.invoke(query)
    return response
    
@tool
def get_workshop_theme_details(topic: str) -> str:
    """
    Gets information about a specific theme or topic from the 'About the Workshop' section.
    """
    print(f"\n*** AGENT TOOL USED: get_workshop_theme_details(topic='{topic}') ***")
    if rag_chain is None:
        return "Error: RAG chain is not initialized."
    query = f"What does the souvenir say about the workshop theme of '{topic}'?"
    response = rag_chain.invoke(query)
    return response


def demonstrate_agentic_ai(llm):
    """
    Demonstrates an AI agent that plans, uses tools, and reasons to answer
    a complex, multi-step question.
    """
    print("\n--- STEP 5: DEMONSTRATING AGENTIC AI (Planning & Tool Use) ---")
    
    print("Initializing 'mistral:instruct' model for the agent...")
    try:
        agent_llm = ChatOllama(model='mistral:instruct', base_url=OLLAMA_EMBED_URL)
        agent_llm.invoke("Hello") # Test connection
        print("Mistral agent LLM connection successful.")
    except Exception as e:
        print(f"Error connecting to Ollama for Mistral: {e}")
        print("Please ensure Ollama is running and you have pulled Mistral:")
        print("> ollama pull mistral:instruct")
        return
    # =================

    tools = [get_speaker_details, get_talk_title, get_workshop_theme_details]
    
    try:
        # Use the stable 'hwchase17/react' prompt
        agent_prompt = hub.pull("hwchase17/react") 
        print("Successfully pulled agent prompt 'hwchase17/react'.")
    except Exception as e:
        print(f"Error pulling prompt from LangChain Hub: {e}")
        print("Please check your internet connection and the prompt name.")
        return

    # This prompt requires the tools to be rendered as text
    prompt_with_tools = agent_prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # Create the agent using 'create_react_agent'
    agent = create_react_agent(agent_llm, tools, prompt_with_tools)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, 
        handle_parsing_errors=True 
    )
    
    # Use a clear, numbered list for the input
    complex_question = """
    Please perform the following three tasks in order:
    1. Get the talk title for "Mr. Kumar Abhinav".
    2. Get the full details for the speaker from "Adobe".
    3. Find out what the workshop says about the theme "Image and Video Processing".
    """
    
    print(f"\nInvoking Agent with clear, multi-step question:\n{complex_question}\n")
    
    try:
        # Run the agent
        response = agent_executor.invoke({"input": complex_question})
        
        print("\n--- AGENT EXECUTION COMPLETE ---")
        print(f"\nFinal Answer:\n{response['output']}")
        
    except Exception as e:
        print(f"\nAn error occurred during agent execution: {e}")
        print("This can sometimes happen if the LLM output is not valid.")
        print("Try re-running the script.")


# ##############################################################################
# --- MAIN EXECUTION ---
# ##############################################################################

if __name__ == "__main__":
    
    print("==============================================")
    print("  AIMLAC-2025 GENERATIVE AI WORKSHOP SCRIPT   ")
    print("  (VSCode / Local Version - v3 with Plots)  ")
    print("==============================================")
    
    # --- Initialize Core Components ---
    print("\nInitializing LLM (ChatOllama)...")
    try:
        llm = ChatOllama(model=LLAMA_MODEL, base_url=OLLAMA_EMBED_URL)
        # Test connection
        llm.invoke("Hello")
        print("LLM connection successful.")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print(f"Please ensure Ollama is running at {OLLAMA_EMBED_URL}")
        print(f"And that you have pulled the model: ollama pull {LLAMA_MODEL}")
        sys.exit(1)
    
    # --- Run Workshop Steps ---
    
    # Step 1: Ingest Knowledge Base (and create plot)
    vectorstore = load_knowledge_base()
    
    if vectorstore is None:
        print("\nFailed to load knowledge base. Exiting.")
        sys.exit(1)
    
    # Step 2: Prompt Engineering
    demonstrate_prompt_engineering(llm)
    
    # Step 3: RAG (and create plot)
    # This function assigns the global 'rag_chain' and 'retriever'
    demonstrate_rag(llm, vectorstore) 
    
    # Step 4: Structured Data Extraction
    demonstrate_structured_extraction(llm)
    
    # Step 5: Agentic AI
    demonstrate_agentic_ai(llm)
    
    print("\n==============================================")
    print("          WORKSHOP DEMO COMPLETE          ")
    print("==============================================")
    print("\nCheck your directory for the plots:")
    print("- vector_distribution_plot.png")
    print("- rag_visualization_plot.png")