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
import pandas as pd  # <-- NEW IMPORT
import plotly.express as px  # <-- NEW IMPORT
from langchain_community.vectorstores import Chroma
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
from sklearn.decomposition import PCA

# --- AZURE OPENAI Imports ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import openai

# ==============================================================================
# VSCode Environment Setup Instructions:
#
# 1. Install the required Python packages:
#    > pip install langchain langchain-openai langchain-core chromadb pypdf scikit-learn pandas plotly
#
# 2. Set your Azure Environment Variables.
#
# 3. Save this script as (e.g.,) `workshop_azure_interactive.py`.
#
# 4. Place your `Souvenir_AIMLAC-2025.pdf` file in the *same directory* as this script.
#
# 5. Run the script from your VSCode terminal:
#    > python workshop_azure_interactive.py
#
# ==============================================================================


# --- Global Configuration ---
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.environ.get("OPENAI_API_VERSION", "2024-05-01-preview")
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_CHAT_DEPLOYMENT = os.environ.get("AZURE_CHAT_DEPLOYMENT")
AZURE_AGENT_DEPLOYMENT = os.environ.get("AZURE_AGENT_DEPLOYMENT")

PDF_PATH = "Souvenir_AIMLAC-2025.pdf"
VECTOR_DB_PATH = "./chroma_db"

# --- Global variables for visualization ---
pca_model = None
all_2d_vectors = None
all_doc_snippets = None # Store snippets for plotting
embeddings_model = None

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module='langsmith')
warnings.filterwarnings("ignore", category=FutureWarning)


# ##############################################################################
# STEP 1: LOAD KNOWLEDGE BASE (The Souvenir)
# ##############################################################################

def plot_vector_distribution_interactive(vectors_2d, snippets):
    """
    Plots the 2D distribution of all document chunk vectors using Plotly.
    """
    print("Generating interactive vector distribution plot...")
    
    # Create a DataFrame for Plotly
    df = pd.DataFrame(vectors_2d, columns=['PC1', 'PC2'])
    df['snippet'] = snippets # Add snippet for hover tooltip
    
    fig = px.scatter(
        df,
        x='PC1',
        y='PC2',
        title='Interactive 2D PCA of All Document Chunk Vectors (Azure OpenAI)',
        hover_data=['snippet'], # Show this data on hover
        opacity=0.7
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(hovermode="closest")
    
    # --- CHANGED: Show in browser and save as HTML ---
    plot_filename = "vector_distribution_plot_azure.html"
    fig.write_html(plot_filename)
    print(f"Interactive vector distribution plot saved to '{plot_filename}'")
    print("\n*** Displaying plot in your browser... ***")
    fig.show()
    # -------------------------------------------------

def load_knowledge_base():
    """
    Loads the AIMLAC-2025 Souvenir PDF, splits it, and ingests it into ChromaDB
    using Azure OpenAI embeddings.
    """
    print("\n--- STEP 1: LOADING KNOWLEDGE BASE (from PDF) ---")
    
    global pca_model, all_2d_vectors, all_doc_snippets, embeddings_model

    print(f"Running locally. Checking for '{PDF_PATH}' in the current directory.")
    if not os.path.exists(PDF_PATH):
        print(f"\nError: '{PDF_PATH}' not found.")
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
        return None

    # 2. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=150
    )
    chunks = text_splitter.split_documents(documents)
    all_doc_snippets = [doc.page_content[:200] + "..." for doc in chunks] # Store for plotting
    print(f"Document split into {len(chunks)} chunks.")
    
    # 3. Initialize embeddings
    print(f"Initializing Azure OpenAI Embeddings (Deployment: {AZURE_EMBEDDING_DEPLOYMENT})...")
    embeddings_model = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        api_version=OPENAI_API_VERSION
    )
    
    # 4. Initialize ChromaDB
    if os.path.exists(VECTOR_DB_PATH):
        print("Deleting old vector store...")
        shutil.rmtree(VECTOR_DB_PATH)
    
    print("Creating new vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=VECTOR_DB_PATH
    )
    vectorstore.persist()
    print("Knowledge base loaded into ChromaDB.")

    # 5. --- Visualization Step ---
    print("Generating visualization for all document vectors...")
    db_data = vectorstore.get(include=['embeddings'])
    all_vectors = np.array(db_data['embeddings'])
    
    if all_vectors.shape[0] > 0:
        pca_model = PCA(n_components=2, random_state=42)
        all_2d_vectors = pca_model.fit_transform(all_vectors)
        
        # Plot interactively
        plot_vector_distribution_interactive(all_2d_vectors, all_doc_snippets)
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

def plot_rag_visualization_interactive(query_2d, retrieved_2d, retrieved_snippets, question):
    """
    Plots the query, its neighbors, and all other vectors using Plotly.
    """
    print("Generating interactive RAG visualization plot...")
    
    # === MODIFICATION START ===
    # Create a DataFrame for all chunks
    df_all = pd.DataFrame(all_2d_vectors, columns=['PC1', 'PC2'])
    df_all['type'] = 'All Chunks'
    df_all['snippet'] = all_doc_snippets
    df_all['size_map'] = 2  # Small, constant size

    # Create a DataFrame for retrieved chunks
    df_retrieved = pd.DataFrame(retrieved_2d, columns=['PC1', 'PC2'])
    df_retrieved['type'] = 'Retrieved Chunk'
    df_retrieved['snippet'] = retrieved_snippets
    df_retrieved['size_map'] = 6  # Medium size

    # Create a DataFrame for the query
    df_query = pd.DataFrame(query_2d, columns=['PC1', 'PC2'])
    df_query['type'] = 'Query Vector'
    df_query['snippet'] = [f"QUERY: {question}"]
    df_query['size_map'] = 10 # Large size
    
    # Combine them
    df_plot = pd.concat([df_all, df_retrieved, df_query])
    
    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='type', # Color points by their type
        symbol='type', # Use different symbols
        title='Interactive 2D PCA of RAG Query and Retrieved Chunks (Azure OpenAI)',
        hover_data=['snippet'], # Show snippet on hover
        size='size_map', # <-- THIS IS THE FIX (was 'type')
        # size_max is no longer needed as we are providing explicit sizes
        color_discrete_map={
            'All Chunks': 'gray',
            'Retrieved Chunk': 'blue',
            'Query Vector': 'red'
        },
        symbol_map={
            'All Chunks': 'circle',
            'Retrieved Chunk': 'circle-open',
            'Query Vector': 'star'
        }
    )
    # === MODIFICATION END ===
    
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(hovermode="closest")

    # --- Show in browser and save as HTML ---
    plot_filename = "rag_visualization_plot_azure.html"
    fig.write_html(plot_filename)
    print(f"Interactive RAG visualization plot saved to '{plot_filename}'")
    print("\n*** Displaying plot in your browser... ***")
    print("*** The script will continue running, but the plot will open in a new browser tab. ***")
    fig.show() # Note: Unlike matplotlib, fig.show() for plotly does not block.
    
def demonstrate_rag(llm, vectorstore):
    """
    Demonstrates the full RAG pipeline: Retrieve, Augment, Generate.
    """
    print("\n--- STEP 3: DEMONSTRATING RAG (Retrieval-Augmented Generation) ---")
    
    global retriever, rag_chain 
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    ) 
    
    question = "What is Kumar Abhinav's talk about, and what is his professional focus?"
    
    # 1. Retrieve: Get relevant docs from ChromaDB
    print(f"\n### 3a: Retrieve (DEBUGGING STEP)")
    print(f"Query: {question}")
    retrieved_docs = retriever.invoke(question)
    retrieved_snippets = [doc.page_content[:200] + "..." for doc in retrieved_docs]
    
    print("Retrieved documents (using MMR):")
    if not retrieved_docs:
        print("!!! RETRIEVAL FAILED - NO DOCUMENTS FOUND !!!")
    for snippet in retrieved_snippets:
        print(f"--- DOC ---\n{snippet}\n---------------")
    
    # 2. RAG Visualization
    if pca_model and embeddings_model:
        query_vector = embeddings_model.embed_query(question)
        query_2d = pca_model.transform(np.array([query_vector]))
        
        retrieved_contents = [doc.page_content for doc in retrieved_docs]
        retrieved_vectors = embeddings_model.embed_documents(retrieved_contents)
        retrieved_2d = pca_model.transform(np.array(retrieved_vectors))
        
        # Plot interactively
        plot_rag_visualization_interactive(query_2d, retrieved_2d, retrieved_snippets, question)
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
    3.  Answer the question using ONLY the information from the relevant context chunks.
    4.  **CRITICAL:** Do NOT blend or combine information from different, unrelated chunks.
    
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
    Demonstrates extracting structured data. We can now use
    .with_structured_output() because Azure OpenAI models support it.
    """
    print("\n--- STEP 4: DEMONSTRATING STRUCTURED EXTRACTION (Ontology & Entities) ---")
    
    if retriever is None or rag_chain is None:
        print("Error: RAG chain not initialized. Run Step 3 first.")
        return

    # 1. Use the LLM with structured output (JSON mode)
    print("Using .with_structured_output() (supported by Azure OpenAI)")
    structured_llm = llm.with_structured_output(SpeakerInfo)

    # 2. Get context from the retriever
    speaker_name = "Dr. Balaji Krishnamurthy"
    context_docs = retriever.invoke(f"Get details for {speaker_name}")
    context_blob = "\n\n".join(doc.page_content for doc in context_docs)
    print(f"Retrieved context for '{speaker_name}' to be used for extraction.")

    # 3. Create a prompt for extraction
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at extracting information. "
         "Extract details for the requested speaker based on the provided context. "
         "You must output in the specified JSON format."),
        ("human", "CONTEXT:\n{context}\n\n"
         "Please extract information for the speaker: {speaker_name}")
    ])
    
    # 4. Create the chain
    extraction_chain = extraction_prompt | structured_llm
    
    print(f"Invoking structured extraction chain for '{speaker_name}'...")
    
    # 5. Invoke the chain
    response_obj = extraction_chain.invoke({
        "context": context_blob,
        "speaker_name": speaker_name
    })
    
    # 6. The 'response_obj' is now a Pydantic object
    print("\nExtracted Object:")
    print(json.dumps(response_obj.dict(), indent=2))
    

    print("\n### 4b: Demonstrating Entity Relationship")
    print("This shows the 'relationship' between 'Dr. B. Vasu' and 'AIMLAC-2025'.")
    
    relation_question = "What is Dr. B. Vasu's role in the AIMLAC-2025 workshop?"
    response = rag_chain.invoke(relation_question)
    
    print(f"QUESTION: {relation_question}")
    print(f"RESPONSE:\n{response}\n")
    print("NOTE: The LLM has extracted the 'Convener' relationship from the PDF text.")


# ##############################################################################
# STEP 5: AGENTIC AI (Planning and Tool Use)
# ##############################################################################

# --- Agent Tools ---
# These are model-agnostic and require no changes.

@tool
def get_speaker_details(speaker_name: str) -> str:
    """
    Fetches all available details for a specific speaker from the AIMLAC-2025 souvenir,
    including their talk title and affiliation.
    """
    print(f"\n*** AGENT TOOL USED: get_speaker_details(speaker_name='{speaker_name}') ***")
    if rag_chain is None:
        return "Error: RAG chain is not initialized."
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
    
    print(f"Initializing Azure OpenAI Agent Model ({AZURE_AGENT_DEPLOYMENT})...")
    try:
        agent_llm = AzureChatOpenAI(
            azure_deployment=AZURE_AGENT_DEPLOYMENT,
            api_version=OPENAI_API_VERSION,
            temperature=0 
        )
        agent_llm.invoke("Hello") # Test connection
        print("Azure OpenAI agent LLM connection successful.")
    except Exception as e:
        print(f"Error connecting to Azure OpenAI for Agent: {e}")
        print(f"Please check your environment variables and deployment name: {AZURE_AGENT_DEPLOYMENT}")
        return
    # ===========================

    tools = [get_speaker_details, get_talk_title, get_workshop_theme_details]
    
    try:
        agent_prompt = hub.pull("hwchase17/react") 
        print("Successfully pulled agent prompt 'hwchase17/react'.")
    except Exception as e:
        print(f"Error pulling prompt from LangChain Hub: {e}")
        print("Please check your internet connection and the prompt name.")
        return

    prompt_with_tools = agent_prompt.partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = create_react_agent(agent_llm, tools, prompt_with_tools)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, 
        handle_parsing_errors=True 
    )
    
    # Use the clear, numbered list for the input
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

def check_env_vars():
    """Checks if all required Azure environment variables are set."""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_EMBEDDING_DEPLOYMENT",
        "AZURE_CHAT_DEPLOYMENT",
        "AZURE_AGENT_DEPLOYMENT"
    ]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        print("="*80)
        print(" ERROR: Missing Environment Variables ".center(80, "="))
        print("\nPlease set the following environment variables in your terminal:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nSee the setup instructions in the script for examples.")
        print("="*80)
        return False
    return True

if __name__ == "__main__":
    
    print("==============================================")
    print("  AIMLAC-2025 GENERATIVE AI WORKSHOP SCRIPT   ")
    print("  (Azure OpenAI + Interactive Plots)        ")
    print("==============================================")
    
    # --- Check for Credentials ---
    if not check_env_vars():
        sys.exit(1)

    # --- Initialize Core Components ---
    print(f"\nInitializing Azure OpenAI Chat Model ({AZURE_CHAT_DEPLOYMENT}) for RAG...")
    try:
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_CHAT_DEPLOYMENT,
            api_version=OPENAI_API_VERSION,
            temperature=0 
        )
        llm.invoke("Hello")
        print("Azure OpenAI chat LLM connection successful.")
    except Exception as e:
        print(f"Error connecting to Azure OpenAI: {e}")
        print(f"Please check your environment variables and deployment name: {AZURE_CHAT_DEPLOYMENT}")
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
    demonstrate_rag(llm, vectorstore) 
    
    # Step 4: Structured Data Extraction
    demonstrate_structured_extraction(llm)
    
    # Step 5: Agentic AI
    demonstrate_agentic_ai(llm)
    
    print("\n==============================================")
    print("          WORKSHOP DEMO COMPLETE          ")
    print("==============================================")
    print("\nCheck your directory for the interactive HTML plots:")
    print("- vector_distribution_plot_azure.html")
    print("- rag_visualization_plot_azure.html")