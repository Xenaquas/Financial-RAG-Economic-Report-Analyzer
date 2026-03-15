import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Added for chunking
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Configuration
FILE_PATH = "economic_data/Annual_Report_2024-25.pdf"
DB_DIR = "vector_db"

# 2. Setup Models (Using your local gemma3:4b from screenshot)
llm = OllamaLLM(model="gemma3:4b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Ingest & Index
if not os.path.exists(DB_DIR):
    print(f"🔄 Processing {FILE_PATH}...")
    loader = PyPDFLoader(FILE_PATH)
    raw_pages = loader.load()  # Load the PDF first

    # FIX: Split pages into smaller chunks to avoid context length error
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        add_start_index=True  # Helps track where the chunk starts
    )
    docs = text_splitter.split_documents(raw_pages)

    print(f"✅ Created {len(docs)} chunks from {len(raw_pages)} pages.")

    # Building Vector Database
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"✅ Indexed chunks into {DB_DIR}")
else:
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    print("🧠 Loaded existing Vector Database.")

# 4. Professional Query Logic
improved_query = (
    "Summarize the Indian economy's growth performance as mentioned in the Annual Report."
)

# 2. Create a "Table-Aware" Prompt Template
template = """You are a Financial Data Analyst. Use the following pieces of retrieved context 
to answer the question. If the data is in a table format, read the columns and rows carefully.

Context:
{context}

Question: {question}

Instructions:
- Look specifically for 'Revised Estimates' or 'RE' and 'Budget Estimates' or 'BE'.
- If you find a table, extract the exact numerical value.
- Mention the specific Table Name or Page Number if available.
- If the information is truly not there, say you can't find it in the provided chunks.

Answer:"""

custom_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# 3. Update the Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="mmr", # Max Marginal Relevance (Finds diverse chunks)
        search_kwargs={"k": 7, "fetch_k": 20}
    ),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

print(f"\n💡 Question:\nSummarize the Indian economy growth performance as mentioned in the Annual Report.")


# 4. Execute
print(f"\n🤖 Analyzing Report with Expert Prompt...")
result = qa_chain.invoke({"query": improved_query})

print(f"\n💡 Answer:\n{result['result']}")

# Displaying Source Attribution
print("\n📍 Data points retrieved from:")
pages_found = sorted(list(set([doc.metadata['page'] + 1 for doc in result["source_documents"]])))
for page in pages_found:
    print(f"- Page {page}")