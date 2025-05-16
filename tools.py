from langchain_community.tools import TavilySearchResults
from langchain_community.tools.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from config import TAVILY_API_KEY, DEFAULT_LLM_MODEL
import requests
from bs4 import BeautifulSoup
import os

# Web search tool
def setup_search_tool():
    search = TavilySearchResults(api_wrapper=TavilySearchAPIWrapper(tavily_api_key=TAVILY_API_KEY))
    return Tool(
        name="web_search",
        description="Search the web for information. Input should be a search query.",
        func=search.invoke
    )

# Web content extraction tool
def extract_content_from_url(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return docs[0].page_content
    except Exception as e:
        return f"Error extracting content: {str(e)}"

def setup_web_extraction_tool():
    return Tool(
        name="web_extractor",
        description="Extract content from a web page. Input should be a URL.",
        func=extract_content_from_url
    )

# PDF extraction tool
def extract_content_from_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            return f"Error: File not found at {pdf_path}"
            
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        return "\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Error extracting content from PDF: {str(e)}"

def setup_pdf_extraction_tool():
    return Tool(
        name="pdf_extractor",
        description="Extract content from a PDF file. Input should be a file path.",
        func=extract_content_from_pdf
    )

# Summarization tool
def summarize_text(text, max_words=300):
    try:
        # Ensure text is not empty
        if not text or len(text.strip()) == 0:
            return "Error: Empty text provided for summarization."
            
        # Initialize LLM
        llm = ChatOpenAI(temperature=0, model=DEFAULT_LLM_MODEL)
        
        # Limit text to avoid token issues
        # This is a simple approach - a more sophisticated chunking strategy 
        # might be better for very long texts
        doc = Document(page_content=text[:25000])
        
        # Use the stuff chain for summarization
        chain = load_summarize_chain(llm, chain_type="stuff")
        summary = chain.invoke([doc])
        
        return summary["output_text"]
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def setup_summarization_tool():
    return Tool(
        name="text_summarizer",
        description="Summarize long text. Input should be text to summarize.",
        func=summarize_text
    )

# Get all tools
def get_all_tools():
    return [
        setup_search_tool(),
        setup_web_extraction_tool(),
        setup_pdf_extraction_tool(),
        setup_summarization_tool()
    ]
