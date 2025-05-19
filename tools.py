from langchain_community.tools import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from crewai.tools import BaseTool
from config import DEFAULT_LLM_MODEL
from bs4 import BeautifulSoup
import os


class SearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web for information. Input should be a search query."

    def _run(self, query: str) -> str:
        search = TavilySearchResults()
        response = search.invoke(query)
        return response


class WebExtractor(BaseTool):
    name: str = "web_extractor"
    description: str = "Extract content from a web page. Input should be a URL."

    def _run(self, url: str) -> str:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            return docs[0].page_content
        except Exception as e:
            return f"Error extracting content: {str(e)}"

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

class ExtractContentFromPDFTool(BaseTool):
    name: str = "pdf_extractor"
    description: str = "Extract content from a PDF file. Input should be a file path."

    def _run(self, pdf_path: str) -> str:
        return extract_content_from_pdf(pdf_path)

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


class SummarizationTool(BaseTool):
    name: str = "text_summarizer"
    description: str = "Summarize long text. Input should be text to summarize."

    def _run(self, text: str) -> str:
        return summarize_text(text)

# Get all tools
def get_all_tools():
    return [
        SearchTool(),
        ExtractContentFromPDFTool(),
        SummarizationTool(),
        WebExtractor()
    ]
