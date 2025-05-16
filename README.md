# ResearchGPT - Autonomous AI Research Agent

This repository contains the code for ResearchGPT, an autonomous AI research agent that can gather information, analyze data, and generate comprehensive research reports with minimal human intervention.

## Features

- Web search capabilities using the Tavily API
- Content extraction from web pages
- Text summarization using OpenAI models
- Document indexing and retrieval using LlamaIndex
- Multi-agent coordination with CrewAI
- Structured report generation

## Architecture

ResearchGPT integrates three powerful frameworks:
1. **LangChain** - For tools and chains
2. **LlamaIndex** - For document indexing and retrieval
3. **CrewAI** - For orchestrating multiple specialized agents

## Requirements

- Python 3.9+
- OpenAI API key
- Tavily API key

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/researchgpt.git
cd researchgpt

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env to add your API keys
```

## Usage

```bash
python run.py --topic "Your research topic here"
```

## Project Structure

```
researchgpt/
├── .env.example         # Example environment variables
├── config.py            # Configuration settings
├── tools.py             # Agent tools (search, extraction, summarization)
├── indexing.py          # Document indexing with LlamaIndex
├── agents.py            # Agent definitions using CrewAI
├── main.py              # Main application logic
├── run.py               # Entry point
└── requirements.txt     # Project dependencies
```

## Extending

The modular architecture allows for easy extension:
- Add new tools (e.g., PDF processing)
- Implement persistent storage with vector databases
- Add memory to agents for better context retention
- Integrate with other APIs and data sources

## License

MIT
