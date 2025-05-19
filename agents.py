from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from tools import SummarizationTool, WebExtractor, ExtractContentFromPDFTool, SearchTool
from config import DEFAULT_LLM_MODEL, VERBOSE


# Initialize the LLM
def get_llm(model=None, temperature=0.2):
    """Get LLM instance with specified parameters"""
    return ChatOpenAI(
        model=model or DEFAULT_LLM_MODEL,
        temperature=temperature
    )

# Create agents
def create_research_crew(research_topic, use_memory=True):
    """Create a crew of agents for research on the specified topic
    
    Args:
        research_topic: The topic to research
        use_memory: Whether to enable agent memory
        
    Returns:
        A CrewAI Crew instance
    """
    # Initialize tools
    search_tool = SearchTool()
    extraction_tool = WebExtractor()
    summarization_tool = SummarizationTool()
    pdf_tool = ExtractContentFromPDFTool()
    
    # Default LLM
    llm = get_llm()
    
    # Create memory if enabled
    researcher_memory = ConversationBufferMemory(memory_key="chat_history") if use_memory else None
    analyst_memory = ConversationBufferMemory(memory_key="chat_history") if use_memory else None
    writer_memory = ConversationBufferMemory(memory_key="chat_history") if use_memory else None
    
    # Research Agent
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Find comprehensive and accurate information about {research_topic}",
        backstory="""You are an expert at finding and collecting relevant information 
                   from various sources. You have years of experience in research methodology
                   and know how to evaluate the credibility of sources. You're thorough
                   and always cite your sources.""",
        verbose=VERBOSE,
        allow_delegation=True,
        tools=[search_tool, extraction_tool, pdf_tool],
        llm=llm,
        memory=researcher_memory
    )
    
    # Analysis Agent - uses a slightly higher temperature for more creative analysis
    analyst = Agent(
        role="Data Analyst and Synthesizer",
        goal=f"Analyze and synthesize information about {research_topic} into meaningful insights",
        backstory="""You excel at breaking down complex information and identifying key 
                   patterns and insights. You can find connections between disparate 
                   pieces of information and organize them into a coherent structure.
                   You're skilled at prioritizing information based on relevance and importance.""",
        verbose=VERBOSE,
        allow_delegation=True,
        tools=[summarization_tool ],
        llm=get_llm(temperature=0.3),
        memory=analyst_memory
    )
    
    # Report Writer Agent
    writer = Agent(
        role="Technical Writer and Editor",
        goal=f"Create a comprehensive, well-structured report about {research_topic}",
        backstory="""You are skilled at organizing information into clear, structured reports 
                   with proper citations. You excel at explaining complex topics in accessible 
                   language without oversimplifying. You have a keen eye for detail and ensure 
                   all content is logically organized with a consistent style.""",
        verbose=VERBOSE,
        allow_delegation=False,
        llm=get_llm(temperature=0.2),
        memory=writer_memory
    )
    
    # Define tasks
    research_task = Task(
        description=f"""
        Research the topic: {research_topic}

        Your job is to gather comprehensive information:

        1. Search for the latest information on this topic
        2. Find at least 3-5 credible sources
        3. Extract relevant information from each source
        4. Include URLs for all sources
        5. Make sure to find information from different perspectives
        6. If the topic has technical aspects, ensure you gather technical details
        7. Note any contradictory information you find

        Your final answer should be a collection of structured findings with proper citations.
        For each source, provide:
        - Source URL or reference
        - Brief description of the source's credibility
        - Key information extracted
        - Date of publication/last update if available
        """,
        expected_output=(
            "A well-structured, multi-paragraph research summary (500-800 words) "
            "with 3-5 citations in markdown format."
        ),
        agent=researcher
    )

    analysis_task = Task(
        description=f"""
        Analyze the research findings on {research_topic}

        Your job is to synthesize and analyze the information:

        1. Identify key themes and patterns in the research
        2. Highlight significant insights and trends
        3. Note any contradictions or gaps in the information
        4. Evaluate the quality and reliability of the information
        5. Prioritize the most important findings
        6. Identify questions that remain unanswered

        Your final answer should be a structured analysis that includes:
        - Executive summary of key findings (3-5 bullet points)
        - Main themes identified
        - Analysis of the quality of available information
        - Gaps in the research that should be addressed
        - Recommendations for further research if appropriate
        """,
        expected_output=(
            "A structured analysis with an executive summary (3–5 bullet points), key themes, "
            "source reliability evaluation, research gaps, and questions for future research."
        ),
        agent=analyst,
        context=[research_task]
    )

    report_task = Task(
        description=f"""
        Create a comprehensive report about {research_topic}

        Your job is to create a professional, well-structured report:

        1. Use the research findings and analysis to create a cohesive report
        2. Include an executive summary at the beginning
        3. Organize information into logical sections with clear headings
        4. Include all sources as citations using a consistent format
        5. Format the report professionally with appropriate use of:
        - Headings and subheadings
        - Bullet points for lists
        - Bold for emphasis
        - Tables if appropriate for data
        6. Include a "Further Research" section with questions that remain unanswered

        Your final answer should be a complete report in Markdown format ready for presentation.
        Make sure the report is comprehensive but concise, focusing on the most valuable information.
        """,
        expected_output=(
            "A professional, markdown-formatted report including an executive summary, clear structure, "
            "proper citations, key findings, and a 'Further Research' section."
        ),
        agent=writer,
        context=[research_task, analysis_task]
    )

    
    # Create the crew
    crew = Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, report_task],
        verbose=VERBOSE,
        process=Process.sequential
    )
    
    return crew
