import asyncio
from crewai import Crew, Process
from agents import create_research_crew
from indexing import ResearchIndex
import argparse
import os
from config import OUTPUT_DIR
import time
from datetime import datetime

async def main():
    """Main application function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ResearchGPT - AI Research Assistant")
    parser.add_argument("--topic", type=str, required=True, help="Research topic")
    parser.add_argument("--memory", action="store_true", help="Enable agent memory")
    parser.add_argument("--persist", action="store_true", help="Enable persistent storage")
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, f"{timestamp}_{args.topic.replace(' ', '_')}")
    os.makedirs(session_dir, exist_ok=True)
    
    # Create research index
    persist_dir = os.path.join(session_dir, "index") if args.persist else None
    research_index = ResearchIndex(persist_dir=persist_dir)
    
    # Create research crew
    print(f"\n{'='*50}")
    print(f"Starting research on: {args.topic}")
    print(f"Memory enabled: {args.memory}")
    print(f"Persistent storage: {args.persist}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # Create and run the research crew
    crew = create_research_crew(args.topic, use_memory=args.memory)
    result = crew.kickoff()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Convert CrewOutput to string
    result_str = str(result)
    
    # Store the final report in the index
    report_metadata = {
        "topic": args.topic,
        "type": "report",
        "created_at": datetime.now().isoformat(),
        "elapsed_time": elapsed_time
    }
    research_index.add_document(content=result_str, metadata=report_metadata)
    
    # Print and save the result
    print("\n")
    print("="*50)
    print(f" RESEARCH REPORT: {args.topic}")
    print("="*50)
    print("\n")
    
    # Save the result to a file
    report_filename = os.path.join(session_dir, f"{args.topic.replace(' ', '_')}_report.md")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(result_str)
    
    # Save metadata
    metadata_filename = os.path.join(session_dir, "metadata.json")
    with open(metadata_filename, "w") as f:
        import json
        json.dump({
            "topic": args.topic,
            "created_at": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "memory_enabled": args.memory,
            "persistent_storage": args.persist
        }, f, indent=2)
    
    print(f"\nResearch completed in {elapsed_time:.2f} seconds")
    print(f"Report saved to {report_filename}")
    print(f"Vector index {'saved to ' + persist_dir if args.persist else 'not persisted'}")

if __name__ == "__main__":
    asyncio.run(main())
