import os
from dotenv import load_dotenv
import asyncio
from main import main

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Check API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to the .env file or environment variables")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not found in environment variables")
        print("Web search functionality will not work without a Tavily API key")
        print("You can get one at https://tavily.com/")
        response = input("Continue without web search capability? (y/n): ")
        if response.lower() != "y":
            exit(1)
    
    # Run the application
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"\nError running the application: {str(e)}")
        import traceback
        traceback.print_exc()