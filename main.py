import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LLAMA_INDEX_EMBEDDING_MODEL"] = "local"
from llama_index.llms.groq import Groq
from llama_index.core.query_engine import PandasQueryEngine
import pandas as pd
from prompts import context, instruction_str, pandas_prompt, system_prompt, welcome_message
from vector_db_manager import VectorDBManager
from lead_collector import LeadCollector
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
import random
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set the global embedding model and LLM to avoid OpenAI dependency
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Groq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY")
)

class SathyabamaAIAssistant:
    def __init__(self):
        self.llm = Settings.llm
        
        # Initialize components
        self.vector_db_manager = VectorDBManager()
        
        # Initialize lead collector
        self.lead_collector = LeadCollector()
        
        # Initialize tools and agent
        self.setup_tools()
        self.setup_agent()
    
    def setup_tools(self):
        """Setup all the tools for the agent"""
        self.tools = []
        
        # Get all query engines from vector database
        query_engines = self.vector_db_manager.get_all_query_engines()
        
        # Define a custom response for missing data
        def no_info_found_response(tool_name: str) -> str:
            return f"No information found for {tool_name}. Please check the official Sathyabama University website for details."

        # Add query engine tools
        if 'syllabus' in query_engines and query_engines['syllabus'] is not None:
            self.tools.append(
                QueryEngineTool(
                    query_engine=query_engines['syllabus'],
                    metadata=ToolMetadata(
                        name="syllabus_search",
                        description="Search for information about academic syllabus, courses, subjects, and curriculum for different departments and programs at Sathyabama University"
                    )
                )
            )
        else:
            self.tools.append(
                FunctionTool.from_defaults(
                    fn=lambda: no_info_found_response("syllabus"),
                    name="syllabus_search",
                    description="Search for information about academic syllabus, courses, subjects, and curriculum for different departments and programs at Sathyabama University (No data available, will return a 'no info found' message)"
                )
            )
        
        if 'admission' in query_engines and query_engines['admission'] is not None:
            self.tools.append(
                QueryEngineTool(
                    query_engine=query_engines['admission'],
                    metadata=ToolMetadata(
                        name="admission_info",
                        description="Get information about admission procedures, eligibility criteria, fees, important dates, and application process for Sathyabama University"
                    )
                )
            )
        else:
            self.tools.append(
                FunctionTool.from_defaults(
                    fn=lambda: no_info_found_response("admission"),
                    name="admission_info",
                    description="Get information about admission procedures, eligibility criteria, fees, important dates, and application process for Sathyabama University (No data available, will return a 'no info found' message)"
                )
            )
        
        if 'food_menu' in query_engines and query_engines['food_menu'] is not None:
            self.tools.append(
                QueryEngineTool(
                    query_engine=query_engines['food_menu'],
                    metadata=ToolMetadata(
                        name="food_menu_info",
                        description="Get information about daily food menu, meal timings, prices, and dining options available at Sathyabama University"
                    )
                )
            )
        else:
            self.tools.append(
                FunctionTool.from_defaults(
                    fn=lambda: no_info_found_response("food menu"),
                    name="food_menu_info",
                    description="Get information about daily food menu, meal timings, prices, and dining options available at Sathyabama University (No data available, will return a 'no info found' message)"
                )
            )
        
        if 'bus_details' in query_engines and query_engines['bus_details'] is not None:
            self.tools.append(
                QueryEngineTool(
                    query_engine=query_engines['bus_details'],
                    metadata=ToolMetadata(
                        name="bus_transport_info",
                        description="Get information about bus routes, timings, fees, and transportation services available for Sathyabama University students"
                    )
                )
            )
        else:
            self.tools.append(
                FunctionTool.from_defaults(
                    fn=lambda: no_info_found_response("bus details"),
                    name="bus_transport_info",
                    description="Get information about bus routes, timings, fees, and transportation services available for Sathyabama University students (No data available, will return a 'no info found' message)"
                )
            )
        
        # Add lead collection tool
        def collect_user_info(user_message: str) -> str:
            """Collect user information from their message"""
            extracted = self.lead_collector.update_lead_info(user_message)
            if extracted:
                return f"Information collected: {extracted}"
            return "No specific information extracted from this message."
        
        self.tools.append(
            FunctionTool.from_defaults(
                fn=collect_user_info,
                name="collect_user_info",
                description="Extract and collect user personal information (name, registration number, phone, email, department) from their messages for better assistance"
            )
        )
        
        # Add contextual question generator tool
        def generate_helpful_questions() -> str:
            """Generate contextual questions to collect missing user information"""
            questions = self.lead_collector.generate_contextual_questions()
            if questions:
                return f"Suggested questions to ask: {random.choice(questions)}"
            return "All essential information has been collected."
        
        self.tools.append(
            FunctionTool.from_defaults(
                fn=generate_helpful_questions,
                name="generate_questions",
                description="Generate natural questions to collect missing user information for better personalized assistance"
            )
        )
        
        # Add lead summary tool
        def get_user_summary() -> str:
            """Get summary of collected user information"""
            return self.lead_collector.get_lead_summary()
        
        self.tools.append(
            FunctionTool.from_defaults(
                fn=get_user_summary,
                name="user_info_summary",
                description="Get a summary of the user information collected so far"
            )
        )
    
    def setup_agent(self):
        """Setup the ReAct agent with all tools"""
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
            context=context,
            system_prompt=system_prompt
        )
    
    def process_query(self, user_input: str) -> str:
        """Process user query and return response"""
        try:
            # Always collect user information from their input
            self.lead_collector.update_lead_info(user_input)
            
            # Get response from agent
            response = self.agent.query(user_input)
            
            # Save lead information periodically
            if len(self.lead_collector.conversation_history) % 3 == 0:
                self.lead_collector.save_lead()
            
            return str(response)
        
        except Exception as e:
            return f"I apologize, but I encountered an error while processing your request. Please try again or contact our support team. Error: {str(e)}"
    
    def get_collected_leads(self):
        """Get all collected leads for admin purposes"""
        return self.lead_collector.get_all_leads()
    
    def start_conversation(self):
        """Start the interactive conversation"""
        print(welcome_message)
        print("\nType 'exit' to end the conversation or 'admin' to view collected leads.\n")
        
        while True:
            try:
                user_input = input("\n You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    # Save final lead information
                    self.lead_collector.save_lead()
                    print("\n Assistant: Thank you for using Sathyabama University AI Assistant! Have a great day! 🎓")
                    break
                
                elif user_input.lower() == 'admin':
                    leads = self.get_collected_leads()
                    print(f"\n Admin: Total leads collected: {len(leads)}")
                    for i, lead in enumerate(leads[-5:], 1):  # Show last 5 leads
                        print(f"\nLead {i}:")
                        for key, value in lead.items():
                            if key not in ['conversation_history', 'last_updated']:
                                print(f"  {key}: {value}")
                    continue
                
                elif not user_input:
                    print("\n Assistant: Please ask me something about Sathyabama University!")
                    continue
                
                # Process the query
                print("\n Assistant: ", end="")
                response = self.process_query(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n Assistant: Goodbye! Thank you for using Sathyabama University AI Assistant!")
                self.lead_collector.save_lead()
                break
            except Exception as e:
                print(f"\n Assistant: I apologize for the technical difficulty. Please try again. Error: {str(e)}")

def main():
    """Main function to run the Sathyabama AI Assistant"""
    print(" Initializing Sathyabama University AI Assistant...")
    print(" Loading knowledge base...")
    
    try:
        assistant = SathyabamaAIAssistant()
        print(" Assistant ready!")
        assistant.start_conversation()
    except Exception as e:
        print(f" Failed to initialize assistant: {str(e)}")
        print("Please check your GROQ_API_KEY in the .env file and try again.")

if __name__ == "__main__":
    main()
