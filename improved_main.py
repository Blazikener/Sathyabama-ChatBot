from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import os
import sys
import sqlite3
from pathlib import Path
from utils.improved_vector_db import get_pdf_engine
from utils.improved_pandas_db import get_pandas_engine
from utils.improved_user_tools import smart_user_details_tool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from datetime import datetime
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM and set global settings
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    logger.error("GROQ_API_KEY not found in environment variables")
    sys.exit(1)

llm = Groq(
    model="llama-3.3-70b-versatile", 
    api_key=api_key
)

# Set global LLM for all LlamaIndex components
Settings.llm = llm

class DataValidator:
    """Validates and sets up required data sources"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.storage_dir = Path("storage")
        
    def setup_directories(self):
        """Create required directories"""
        directories = [
            self.data_dir,
            self.data_dir / "syllabus",
            self.data_dir / "syllabus" / "ai",
            self.data_dir / "syllabus" / "cse", 
            self.data_dir / "syllabus" / "ece",
            self.storage_dir,
            self.storage_dir / "syllabus",
            self.storage_dir / "admission"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Directory created/verified: {directory}")
    
    def create_sample_data(self):
        """Create sample data files if they don't exist"""
        
        # Sample syllabus content
        ai_syllabus = """
# AI Department Syllabus - Sathyabama University

## Semester 1
- Mathematics for AI
- Programming Fundamentals
- Statistics and Probability
- Introduction to AI

## Semester 2  
- Data Structures and Algorithms
- Machine Learning Basics
- Linear Algebra
- Database Systems

## Semester 3
- Deep Learning
- Natural Language Processing
- Computer Vision
- Neural Networks

## Semester 4
- Advanced ML Algorithms
- AI Ethics
- Robotics
- Capstone Project

## Course Details
Each course is 4 credits with both theory and practical components.
Total program duration: 4 years (8 semesters)
Minimum CGPA required: 6.0
"""
        
        # Sample admission info
        admission_content = """
# Sathyabama University Admission Information

## AI Department Admission Requirements

### Eligibility Criteria:
- 12th grade with minimum 75% marks
- Mathematics and Physics mandatory
- Valid entrance exam score (JEE/SATHYABAMA-EET)

### Application Process:
1. Online application at www.sathyabama.ac.in
2. Document verification
3. Entrance exam
4. Counseling and seat allocation

### Important Dates:
- Application starts: May 1st
- Last date to apply: June 30th  
- Entrance exam: July 15th
- Results: July 30th
- Admission starts: August 1st

### Fee Structure:
- Tuition fee: ₹2,50,000 per year
- Hostel fee: ₹80,000 per year
- Other charges: ₹20,000 per year

### Contact Information:
- Admission Office: +91-44-24503200
- Email: admissions@sathyabama.ac.in
"""
        
        # Create sample files
        files_to_create = [
            (self.data_dir / "syllabus" / "ai" / "ai_syllabus.txt", ai_syllabus),
            (self.data_dir / "admission.pdf", admission_content),  # Will be txt for simplicity
        ]
        
        for file_path, content in files_to_create:
            if not file_path.exists():
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✓ Sample file created: {file_path}")
        
        # Create sample CSV files
        food_menu_csv = """Day,Breakfast,Lunch,Dinner,Price
Monday,Idli Sambar,Rice Dal Curry,Chapati Curry,₹150
Tuesday,Dosa Chutney,Biryani,Rice Sambar,₹160
Wednesday,Poha,Rice Rasam,Chapati Dal,₹140
Thursday,Upma,Curd Rice,Rice Curry,₹150
Friday,Paratha,Pulao,Chapati Vegetables,₹170
Saturday,Pongal,Rice Curry,Dosa Sambar,₹155
Sunday,Special Breakfast,Special Lunch,Special Dinner,₹200"""
        
        bus_schedule_csv = """Route,Departure_Time,Arrival_Time,Stops,Fare
Chennai Central,06:00,07:30,"Central,Egmore,Saidapet,Guindy,Tambaram",₹25
Tambaram,07:00,08:00,"Tambaram,Chrompet,Pallavaram,Guindy",₹15
Velachery,07:15,08:15,"Velachery,Guindy,Saidapet",₹20
OMR,06:45,08:00,"Sholinganallur,Navalur,Siruseri,Guindy",₹30
GST Road,07:30,08:30,"Chrompet,Pallavaram,Airport,Guindy",₹25"""
        
        csv_files = [
            (self.data_dir / "food_menu.csv", food_menu_csv),
            (self.data_dir / "bus_schedule.csv", bus_schedule_csv)
        ]
        
        for file_path, content in csv_files:
            if not file_path.exists():
                file_path.write_text(content, encoding='utf-8')
                logger.info(f"✓ Sample CSV created: {file_path}")
    
    def validate_all_sources(self):
        """Validate all required data sources"""
        self.setup_directories()
        self.create_sample_data()
        
        required_files = [
            self.data_dir / "syllabus" / "ai" / "ai_syllabus.txt",
            self.data_dir / "admission.pdf",
            self.data_dir / "food_menu.csv", 
            self.data_dir / "bus_schedule.csv"
        ]
        
        all_valid = True
        for file_path in required_files:
            if file_path.exists():
                logger.info(f"✓ Data source validated: {file_path}")
            else:
                logger.error(f"✗ Missing data source: {file_path}")
                all_valid = False
        
        return all_valid

def create_robust_query_engine(path: str, storage_name: str, description: str):
    """Create query engine with comprehensive error handling"""
    try:
        logger.info(f"Creating query engine for {storage_name}")
        engine = get_pdf_engine(path, storage_name)
        
        tool = QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=storage_name.split("/")[-1] + "_data",
                description=description
            ),
        )
        logger.info(f"✓ {storage_name} engine created successfully")
        return tool
        
    except Exception as e:
        logger.error(f"✗ Failed to create {storage_name} engine: {str(e)}")
        # Create a fallback tool that explains the limitation
        def fallback_response(query: str) -> str:
            return f"I apologize, but the {storage_name.split('/')[-1]} information is currently unavailable due to a technical issue. Please contact the university directly for this information."
        
        from llama_index.core.tools import FunctionTool
        return FunctionTool.from_defaults(
            fn=fallback_response,
            name=storage_name.split("/")[-1] + "_fallback",
            description=f"Fallback for {description}"
        )

def initialize_tools():
    """Initialize all tools with validation"""
    # Validate data sources first
    validator = DataValidator()
    if not validator.validate_all_sources():
        logger.warning("Some data sources are missing, but continuing with available ones")
    
    tools = [smart_user_details_tool]
    
    # Syllabus engine (directory of PDFs/text files)
    syllabus_tool = create_robust_query_engine(
        path="data/syllabus/",
        storage_name="storage/syllabus",
        description="Complete course syllabus for all departments including AI, CSE, ECE with semester-wise subjects, credits, and course details"
    )
    tools.append(syllabus_tool)
    
    # Admission engine (single PDF/text file)
    admission_tool = create_robust_query_engine(
        path="data/admission.pdf",
        storage_name="storage/admission", 
        description="Detailed admission procedures, eligibility criteria, application process, important dates, fee structure and contact information"
    )
    tools.append(admission_tool)
    
    # Food menu engine (CSV)
    try:
        food_menu_engine = get_pandas_engine("data/food_menu.csv")
        tools.append(QueryEngineTool(
            query_engine=food_menu_engine,
            metadata=ToolMetadata(
                name="food_menu_data",
                description="Daily food menu with breakfast, lunch, dinner options and prices for each day of the week"
            ),
        ))
        logger.info("✓ Food menu engine loaded")
    except Exception as e:
        logger.error(f"✗ Error creating food menu engine: {str(e)}")
    
    # Bus schedule engine (CSV)
    try:
        bus_engine = get_pandas_engine("data/bus_schedule.csv")
        tools.append(QueryEngineTool(
            query_engine=bus_engine,
            metadata=ToolMetadata(
                name="bus_schedule_data",
                description="Bus routes, departure and arrival times, stops, and fare information for transportation to university"
            ),
        ))
        logger.info("✓ Bus schedule engine loaded")
    except Exception as e:
        logger.error(f"✗ Error creating bus engine: {str(e)}")
    
    return tools

# Load privacy policy
try:
    with open("privacy_policy.txt", "r") as f:
        privacy_policy = f.read()
except:
    privacy_policy = ("Personal information is collected for official university purposes only. "
                      "Data is stored securely and never shared with third parties. "
                      "We only collect information from new users for lead generation.")
    logger.info("Using default privacy policy")

# Enhanced agent context
def get_tools_description(tools):
    """Generate description of available tools"""
    descriptions = []
    for tool in tools:
        if hasattr(tool, 'metadata') and tool.metadata:
            descriptions.append(f"- {tool.metadata.name}: {tool.metadata.description}")
    return "\n".join(descriptions)

def create_agent():
    """Create the enhanced agent"""
    tools = initialize_tools()
    
    context = f"""
You are SATBOT, the official AI assistant for Sathyabama University.

CRITICAL INSTRUCTIONS:
1. ALWAYS use the available RAG tools to search for specific information before responding
2. When users ask about syllabus, admission, food menu, or bus schedules, MUST query the respective tools
3. Provide specific, detailed answers based on retrieved data from the tools
4. If a tool search fails, inform the user and suggest contacting the university directly
5. Only collect user details for NEW users (the system will check automatically)
6. Be transparent about data sources and cite them in responses
7. For queries outside university matters, politely decline and redirect to university topics

Available Data Sources:
{get_tools_description(tools)}

Response Format:
- Start with specific information retrieved from RAG tools
- Provide concrete details (dates, prices, requirements, etc.)
- Cite the data source when possible
- Offer actionable next steps
- Keep responses concise but comprehensive

Privacy Policy: {privacy_policy}

Remember: Your primary goal is to provide accurate, specific information about Sathyabama University using the available data sources.
"""
    
    try:
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            context=context,
            max_iterations=10  # Increased for better tool usage
        )
        logger.info("✓ SATBOT agent initialized successfully")
        return agent
    except Exception as e:
        logger.error(f"✗ Failed to create agent: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def log_interaction(prompt: str, response: str):
    """Log all interactions to a file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] Q: {prompt}\nA: {response}\n{'='*50}\n"
    
    os.makedirs("data", exist_ok=True)
    with open("data/chat_log.txt", "a", encoding="utf-8") as f:
        f.write(log_entry)

if __name__ == "__main__":
    print("\n🤖 SATBOT: Welcome to Sathyabama AI Assistant!")
    print("I can help you with:")
    print("- Course syllabus for all departments")
    print("- Admission procedures and requirements") 
    print("- Food menu and prices")
    print("- Bus schedules and routes")
    print("\nHow can I help you today?")
    print("(Type 'exit' to end the conversation)\n")
    
    agent = create_agent()
    
    while True:
        try:
            prompt = input("👤 USER: ")
            if prompt.lower() in ['exit', 'quit']:
                break
                
            response = agent.query(prompt)
            print(f"\n🤖 SATBOT: {response}\n")
            log_interaction(prompt, str(response))
            
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            traceback.print_exc()
            print("\n🤖 SATBOT: I encountered an unexpected issue. Please try again or rephrase your question.\n")

