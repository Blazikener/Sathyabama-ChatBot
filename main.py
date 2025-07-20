from dotenv import load_dotenv
from llama_index.llms.groq import Groq
import os
import sys
from utils.vector_db import get_pdf_engine
from utils.pandas_db import get_pandas_engine
from utils.user_tools import user_details_tool
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from datetime import datetime
from llama_index.core.llms import ChatMessage, LLM, CompletionResponse, ChatResponse, CompletionResponseGen, ChatResponseGen
from llama_index.core.llms import LLMMetadata
from typing import Optional, List, Generator, Any, AsyncGenerator
import traceback
import asyncio

# Load environment variables
load_dotenv()

# Create a complete implementation of the LLM interface
class GroqWrapper(LLM):
    def __init__(self, model: str, api_key: str):
        super().__init__()
        self.model = model
        self.api_key = api_key
        self._groq = None
        
    @property
    def groq(self):
        if self._groq is None:
            try:
                from llama_index.llms.groq import Groq
                self._groq = Groq(model=self.model, api_key=self.api_key)
            except ImportError:
                raise RuntimeError("Groq module not installed. Run: pip install llama-index-llms-groq")
        return self._groq
    
    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=1024,
            model_name=self.model
        )
        
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        try:
            return self.groq.complete(prompt, **kwargs)
        except Exception as e:
            print(f"Groq API error: {str(e)}")
            return CompletionResponse(text="I'm having trouble connecting to the AI service. Please try again later.")
            
    def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
        try:
            return self.groq.stream_complete(prompt, **kwargs)
        except Exception as e:
            print(f"Groq streaming error: {str(e)}")
            def error_generator():
                yield CompletionResponse(text="Service unavailable", delta="Service unavailable")
            return error_generator()
    
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        try:
            return self.groq.chat(messages, **kwargs)
        except Exception as e:
            print(f"Groq chat error: {str(e)}")
            return ChatResponse(message=ChatMessage(role="assistant", content="I'm having trouble connecting to the AI service."))
            
    def stream_chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponseGen:
        try:
            return self.groq.stream_chat(messages, **kwargs)
        except Exception as e:
            print(f"Groq stream chat error: {str(e)}")
            def error_generator():
                yield ChatResponse(message=ChatMessage(role="assistant", content="Service unavailable"), delta="Service unavailable")
            return error_generator()
    
    # Async methods (stubs for now)
    async def acomplete(self, prompt: str, **kwargs) -> CompletionResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.complete, prompt, **kwargs)
        
    async def astream_complete(self, prompt: str, **kwargs) -> AsyncGenerator[CompletionResponse, None]:
        # Simple implementation that wraps sync version
        response = self.complete(prompt, **kwargs)
        yield response
        
    async def achat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.chat, messages, **kwargs)
        
    async def astream_chat(self, messages: List[ChatMessage], **kwargs) -> AsyncGenerator[ChatResponse, None]:
        # Simple implementation that wraps sync version
        response = self.chat(messages, **kwargs)
        yield response

# Initialize LLM
api_key = os.getenv("GROQ_API_KEY")
model = "llama3-70b-8192" if api_key else "llama3-8b-8192"
llm = Groq(
    model="llama-3.3-70b-versatile", 
    api_key=os.getenv("GROQ_API_KEY")
)

def create_query_engine(path: str, storage_name: str, description: str):
    """Create query engine with proper error handling"""
    try:
        engine = get_pdf_engine(path, storage_name)
        return QueryEngineTool(
            query_engine=engine,
            metadata=ToolMetadata(
                name=storage_name.split("/")[-1] + "_data",
                description=description
            ),
        )
    except Exception as e:
        print(f"Error creating {storage_name} engine: {str(e)}")
        return None

# Create query engines
tools = [user_details_tool]

# Syllabus engine (directory of PDFs)
syllabus_tool = create_query_engine(
    path="data/syllabus/",
    storage_name="storage/syllabus",
    description="Course syllabus for all departments and semesters"
)
if syllabus_tool:
    tools.append(syllabus_tool)
else:
    print(" Syllabus engine not available")

# Admission engine (single PDF)
admission_tool = create_query_engine(
    path="data/admission.pdf",
    storage_name="storage/admission",
    description="Admission procedures, requirements, and deadlines"
)
if admission_tool:
    tools.append(admission_tool)
else:
    print(" Admission engine not available")

# Food menu engine (CSV)
try:
    food_menu_engine = get_pandas_engine("data/food_menu.csv")
    tools.append(QueryEngineTool(
        query_engine=food_menu_engine,
        metadata=ToolMetadata(
            name="food_menu_data",
            description="Daily food menu with prices and meal options"
        ),
    ))
    print(" Food menu engine loaded")
except Exception as e:
    print(f" Error creating food menu engine: {str(e)}")

# Bus schedule engine (CSV)
try:
    bus_engine = get_pandas_engine("data/bus_schedule.csv")
    tools.append(QueryEngineTool(
        query_engine=bus_engine,
        metadata=ToolMetadata(
            name="bus_schedule_data",
            description="Bus routes, timings, and stop information"
        ),
    ))
    print(" Bus schedule engine loaded")
except Exception as e:
    print(f" Error creating bus engine: {str(e)}")

# Load privacy policy
try:
    with open("privacy_policy.txt", "r") as f:
        privacy_policy = f.read()
except:
    privacy_policy = ("Personal information is collected for official university purposes only. "
                      "Data is stored securely and never shared with third parties.")
    print(" Using default privacy policy")

# Agent context
context = f"""
You are SATBOT, the official AI assistant for Sathyabama University. 
Your primary role is to assist students and parents with university-related queries.

Key guidelines:
1. Be polite, patient, and helpful at all times
2. When users provide personal information (name, reg no, phone, email), 
   capture it using the user_details tool
3. Don't mention that you're storing information unless explicitly asked
4. For any query outside university matters, politely decline to answer.
5. Make sure to give accurate and up-to-date information while keep the answer concise.
6. Privacy Policy: {privacy_policy}
"""

# Create agent
try:
    agent = ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        context=context,
        max_iterations=8 
    )
    print(" SATBOT agent initialized successfully")
except Exception as e:
    print(f" Failed to create agent: {str(e)}")
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
    print("\n SATBOT: Welcome to Sathyabama AI Assistant! How can I help you today?")
    print("(Type 'exit' to end the conversation)\n")
    
    while True:
        try:
            prompt = input(" USER: ")
            if prompt.lower() in ['exit', 'quit']:
                break
                
            response = agent.query(prompt)
            print(f"\n SATBOT: {response}\n")
            log_interaction(prompt, str(response))
            
        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as e:
            print(f"\n Error: {str(e)}")
            traceback.print_exc()
            print("\n SATBOT: I encountered an unexpected issue. Please try again or rephrase your question.\n")