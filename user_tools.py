import os
import csv
import re
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

class UserDetails(BaseModel):
    name: Optional[str] = Field(None, description="User's full name")
    reg_no: Optional[str] = Field(None, description="University registration number")
    phone: Optional[str] = Field(None, description="Contact phone number")
    email: Optional[str] = Field(None, description="Email address")
    context: Optional[str] = Field(None, description="Conversation snippet where details were obtained")

def validate_phone(phone: str) -> bool:
    """Validate Indian phone number format"""
    return re.match(r"^[6-9]\d{9}$", phone) is not None

def validate_email(email: str) -> bool:
    """Validate email format"""
    return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email) is not None

def validate_reg_no(reg_no: str) -> bool:
    """Validate registration number format (example)"""
    return re.match(r"^[A-Z]{2,3}\d{5}$", reg_no, re.IGNORECASE) is not None

def capture_user_details(**kwargs) -> str:
    """Captures user details and saves to CSV file"""
    # Extract properties from either 'properties' or direct kwargs
    properties = kwargs.get('properties', kwargs)
    
    # Extract values
    name = properties.get('name')
    reg_no = properties.get('reg_no')
    phone = properties.get('phone')
    email = properties.get('email')
    context = properties.get('context')
    
    file_path = "data/user_details.csv"
    
    # Validate inputs
    validations = []
    if phone and not validate_phone(phone):
        validations.append("Invalid phone number format")
    if email and not validate_email(email):
        validations.append("Invalid email format")
    if reg_no and not validate_reg_no(reg_no):
        validations.append("Invalid registration number format")
    
    if validations:
        return f"Validation errors: {', '.join(validations)}"
    
    # Prepare data record
    record = {
        "timestamp": datetime.now().isoformat(),
        "name": name,
        "reg_no": reg_no,
        "phone": phone,
        "email": email,
        "context": context
    }
    
    # Remove empty fields
    record = {k: v for k, v in record.items() if v is not None}
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write to CSV
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    
    return "Your information has been recorded. Thank you!"

user_details_tool = FunctionTool.from_defaults(
    fn=capture_user_details,
    name="capture_user_details",
    description="Captures user personal information when provided in conversation",
    fn_schema=UserDetails
)