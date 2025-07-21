import os
import csv
import re
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool
import logging

logger = logging.getLogger(__name__)

class UserDetails(BaseModel):
    name: Optional[str] = Field(None, description="User's full name")
    reg_no: Optional[str] = Field(None, description="University registration number")
    phone: Optional[str] = Field(None, description="Contact phone number")
    email: Optional[str] = Field(None, description="Email address")
    context: Optional[str] = Field(None, description="Conversation snippet where details were obtained")

class UserManager:
    """Manages user state and lead collection"""
    
    def __init__(self, db_path="data/users.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize user database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                reg_no TEXT UNIQUE,
                phone TEXT UNIQUE,
                email TEXT UNIQUE,
                first_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_interaction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                interaction_count INTEGER DEFAULT 1,
                is_lead BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                context TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✓ User database initialized")
    
    def extract_identifiers(self, user_details: Dict[str, Any]) -> List[str]:
        """Extract unique identifiers from user details"""
        identifiers = []
        
        if user_details.get('reg_no'):
            identifiers.append(('reg_no', user_details['reg_no']))
        if user_details.get('phone'):
            identifiers.append(('phone', user_details['phone']))
        if user_details.get('email'):
            identifiers.append(('email', user_details['email']))
        
        return identifiers
    
    def find_existing_user(self, identifiers: List[tuple]) -> Optional[int]:
        """Find existing user by any identifier"""
        if not identifiers:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for field, value in identifiers:
            cursor.execute(f'SELECT id FROM users WHERE {field} = ?', (value,))
            result = cursor.fetchone()
            if result:
                conn.close()
                return result[0]
        
        conn.close()
        return None
    
    def create_new_user(self, user_details: Dict[str, Any]) -> int:
        """Create new user record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO users (name, reg_no, phone, email, is_lead)
            VALUES (?, ?, ?, ?, 1)
        ''', (
            user_details.get('name'),
            user_details.get('reg_no'),
            user_details.get('phone'),
            user_details.get('email')
        ))
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"✓ New user created with ID: {user_id}")
        return user_id
    
    def update_existing_user(self, user_id: int, user_details: Dict[str, Any]) -> None:
        """Update existing user record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update fields that are not None
        updates = []
        values = []
        
        for field in ['name', 'reg_no', 'phone', 'email']:
            if user_details.get(field):
                updates.append(f"{field} = ?")
                values.append(user_details[field])
        
        if updates:
            values.append(user_id)
            cursor.execute(f'''
                UPDATE users 
                SET {', '.join(updates)}, 
                    last_interaction = CURRENT_TIMESTAMP,
                    interaction_count = interaction_count + 1
                WHERE id = ?
            ''', values)
        else:
            cursor.execute('''
                UPDATE users 
                SET last_interaction = CURRENT_TIMESTAMP,
                    interaction_count = interaction_count + 1
                WHERE id = ?
            ''', (user_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"✓ Updated existing user ID: {user_id}")
    
    def log_interaction(self, user_id: int, context: str) -> None:
        """Log user interaction"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO interactions (user_id, context)
            VALUES (?, ?)
        ''', (user_id, context))
        
        conn.commit()
        conn.close()
    
    def is_new_user(self, user_details: Dict[str, Any]) -> bool:
        """Check if user is new based on identifiers"""
        identifiers = self.extract_identifiers(user_details)
        return self.find_existing_user(identifiers) is None

def validate_phone(phone: str) -> bool:
    """Validate Indian phone number format"""
    if not phone:
        return True  # Optional field
    # Remove spaces and special characters
    clean_phone = re.sub(r'[^\d]', '', phone)
    return re.match(r"^[6-9]\d{9}$", clean_phone) is not None

def validate_email(email: str) -> bool:
    """Validate email format"""
    if not email:
        return True  # Optional field
    return re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email) is not None

def validate_reg_no(reg_no: str) -> bool:
    """Validate registration number format"""
    if not reg_no:
        return True  # Optional field
    # More flexible regex for different formats
    return re.match(r"^[A-Z0-9]{5,15}$", reg_no, re.IGNORECASE) is not None

def smart_capture_user_details(**kwargs) -> str:
    """Smart user details capture with new/existing user logic"""
    
    # Extract properties from either 'properties' or direct kwargs
    properties = kwargs.get('properties', kwargs)
    
    # Extract values
    user_details = {
        'name': properties.get('name'),
        'reg_no': properties.get('reg_no'),
        'phone': properties.get('phone'),
        'email': properties.get('email'),
        'context': properties.get('context', 'User provided personal information')
    }
    
    # Remove None values
    user_details = {k: v for k, v in user_details.items() if v is not None}
    
    # Validate inputs
    validations = []
    if user_details.get('phone') and not validate_phone(user_details['phone']):
        validations.append("Invalid phone number format (should be 10 digits starting with 6-9)")
    if user_details.get('email') and not validate_email(user_details['email']):
        validations.append("Invalid email format")
    if user_details.get('reg_no') and not validate_reg_no(user_details['reg_no']):
        validations.append("Invalid registration number format")
    
    if validations:
        return f"I noticed some formatting issues: {', '.join(validations)}. Please provide the correct format."
    
    # Check if we have any meaningful data to store
    meaningful_fields = ['name', 'reg_no', 'phone', 'email']
    if not any(user_details.get(field) for field in meaningful_fields):
        return "Thank you for the information!"
    
    try:
        user_manager = UserManager()
        
        # Check if user is new
        if user_manager.is_new_user(user_details):
            # New user - capture as lead
            user_id = user_manager.create_new_user(user_details)
            user_manager.log_interaction(user_id, user_details.get('context', ''))
            
            # Also save to CSV for backward compatibility
            save_to_csv(user_details)
            
            logger.info(f"✓ New lead captured: {user_details.get('name', 'Unknown')}")
            return "Thank you for providing your information! I've recorded your details and I'm here to help with any questions about Sathyabama University."
        
        else:
            # Existing user - just update interaction
            identifiers = user_manager.extract_identifiers(user_details)
            user_id = user_manager.find_existing_user(identifiers)
            user_manager.update_existing_user(user_id, user_details)
            user_manager.log_interaction(user_id, user_details.get('context', ''))
            
            logger.info(f"✓ Existing user interaction logged: {user_details.get('name', 'Unknown')}")
            return "Good to see you again! How can I help you today?"
    
    except Exception as e:
        logger.error(f"Error in smart_capture_user_details: {str(e)}")
        # Fallback to simple CSV storage
        save_to_csv(user_details)
        return "Your information has been recorded. Thank you!"

def save_to_csv(user_details: Dict[str, Any]) -> None:
    """Save user details to CSV (backward compatibility)"""
    file_path = "data/user_details.csv"
    
    # Prepare data record
    record = {
        "timestamp": datetime.now().isoformat(),
        **user_details
    }
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Write to CSV
    file_exists = os.path.isfile(file_path)
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

# Create the smart user details tool
smart_user_details_tool = FunctionTool.from_defaults(
    fn=smart_capture_user_details,
    name="capture_user_details",
    description="Intelligently captures user personal information, only treating new users as leads while updating existing user interactions",
    fn_schema=UserDetails
)

def get_user_stats() -> Dict[str, Any]:
    """Get user statistics for monitoring"""
    try:
        user_manager = UserManager()
        conn = sqlite3.connect(user_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_lead = 1')
        total_leads = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interactions')
        total_interactions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_users': total_users,
            'total_leads': total_leads,
            'total_interactions': total_interactions
        }
    except Exception as e:
        logger.error(f"Error getting user stats: {e}")
        return {'error': str(e)}

