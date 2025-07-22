import os
import json
from datetime import datetime
import re

class LeadCollector:
    def __init__(self):
        self.leads_file = "data/collected_leads.json"
        self.conversation_history = []
        self.current_lead = {}
        
    def extract_personal_info(self, user_input):
        """Extract personal information from user input using regex patterns"""
        extracted_info = {}
        
        # Extract name patterns
        name_patterns = [
            r"my name is (\w+(?:\s+\w+)*)",
            r"i am (\w+(?:\s+\w+)*)",
            r"i'm (\w+(?:\s+\w+)*)",
            r"call me (\w+(?:\s+\w+)*)",
            r"this is (\w+(?:\s+\w+)*)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                extracted_info['name'] = match.group(1).title()
                break
        
        # Extract registration number patterns
        reg_patterns = [
            r"reg(?:istration)?\s*(?:no|number|num)?\s*:?\s*([a-zA-Z0-9]+)",
            r"registration\s+(?:is\s+)?([a-zA-Z0-9]+)",
            r"my\s+reg\s+(?:is\s+)?([a-zA-Z0-9]+)",
            r"student\s+id\s*:?\s*([a-zA-Z0-9]+)"
        ]
        
        for pattern in reg_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                extracted_info['registration_number'] = match.group(1).upper()
                break
        
        # Extract phone number patterns
        phone_patterns = [
            r"phone\s*(?:number|no)?\s*:?\s*([+]?[0-9\s\-\(\)]{10,15})",
            r"mobile\s*(?:number|no)?\s*:?\s*([+]?[0-9\s\-\(\)]{10,15})",
            r"contact\s*(?:number|no)?\s*:?\s*([+]?[0-9\s\-\(\)]{10,15})",
            r"call\s+me\s+(?:at\s+)?([+]?[0-9\s\-\(\)]{10,15})",
            r"my\s+number\s+is\s+([+]?[0-9\s\-\(\)]{10,15})"
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                phone = re.sub(r'[^\d+]', '', match.group(1))
                if len(phone) >= 10:
                    extracted_info['phone_number'] = phone
                break
        
        # Extract email patterns
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        email_match = re.search(email_pattern, user_input)
        if email_match:
            extracted_info['email'] = email_match.group(0).lower()
        
        # Extract department/course information
        dept_patterns = [
            r"(?:studying|from|in)\s+([a-zA-Z\s]+?)(?:\s+department|\s+dept)",
            r"([a-zA-Z\s]+?)\s+(?:department|dept)",
            r"(?:course|branch)\s*:?\s*([a-zA-Z\s]+)",
            r"(?:cse|ece|eee|mech|civil|it|computer science|electronics|electrical|mechanical|information technology)"
        ]
        
        for pattern in dept_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                extracted_info['department'] = match.group(1).strip().title()
                break
        
        # Extract year/semester information
        year_patterns = [
            r"(?:year|semester|sem)\s*:?\s*([1-4])",
            r"([1-4])(?:st|nd|rd|th)\s+(?:year|semester|sem)",
            r"(?:first|second|third|fourth)\s+(?:year|semester)"
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                if match.group(1).isdigit():
                    extracted_info['year'] = match.group(1)
                else:
                    year_map = {'first': '1', 'second': '2', 'third': '3', 'fourth': '4'}
                    for word, num in year_map.items():
                        if word in match.group(0):
                            extracted_info['year'] = num
                            break
                break
        
        return extracted_info
    
    def update_lead_info(self, user_input):
        """Update current lead information with extracted data"""
        extracted = self.extract_personal_info(user_input)
        
        # Update current lead with new information
        for key, value in extracted.items():
            if value and value.strip():
                self.current_lead[key] = value
        
        # Add timestamp
        self.current_lead['last_updated'] = datetime.now().isoformat()
        
        # Store conversation for context
        self.conversation_history.append({
            'timestamp': datetime.now().isoformat(),
            'user_input': user_input,
            'extracted_info': extracted
        })
        
        return extracted
    
    def save_lead(self):
        """Save the current lead to file"""
        if not self.current_lead:
            return False
        
        # Load existing leads
        leads = []
        if os.path.exists(self.leads_file):
            try:
                with open(self.leads_file, 'r') as f:
                    leads = json.load(f)
            except:
                leads = []
        
        # Add current lead with unique ID
        lead_id = f"lead_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_lead['lead_id'] = lead_id
        self.current_lead['conversation_history'] = self.conversation_history
        
        leads.append(self.current_lead.copy())
        
        # Save to file
        os.makedirs(os.path.dirname(self.leads_file), exist_ok=True)
        with open(self.leads_file, 'w') as f:
            json.dump(leads, f, indent=2)
        
        return True
    
    def get_lead_summary(self):
        """Get summary of collected lead information"""
        if not self.current_lead:
            return "No information collected yet."
        
        summary = "Collected Information:\n"
        for key, value in self.current_lead.items():
            if key not in ['last_updated', 'lead_id', 'conversation_history']:
                summary += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        return summary
    
    def generate_contextual_questions(self):
        """Generate questions to collect missing information naturally"""
        questions = []
        
        if 'name' not in self.current_lead:
            questions.extend([
                "What should I call you?",
                "May I know your name for better assistance?",
                "Could you please tell me your name?"
            ])
        
        if 'registration_number' not in self.current_lead:
            questions.extend([
                "What's your student registration number?",
                "Could you provide your student ID for specific information?",
                "What's your reg number so I can help you better?"
            ])
        
        if 'department' not in self.current_lead:
            questions.extend([
                "Which department are you from?",
                "What's your course or branch of study?",
                "Are you from CSE, ECE, Mechanical, or another department?"
            ])
        
        if 'year' not in self.current_lead:
            questions.extend([
                "Which year are you currently in?",
                "Are you a first year, second year, third year, or final year student?",
                "What semester are you in?"
            ])
        
        if 'phone_number' not in self.current_lead:
            questions.extend([
                "Could you share your contact number for any follow-up?",
                "What's your mobile number in case we need to reach you?",
                "May I have your phone number for future assistance?"
            ])
        
        if 'email' not in self.current_lead:
            questions.extend([
                "What's your email address?",
                "Could you provide your email for sending detailed information?",
                "May I have your email ID for sharing documents?"
            ])
        
        return questions
    
    def should_ask_for_info(self):
        """Determine if we should ask for more information"""
        essential_fields = ['name', 'registration_number', 'department']
        missing_essential = [field for field in essential_fields if field not in self.current_lead]
        
        # Ask for info if missing essential fields and conversation is progressing
        return len(missing_essential) > 0 and len(self.conversation_history) > 1
    
    def get_all_leads(self):
        """Get all collected leads"""
        if os.path.exists(self.leads_file):
            try:
                with open(self.leads_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []

