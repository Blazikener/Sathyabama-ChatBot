Sathyabama AI Assistant - Setup and Usage Guide

This document provides instructions on how to set up and run the Sathyabama AI Assistant chatbot.


Setup Instructions

1. Set up environment variables:
Open the .env file and replace your_groq_api_key_here with your actual Groq API key.

2. Install dependencies:

3. Ingest data:
Run the data ingestion script to create and populate the vector databases. This step will create syllabus_index, admission_index, food_menu_index, and bus_details_index directories.

Usage

To start the Sathyabama AI Assistant, run the main.py file:

python3 main.py

The chatbot will greet you, and you can start asking questions related to Sathyabama University. You can type exit to end the conversation or admin to view collected leads.

Example Queries:

•
"What is the syllabus for CSE first year?"

•
"Tell me about the admission process."

•
"What's for lunch today?"

•
"Can I get details about bus route R3?"

•
"My name is John and my registration number is 12345."

Lead Collection

The chatbot is designed to collect user information (name, registration number, phone, email, department, year) during the conversation. This information is saved in data/collected_leads.json.

To view the collected leads during a conversation, type admin when prompted for input.

Customization

•
Data: You can update the data in the data/ directory (e.g., syllabus.txt, admission_details.txt, food_menu.csv, bus_details.csv) to reflect the latest information. After updating, re-run python3 ingest_data.py to refresh the vector databases.

•
Prompts: Modify prompts.py to change the chatbot's personality, system prompts, or welcome messages.

•
Tools: Extend main.py to add more tools or integrate with other systems.

•
Lead Collection: Adjust lead_collector.py to modify the information extraction patterns or add new fields to collect.

