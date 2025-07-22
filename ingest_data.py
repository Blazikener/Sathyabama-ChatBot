from vector_db_manager import VectorDBManager
import os

def ingest_data():
    print("Starting data ingestion...")
    
    vector_db_manager = VectorDBManager()
    
    # Ingest syllabus data
    print("Ingesting syllabus data...")
    syllabus_engine = vector_db_manager.create_syllabus_index(os.path.join("data", "syllabus.txt"))
    if syllabus_engine:
        print("Syllabus data ingested successfully.")
    else:
        print("Failed to ingest syllabus data. Make sure data/syllabus.txt exists.")
        
    # Ingest admission details
    print("Ingesting admission details...")
    admission_engine = vector_db_manager.create_admission_index(os.path.join("data", "admission_details.txt"))
    if admission_engine:
        print("Admission details ingested successfully.")
    else:
        print("Failed to ingest admission details. Make sure data/admission_details.txt exists.")
        
    # Ingest food menu data
    print("Ingesting food menu data...")
    food_menu_engine = vector_db_manager.create_food_menu_index(os.path.join("data", "food_menu.csv"))
    if food_menu_engine:
        print("Food menu data ingested successfully.")
    else:
        print("Failed to ingest food menu data. Make sure data/food_menu.csv exists.")
        
    # Ingest bus details data
    print("Ingesting bus details data...")
    bus_details_engine = vector_db_manager.create_bus_details_index(os.path.join("data", "bus_details.csv"))
    if bus_details_engine:
        print("Bus details data ingested successfully.")
    else:
        print("Failed to ingest bus details data. Make sure data/bus_details.csv exists.")
        
    print("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
