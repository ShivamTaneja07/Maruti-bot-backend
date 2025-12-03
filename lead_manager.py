import pandas as pd
import os
from datetime import datetime

# ✅ UPDATED FILENAME
LEADS_FILE_PATH = 'contact_leads.xlsx'

def save_recommendation_lead(data: dict):
    """
    Saves the lead data from the car recommendation form to contact_leads.xlsx.
    The file is created automatically if it does not exist.

    Args:
        data (dict): A dictionary containing the form data.
    """
    try:
        # Prepare the new data with a timestamp
        new_lead_data = {
            'timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'phone_number': [data.get('contactNumber')],
            'budget': [data.get('budget')],
            'seating': [data.get('seating')],
            'type': [data.get('type')],
            'transmission': [data.get('transmission')],
            'fuel_type': [data.get('fuelType')],
        }
        
        new_df = pd.DataFrame(new_lead_data)

        # Check if the file already exists
        if os.path.exists(LEADS_FILE_PATH):
            # If it exists, read it and append the new data
            existing_df = pd.read_excel(LEADS_FILE_PATH)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # If it doesn't exist, the new data is all we have
            combined_df = new_df

        # Save the combined data back to the Excel file
        combined_df.to_excel(LEADS_FILE_PATH, index=False)
        
        print(f"✅ Successfully saved lead to {LEADS_FILE_PATH} for: {data.get('contactNumber')}")

    except Exception as e:
        print(f"❌ Error saving lead to Excel: {e}")