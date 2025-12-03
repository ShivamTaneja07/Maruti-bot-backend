import pandas as pd
import re

# The path to your data file. Make sure it's correct.
CAR_DATA_PATH = 'data/maruti.csv' 

def find_matching_cars(preferences: dict):
    """
    Cleans the car dataset, filters based on user preferences, and returns matches.
    """
    try:
        df = pd.read_csv(CAR_DATA_PATH)

        # --- Data Cleaning Section (No changes here) ---
        df['ex_showroom_price'] = df['ex_showroom_price'].astype(str).str.replace(r'\s*Lakh', '', regex=True)
        df['ex_showroom_price'] = pd.to_numeric(df['ex_showroom_price'], errors='coerce') * 100000
        df['seating_capacity'] = df['seating_capacity'].astype(str).str.extract(r'(\d+)').astype(float)
        df.dropna(subset=['ex_showroom_price', 'seating_capacity', 'body_type', 'fuel', 'transmission'], inplace=True)
        df['seating_capacity'] = df['seating_capacity'].astype(int)

        # --- Filtering Section ---
        budget = float(preferences.get('budget', 0))
        if budget > 0:
            filtered_df = df[df['ex_showroom_price'] <= budget].copy()
        else:
            filtered_df = df.copy()

        seating_preference = preferences.get('seating')
        if seating_preference:
            seating = int(str(seating_preference).replace('+', ''))
            filtered_df = filtered_df[filtered_df['seating_capacity'] >= seating]
        
        # ✅ UPDATED: Logic to handle a list of car types
        car_types = preferences.get('type')
        # Check if the list of car types is not empty
        if car_types and isinstance(car_types, list) and len(car_types) > 0:
            # Make the comparison case-insensitive
            lowercase_types = [t.lower() for t in car_types]
            filtered_df = filtered_df[filtered_df['body_type'].str.lower().isin(lowercase_types)]

        transmission = preferences.get('transmission')
        if transmission:
            filtered_df = filtered_df[filtered_df['transmission'].str.lower() == transmission.lower()]

        fuel_type = preferences.get('fuelType')
        if fuel_type:
            filtered_df = filtered_df[filtered_df['fuel'].str.contains(fuel_type, case=False, na=False)]

        # --- Final Output Section (No changes here) ---
        recommended_cars = filtered_df.sort_values(by='ex_showroom_price', ascending=True)
        output_columns = [
            'make', 'model', 'version', 'ex_showroom_price', 'fuel', 
            'transmission', 'seating_capacity', 'body_type'
        ]
        existing_output_columns = [col for col in output_columns if col in recommended_cars.columns]
        result = recommended_cars[existing_output_columns].to_dict('records')

        print(f"✅ Found {len(result)} matching cars after cleaning and filtering.")
        return result

    except FileNotFoundError:
        print(f"❌ Error: The file {CAR_DATA_PATH} was not found.")
        return []
    except Exception as e:
        print(f"❌ An unexpected error occurred in recommendation engine: {e}")
        return []