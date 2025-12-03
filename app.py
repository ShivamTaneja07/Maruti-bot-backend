import os
from openpyxl import load_workbook
from dotenv import load_dotenv
import joblib
import faiss
import pandas as pd
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import asyncio
import re
import random
import html
import json
from datetime import datetime
import numpy as np
from lead_manager import save_recommendation_lead
from recommendation_engine import find_matching_cars
from typing import Optional
from typing import Optional, List
load_dotenv()
# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o-mini"
POWERFUL_MODEL_NAME = "gpt-4o"
TTS_MODEL = "tts-1"
ARENA_NUMBER = "18001021800"
NEXA_NUMBER = "18001026392"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# --- INITIALIZATION ---
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
corrected_answer_cache = {}
embedding_model = None
classifier_model = None
car_data_df = pd.DataFrame()
CAR_VARIANTS_DATA = {}

async def get_mentioned_cars_from_ai(question: str, language: str) -> list[str]:
    """
    Uses the AI to reliably identify car models mentioned in a question,
    regardless of the language.
    """
    valid_models_str = ", ".join(CAR_LINKS_AND_BROCHURES.keys())
    
    system_prompt = f"""
    You are an expert entity extractor. Your task is to identify specific Maruti Suzuki car models mentioned in the user's question.
    The user's question is in {language}.
    The list of valid models is: {valid_models_str}.

    Respond ONLY with a comma-separated list of the valid models you found, using their exact English names from the list.
    For example, if the user asks "स्विफ्ट और ब्रेजा के बारे में बताओ", you should respond with "Swift,Brezza".
    If no valid model is found, respond with the single word "None".
    """
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            temperature=0,
            max_tokens=50
        )
        result = response.choices[0].message.content.strip()
        
        if result.lower() == "none":
            return []
            
        # Clean up and return a list of identified car models
        return [car.strip() for car in result.split(',') if car.strip() in CAR_LINKS_AND_BROCHURES]
        
    except Exception as e:
        print(f"Error in get_mentioned_cars_from_ai: {e}")
        return []
async def translate_text(text: str, language: str) -> str:
    """Translates a given text to the target language using the AI."""
    # Avoids an unnecessary API call if the target language is English
    if "english" in language.lower():
        return text
    try:
        prompt = f"You are a professional translator. Translate the following English text into the language: {language}. Provide ONLY the direct translation, nothing else. Do not add any extra phrases or explanations. Text to translate: '{text}'"
        
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation failed for '{text}': {e}")
        return text 

# Conversation Logging Function
# Logs each conversation turn (user question + bot answer) into an Excel file.
# - Creates/updates `conversation_log.xlsx`
# - Each log entry includes timestamp, user question, and bot answer
def log_conversation(user_question: str, bot_answer: str):
    """Logs a single turn of the conversation to an Excel file."""
    filename = "conversation_log.xlsx"
    log_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_question": user_question,
        "bot_answer": bot_answer
    }
    log_to_xlsx(filename, log_data)
    
# --- TRANSLATIONS FOR CALCULATORS ---
TRANSLATIONS = {
    "English": {
        "emi_title": "EMI Calculation Summary", "formula_label": "The formula used is:", "header_detail": "Detail", "header_value": "Value",
        "loan_amount_label": "Loan Amount (P)", "interest_rate_label": "Interest Rate (r)", "tenure_label": "Loan Tenure (n)",
        "bank_label": "Bank", "result_label": "Calculated EMI", "disclaimer": "Disclaimer: This is an approximate calculation."
    },
    "Hindi": {
        "emi_title": "ईएमआई गणना सारांश", "formula_label": "प्रयुक्त सूत्र है:", "header_detail": "विवरण", "header_value": "मूल्य",
        "loan_amount_label": "ऋण राशि (P)", "interest_rate_label": "ब्याज दर (r)", "tenure_label": "ऋण अवधि (n)",
        "bank_label": "बैंक", "result_label": "गणित ईएमआई", "disclaimer": "अस्वीकरण: यह एक अनुमानित गणना है।"
    }
}

# Car Models Loader
# Reads the list of available car models from the `static` directory.
# Each subfolder inside `static/` is treated as a car model.
# - Returns a sorted list of folder names (car models).
def get_car_models_from_folders():
    static_dir = 'static'
    if not os.path.isdir(static_dir): return []
    return sorted([name for name in os.listdir(static_dir) if os.path.isdir(os.path.join(static_dir, name))])

AVAILABLE_CAR_MODELS = get_car_models_from_folders()
CAR_MODELS_STR = ", ".join(AVAILABLE_CAR_MODELS)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    question: str
    language: str = "English"
    session_state: dict = Field(default_factory=dict)

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback_type: str
    message_id: str

class RegenerateRequest(BaseModel):
    question: str
    previous_answer: str
    language: str

class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"   

class ContactFormRequest(BaseModel):
    name: str
    phone_number: str
    pincode: str
    email: str

class TestDriveLeadRequest(BaseModel):
    name: str
    phone_number: str
    pincode: str
    choice: str # This will be "Arena" or "Nexa"
    model: str  
    
class SimpleEMICalculatorRequest(BaseModel):
    loan_amount: float
    interest_rate: float
    loan_tenure_years: int
    language: str

class AdvancedTcoRequest(BaseModel):
    model: str
    version: str
    ex_showroom_price: float
    ownership_years: int
    daily_run_km: int
    total_km_driven: int
    fuel_price_per_liter: float
    annual_maintenance: int
    annual_insurance: int
    language: str

class ContactFormRequest(BaseModel):
    name: str
    phone_number: str
    pincode: str
    email: str
    
class RecommendationRequest(BaseModel):
    # --- Mandatory Fields ---
    budget: str
    contactNumber: str
    language: str
    # --- Optional Fields ---
    seating: Optional[str] = None
    type: Optional[List[str]] = None 
    transmission: Optional[str] = None
    fuelType: Optional[str] = None


# Car Links & Brochures Reference
# Stores key resources for each car model:
# - PDF brochure path (local static file)
# - Official website URL
# - Default test drive channel ("Arena" or "Nexa")
# This is used to provide users quick access to brochures, web pages, and test drive bookings.

CAR_LINKS_AND_BROCHURES = {
    "AltoK10": {"brochure_path": "/static/brochures/AltoK10_brochure.pdf", "website_url": "https://www.marutisuzuki.com/alto-k10", "test_drive_url": "Arena"},
    "Baleno": {"brochure_path": "/static/brochures/Baleno_brochure.pdf", "website_url": "https://www.nexaexperience.com/baleno", "test_drive_url": "Nexa"},
    "Brezza": {"brochure_path": "/static/brochures/Brezza_brochure.pdf", "website_url": "https://www.marutisuzuki.com/brezza", "test_drive_url": "Arena"},
    "Celerio": {"brochure_path": "/static/brochures/Celerio_brochure.pdf", "website_url": "https://www.marutisuzuki.com/celerio", "test_drive_url": "Arena"},
    "Ciaz": {"brochure_path": "/static/brochures/Ciaz_brochure.pdf", "website_url": "https://www.nexaexperience.com/ciaz", "test_drive_url": "Nexa"},
    "Dzire": {"brochure_path": "/static/brochures/Dzire_brochure.pdf", "website_url": "https://www.marutisuzuki.com/dzire", "test_drive_url": "Arena"},
    "EECO": {"brochure_path": "/static/brochures/EECO_brochure.pdf", "website_url": "https://www.marutisuzuki.com/eeco", "test_drive_url": "Arena"},
    "Ertiga": {"brochure_path": "/static/brochures/Ertiga_brochure.pdf", "website_url": "https://www.marutisuzuki.com/ertiga", "test_drive_url": "Arena"},
    "Fronx": {"brochure_path": "/static/brochures/Fronx_brochure.pdf", "website_url": "https://www.nexaexperience.com/fronx", "test_drive_url": "Nexa"},
    "Grand Vitara": {"brochure_path": "/static/brochures/Grand_Vitara_brochure.pdf", "website_url": "https://www.nexaexperience.com/grand-vitara", "test_drive_url": "Nexa"},
    "Ignis": {"brochure_path": "/static/brochures/Ignis_brochure.pdf", "website_url": "https://www.nexaexperience.com/ignis", "test_drive_url": "Nexa"},
    "Invicto": {"brochure_path": "/static/brochures/Invicto_brochure.pdf", "website_url": "https://www.nexaexperience.com/invicto", "test_drive_url": "Nexa"},
    "Jimny": {"brochure_path": "/static/brochures/Jimny_brochure.pdf", "website_url": "https://www.nexaexperience.com/jimny", "test_drive_url": "Nexa"},
    "Spresso": {"brochure_path": "/static/brochures/Spresso_brochure.pdf", "website_url": "https://www.marutisuzuki.com/s-presso", "test_drive_url": "Arena"},
    "Swift": {"brochure_path": "/static/brochures/Swift_brochure.pdf", "website_url": "https://www.marutisuzuki.com/swift", "test_drive_url": "Arena"},
    "WagonR": {"brochure_path": "/static/brochures/WagonR_brochure.pdf", "website_url": "https://www.marutisuzuki.com/wagonr", "test_drive_url": "Arena"},
    "XL6": {"brochure_path": "/static/brochures/XL6_brochure.pdf", "website_url": "https://www.nexaexperience.com/xl6", "test_drive_url": "Nexa"}
}
    
# Excel Logging Utility   
#- Creates the file if it doesn’t exist.
# - If it exists, reads it in, appends the new row, and writes it back.
# - Handles common errors like permission issues (e.g. file open in Excel).

def log_to_xlsx(filename: str, data: dict):
    df_new_row = pd.DataFrame([data])
    try:
        if not os.path.exists(filename):
            df_new_row.to_excel(filename, index=False, engine='openpyxl')
        else:
            existing_df = pd.read_excel(filename, engine='openpyxl')
            updated_df = pd.concat([existing_df, df_new_row], ignore_index=True)
            updated_df.to_excel(filename, index=False, engine='openpyxl')
    except PermissionError:
        print(f" PERMISSION DENIED: Could not write to {filename}. Is the file open in Excel?")
    except Exception as e:
        print(f" An unexpected error occurred while writing to {filename}: {e}")


# --- API Endpoints ---
   #  car_recommendation (get_recommendation)
@app.post("/api/get_recommendation")
async def get_recommendation(request: RecommendationRequest):
    """
    Receives preferences, finds cars, and streams the AI-generated 
    response back to the client.
    """
    preferences = request.dict()
    language = request.language

    # 1. Save the lead 
    save_recommendation_lead(preferences)
    
    # 2. Find matching cars 
    recommended_cars = find_matching_cars(preferences)

    # 3. Create the prompt for the AI 
    if not recommended_cars:
        prompt = f"You are a helpful car assistant. Inform the user that based on their specific criteria, you couldn't find a perfect match right now. Suggest they try adjusting their budget or other preferences. The response MUST be entirely in this language: {language}."
    else:
        prompt = f"""
You are a helpful car dealership assistant. Your task is to present a list of recommended cars to a user in a clean and easy-to-read table format.

Instructions:
1.  Start with a friendly opening sentence in {language}.
2.  The entire response, including the table headers, MUST be in this language: {language}.
3.  Create a Markdown table using the provided car data.
4.  The table should have the following columns: Make, Model, Version, and Price. Please translate these headers into {language}.
5.  Format the price in a user-friendly way (e.g., with commas).

Here is the raw data for the recommended cars:
{json.dumps(recommended_cars, indent=2)}
"""

    #  Define an async generator to stream the response
# Streams AI-generated responses back to the client in real-time.
# Uses OpenAI's `stream=True` option so tokens are yielded as they arrive.
# Wrapped in a FastAPI `StreamingResponse` for SSE-style streaming.(SSE-Server sent events)

    async def stream_generator():
        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                stream=True  #  This is the key change to enable streaming
            )
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield content
        except Exception as e:
            print(f"AI streaming failed for recommendation: {e}")
            yield "Sorry, an error occurred while finding recommendations."
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# Feedback Endpoint
# Handles user feedback on bot answers.
# If "up": log successful answers, cache them for reuse.
# If "down": log failed answers so they can be reviewed later.
# Returns JSONResponse with status + message.
@app.post("/api/feedback")
async def handle_feedback(req: FeedbackRequest):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if req.feedback_type == "up":
        log_data = {"timestamp": timestamp, "question": req.question, "successful_answer": req.answer, "feedback": "up"}
        log_to_xlsx("thumbs_up_log.xlsx", log_data)
        corrected_answer_cache[req.question.lower().strip()] = req.answer
        return JSONResponse(content={"status": "success", "message": "Successful feedback logged."})
    elif req.feedback_type == "down":
        log_data = {"timestamp": timestamp, "question": req.question, "failed_answer": req.answer, "feedback": "down_retry_triggered"}
        log_to_xlsx("thumbs_down_log.xlsx", log_data)
        return JSONResponse(content={"status": "success", "message": "Failed attempt logged."})
    return JSONResponse(content={"status": "error", "message": "Invalid feedback type."}, status_code=400)


# Save Contact Endpoint
# Stores contact form submissions into an Excel file.
# - Captures name, phone number, pincode, and email.
# - Adds a timestamp for tracking.
# - Logs everything to `contact_leads.xlsx`. 

@app.post("/api/save_contact")
async def save_contact(req: ContactFormRequest):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = { "timestamp": timestamp, "name": req.name, "phone_number": req.phone_number,"pincode":req.pincode,"email": req.email }
    log_to_xlsx("contact_leads.xlsx", log_data)
    return JSONResponse(content={"status": "success", "message": "Contact details saved."})

# Save Test Drive Lead Endpoint
# Stores test drive requests into an Excel file.
# - Captures name, phone, pincode, dealership choice (Arena/Nexa), and car model.
# - Adds timestamp for tracking.
# - Saves data to `contact_leads.xlsx` (could later be split into a dedicated file if needed)

@app.post("/api/save_test_drive_lead")
async def save_test_drive_lead(req: TestDriveLeadRequest):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = {
        "timestamp": timestamp,
        "name": req.name,
        "phone_number": req.phone_number,
        "pincode": req.pincode,
        "choice": req.choice,
        "model": req.model 
    }
    log_to_xlsx("contact_leads.xlsx", log_data)
    return JSONResponse(content={"status": "success", "message": "Test drive lead saved."})

# Car Variants API Endpoint
# Returns all available car variant data as JSON.
# This can be used by frontend apps to populate dropdowns, tables, or recommendation engines.

@app.get("/api/car_variants")
async def get_car_variants():
    return JSONResponse(content=CAR_VARIANTS_DATA)
CAR_MILEAGE = {
    "AltoK10": 24.5, "Baleno": 22.9, "Brezza": 19.8, "Celerio": 25.5, "Ciaz": 20.5, "Dzire": 22.6,
    "EECO": 19.7, "Ertiga": 20.3, "Fronx": 22.0, "Grand Vitara": 21.0, "Ignis": 20.8, "Invicto": 23.2,
    "Jimny": 16.9, "Spresso": 25.0, "Swift": 22.5, "WagonR": 25.0, "XL6": 20.9
}

# Car Test Drive Channels Endpoint

# Returns a mapping of car models to their default test drive channel.
# - Uses `CAR_LINKS_AND_BROCHURES` as the source.
# - Output format: { "Model": "Arena" or "Nexa" }
# - Useful for frontend to know which dealership channel to direct users to.

@app.get("/api/car_channels")
async def get_car_channels():
    channel_mapping = {
        model: details["test_drive_url"]
        for model, details in CAR_LINKS_AND_BROCHURES.items()
    }
    return JSONResponse(content=channel_mapping)

 #Advanced TCO (Total Cost of Ownership) Calculation Endpoint
# Calculates the complete cost of owning a car over a number of years.
# - Includes fuel, maintenance, insurance, and depreciation.
# - Adjusts resale value based on expected vs actual mileage.
# - Converts the result into a monthly cost.
# - Sends the detailed results to the AI to get a user-friendly, language-specific summary.

@app.post("/api/calculate_tco")
async def calculate_advanced_tco(req: AdvancedTcoRequest):
    # --- 1. Perform all calculations
    mileage = CAR_MILEAGE.get(req.model, 20.0)
    annual_run_km = req.daily_run_km * 365
    future_km_run = annual_run_km * req.ownership_years
    total_fuel_cost = (future_km_run / mileage) * req.fuel_price_per_liter
    total_maintenance = req.annual_maintenance * req.ownership_years
    total_insurance = req.annual_insurance * req.ownership_years
    total_running_cost = total_fuel_cost + total_maintenance + total_insurance
    initial_price = req.ex_showroom_price
    depreciation_rates = [0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.08]
    resale_value = initial_price
    for i in range(min(req.ownership_years, len(depreciation_rates))):
        resale_value *= (1 - depreciation_rates[i])
    total_driven_at_sale = req.total_km_driven + future_km_run
    avg_km_at_sale = 15000 * req.ownership_years
    if total_driven_at_sale > avg_km_at_sale * 1.2: resale_value *= 0.95
    elif total_driven_at_sale < avg_km_at_sale * 0.8: resale_value *= 1.05
    net_total_cost = (initial_price + total_running_cost) - resale_value
    cost_per_month = net_total_cost / (req.ownership_years * 12) if req.ownership_years > 0 else 0
    # --- 2. Create a detailed prompt with the results for the AI ---
    prompt_for_translation = f"""
    Here is the result of a TCO calculation. Please present this information to the user in a summarized, easy-to-read format.
    The response MUST be entirely in the user's language: {req.language}.
    Do not include the "Disclaimer" in your response.

    Calculation Details:
    - Car Model: {req.model} {req.version}
    - Ownership Period: {req.ownership_years} years
    - Initial Price: {initial_price:,.0f}
    - Total Running Costs: {total_running_cost:,.0f} (Fuel: {total_fuel_cost:,.0f}, Maintenance: {total_maintenance:,.0f}, Insurance: {total_insurance:,.0f})
    - Estimated Resale Value (Credit): {resale_value:,.0f}
    - Effective Cost Per Month: {cost_per_month:,.0f}
    """

    # Call the AI to get the final, translated response 
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": f"You are a helpful car assistant. Your response must be in {req.language}."}, 
                  {"role": "user", "content": prompt_for_translation}],
        temperature=0.5
    )
    final_result = response.choices[0].message.content
    
    return JSONResponse(content={"result": final_result})

# Regenerate Answer Endpoint
# Handles cases where the previous AI response was unsatisfactory.
# - Streams a completely new answer to the user.
# - Differentiates between general queries and car filtering queries.
# - Uses a streaming response so the frontend can show the answer in real-time.

@app.post("/api/regenerate")
async def regenerate_answer(req: RegenerateRequest):
    async def stream_regenerated_answer():
        try:
            decision = await get_ai_router_decision(req.question, {})
            tool_used = decision.get("tool", "general_query")
            system_prompt = f"You are an expert on Maruti Suzuki cars. Your final response MUST be in {req.language}."
            if tool_used == "filter_cars":
                user_prompt = f"""The user's search for '{req.question}' led to an unsatisfactory answer: '{req.previous_answer}'. Instead of trying to answer directly again, your task is to provide a helpful, conversational response. If the previous answer was "No cars found," suggest relaxing the criteria (e.g., "Would you like to increase your budget slightly?"). Offer to search for something else or suggest a popular model. Do NOT repeat the previous answer. Be proactive and helpful."""
            else:
                user_prompt = f"""Our previous attempt to answer the user's question '{req.question}' was not good. Please answer the question now using your own extensive knowledge base. Ignore our previous failed attempt and provide a completely new, comprehensive, and accurate response. The previous failed answer (for your reference only) was: '{req.previous_answer}'."""
            stream = await client.chat.completions.create(model=POWERFUL_MODEL_NAME, messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], stream=True, temperature=0.7)
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield content
                    await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error during regeneration: {e}")
            yield "I apologize, but I encountered an error while trying to generate a new answer. Please try asking in a different way."
    return StreamingResponse(stream_regenerated_answer(), media_type="text/event-stream")

# Load and Process Car Data
# Loads raw CSV data, cleans it, and prepares structures for TCO calculations and semantic search.
# - Cleans column names.
# - Converts price and seating columns to numeric values.
# - Builds a dictionary of car variants for easy lookup in TCO calculators.
# - Converts each row into a text chunk for semantic search and builds a FAISS index.

def load_and_process_data(path: str):
    global car_data_df, index, stored_chunks, CAR_VARIANTS_DATA
    try:
        df = pd.read_csv(path, encoding='latin-1')
        df.columns = [col.strip().lower().replace(' ', '_').replace('/', '_').replace('.', '').replace('(', '').replace(')', '') for col in df.columns]
        if 'ex_showroom_price' in df.columns:
            price_str = df['ex_showroom_price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df['numeric_price'] = pd.to_numeric(price_str, errors='coerce') * 100000
        if 'seating_capacity' in df.columns:
            df['numeric_seating'] = pd.to_numeric(df['seating_capacity'], errors='coerce')
        car_data_df = df.copy()
        temp_variants = {}
        for _, row in df.dropna(subset=['model', 'numeric_price', 'version']).iterrows():
            model_name = row['model'].strip().upper()
            if model_name not in temp_variants:
                temp_variants[model_name] = []
            temp_variants[model_name].append({"version": row['version'], "price": row['numeric_price']})
        CAR_VARIANTS_DATA = temp_variants
        print("Car variants data structure created for TCO calculator.")
        print(" Successfully loaded and processed CLEANED car data.")
        chunks = [", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip() != ""]) for _, row in df.iterrows()]
        index, stored_chunks = build_faiss_index(chunks)
        if index:
            print("Successfully built FAISS index for semantic search.")
        else:
            print(" FAISS index could not be built (no data).")
    except Exception as e:
        print(f" FATAL: An unexpected error occurred while loading data: {e}")
        index, stored_chunks = None, []
      
# Generate Text Embeddings
# Converts a list of text strings into vector embeddings for semantic search or AI tasks.
# - Uses the global EMBED_MODEL (e.g., SentenceTransformer or similar)
# - Returns normalized NumPy embeddings for better similarity calculations.

def embed_texts(texts):
    return EMBED_MODEL.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# Build FAISS Index for Semantic Search
# Converts a list of text chunks into embeddings and builds a FAISS index for fast similarity search.
# - Each chunk is encoded into a vector using the global EMBED_MODEL.
# - Normalized embeddings are used for cosine similarity.
# - Returns both the FAISS index and the original chunks for reference.

def build_faiss_index(chunks):
    if not chunks: return None, []
    embeddings = EMBED_MODEL.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks

# Tool: Filter Cars by Criteria
# Filters the car dataset based on user-specified criteria:
# - Price range (min/max)
# - Fuel type (e.g., Petrol, Diesel)
# - Transmission type (e.g., Manual, Automatic)
# - Seating capacity
# Returns a Markdown table of matching cars, sorted by price.

def tool_filter_cars(max_price=None, min_price=None, fuel_type=None, transmission=None, seating_capacity=None):
    if car_data_df.empty: return "Sorry, the car database is currently unavailable."
    temp_df = car_data_df.copy()
    if max_price is not None: temp_df = temp_df[temp_df['numeric_price'] <= float(max_price * 100000)]
    if min_price is not None: temp_df = temp_df[temp_df['numeric_price'] >= float(min_price * 100000)]
    if seating_capacity is not None: temp_df = temp_df[temp_df['numeric_seating'] == int(seating_capacity)]
    if fuel_type and 'fuel' in temp_df.columns: temp_df = temp_df[temp_df['fuel'].str.contains(fuel_type, case=False, na=False)]
    if transmission and 'transmission' in temp_df.columns: temp_df = temp_df[temp_df['transmission'].str.contains(transmission, case=False, na=False)]
    if temp_df.empty: return "No cars were found that match your specific criteria. You might want to try a broader search."
    display_columns = ['model', 'version', 'ex_showroom_price', 'fuel', 'transmission', 'seating_capacity']
    existing_columns = [col for col in display_columns if col in temp_df.columns]
    if not existing_columns: return temp_df.to_markdown(index=False)
    # Convert price back to Lakhs for display
    temp_df['ex_showroom_price'] = (temp_df['numeric_price'] / 100000).round(2).astype(str) + ' Lakh'
    return temp_df.sort_values(by='numeric_price')[existing_columns].to_markdown(index=False)

# EMI Calculation Endpoint
# Calculates the monthly EMI for a given loan amount, interest rate, and tenure.
# - Supports multiple languages by translating label texts via AI.
# - Returns the result as a Markdown table for display in chat interfaces.

@app.post("/api/calculate_emi")
async def calculate_emi_api(req: SimpleEMICalculatorRequest):
    if req.loan_amount <= 0:
        return JSONResponse(status_code=400, content={"result": "Loan amount must be greater than zero."})

    monthly_rate = (req.interest_rate / 12) / 100
    tenure_months = req.loan_tenure_years * 12
    emi = 0
    if monthly_rate > 0:
        emi = (req.loan_amount * monthly_rate * (1 + monthly_rate)**tenure_months) / ((1 + monthly_rate)**tenure_months - 1)
    elif tenure_months > 0:
        emi = req.loan_amount / tenure_months

    # Define the English labels that need translation
    labels = {
        "emi_title": "EMI Calculation Summary",
        "formula_label": "The formula used is:",
        "header_detail": "Detail",
        "header_value": "Value",
        "loan_amount_label": "Loan Amount (P)",
        "interest_rate_label": "Interest Rate (r)",
        "tenure_label": "Loan Tenure (n)",
        "result_label": "Calculated EMI",
        "disclaimer": "Disclaimer: This is an approximate calculation."
    }

    # If language is not English, get translations from the AI
    if req.language.lower() != 'english':
        try:
            translation_prompt = f"""
            You are a professional JSON translator. Translate the values of the following JSON object into the language: {req.language}.
            Return ONLY a valid JSON object with the translated values. Do not change the keys.

            Input JSON:
            {json.dumps(labels)}
            """
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": translation_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            translated_labels = json.loads(response.choices[0].message.content)
            labels.update(translated_labels) # Update labels with translations
        except Exception as e:
            print(f"Error during EMI label translation: {e}")
            # If translation fails, we will proceed with English labels
    
    # Build the final response using the (now translated) labels and calculated values
    result_md = f"""### {labels['emi_title']}
{labels['formula_label']}
$$ EMI = P \\times r \\times \\frac{{(1+r)^n}}{{(1+r)^n - 1}} $$
| {labels['header_detail']} | {labels['header_value']} |
| :--- | :--- |
| {labels['loan_amount_label']} | ₹ {req.loan_amount:,.2f} |
| {labels['interest_rate_label']} | {req.interest_rate}% p.a. |
| {labels['tenure_label']} | {req.loan_tenure_years} Years |
| **{labels['result_label']}** | **₹ {emi:,.2f} / month** |

*{labels['disclaimer']}*
"""
    return JSONResponse(content={"result": result_md})

# Text-to-Speech Endpoint
# Converts user-provided text into spoken audio.
# - Streams audio in real-time to the frontend.
# - Supports different voices via the 'voice' parameter.
@app.post("/api/tts")
async def tts_api(req: TTSRequest):
    async def audio_stream_generator():
        try:
            async with client.audio.speech.with_streaming_response.create(model=TTS_MODEL, voice=req.voice, input=req.text) as response:
                async for chunk in response.iter_bytes(chunk_size=4096):
                    yield chunk
        except Exception as e:
            print(f"Error during TTS generation: {e}")
    return StreamingResponse(audio_stream_generator(), media_type="audio/mpeg")


# Extract Color from Filename
# Infers the car color from the image filename.
# - Removes the model name from the filename.
# - Cleans up separators like '-' or '_'.
# - Capitalizes each word for display.
# - Returns "Default" if no color can be inferred.

def extract_color_from_filename(filename, model_name):
    try:
        base_name = os.path.splitext(filename)[0]
        color_part = re.sub(re.escape(model_name), '', base_name, flags=re.IGNORECASE).lstrip('-_ ')
        return color_part.replace('-', ' ').replace('_', ' ').title() if color_part else "Default"
    except: return "Default"

# AI Request Router
# Analyzes a user's query and decides which tool the chatbot should use.
# - Routes queries to tools like `filter_cars`, `get_car_images`, or `general_query`.
# - Returns a JSON object with the chosen tool and any required arguments.
# - Can handle a pre-defined flow override via session_state.

async def get_ai_router_decision(query: str, session_state: dict):
    if session_state.get("flow") == "car_suggestion": return {"tool": "continue_suggestion_flow", "args": {}}
    system_prompt = f"""You are an intelligent request router for a Maruti Suzuki chatbot. Your job is to analyze the user's query and decide which tool to use. Respond in valid JSON.
Available tools:
1. `filter_cars`: For finding cars by criteria (price, fuel, transmission, seating). `args`: `max_price` (number, in Lakhs), `min_price` (number, in Lakhs), `fuel_type` (string: "Petrol", "CNG"), `transmission` (string: "Manual", "Automatic"), `seating_capacity` (number).
2. `get_car_images`: For requests of "images", "pictures", or "colors" of a specific car. `args`: `car_model` (string). Valid models: {CAR_MODELS_STR}.
3. `general_query`: Fallback for conversational questions, greetings, history,requests for brochures or website links, etc. `args`: {{}}
User Query: "{query}" """
    try:
        response = await client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "system", "content": system_prompt}], response_format={"type": "json_object"}, temperature=0)
        decision = json.loads(response.choices[0].message.content)
        if 'args' not in decision: decision['args'] = {}
        return decision
    except Exception as e:
        print(f"AI Router Error: {e}")
        return {"tool": "general_query", "args": {}}

# Ask OpenAI for Detailed Answer
# Uses AI to generate a detailed, context-aware answer to the user's question.
# - Incorporates conversation history for follow-ups.
# - Uses a special `[CAR_IMAGE: ModelName]` tag to indicate images.
# - Enforces the response language.
# - Streams the AI's response in real-time.

async def ask_openai_for_details(query: str, context: str, language: str, history: list):
    system_prompt = f"""You are a helpful Maruti Suzuki cars expert assistant.

**CRITICAL IMAGE RULE:**
- To display a car image, you MUST ONLY use the special tag: `[CAR_IMAGE: ModelName]`.
- Correct example: `Here is the Swift [CAR_IMAGE: Swift]`.
- NEVER use Markdown's `![alt](link)` syntax. It is disabled and will show up as broken text.
- Bad example (DO NOT DO THIS): `![A picture of the Swift](https://example.com/swift.png)`

The list of valid models for the tag is: {CAR_MODELS_STR}.
Your final response MUST be in {language}."""
    # Build the full conversation history for the AI model
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add the previous turns of the conversation
    if history:
        messages.extend(history[:-1]) # Add all history except the latest user question

    # Add the final user question with the retrieved context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })
    
    # Call the API with the complete context and history
    stream = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        temperature=0.5
    )
    async for chunk in stream:
        if content := chunk.choices[0].delta.content:
            yield content
            
#Stream and Process Final AI Answer
# Processes the AI's answer text to:
# 1. Identify [CAR_IMAGE: ModelName] tags.
# 2. Replace each tag with a fully functional HTML carousel of images.
# 3. Include color selector buttons to switch between car colors.
# 4. Stream the resulting HTML/text back to the frontend incrementally.

async def stream_and_process_final_answer(answer_text, request):
    last_match_end = 0
    for match in re.finditer(r'\[CAR_IMAGE:\s*([^\]]+)\]', answer_text):
        yield answer_text[last_match_end:match.start()]
        car_name_from_tag = match.group(1).strip()
        folder_name = next((m for m in AVAILABLE_CAR_MODELS if m.lower() in car_name_from_tag.lower()), None)
        if folder_name:
            image_path_dir = os.path.join('static', folder_name)
            images_with_colors = []
            if os.path.isdir(image_path_dir):
                for f in os.listdir(image_path_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images_with_colors.append({"file": f, "color": extract_color_from_filename(f, folder_name)})
            if images_with_colors:
                carousel_id = f"carousel-{random.randint(1000,9999)}"; carousel_html = f'<div id="{carousel_id}" class="carousel-container">'
                for i, img in enumerate(images_with_colors):
                    url = str(request.url_for('static', path=f"{folder_name}/{img['file']}"))
                    carousel_html += f'<img src="{url}" alt="{folder_name} {img["color"]}" class="carousel-item {"active" if i==0 else ""}" data-color="{img["color"].lower()}" data-model="{folder_name}">'
                if len(images_with_colors) > 1: carousel_html += f'<button class="carousel-btn prev" onclick="navigateCarousel(\'{carousel_id}\',-1)">&#10094;</button><button class="carousel-btn next" onclick="navigateCarousel(\'{carousel_id}\',1)">&#10095;</button>'
                carousel_html += '</div>'; color_selector_html = '<div class="color-selector">'
                for img in images_with_colors:
                    js = f"document.querySelectorAll('#{carousel_id} .carousel-item').forEach(i=>i.classList.toggle('active',i.getAttribute('data-color')==='{img['color'].lower()}'))"
                    color_selector_html += f'<button class="color-btn" onclick="{html.escape(js)}">{img["color"]}</button>'
                color_selector_html += '</div>'; yield carousel_html + color_selector_html
        last_match_end = match.end()
    yield answer_text[last_match_end:]
    


# Streams an interactive AI response to the frontend Handles:
#   - Static greetings in multiple languages
#   - System messages (e.g., confirmations)
#   - Cached/corrected answers
#   - Filtered car searches
#   - Car images with carousel HTML
#   - General queries using RAG + context + conversation history
#   - Appends links for brochures, test drives, and websites
#   - Streams chunks progressively and updates session state
   
async def rewrite_query_for_rag(history: list) -> str:
    """Uses an LLM to rewrite a follow-up query to be self-contained."""
    if len(history) <= 1:
        return history[0]['content']

    messages = [
        {"role": "system", "content": "You are an expert at rewriting a user's question to be a standalone, self-contained question based on the chat history. Only output the rewritten question and nothing else."},
        *history
    ]

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0, n=1
        )
        rewritten_question = response.choices[0].message.content.strip()
        print(f"Original Query: '{history[-1]['content']}' -> Rewritten Query: '{rewritten_question}'")
        return rewritten_question
    except Exception as e:
        print(f" Query rewriting failed: {e}")
        return history[-1]['content']

async def stream_generator(question: str, language: str, request: Request, session_state: dict):
    
    if question == "GET_GREETING":
        static_greetings = {
        "Assamese": "মাৰুতি সুজুকীলৈ স্বাগতম! মই আপোনাৰ ভাৰ্চুৱেল সহায়ক। মই আজি আপোনাক কেনেদৰে সহায় কৰিব পাৰোঁ?",
        "Bengali": "মারুতি সুজুকিতে স্বাগতম! আমি আপনার ভার্চুয়াল সহকারী। আমি আজ আপনাকে কিভাবে সাহায্য করতে পারি?",
        "Bodo": "मारुति सुजुकीआव बरायबाय! आं नोंथांनि भर्चुयेल मददगिरि। आं दिनै नोंथांनो माबोरै मदद खालामनो हायो?",
        "Dogri": "मारुति सुजुकी च तुंदा स्वागत है! मैं तुंदा वर्चुअल असिस्टेंट आं। मैं अज्ज तुंदी क्या मदद करी सकदा आं?",
        "English": "Welcome to Maruti Suzuki! I'm your virtual assistant. How can I help you today?",
        "Gujarati": "મારુતિ સુઝુકીમાં આપનું સ્વાગત છે! હું તમારો વર્ચ્યુઅલ સહાયક છું। હું આજે તમને કેવી રીતે મદદ કરી શકું?",
        "Hindi": "मारुति सुजुकी में आपका स्वागत है! मैं आपका वर्चुअल असिस्टेंट हूं। मैं आज आपकी कैसे मदद कर सकता हूं?",
        "Kannada": "ಮಾರುತಿ ಸುಜುಕಿಗೆ ಸ್ವಾಗತ! ನಾನು ನಿಮ್ಮ ವರ್ಚುವಲ್ ಸಹಾಯಕ. ನಾನು ಇಂದು ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?",
        "Kashmiri": "मारुति सुजुकी मंज़ छुव स्वागत! ब छुस तोहिन्द वर्चुअल असिस्टेंट। ब क्यथ पऺठ्य छि करितोहि मदद ॳज़?",
        "Konkani": "मारुती सुझुकींत तुमकां येवकार! हांव तुमचो व्हर्च्युअल सहाय्यक। आयज हांव तुमकां कशी मजत करूं?",
        "Maithili": "मारुति सुजुकी मे अहाँक स्वागत अछि! हम अहाँक वर्चुअल सहायक छी। हम आइ अहाँके कोना मदद क' सकैत छी?",
        "Malayalam": "മാരുതി സുസുക്കിയിലേക്ക് സ്വാഗതം! ഞാൻ നിങ്ങളുടെ വെർച്വൽ അസിസ്റ്റന്റാണ്. ഇന്ന് ഞാൻ നിങ്ങളെ എങ്ങനെ സഹായിക്കും?",
        "Manipuri": "মারুতি সুজুকীদা তরাম্না ওকচরি! ঐ नोंगी भर्चुएल असिस्तेन्तনি। ঙসি ঐনা नोंबू করি মতেং পাংবগে?",
        "Marathi": "मारुती सुझुकीमध्ये आपले स्वागत आहे! मी तुमचा व्हर्च्युअल सहाय्यक आहे। मी आज तुम्हाला कशी मदत करू शकेन?",
        "Nepali": "मारुति सुजुकीमा स्वागत छ! म तपाईंको भर्चुअल सहायक हुँ। म आज तपाईंलाई कसरी मद्दत गर्न सक्छु?",
        "Oriya": "ମାରୁତି ସୁଜୁକିକୁ ସ୍ୱାଗତ! ମୁଁ ଆପଣଙ୍କର ଭର୍ଚ୍ଆଲ୍ ସହାୟକ। ମୁଁ ଆଜି ଆପଣଙ୍କୁ କିପରି ସାହାଯ୍ୟ କରିପାରେ?",
        "Punjabi": "ਮਾਰੂਤੀ ਸੁਜ਼ੂਕੀ ਵਿੱਚ ਤੁਹਾਡਾ ਸੁਆଗਤ ਹੈ! ਮੈਂ ਤੁਹਾਡਾ ਵਰਚੁਅਲ ਸਹਾਇਕ ਹਾਂ। ਮੈਂ ਅੱਜ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "Sanskrit": "मारुतिसुजुकीमध्ये स्वागतम्! अहं भवतः आभासी सहायकः अस्मि। अद्य अहं भवतः कथं साहाय्यं कर्तुं शक्नोमि?",
        "Santali": "मारुति सुजुकी रे सगुन दाराम! इंञ आमरेन वर्चुअल असिस्टेंट काना। तिहिंञ् चेद् लेकाते इंञ आमरेन गोड़ो दाड़ेयामा?",
        "Sindhi": "ماروتي سوزوڪي ۾ ڀليڪار! مان توهان جو ورچوئل اسسٽنٽ آهيان. اڄ مان توهان جي ڪيئن مدد ڪري سگهان ٿو؟",
        "Tamil": "மாருதி சுஸுகிக்கு வரவேற்கிறோம்! நான் உங்கள் ভার্চுவல் உதவியாளர். இன்று நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "Telugu": "మారుతి సుజుకికి స్వాగతం! నేను మీ వర్చువల్ అసిస్టెంట్‌ని. ఈ రోజు నేను మీకు ఎలా సహాయం చేయగలను?",
        "Urdu": "ماروتی سوزوکی میں خوش آمدید! میں آپ کا ورچوئল اسسٹنٹ ہوں۔ میں آج آپ کی کس طرح مدد کر سکتا ہوں؟",
        }
        greeting = static_greetings.get(language, static_greetings["English"])
        yield greeting
        return
    SYSTEM_MESSAGES_ENGLISH = {
        "POST_CONTACT_CONFIRMATION": "Thank you! Our team will contact you shortly.",
        "POST_TEST_DRIVE_CONFIRMATION": "Thank you! We've saved your details and our team will contact you shortly.",
        "PRE_RECOMMENDATION_MESSAGE": "Thank you! I'm finding the best cars based on your preferences...",
    }

    if question in SYSTEM_MESSAGES_ENGLISH:
        english_text = SYSTEM_MESSAGES_ENGLISH[question]
    
    # Create a prompt to ask the AI for a direct translation
        translation_prompt = f"You are a professional translator. Translate the following English text into the language: {language}. Provide only the direct translation, nothing else. Text to translate: '{english_text}'"
    
    # Call the AI to get the translation and stream it back
        try:
            stream = await client.chat.completions.create(
                model=MODEL_NAME, # Use the fast model for this simple task
                messages=[{"role": "user", "content": translation_prompt}],
                stream=True,
                temperature=0.1 # Low temperature for accurate translation
            )
            async for chunk in stream:
                if content := chunk.choices[0].delta.content:
                    yield content
            return # Stop further processing
        except Exception as e:
            print(f" Translation failed for system message: {e}")
            yield english_text # Fallback to English if translation fails
            return
    if not stored_chunks:
        yield "I'm sorry, my knowledge base is currently unavailable. Please try again later."
        return
    if cached_answer := corrected_answer_cache.get(question.lower().strip()):
        yield cached_answer
        return

    # Initialize or retrieve conversation history
    history = session_state.get("history", [])
    history.append({"role": "user", "content": question})
    if len(history) > 10:
        history = history[-10:]

    decision = await get_ai_router_decision(question, session_state)
    tool, args = decision.get("tool"), decision.get("args", {})
    full_response_buffer = ""
    if tool == "filter_cars":
        filtered_results = tool_filter_cars(**args)
        final_prompt = f"The user asked: '{question}'. Based on our database, here is the list of cars: {filtered_results}. Present this in a friendly format in {language}."
        stream = await client.chat.completions.create(model=MODEL_NAME, messages=[{"role": "user", "content": final_prompt}], stream=True)
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                full_response_buffer += content

    elif tool == "get_car_images" and args.get("car_model"):
        car_model = args.get("car_model")
        full_response_buffer = f"Here are the images for the {car_model}:\n[CAR_IMAGE:{car_model}]"

    else:  # General queries will use memory and RAG
        standalone_question = await rewrite_query_for_rag(history)
        
        q_emb = embed_texts([standalone_question])
        _, I = index.search(q_emb, k=7)
        retrieved = "\n\n".join([stored_chunks[i] for i in I[0]])
        
        system_prompt = f"""You are a helpful Maruti Suzuki cars expert assistant.
1. Use the 'Context' provided below to answer the user's latest question.
2. Leverage the 'Conversation History' to understand follow-up questions like "tell me more".
3. When mentioning a Maruti Suzuki model for the first time, add its image tag: [CAR_IMAGE: ModelName]. Valid models: {CAR_MODELS_STR}.
4. Your final response MUST be in {language}."""

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[:-1]) 
        messages.append({"role": "user", "content": f"Context:\n{retrieved}\n\nQuestion: {standalone_question}"})
        
        stream = await client.chat.completions.create(model=MODEL_NAME, messages=messages, stream=True, temperature=0.5)
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                full_response_buffer += content
        # Call the AI to reliably identify car models, regardless of language
        identified_cars = await get_mentioned_cars_from_ai(standalone_question, language)
        
        contact_message_html = ""
        if identified_cars:
            found_arena = False
            found_nexa = False
            # Look up the channel for each car the AI identified
            for car_model in identified_cars:
                details = CAR_LINKS_AND_BROCHURES.get(car_model)
                if details:
                    if details["test_drive_url"] == "Arena":
                        found_arena = True
                    elif details["test_drive_url"] == "Nexa":
                        found_nexa = True
            
            contact_items = []
            if found_arena:
                arena_link = f'<b>Arena:</b> <a href="tel:{ARENA_NUMBER}">1800-102-1800</a>'
                contact_items.append(arena_link)
            if found_nexa:
                nexa_link = f'<b>Nexa:</b> <a href="tel:{NEXA_NUMBER}">1800-102-6392</a>'
                contact_items.append(nexa_link)
            
            if contact_items:
            # 1. This line DEFINES the variable with the English text.
                intro_text_english = "For more queries, you can contact us:"
            
            # 2. This line USES the variable in the function call.
                translated_intro = await translate_text(intro_text_english, language)

            # The rest of the code builds the final HTML
                contact_details = "<br>".join(contact_items)
                contact_message_html = f"""
                \n\n<div class="chatbot-contact-info">
                {translated_intro}<br>{contact_details}
                </div>
                """
                full_response_buffer += contact_message_html
        mentioned_car_key = None
        # Use the rewritten question to check for a car model
        for car_key in CAR_LINKS_AND_BROCHURES.keys():
            if car_key.lower() in standalone_question.lower():
                mentioned_car_key = car_key
                break
        
        if mentioned_car_key:
            links = CAR_LINKS_AND_BROCHURES[mentioned_car_key]
            
            base_url = str(request.base_url).rstrip('/')
            brochure_full_url = f"{base_url}{links['brochure_path']}"
            html_links = f"""
\n\n---
\n<div class="chatbot-links">
<button class="chatbot-link-button js-test-drive-btn" data-channel="{links['test_drive_url']}" data-model="{mentioned_car_key}">Book a Test Drive</button>
<a href="{brochure_full_url}" class="chatbot-link-button">Download Brochure</a>
<a href="{links['website_url']}" class="chatbot-link-button">Visit Official Website</a>
</div>
"""
            full_response_buffer += html_links
    
    # Stream the final answer and then send the updated state 
    async for chunk in stream_and_process_final_answer(full_response_buffer, request):
        yield chunk
        await asyncio.sleep(0.01)

    # After streaming, update history, log, and send state back to client 
    history.append({"role": "assistant", "content": full_response_buffer})
    log_conversation(user_question=question, bot_answer=full_response_buffer)
    
    session_state["history"] = history
    yield f"||SESSION_STATE_UPDATE||{json.dumps(session_state)}"

@app.post("/api/ask")
async def ask_api_stream(req: ChatRequest, request: Request):
    return StreamingResponse(stream_generator(req.question, req.language, request, req.session_state), media_type="text/event-stream")

@app.on_event("startup")
async def startup_event():
    global embedding_model, classifier_model
    print(" Starting up and loading all models and data...")
    load_and_process_data("data/maruti.csv")

    try:
        model_package = joblib.load('critic_model.pkl')
        embedding_model = model_package['embedding_model']
        classifier_model = model_package['classifier']
        print("Critic model package loaded successfully.")
    except FileNotFoundError:
        print("WARN: critic_model.pkl not found. The critic will be disabled.")
    except Exception as e:
        print(f"ERROR loading critic model: {e}")