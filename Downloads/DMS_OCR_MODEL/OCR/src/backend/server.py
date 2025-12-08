import os
import shutil
import time
import json
import traceback
import re
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from openai import AzureOpenAI
import httpx
from dateutil import parser

# --- CONFIGURATION ---
STORAGE_ROOT = "corporate_archive"
os.makedirs(STORAGE_ROOT, exist_ok=True)

# Folder creation configuration
BASE_STORAGE_PATH = "./DMS_Storage"
ALLOWED_CATEGORIES = {
    "finance", "legal", "hr", "compliance", "operations", 
    "procurement", "land / permit / noc", "general business docs"
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- AZURE GPT-4o SETUP ---
AZURE_ENDPOINT = "#"
AZURE_API_KEY = "#"
AZURE_API_VERSION = "#"
AZURE_DEPLOYMENT_NAME = "gpt-4o"

azure_client = None

if AZURE_API_KEY and AZURE_ENDPOINT:
    try:
        # Robust http client to avoid proxy errors
        http_client = httpx.Client()
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT, 
            api_key=AZURE_API_KEY,  
            api_version=AZURE_API_VERSION,
            http_client=http_client
        )
        print(f"[INFO] Azure GPT-4o: CONNECTED")
    except Exception as e:
        print(f"[ERROR] Azure Connection Failed: {e}")
else:
    print("[WARN] Azure GPT-4o: NOT CONNECTED")

print("\n[INFO] Initializing Backend Modules...")

# --- LOAD OCR MODULES ---
try: import _ppt_ocr
except: _ppt_ocr = None
try: import _csv_ocr
except: _csv_ocr = None
try: import _pdf_ocr 
except: _pdf_ocr = None
try: import _image_ocr 
except: _image_ocr = None
try: import _docx_ocr 
except: _docx_ocr = None

# --- LOAD TIKA ---
try:
    from tika import parser as tika_parser
    print("   [+] Apache Tika: LOADED")
    TIKA_AVAILABLE = True
except ImportError:
    print("   [-] Apache Tika: NOT FOUND")
    TIKA_AVAILABLE = False

try:
    import pandas as pd
    from bs4 import BeautifulSoup
except ImportError: 
    pass

print("[INFO] Backend Initialization Complete.\n")

# --- HELPERS ---

def sanitize_name(name):
    """Removes illegal characters from folder/file names."""
    if not name: return "Unknown"
    # Allow alphanumeric, spaces, hyphens, underscores
    clean = re.sub(r'[<>:"/\\|?*]', '', str(name)).strip()
    return clean if clean else "Unknown"

def run_tika_internal(filepath):
    if not TIKA_AVAILABLE: 
        return {"text": "", "metadata": {}, "status": "Tika Not Installed"}
    try:
        parsed = tika_parser.from_file(filepath)
        content = parsed.get("content", "") or ""
        return {"text": content.strip(), "metadata": parsed.get("metadata", {}), "status": "Success"}
    except Exception as e:
        return {"text": "", "metadata": {}, "status": f"Error: {e}"}

def sanitize_filename(name: str) -> str:
    """
    Prevents directory traversal and removes illegal characters.
    """
    if not name:
        return "Untitled"
    # Remove traversal attempts (../)
    name = name.replace("..", "")
    # Remove illegal OS characters (Windows/Linux safe)
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # Strip whitespace and replace spaces with underscores
    return name.strip().replace(" ", "_")

def normalize_date(date_str: Optional[str]):
    """
    Parses 'Jan 1st 2025', '2025-01-01', '01/01/25' etc.
    Returns (YYYY-MM-DD, Year_String)
    """
    if not date_str:
        now = datetime.now()
        return now.strftime("%Y-%m-%d"), str(now.year)
    
    try:
        # fuzzy=True allows it to ignore extra text like "Date: 2025-01-01"
        dt = parser.parse(date_str, fuzzy=True)
        return dt.strftime("%Y-%m-%d"), str(dt.year)
    except (ValueError, TypeError):
        # Fallback: If the date is complete gibberish, use today's year
        print(f"⚠️ Warning: Could not parse date '{date_str}'. Using current year.")
        now = datetime.now()
        return date_str, str(now.year)

def determine_category(raw_cat: Optional[str]) -> str:
    if not raw_cat:
        return "Others"
    
    clean_cat = raw_cat.lower().strip()
    
    if clean_cat in ALLOWED_CATEGORIES:
        return clean_cat.title()
    
    for allowed in ALLOWED_CATEGORIES:
        if allowed in clean_cat:
            return allowed.title()
            
    return "Others"

def create_folder_sync(path: str):
    os.makedirs(path, exist_ok=True)

class DocumentInput(BaseModel):
    document_id: Optional[str] = Field(None, alias="Document ID")
    category: Optional[str] = Field(None, alias="Document Category")
    doc_type: Optional[str] = Field("General", alias="Document Type")
    author: Optional[str] = Field("Unknown_Org", alias="Author / Issuer")
    vendor: Optional[str] = Field(None, alias="Vendor / Client")
    date_str: Optional[str] = Field(None, alias="Date")
    title: Optional[str] = Field(None, alias="Title")
    classification: Optional[str] = Field(None, alias="Classification Category")
    file_type: str = Field(..., alias="file_type")

    class Config:
        populate_by_name = True

class ProcessedDocument(BaseModel):
    Organisation_name: str
    Document_category: str
    Document_type: str
    date: str
    file_type: str
    storage_path: str
    status: str

async def analyze_with_azure_gpt(text_content: str, original_path: str, filename: str, page_count: int = 1) -> Dict[str, Any]:
    """
    Unified extraction of:
    1. Internal Classification (for folder sorting)
    2. Universal Metadata (15 standard fields)
    3. Specific Entities (Money, Vendors, IDs)
    """
    default_res = {
        "category": "Unsorted", 
        "path": f"Unsorted/{filename}", 
        "entities": [],
        "metadata": {}
    }
    
    if not azure_client or len(text_content) < 10:
        return default_res

    try:
        print(f"   [...] Analyzing with Azure GPT-4o (Unified Extraction)...")
        
        prompt = f"""
        You are a Senior Corporate Archivist. Analyze the document text below and extract comprehensive metadata.

        --- SECTION 1: INTERNAL CLASSIFICATION (For Filing) ---
        Determine the filing location based on content:
        - Department: (Finance, Legal, HR, Technical, Operations, Marketing, Sales, Exec).
        - Doc Type: (Invoice, Contract, Report, Resume, Email, Meeting Minutes, Spec Sheet).
        - Fiscal Year: The primary year associated with the document content (YYYY).

        --- SECTION 2: UNIVERSAL METADATA (Compulsory) ---
        Extract these fields. If not explicitly found, infer from context or return "N/A".
        1. Document Type: (Specific type e.g., "MSA Contract")
        2. Document Category: (High level e.g., "Legal")
        3. Title: (Formal title or Subject line)
        4. Description / Summary: (1 sentence summary)
        5. Owner: (Person or Dept responsible)
        6. Author / Issuer: (Entity who created it)
        7. Department / Asset / Project: (Associated codes or names)
        8. Tags: (5 AI-suggested keywords)
        9. Created Date: (Date generated/signed. INCLUDE TIME if available e.g. YYYY-MM-DD HH:MM)
        10. Effective Date: (Start date. INCLUDE TIME if available)
        11. Expiry Date: (End date)
        12. Version: (e.g., v1.0, Final, Draft)
        13. Classification Category: DETERMINE SECURITY LEVEL & REASON.
            - Format: "LEVEL - REASON"
            - Levels:
              * "Public": Marketing, Press.
              * "Internal": Standard procedures, Memos.
              * "Confidential": Invoices, Contracts, PII.
              * "Restricted": Passwords, Trade Secrets.
            - Example Output: "Confidential - Contains Payroll Data"
        *(File Type and Page Count are handled systematically)*

        --- SECTION 3: SPECIFIC ENTITIES ---
        Extract granular details if present:
        - Vendor / Client Names
        - Total Amount (with currency symbol)
        - Tax Amount
        - Document IDs (Invoice #, PO #, Contract Ref)
        - Project Codes
        - Action Items / Next Steps

        --- OUTPUT JSON FORMAT ---
        Return ONLY valid JSON:
        {{
            "classification": {{
                "department": "...",
                "doc_type": "...",
                "year": "..."
            }},
            "universal": {{
                "document_type": "...",
                "document_category": "...",
                "title": "...",
                "summary": "...",
                "owner": "...",
                "author": "...",
                "dept_project": "...",
                "tags": ["..."],
                "created_date": "...",
                "effective_date": "...",
                "expiry_date": "...",
                "version": "...",
                "classification_category": "..."
            }},
            "specifics": {{
                "vendor_client": ["..."],
                "amounts": {{ "total": "...", "tax": "..." }},
                "doc_ids": ["..."],
                "project_codes": ["..."],
                "action_items": ["..."]
            }}
        }}

        Document Text (Excerpt):
        {text_content[:30000]} 
        """

        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "You are a JSON-only extraction engine."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Parse AI Response
        ai_data = json.loads(response.choices[0].message.content)
        
        cls = ai_data.get("classification", {})
        uni = ai_data.get("universal", {})
        spec = ai_data.get("specifics", {})
        
        # --- 1. INTELLIGENT FILING ---
        dept = sanitize_name(cls.get("department", "Unsorted"))
        year = sanitize_name(cls.get("year", "General"))
        dtype = sanitize_name(cls.get("doc_type", "Misc"))

        # Create Directory: Root / Dept / Year / Type
        target_dir = os.path.join(STORAGE_ROOT, dept, year, dtype)
        os.makedirs(target_dir, exist_ok=True)
        final_path = os.path.join(target_dir, filename)
        
        # Physical Copy
        shutil.copy2(original_path, final_path)
        
        # --- 2. FORMAT ENTITIES FOR FRONTEND ---
        formatted_entities = []
        
        def add(label, value):
            if isinstance(value, list):
                val_str = ", ".join([str(v) for v in value if v])
                if val_str: formatted_entities.append({"label": label, "text": val_str})
            elif isinstance(value, dict):
                # Handle nested dicts like amounts
                for k, v in value.items():
                    if v and str(v).lower() != "n/a": 
                        formatted_entities.append({"label": f"{label} ({k.upper()})", "text": str(v)})
            elif isinstance(value, str) and value and value.lower() not in ["null", "none", "", "n/a"]:
                formatted_entities.append({"label": label, "text": value})

        # --- UNIVERSAL SET ---
        add("Document Type", uni.get("document_type"))
        add("Document Category", uni.get("document_category"))
        add("Title", uni.get("title"))
        add("Description / Summary", uni.get("summary"))
        add("Owner", uni.get("owner"))
        add("Author / Issuer", uni.get("author"))
        add("Department / Asset / Project", uni.get("dept_project"))
        add("Tags", uni.get("tags"))
        
        # --- DATE LOGIC ---
        c_date = uni.get("created_date", "N/A")
        e_date = uni.get("effective_date", "N/A")
        
        # Helper to normalize for comparison
        def norm(s): return str(s).strip().lower().replace(" ", "").replace("-", "").replace("/", "") if s else ""
        
        # If both exist and are effectively the same, collapse them
        if norm(c_date) not in ["na", ""] and norm(c_date) == norm(e_date):
            add("Date", c_date)
        else:
            add("Created Date", c_date)
            add("Effective Date", e_date)

        add("Expiry Date", uni.get("expiry_date"))
        add("Version", uni.get("version"))
        add("Classification Category", uni.get("classification_category"))
        
        # Deterministic Fields
        file_ext = os.path.splitext(filename)[1].upper().replace(".", "")
        add("File Type", file_ext)
        add("Page Count", str(page_count))

        # --- SPECIFIC ENTITIES ---
        add("Vendor / Client", spec.get("vendor_client"))
        add("Document ID", spec.get("doc_ids"))
        add("Project Code", spec.get("project_codes"))
        add("Amount", spec.get("amounts")) # Handles total/tax nesting
        add("Action Items", spec.get("action_items"))

        # --- INTERNAL CLASSIFICATION META ---
        add("Filing Department", dept)
        add("Filing Year", year)

        return {
            "category": f"{dept} > {dtype}", 
            "path": f"{STORAGE_ROOT}/{dept}/{year}/{dtype}/{filename}",
            "entities": formatted_entities
        }

    except Exception as e:
        print(f"   [ERROR] Azure Analysis Failed: {e}")
        traceback.print_exc()
        return default_res

# --- WRAPPERS ---
def wrap_simple_result(content):
    return {
        "total_accuracy": 100.0, "extracted_text": content, 
        "per_slide_metrics": [], "ner_entities": []
    }

# --- SIMPLE FORMAT HANDLERS ---
def _txt_OCR(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f: c = f.read()
        return 0.1, wrap_simple_result(c), "Native IO", "Success"
    except Exception as e: return 0, {}, "Error", str(e)

def _xml_ocr(filepath):
    if not BeautifulSoup: return 0, {}, "Missing BS4", "Error"
    try:
        with open(filepath, 'r', encoding='utf-8') as f: 
            return 0.2, wrap_simple_result(BeautifulSoup(f, 'xml').get_text(separator=' ')), "BS4 XML", "Success"
    except Exception as e: return 0, {}, "Error", str(e)

def _html_ocr(filepath):
    if not BeautifulSoup: return 0, {}, "Missing BS4", "Error"
    try:
        with open(filepath, 'r', encoding='utf-8') as f: 
            return 0.2, wrap_simple_result(BeautifulSoup(f, 'html.parser').get_text(separator=' ')), "BS4 HTML", "Success"
    except Exception as e: return 0, {}, "Error", str(e)

# --- MAIN ENDPOINT ---
@app.post("/api/benchmark")
async def run_benchmark(files: List[UploadFile] = File(...), config: str = Form(...)):
    results = []

    for file in files:
        file_location = os.path.join(TEMP_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        filename = file.filename.lower()
        ext = os.path.splitext(filename)[1]
        
        print(f"\n" + "="*50)
        print(f"DEBUG: Processing file: {file.filename}")
        print(f"DEBUG: File saved to {file_location}")
        print(f"DEBUG: File extension: {ext}")
        
        # 1. Raw Text Extraction (Tika)
        print("DEBUG: Starting Tika text extraction...")
        tika_res = run_tika_internal(file_location)
        print(f"DEBUG: Tika extraction complete. Text length: {len(tika_res.get('text', ''))} chars")
        
        # 2. Specialized OCR
        duration = 0
        detailed_data = {}
        script_used = "Tika Only"
        status = "Success" # Default

        if ext == '.pdf' and _pdf_ocr:
            duration, detailed_data, script_used, status = _pdf_ocr.execute_pdf_ocr(file_location)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff'] and _image_ocr:
            duration, detailed_data, script_used, status = _image_ocr.execute_image_ocr(file_location)
        elif ext == '.pptx' and _ppt_ocr:
            duration, detailed_data, script_used, status = _ppt_ocr.execute_ppt_ocr(file_location)
        elif ext == '.docx' and _docx_ocr:
            duration, detailed_data, script_used, status = _docx_ocr.execute_docx_ocr(file_location)
        elif ext == '.csv' and _csv_ocr:
            duration, detailed_data, script_used, status = _csv_ocr.execute_csv_ocr(file_location)
        elif ext == '.txt':
            duration, detailed_data, script_used, status = _txt_OCR(file_location)
        elif ext == '.xml':
            duration, detailed_data, script_used, status = _xml_ocr(file_location)
        elif ext == '.html':
            duration, detailed_data, script_used, status = _html_ocr(file_location)
        else:
            detailed_data = wrap_simple_result(tika_res['text'])
            status = "Fallback to Tika"

        # 3. Consolidate Text & Metrics
        combined_text = detailed_data.get("extracted_text", "") or tika_res['text']
        
        # Extract Page Count from specialized engines (PDF/PPT/Images)
        page_count = detailed_data.get("total_slides", 1) 

        # 4. Azure GPT-4o Analysis (Unified Extraction)
        print("\nDEBUG: Starting Azure GPT-4o analysis...")
        analysis = await analyze_with_azure_gpt(combined_text, file_location, file.filename, page_count)
        
        print(f"DEBUG: Azure analysis complete. Smart Filed path: {analysis.get('path', 'N/A')}")
        
        # Debug folder creation process
        print("\nDEBUG: Starting folder creation process...")
        try:
            # Create document input for folder creation
            doc_input = DocumentInput(
                document_id=analysis.get("metadata", {}).get("document_id", ""),
                category=analysis.get("category", ""),
                doc_type=analysis.get("metadata", {}).get("document_type", "General"),
                author=analysis.get("metadata", {}).get("author", ""),
                vendor=analysis.get("metadata", {}).get("vendor_client", [""])[0],
                date_str=analysis.get("metadata", {}).get("created_date", ""),
                title=analysis.get("metadata", {}).get("title", ""),
                classification=analysis.get("metadata", {}).get("classification_category", ""),
                file_type=os.path.splitext(file.filename)[1][1:].upper()
            )
            print(f"DEBUG: Created DocumentInput: {doc_input.dict()}")
            
            # Try local folder creation first
            folder_result = await try_local_folder_creation(doc_input)
            
            # If local fails, try Azure fallback
            if not folder_result.get('success'):
                print("DEBUG: Local folder creation failed, trying Azure fallback...")
                folder_result = await try_azure_folder_creation(analysis, file.filename)
                
                # If Azure fails, use fallback to Unsorted
                if not folder_result.get('success'):
                    fallback_path = os.path.join(BASE_STORAGE_PATH, "Unsorted")
                    os.makedirs(fallback_path, exist_ok=True)
                    print(f"DEBUG: Using fallback folder: {fallback_path}")
                    folder_result = {
                        "success": True,
                        "method": "fallback",
                        "result": {"storage_path": fallback_path},
                        "error": None
                    }
            
            print(f"DEBUG: Final folder path: {folder_result.get('result', {}).get('storage_path', 'N/A')}")
            
        except Exception as e:
            print(f"ERROR in folder creation: {str(e)}")
            traceback.print_exc()
        
        # Merge basic NER (if any) with Advanced GPT Entities
        final_entities = detailed_data.get("ner_entities", []) + analysis.get("entities", [])

        results.append({
            "fileName": file.filename,
            "engine": f"{script_used} + Azure GPT",
            "time_seconds": duration,
            "status": status,
            "extracted_length": len(combined_text),
            "accuracy": detailed_data.get("total_accuracy", 100),
            "text_preview": combined_text,
            "slides_data": detailed_data.get("per_slide_metrics", []),
            "ner_entities": final_entities, # Populates the "Smart Entities" tab
            "doc_category": analysis["category"],
            "storage_path": analysis["path"]
        })
        
        # Cleanup
        if os.path.exists(file_location):
            try: os.remove(file_location)
            except: pass

    return {"results": results}

@app.post("/organize", response_model=List[ProcessedDocument])
async def organize_documents(docs: List[DocumentInput]):
    results = []

    for doc in docs:
        try:
            # Normalize Data
            org_name = doc.author if doc.author else (doc.vendor or "Unknown_Org")
            final_category = determine_category(doc.category)
            
            # Smart date parser
            final_date, year = normalize_date(doc.date_str)
            
            # Sanitize inputs for safety
            safe_doc_type = sanitize_filename(doc.doc_type)

            # Construct Path
            relative_path = os.path.join(
                final_category,
                year,
                safe_doc_type
            )
            full_path = os.path.join(BASE_STORAGE_PATH, relative_path)

            # Create Folders
            await run_in_threadpool(create_folder_sync, full_path)

            output_obj = ProcessedDocument(
                Organisation_name=org_name,
                Document_category=final_category,
                Document_type=safe_doc_type,
                date=final_date,
                file_type=doc.file_type,
                storage_path=full_path,
                status="Folder Created"
            )
            results.append(output_obj)
        except Exception as e:
            print(f"Error processing document {doc.document_id}: {str(e)}")
            continue

    return results

async def try_local_folder_creation(doc_input: DocumentInput) -> dict:
    """Attempt to create folder structure using local model first"""
    try:
        print("\n=== Starting Local Model Folder Creation ===")
        print(f"Attempting to create folder for document: {doc_input.title or 'Untitled'}")
        print(f"Category: {doc_input.category}, Type: {doc_input.doc_type}")
        
        folder_result = await organize_documents([doc_input])
        
        if folder_result and folder_result[0].status == "Folder Created":
            print(" Local model successfully created folder structure")
            print(f"  Path: {folder_result[0].storage_path}")
            return {
                "success": True,
                "method": "local",
                "result": folder_result[0].dict(),
                "error": None
            }
            
    except Exception as e:
        error_msg = f"Local folder creation failed: {str(e)}"
        print(f" {error_msg}")
        return {
            "success": False,
            "method": "local",
            "result": None,
            "error": error_msg
        }
        
    error_msg = "Local folder creation returned no result"
    print(f" {error_msg}")
    return {
        "success": False,
        "method": "local",
        "result": None,
        "error": error_msg
    }

async def try_azure_folder_creation(analysis_result: dict, filename: str) -> dict:
    """Fallback to Azure OpenAI for folder creation if local method fails"""
    try:
        print("\n=== Starting Azure OpenAI Folder Creation ===")
        print("Local model failed, falling back to Azure OpenAI...")
        
        # Prepare document for folder creation using Azure's analysis
        doc_input = DocumentInput(
            document_id=analysis_result.get("metadata", {}).get("document_id", ""),
            category=analysis_result.get("category", ""),
            doc_type=analysis_result.get("metadata", {}).get("document_type", "General"),
            author=analysis_result.get("metadata", {}).get("author", ""),
            vendor=analysis_result.get("metadata", {}).get("vendor_client", [""])[0],
            date_str=analysis_result.get("metadata", {}).get("created_date", ""),
            title=analysis_result.get("metadata", {}).get("title", ""),
            classification=analysis_result.get("metadata", {}).get("classification_category", ""),
            file_type=os.path.splitext(filename)[1][1:].upper()
        )
        
        print(f"Azure OpenAI processing document: {doc_input.title or 'Untitled'}")
        print(f"Extracted category: {doc_input.category}, Type: {doc_input.doc_type}")
        
        folder_result = await organize_documents([doc_input])
        
        if folder_result and folder_result[0].status == "Folder Created":
            print(" Azure OpenAI successfully created folder structure")
            print(f"  Path: {folder_result[0].storage_path}")
            return {
                "success": True,
                "method": "azure",
                "result": folder_result[0].dict(),
                "error": None
            }
            
    except Exception as e:
        error_msg = f"Azure folder creation failed: {str(e)}"
        print(f" {error_msg}")
        return {
            "success": False,
            "method": "azure",
            "result": None,
            "error": error_msg
        }
        
    error_msg = "Azure folder creation returned no result"
    print(f" {error_msg}")
    return {
        "success": False,
        "method": "azure",
        "result": None,
        "error": error_msg
    }

@app.post("/process_document")
async def process_document(file: UploadFile = File(...)):
    temp_path = ""
    try:
        # Save the uploaded file temporarily
        temp_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Step 1: Extract text content
        text_content = ""
        if file.filename.lower().endswith(('.txt', '.md', '.log')):
            with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read()
        elif TIKA_AVAILABLE:
            tika_result = run_tika_internal(temp_path)
            text_content = tika_result.get('text', '')
        
        # Step 2: Extract metadata and entities using Azure GPT
        print("Extracting metadata and entities...")
        analysis_result = await analyze_with_azure_gpt(
            text_content=text_content,
            original_path=temp_path,
            filename=file.filename,
            page_count=1
        )
        
        # Prepare document for folder creation with extracted entities
        doc_input = DocumentInput(
            document_id=analysis_result.get("metadata", {}).get("document_id", ""),
            category=analysis_result.get("category", ""),
            doc_type=analysis_result.get("metadata", {}).get("document_type", "General"),
            author=analysis_result.get("metadata", {}).get("author", ""),
            vendor=analysis_result.get("metadata", {}).get("vendor_client", [""])[0],
            date_str=analysis_result.get("metadata", {}).get("created_date", ""),
            title=analysis_result.get("metadata", {}).get("title", ""),
            classification=analysis_result.get("metadata", {}).get("classification_category", ""),
            file_type=os.path.splitext(file.filename)[1][1:].upper()
        )
        
        # Step 3: First attempt - Local model folder creation using extracted entities
        print("Attempting folder creation with local model...")
        folder_result = await try_local_folder_creation(doc_input)
        
        # Step 4: If local model fails, fall back to Azure OpenAI for folder creation
        if not folder_result.get('success'):
            print(f"Local folder creation failed: {folder_result.get('error')}")
            print("Falling back to Azure OpenAI for folder creation...")
            
            # Use Azure's analysis to generate folder structure
            azure_folder_result = await try_azure_folder_creation(analysis_result, file.filename)
            
            if azure_folder_result.get('success'):
                print("Azure folder creation successful")
                folder_result = azure_folder_result
            else:
                print(f"Azure folder creation failed: {azure_folder_result.get('error')}")
                
                # Final fallback: Create a folder in "Unsorted"
                print("\n Both local and Azure folder creation failed. Creating fallback location...")
                fallback_path = os.path.join(BASE_STORAGE_PATH, "Unsorted")
                os.makedirs(fallback_path, exist_ok=True)
                print(f" Created fallback folder: {fallback_path}")
                folder_result = {
                    "success": True,
                    "method": "fallback",
                    "result": {
                        "storage_path": fallback_path,
                        "status": "Created fallback folder in 'Unsorted'"
                    },
                    "error": None
                }
        
        # Move the file to the created folder
        if folder_result.get('success'):
            target_path = os.path.join(
                folder_result['result']['storage_path'], 
                os.path.basename(temp_path)
            )
            shutil.move(temp_path, target_path)
            folder_result['result']['file_location'] = target_path
        
        return {
            "analysis": analysis_result,
            "folder_creation": folder_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
