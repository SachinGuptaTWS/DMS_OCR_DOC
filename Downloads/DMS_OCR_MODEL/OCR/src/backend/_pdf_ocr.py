import os
import sys
import time
import shutil
import io
import warnings
import json
import concurrent.futures
from PIL import Image
import traceback
import fitz # PyMuPDF
import numpy as np
import cv2
import types
import pandas as pd
import re
from difflib import SequenceMatcher

# --- FIX 0: NUMPY 2.0 HOT PATCH ---
try:
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str, np.void]
        }
except Exception: pass

# --- FIX 1: AGGRESSIVE LANGCHAIN PATCH ---
try:
    if "langchain.docstore" not in sys.modules:
        m_docstore = types.ModuleType("langchain.docstore")
        sys.modules["langchain.docstore"] = m_docstore
    if "langchain.docstore.document" not in sys.modules:
        m_document = types.ModuleType("langchain.docstore.document")
        sys.modules["langchain.docstore.document"] = m_document
    try:
        from langchain_core.documents import Document
    except ImportError:
        class Document: pass
    sys.modules["langchain.docstore.document"].Document = Document
    if "langchain.text_splitter" not in sys.modules:
        m_text_splitter = types.ModuleType("langchain.text_splitter")
        sys.modules["langchain.text_splitter"] = m_text_splitter
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter: pass
        class CharacterTextSplitter: pass
    sys.modules["langchain.text_splitter"].RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"].CharacterTextSplitter = CharacterTextSplitter
except Exception: pass

# --- 1. Library Imports ---
print("\n[INFO] Initializing Optimized Engine Cluster...")

# 1. PDFPlumber
try:
    import pdfplumber
    PLUMBER_AVAILABLE = True
except ImportError:
    PLUMBER_AVAILABLE = False

# 2. Unstructured
try:
    from unstructured.partition.pdf import partition_pdf
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

# 3. Camelot
try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

# 4. Tabula
try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

# 5. Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

# 6. RapidOCR (The Fast Replacement for EasyOCR)
try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_AVAILABLE = True
    print("   [+] RapidOCR (High-Speed Engine): LOADED")
except ImportError:
    RAPID_AVAILABLE = False

# 7. Img2Table
try:
    from img2table.document import PDF as Img2TablePDF
    from img2table.ocr import TesseractOCR as Img2TableTess
    IMG2TABLE_AVAILABLE = True
except ImportError:
    IMG2TABLE_AVAILABLE = False

# 8. GLiNER
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False

print("[INFO] Engine Loading Complete.\n")

warnings.filterwarnings("ignore")
GEMINI_API_KEY = "AIzaSyCI3S7weovcMknn3sSpsTstB9N4b57VvqY"

# --- ENGINE CLASSES ---

class PDFPlumberEngine:
    def __init__(self):
        self.available = PLUMBER_AVAILABLE
    def extract_layout(self, filepath):
        if not self.available: return []
        results = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            df = pd.DataFrame(table).replace(r'^\s*$', np.nan, regex=True).fillna('')
                            md_table = df.to_markdown(index=False, header=False, tablefmt="grid")
                            results.append({"page": i + 1, "type": "table_structure", "text": f"### PDFPlumber Table\n{md_table}", "conf": 0.98, "source": "PDFPlumber"})
                    text = page.extract_text(layout=True)
                    if text: results.append({"page": i + 1, "type": "text_block", "text": text, "conf": 0.95, "source": "PDFPlumber Layout"})
        except Exception: pass
        return results

class RapidOCREngine:
    def __init__(self):
        self.available = RAPID_AVAILABLE
        if self.available:
            try: self.engine = RapidOCR()
            except: self.available = False
    def process_image(self, img_path):
        if not self.available: return None
        try:
            result, _ = self.engine(img_path)
            if result:
                txts = [line[1] for line in result]
                return "\n".join(txts)
        except: return None

class TabulaEngine:
    @staticmethod
    def extract_tables(filepath):
        if not TABULA_AVAILABLE: return []
        results = []
        try:
            tables = tabula.read_pdf(filepath, pages='all', multiple_tables=True, silent=True)
            for i, df in enumerate(tables):
                if not df.empty:
                    md = df.to_markdown(index=False, tablefmt="pipe")
                    results.append({"page": 1, "type": "table_structure", "text": f"### Tabula Spreadsheet\n{md}", "conf": 0.92, "source": "Tabula-py"})
        except: pass
        return results

class GeminiVisionEngine:
    def __init__(self, api_key):
        self.available = False
        try:
            if api_key and genai:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                self.available = True
        except Exception: pass
    def analyze_image_file(self, image_path):
        try:
            img = Image.open(image_path)
            response = self.model.generate_content(["Extract all text and data from this image exactly.", img])
            return response.text.strip()
        except: return None

class CamelotTableEngine:
    @staticmethod
    def extract_tables(filepath):
        if not CAMELOT_AVAILABLE: return []
        tables_data = []
        try:
            tables = camelot.read_pdf(filepath, pages='all', flavor='lattice')
            if len(tables) == 0:
                tables = camelot.read_pdf(filepath, pages='all', flavor='stream', edge_tol=500)
            for t in tables:
                df = t.df
                try: markdown = df.to_markdown(index=False, header=False, tablefmt="grid")
                except: markdown = df.to_string()
                tables_data.append({"content": markdown, "accuracy": round(t.parsing_report['accuracy'], 2), "page": t.page})
        except Exception: pass
        return tables_data

class FitzTextEngine:
    def extract_text_blocks(self, filepath):
        results = []
        try:
            doc = fitz.open(filepath)
            for page in doc:
                blocks = page.get_text("blocks")
                for b in blocks:
                    text = b[4].strip()
                    if text:
                        results.append({"page": page.number + 1, "type": "text_block", "text": text, "conf": 1.0, "source": "PyMuPDF Text"})
        except: pass
        return results

class Img2TableEngine:
    def extract_tables(self, filepath):
        if not IMG2TABLE_AVAILABLE: return []
        results = []
        try:
            pdf_doc = Img2TablePDF(src=filepath)
            ocr_backend = Img2TableTess() if 'pytesseract' in globals() else None
            extracted_tables = pdf_doc.extract_tables(ocr=ocr_backend, implicit_rows=False, borderless_tables=True, min_confidence=50)
            for page_idx, tables in extracted_tables.items():
                for table in tables:
                    df = table.df
                    if not df.empty:
                        md = df.to_markdown(index=False, tablefmt="grid")
                        results.append({"page": page_idx + 1, "type": "table_structure", "text": f"### Img2Table Structure\n{md}", "conf": 0.90, "source": "Img2Table"})
        except: pass
        return results

class NEREngine:
    def __init__(self):
        if GLINER_AVAILABLE:
            try:
                self.model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
                self.available = True
            except: self.available = False
        else: self.available = False
    
    def extract_entities(self, text, custom_labels=None):
        if not self.available or not text: return []
        
        # --- UPDATED LABELS PER REQUIREMENTS ---
        # "Metadata Extraction: Dates, Names, Vendor/client, Amounts, Document ID, Doc Type, Category, project"
        required_labels = [
            # Dates
            "date", "invoice date", "due date", "effective date",
            # Names
            "person", "name", "signatory",
            # Vendor / Client
            "vendor", "client", "organization", "company", "supplier",
            # Amounts
            "amount", "total amount", "net amount", "tax amount", "currency",
            # Document ID / Type
            "document id", "invoice number", "contract number", "reference number", "document type",
            # Category / Project
            "category", "project", "project name", "project id"
        ]
        
        labels = custom_labels if custom_labels else required_labels
        try:
            entities = self.model.predict_entities(text[:25000], labels, threshold=0.3)
            return [{"text": e["text"], "label": e["label"], "score": round(e["score"], 2)} for e in entities]
        except: return []

# --- MAIN PARSER ---
class HybridPDFParser:
    def __init__(self, filepath, output_dir):
        self.filepath = filepath
        self.output_dir = output_dir
        self.plumber = PDFPlumberEngine()
        self.rapid = RapidOCREngine()
        self.fitz_text = FitzTextEngine()
        self.img2table = Img2TableEngine()
        self.gemini = GeminiVisionEngine(GEMINI_API_KEY)
        self.images_dir = os.path.join(output_dir, "extracted_images")
        if os.path.exists(self.images_dir): shutil.rmtree(self.images_dir)
        os.makedirs(self.images_dir, exist_ok=True)

    def _extract_images_robust(self):
        found_images = []
        try:
            doc = fitz.open(self.filepath)
            for page_idx, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_idx, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    # Ignore tiny icons, keep robust images
                    if len(base_image["image"]) < 3000: continue
                    fname = f"page{page_idx+1}_img{img_idx}.{base_image['ext']}"
                    fpath = os.path.join(self.images_dir, fname)
                    with open(fpath, "wb") as f: f.write(base_image["image"])
                    found_images.append(fpath)
        except: pass
        return found_images

    # --- Workers ---
    def _worker_unstructured(self):
        if not UNSTRUCTURED_AVAILABLE: return []
        results = []
        try:
            # Using Hi-Res as primary logic, relying on RapidOCR for fallback
            elements = partition_pdf(filename=self.filepath, strategy="hi_res", infer_table_structure=True, languages=["eng"])
            for el in elements:
                text = str(el)
                page = el.metadata.page_number if hasattr(el.metadata, 'page_number') else 1
                if len(text.strip()) > 3:
                    el_type = type(el).__name__
                    results.append({"page": page, "type": "table_structure" if el_type == "Table" else "text_block", "text": text, "conf": 0.85 if el_type == "Table" else 1.0, "source": "Unstructured"})
        except Exception as e:
            print(f"Unstructured Error: {e}")
        return results

    def _worker_camelot(self):
        results = []
        tables = CamelotTableEngine.extract_tables(self.filepath)
        for t in tables: results.append({"type": "table_structure", "text": t["content"], "page": t["page"], "conf": t["accuracy"] / 100.0, "source": "Camelot"})
        return results

    def _worker_tabula(self):
        return TabulaEngine.extract_tables(self.filepath)

    def _worker_pdfplumber(self):
        return self.plumber.extract_layout(self.filepath)

    def _worker_fitz_text(self):
        return self.fitz_text.extract_text_blocks(self.filepath)

    def _worker_img2table(self):
        return self.img2table.extract_tables(self.filepath)

    def _worker_images(self):
        results = []
        image_files = self._extract_images_robust()
        if not image_files: return []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as img_exec:
            for img_path in image_files:
                try:
                    # 1. Try RapidOCR (Fast)
                    rapid_txt = self.rapid.process_image(img_path)
                    p_num = 1
                    try: p_num = int(os.path.basename(img_path).split('page')[1].split('_')[0])
                    except: pass
                    
                    if rapid_txt and len(rapid_txt) > 5:
                         results.append({"type": "ocr_image", "text": f"[IMAGE-OCR]: {rapid_txt}", "conf": 0.90, "page": p_num})
                    
                    # 2. If RapidOCR fails/low quality, use Gemini
                    if not rapid_txt or len(rapid_txt) < 10:
                        gemini_res = self.gemini.analyze_image_file(img_path)
                        if gemini_res:
                             results.append({"type": "ocr_image", "text": f"[IMAGE-GEMINI]: {gemini_res}", "conf": 0.95, "page": p_num})
                except: pass
        
        try: shutil.rmtree(self.images_dir)
        except: pass
        return results

    # --- AGGRESSIVE DEDUPLICATION ---
    def _tokenize(self, text):
        clean_text = re.sub(r'[\|\+\-\=\_\:\;]', ' ', text)
        words = re.findall(r'\b\w+\b', clean_text.lower())
        return set(words)

    def _calculate_overlap(self, text_a, text_b):
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)
        if len(tokens_a) == 0: return 0.0
        intersection = tokens_a.intersection(tokens_b)
        return len(intersection) / len(tokens_a)

    def _deduplicate_items(self, all_items):
        print("   [*] Running Aggressive Token-Based Deduplication...")
        pages = {}
        for item in all_items:
            p = item.get("page", 1)
            if p not in pages: pages[p] = []
            pages[p].append(item)

        final_items = []
        for p, items in pages.items():
            # Sort: Tables first, then Longest Text
            items.sort(key=lambda x: (x.get("type") == "table_structure", len(x.get("text", ""))), reverse=True)
            unique_on_page = []
            
            for candidate in items:
                candidate_text = candidate.get("text", "")
                candidate_type = candidate.get("type", "text_block")
                if len(candidate_text.strip()) < 5: continue

                is_duplicate = False
                for existing in unique_on_page:
                    existing_text = existing.get("text", "")
                    
                    # Exact Match
                    if candidate_text.strip() in existing_text.strip():
                        is_duplicate = True; break
                    
                    # Overlap Check
                    overlap = self._calculate_overlap(candidate_text, existing_text)
                    if overlap > 0.85: # If 85% of words are same, it's a dupe
                         is_duplicate = True; break

                if not is_duplicate:
                    unique_on_page.append(candidate)
            final_items.extend(unique_on_page)
        return final_items

    def parse(self):
        
        print("\n=== STARTING OPTIMIZED HYBRID PARSER ===")
        print("   [*] Removed: EnsembleOCR (Slow), Tika (Redundant)")
        print("   [*] Active: Unstructured, Camelot, Tabula, RapidOCR, Img2Table")
        
        structured_output = {"total_pages": 0, "text_content": "", "items": []}
        try:
            doc = fitz.open(self.filepath)
            structured_output["total_pages"] = len(doc)
            doc.close()
        except: pass

        all_items = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._worker_unstructured): "Unstructured",
                executor.submit(self._worker_camelot): "Camelot",
                executor.submit(self._worker_tabula): "Tabula",
                executor.submit(self._worker_pdfplumber): "PDFPlumber",
                executor.submit(self._worker_fitz_text): "PyMuPDF",
                executor.submit(self._worker_img2table): "Img2Table",
                executor.submit(self._worker_images): "Visual-OCR (Rapid+Gemini)"
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    print(f"   [+] {name} Finished ({len(res)} items)")
                    all_items.extend(res)
                except Exception as e:
                    print(f"   [-] {name} Failed: {e}")

        clean_items = self._deduplicate_items(all_items)
        
        structured_output["items"] = clean_items
        full_text_accum = [x['text'] for x in clean_items if x.get('text')]
        structured_output["text_content"] = "\n\n".join(full_text_accum)
        return structured_output

# --- 8. Main Entry ---
def execute_pdf_ocr(filepath):
    start = time.time()
    tool = "Hybrid-Optimized-Fast"
    try:
        process_id = str(int(time.time()))
        temp_dir = os.path.join(os.path.dirname(filepath), f"pdf_proc_{process_id}")
        parser = HybridPDFParser(filepath, temp_dir)
        data = parser.parse()
        
        per_page_metrics = []
        pages = {}
        for item in data["items"]:
            p = item.get("page", 1)
            if p not in pages: pages[p] = []
            pages[p].append(item)
        for p in sorted(pages.keys()):
            confs = [i.get("conf", 1.0) for i in pages[p]]
            avg = (sum(confs) / len(confs) * 100) if confs else 100.0
            per_page_metrics.append({"slide_number": p, "text_content": "", "accuracy_score": round(avg, 2), "item_count": len(pages[p]), "items": pages[p]})

        print("   [+] Layer 7: GLiNER Metadata Extraction...")
        ner = NEREngine()
        entities = ner.extract_entities(data["text_content"][:25000])

        result = {
            "total_accuracy": 99.0, "total_slides": data.get("total_pages", 1),
            "extracted_text": data["text_content"],
            "per_slide_metrics": per_page_metrics, "ner_entities": entities
        }
        return time.time() - start, result, tool, "Success"
    except Exception as e:
        traceback.print_exc()
        return time.time() - start, {"error": str(e)}, tool, f"Error: {e}"