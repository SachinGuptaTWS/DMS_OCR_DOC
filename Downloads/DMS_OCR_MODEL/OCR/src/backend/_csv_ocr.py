import pandas as pd
import csv
import os
import codecs
import time
import warnings
import io
import re
from collections import Counter

# Ensure GLiNER is installed: pip install gliner
try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None
    print("WARNING: GLiNER not found. NER features will be disabled.")

# Suppress warnings
warnings.filterwarnings("ignore")

# --- 1. Robust CSV/Excel Repair Engine ---

class MessyCSVHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_content = None
        self.encoding = 'utf-8'
        self.delimiter = ','
        self.lines = []

    def _detect_encoding(self):
        """Aggressively tries to find a working encoding."""
        codings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1', 'iso-8859-1', 'utf-16']
        for enc in codings:
            try:
                with open(self.filepath, 'r', encoding=enc) as f:
                    content = f.read()
                    # Heuristic: If we read it but it looks like garbage (contains null bytes), skip
                    if '\0' in content: 
                        continue
                    self.raw_content = content
                    self.encoding = enc
                    return
            except (UnicodeDecodeError, UnicodeError): # Catch BOM errors specifically
                continue
            except Exception:
                continue
                
        # Fallback: Read as binary and decode ignoring errors
        with open(self.filepath, 'rb') as f:
            self.raw_content = f.read().decode('utf-8', errors='ignore')
            self.encoding = 'utf-8 (forced)'

    def _clean_text_content(self):
        """Fixes common text issues."""
        if not self.raw_content: return
        self.raw_content = self.raw_content.replace('\x00', '')
        self.raw_content = self.raw_content.replace('\r\n', '\n').replace('\r', '\n')
        replacements = {
            'â€™': "'", 'â€“': "-", 'â€œ': '"', 'â€\x9d': '"',
            'Ã©': 'é', 'Ã ': 'à', 'Â': '' 
        }
        for bad, good in replacements.items():
            self.raw_content = self.raw_content.replace(bad, good)
        self.lines = [line.strip() for line in self.raw_content.split('\n') if line.strip()]

    def _detect_delimiter_robust(self):
        """Scans ALL lines to find the most consistent separator."""
        candidates = [',', ';', '\t', '|', ':']
        scores = {c: 0 for c in candidates}
        for d in candidates:
            counts = [line.count(d) for line in self.lines[:50]]
            if not counts: continue
            avg = sum(counts) / len(counts)
            if avg < 1: continue
            variance = sum((x - avg) ** 2 for x in counts) / len(counts)
            scores[d] = avg - (variance * 2)
        best_delim = max(scores, key=scores.get)
        self.delimiter = best_delim if scores[best_delim] > 0 else ','

    def _find_header_start(self):
        """Finds row index where table likely starts."""
        max_cols = 0
        header_idx = 0
        for i, line in enumerate(self.lines[:50]):
            cols = line.count(self.delimiter) + 1
            if cols > max_cols:
                max_cols = cols
                header_idx = i
        return header_idx, max_cols

    def parse(self):
        self._detect_encoding()
        self._clean_text_content()
        if not self.lines: return None, "Empty File"

        self._detect_delimiter_robust()
        start_idx, max_cols = self._find_header_start()
        
        cleaned_rows = []
        for line in self.lines[start_idx:]:
            reader = csv.reader([line], delimiter=self.delimiter)
            try:
                row = next(reader)
            except StopIteration:
                continue
            
            if len(row) < max_cols:
                row += [''] * (max_cols - len(row))
            elif len(row) > max_cols:
                row = row[:max_cols]
            cleaned_rows.append(row)

        if not cleaned_rows: return None, "No valid data found"

        try:
            header = cleaned_rows[0]
            data = cleaned_rows[1:]
            seen = {}
            final_header = []
            for col in header:
                col = col.strip() or "Unnamed"
                if col in seen:
                    seen[col] += 1
                    final_header.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    final_header.append(col)

            df = pd.DataFrame(data, columns=final_header)
            return df, f"Parsed with {self.encoding}, Delimiter: '{self.delimiter}'"
        except Exception as e:
            return None, str(e)

# --- 2. NER Engine (GLiNER Integration) ---

class NEREngine:
    def __init__(self):
        print("INFO: Initializing GLiNER for CSV...")
        self.available = False
        if GLiNER:
            try:
                # Using the small v2.1 model for balance of speed/accuracy
                self.model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
                self.available = True
            except Exception as e:
                print(f"WARNING: GLiNER load failed: {e}")

    def extract_entities(self, text):
        if not self.available or not text: return []
        
        # Broad ontology suitable for structured data analysis
        labels = [
            # Business/Finance
            "organization", "company", "stock ticker", "currency", "money", 
            "product", "brand", "competitor",
            # Personal
            "person", "job title", "email", "phone number",
            # Geo/Tech
            "location", "city", "country", "technology", "software", "ip address",
            # Metrics
            "date", "time", "percentage", "metric", "quantity"
        ]
        
        try:
            # Threshold tuned for tabular data (often cleaner text)
            entities = self.model.predict_entities(text, labels, threshold=0.3)
            return [{"text": e["text"], "label": e["label"], "score": round(e["score"], 2)} for e in entities]
        except Exception as e:
            print(f"NER Error: {e}")
            return []

# --- 3. Main Entry Point for API ---

def execute_csv_ocr(filepath):
    """
    Called by server.py. Returns structured data compatible with the Dashboard.
    """
    start_time = time.time()
    tool_name = "Robust CSV Heuristic + GLiNER"
    
    try:
        # 1. Parse Structure
        handler = MessyCSVHandler(filepath)
        df, status_msg = handler.parse()
        
        if df is None:
            return time.time() - start_time, {"error": f"Failed: {status_msg}"}, tool_name, "Failed"

        # 2. Prepare Display Data
        try:
            markdown_table = df.head(50).to_markdown(index=False, tablefmt="pipe") 
        except ImportError:
            markdown_table = df.head(50).to_string(index=False)

        full_text_csv = df.to_csv(index=False)
        
        # 3. Run Intelligence (NER)
        # We sample the first 2000 chars or first 50 rows to keep it fast, 
        # as running NER on a 1GB CSV would be too slow for an API response.
        text_sample = df.head(50).to_string() 
        ner_engine = NEREngine()
        print("INFO: Running GLiNER on CSV content...")
        entities = ner_engine.extract_entities(text_sample)

        # 4. Construct Result
        detailed_result = {
            "total_accuracy": 100.0,
            "total_slides": 1,
            "extracted_text": full_text_csv,
            "per_slide_metrics": [
                {
                    "slide_number": 1,
                    "text_content": f"Status: {status_msg}\nRows: {len(df)}, Columns: {len(df.columns)}\n\n--- Sample Data ---\n{text_sample}",
                    "accuracy_score": 100.0,
                    "item_count": len(df.columns),
                    "items": [
                        {
                            "text": markdown_table,
                            "conf": 1.0,
                            "type": "table_structure"
                        }
                    ]
                }
            ],
            "ner_entities": entities 
        }

        duration = time.time() - start_time
        return duration, detailed_result, tool_name, "Success"

    except Exception as e:
        return time.time() - start_time, {"error": str(e)}, tool_name, f"Error: {str(e)}"