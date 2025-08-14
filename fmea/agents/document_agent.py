# agents/document_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
import logging
from io import BytesIO
import base64

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# Image processing  
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Excel processing
try:
    import openpyxl
    HAS_EXCEL = True
except ImportError:
    HAS_EXCEL = False

class DocumentAgent:
    """
    Handles document processing and data extraction
    Implements the DocumentAgent from the class diagram
    """
    
    def __init__(self):
        self.pdf_parser = PDFParser() if HAS_PDF else None
        self.excel_parser = ExcelParser() if HAS_EXCEL else None
        self.logger = logging.getLogger(__name__)
        
        # Initialize regex patterns for different types of data
        self.setup_extraction_patterns()
    
    def setup_extraction_patterns(self):
        """Setup regex patterns for extracting different types of information"""
        self.patterns = {
            'failure_modes': [
                r'(?:failure|fail|malfunction|breakdown|error)(?:\s+(?:mode|type|condition))?[:\s]*([^.\n]+)',
                r'(?:when|if)\s+([^,.\n]+)\s+(?:fails|breaks|malfunctions)',
                r'(?:risk|hazard|problem)[:\s]*([^.\n]+)',
            ],
            'components': [
                r'(?:component|part|system|assembly|module)[:\s]*([^.\n]+)',
                r'(?:motor|bearing|switch|valve|pump|sensor|controller)(?:\s+\w+)*',
                r'(?:X|Y|Z)[\-\s]*axis(?:\s+\w+)*',
            ],
            'technical_specs': [
                r'(?:specification|spec|requirement)[:\s]*([^.\n]+)',
                r'(?:tolerance|precision|accuracy)[:\s]*([^.\n]+)',
                r'(\d+(?:\.\d+)?)\s*(?:mm|inch|°|rpm|psi|v|amp|hz)',
            ],
            'maintenance': [
                r'(?:maintenance|service|lubrication|cleaning)[:\s]*([^.\n]+)',
                r'(?:every|after)\s+(\d+(?:\.\d+)?)\s*(?:hours|days|months|cycles)',
            ]
        }
    
    def extract_text(self, file) -> str:
        """Extract text from various file types"""
        try:
            if file.name.endswith('.pdf'):
                return self.extract_text_from_pdf(file)
            elif file.name.endswith('.txt'):
                return str(file.read(), "utf-8")
            elif file.name.endswith(('.png', '.jpg', '.jpeg')):
                return self.extract_text_from_image(file)
            else:
                self.logger.warning(f"Unsupported file type for text extraction: {file.name}")
                return ""
        except Exception as e:
            self.logger.error(f"Text extraction failed for {file.name}: {str(e)}")
            return ""
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF files"""
        if not HAS_PDF:
            self.logger.error("PDF processing libraries not available")
            return ""
        
        try:
            text = ""
            file.seek(0)
            
            # Try with pdfplumber first (better for complex layouts)
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            except Exception:
                # Fallback to PyPDF2
                file.seek(0)
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            return text.strip()
            
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {str(e)}")
            return ""
    
    def extract_text_from_image(self, file) -> str:
        """Extract text from images using OCR"""
        if not HAS_OCR:
            self.logger.error("OCR libraries not available")
            return ""
        
        try:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {str(e)}")
            return ""
    
    def extract_tables_from_pdf(self, file) -> List[pd.DataFrame]:
        """Extract tables from PDF files"""
        if not HAS_PDF:
            return []
        
        tables = []
        try:
            file.seek(0)
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:  # Ensure table has header and data
                            df = pd.DataFrame(table[1:], columns=table[0])
                            # Clean the dataframe
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            if not df.empty:
                                tables.append(df)
        except Exception as e:
            self.logger.error(f"Table extraction failed: {str(e)}")
        
        return tables
    
    def read_excel(self, file) -> pd.DataFrame:
        """Read Excel/CSV files"""
        try:
            if file.name.endswith('.csv'):
                return pd.read_csv(file)
            elif file.name.endswith(('.xlsx', '.xls')):
                if HAS_EXCEL:
                    return pd.read_excel(file)
                else:
                    self.logger.error("Excel processing libraries not available")
                    return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Excel/CSV reading failed: {str(e)}")
            return pd.DataFrame()
    
    def extract_failure_modes(self, text: str) -> List[Dict[str, str]]:
        """Extract potential failure modes from text"""
        failure_modes = []
        
        for pattern in self.patterns['failure_modes']:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                failure_mode = match.group(1).strip()
                if len(failure_mode) > 5:  # Filter out very short matches
                    failure_modes.append({
                        'failure_mode': failure_mode,
                        'source': 'document_extraction',
                        'confidence': 0.7
                    })
        
        # Remove duplicates
        seen = set()
        unique_failure_modes = []
        for fm in failure_modes:
            key = fm['failure_mode'].lower()
            if key not in seen:
                seen.add(key)
                unique_failure_modes.append(fm)
        
        return unique_failure_modes
    
    def extract_technical_specs(self, text: str) -> Dict[str, Any]:
        """Extract technical specifications from text"""
        specs = {
            'components': [],
            'specifications': [],
            'maintenance_info': [],
            'measurements': []
        }
        
        # Extract components
        for pattern in self.patterns['components']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                component = match.group(0).strip()
                if component not in specs['components']:
                    specs['components'].append(component)
        
        # Extract specifications
        for pattern in self.patterns['technical_specs']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                spec = match.group(0).strip()
                specs['specifications'].append(spec)
        
        # Extract maintenance information
        for pattern in self.patterns['maintenance']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                maintenance = match.group(0).strip()
                specs['maintenance_info'].append(maintenance)
        
        return specs
    
    def extract_data(self, file_data: Dict) -> Dict[str, Any]:
        """Main extraction method called by coordinator"""
        return {
            'text': self.extract_text(file_data['file']),
            'tables': self.extract_tables_from_pdf(file_data['file']) if file_data['file'].name.endswith('.pdf') else [],
            'type': file_data.get('type', 'unknown')
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            'name': 'DocumentAgent',
            'pdf_support': HAS_PDF,
            'ocr_support': HAS_OCR,
            'excel_support': HAS_EXCEL,
            'active': True
        }


class PDFParser:
    """Specialized PDF parsing functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_service_bulletins(self, text: str) -> List[Dict[str, Any]]:
        """Extract service bulletin information from TORMACH documents"""
        bulletins = []
        
        # Pattern for service bulletin headers
        sb_pattern = r'(?:Service\s+Bulletin|SB)[\s#]*(\d+)[:\s]*([^\n]+)'
        matches = re.finditer(sb_pattern, text, re.IGNORECASE | re.MULTILINE)
        
        for match in matches:
            bulletin_id = match.group(1)
            title = match.group(2).strip()
            
            bulletins.append({
                'id': bulletin_id,
                'title': title,
                'type': 'service_bulletin'
            })
        
        return bulletins
    
    def extract_part_numbers(self, text: str) -> List[str]:
        """Extract part numbers from technical documents"""
        # Common part number patterns
        patterns = [
            r'(?:PN|Part\s+Number|P/N)[:\s]*([A-Z0-9\-]+)',
            r'\b(\d{5,6})\b',  # 5-6 digit numbers (common in TORMACH docs)
            r'(?:Tormach\s+PN|PN)\s*([A-Z0-9\-]+)',
        ]
        
        part_numbers = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                part_num = match.group(1).strip()
                if part_num not in part_numbers:
                    part_numbers.append(part_num)
        
        return part_numbers
    
    def extract_procedures(self, text: str) -> List[Dict[str, Any]]:
        """Extract step-by-step procedures from documents"""
        procedures = []
        
        # Look for numbered steps
        step_pattern = r'(\d+)\.\s*([^\n]+(?:\n(?!\d+\.)[^\n]*)*)'
        matches = re.finditer(step_pattern, text, re.MULTILINE)
        
        current_procedure = []
        for match in matches:
            step_num = int(match.group(1))
            step_text = match.group(2).strip()
            
            if step_num == 1 and current_procedure:
                # New procedure starting, save previous one
                procedures.append({
                    'steps': current_procedure,
                    'type': 'maintenance_procedure'
                })
                current_procedure = []
            
            current_procedure.append({
                'step': step_num,
                'description': step_text
            })
        
        # Add the last procedure
        if current_procedure:
            procedures.append({
                'steps': current_procedure,
                'type': 'maintenance_procedure'
            })
        
        return procedures


class ExcelParser:
    """Specialized Excel parsing functionality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_fmea_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse FMEA data from Excel files"""
        fmea_columns = {
            'component': ['component', 'system', 'item', 'part'],
            'failure_mode': ['failure mode', 'failure', 'mode', 'failure_mode'],
            'effect': ['effect', 'failure effect', 'consequence'],
            'cause': ['cause', 'failure cause', 'root cause', 'potential cause'],
            'severity': ['severity', 'sev', 's'],
            'occurrence': ['occurrence', 'occ', 'o', 'probability'],
            'detection': ['detection', 'det', 'd', 'detectability'],
            'rpn': ['rpn', 'risk priority number', 'risk']
        }
        
        # Map actual columns to standard FMEA columns
        column_mapping = {}
        df_columns_lower = [col.lower().strip() for col in df.columns]
        
        for standard_col, possible_names in fmea_columns.items():
            for possible_name in possible_names:
                for i, df_col in enumerate(df_columns_lower):
                    if possible_name in df_col or df_col in possible_name:
                        column_mapping[standard_col] = df.columns[i]
                        break
                if standard_col in column_mapping:
                    break
        
        # Create standardized FMEA dataframe
        standardized_df = pd.DataFrame()
        for standard_col, actual_col in column_mapping.items():
            standardized_df[standard_col] = df[actual_col]
        
        return {
            'dataframe': standardized_df,
            'column_mapping': column_mapping,
            'is_fmea': len(column_mapping) >= 4
        }
    
    def extract_numerical_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract numerical patterns from Excel data"""
        numerical_summary = {}
        
        for column in df.columns:
            if df[column].dtype in ['int64', 'float64']:
                numerical_summary[column] = {
                    'mean': df[column].mean(),
                    'std': df[column].std(),
                    'min': df[column].min(),
                    'max': df[column].max(),
                    'count': df[column].count()
                }
        
        return numerical_summary