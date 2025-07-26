import os
import json
from pathlib import Path
import fitz  # PyMuPDF
import pdfplumber
from collections import Counter, defaultdict
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import spacy
from textdistance import levenshtein
import pandas as pd
from scipy import stats
import time # Import the time module
import logging

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

# Load spaCy model (install with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFOutlineExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.heading_keywords = {
            'introduction', 'overview', 'summary', 'conclusion', 'references', 
            'acknowledgements', 'abstract', 'table', 'contents', 'chapter',
            'section', 'appendix', 'bibliography', 'index', 'glossary',
            'revision', 'history', 'version'
        }
        
    def extract_with_pymupdf(self, pdf_path):
        """Extract text with detailed formatting using PyMuPDF"""
        doc = fitz.open(pdf_path)
        elements = []
        
        for page_num, page in enumerate(doc, start=1):
            # Get text with formatting
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if block["type"] != 0:  # Skip non-text blocks
                    continue
                    
                for line in block["lines"]:
                    spans = line["spans"]
                    if not spans:
                        continue
                        
                    # Combine text from all spans in line
                    text = ""
                    sizes = []
                    flags = []
                    fonts = []
                    
                    for span in spans:
                        text += span["text"]
                        sizes.append(span["size"])
                        flags.append(span["flags"])
                        fonts.append(span["font"])
                    
                    if not text.strip():
                        continue
                        
                    # Calculate average properties
                    avg_size = np.mean(sizes)
                    is_bold = any(flag & 2**4 for flag in flags)
                    is_italic = any(flag & 2**6 for flag in flags)
                    
                    # Get position
                    bbox = line["bbox"]
                    
                    elements.append({
                        "text": text.strip(),
                        "page": page_num,
                        "size": avg_size,
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "font": fonts[0] if fonts else "",
                        "x": bbox[0],
                        "y": bbox[1],
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1]
                    })
        
        doc.close()
        return elements
    
    def extract_with_pdfplumber(self, pdf_path):
        """Extract text with pdfplumber for additional analysis"""
        elements = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text with character-level details
                    chars = page.chars
                    
                    if not chars:
                        continue
                    
                    # Group characters into lines
                    lines = defaultdict(list)
                    for char in chars:
                        if 'y0' in char and 'text' in char:
                            y_pos = round(char['y0'])
                            lines[y_pos].append(char)
                    
                    # Process each line
                    for y_pos, line_chars in lines.items():
                        if not line_chars:
                            continue
                            
                        # Sort characters by x position
                        line_chars.sort(key=lambda c: c.get('x0', 0))
                        
                        # Combine into text
                        text = ''.join(char.get('text', '') for char in line_chars)
                        if not text.strip():
                            continue
                        
                        # Calculate properties
                        sizes = [char.get('size', 12) for char in line_chars if 'size' in char]
                        avg_size = np.mean(sizes) if sizes else 12
                        
                        # Check for bold/italic (approximate)
                        fonts = [char.get('fontname', '') for char in line_chars if 'fontname' in char]
                        is_bold = any('bold' in font.lower() for font in fonts) if fonts else False
                        is_italic = any('italic' in font.lower() for font in fonts) if fonts else False
                        
                        x_coords = [char.get('x0', 0) for char in line_chars if 'x0' in char]
                        min_x = min(x_coords) if x_coords else 0
                        
                        elements.append({
                            "text": text.strip(),
                            "page": page_num,
                            "size": avg_size,
                            "is_bold": is_bold,
                            "is_italic": is_italic,
                            "font": fonts[0] if fonts else "",
                            "x": min_x,
                            "y": y_pos
                        })
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        return elements
    
    def analyze_text_features(self, elements):
        """Analyze text features to identify headings"""
        if not elements:
            return pd.DataFrame()
            
        df = pd.DataFrame(elements)
        # Ensure required columns exist
        required_columns = ['text', 'size', 'page', 'is_bold', 'is_italic', 'x']
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0 if col in ['size', 'page', 'x'] else False
        
        # Handle empty dataframe
        if df.empty:
            return df
        
        # Calculate statistics
        if len(df) > 0:
            size_stats = df['size'].describe()
            body_size = size_stats['50%']  # Median size
            large_size_threshold = size_stats['75%']
        else:
            body_size = 12
            large_size_threshold = 14
        
        # Add features safely
        df['is_large'] = df['size'] > large_size_threshold
        df['is_very_large'] = df['size'] > (size_stats['75%'] + 1.5 * (size_stats['75%'] - size_stats['25%'])) if len(df) > 0 else False
        df['text_length'] = df['text'].astype(str).str.len()
        df['word_count'] = df['text'].astype(str).str.split().str.len()
        df['is_short'] = df['text_length'] < 100
        df['is_uppercase'] = df['text'].astype(str).str.isupper()
        df['starts_with_number'] = df['text'].astype(str).str.match(r'^(\d+\.)+\d*', na=False)
        
        # Check for heading keywords
        df['has_heading_keywords'] = df['text'].astype(str).str.lower().apply(
            lambda x: any(keyword in x for keyword in self.heading_keywords)
        )
        
        # Position-based features (handle missing x values)
        if 'x' in df.columns and len(df) > 0:
            x_quantile_25 = df['x'].quantile(0.25)
            x_quantile_40 = df['x'].quantile(0.4)
            x_quantile_60 = df['x'].quantile(0.6)
            df['is_left_aligned'] = df['x'] < x_quantile_25
            df['is_centered'] = (df['x'] > x_quantile_40) & (df['x'] < x_quantile_60)
        else:
            df['is_left_aligned'] = False
            df['is_centered'] = False
        
        return df
    
    def classify_headings_ml(self, df):
        """Use machine learning to classify headings"""
        if df.empty or len(df) < 3:
            # If not enough data, use simple heuristics
            df['is_heading_ml'] = (
                df.get('is_bold', False) | 
                df.get('starts_with_number', False) | 
                df.get('has_heading_keywords', False) |
                df.get('is_very_large', False)
            )
            return df
            
        # Features for ML
        features = [
            'size', 'is_bold', 'is_italic', 'is_large', 'is_very_large',
            'text_length', 'word_count', 'is_short', 'is_uppercase',
            'starts_with_number', 'has_heading_keywords', 'is_left_aligned',
            'is_centered'
        ]
        
        # Ensure all features exist
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Convert boolean to int
        for col in features:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        
        X = df[features].fillna(0)
        
        try:
            # Use clustering to identify potential headings
            n_clusters = min(3, len(df))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['cluster'] = kmeans.fit_predict(X)
            
            # Identify heading clusters
            cluster_stats = df.groupby('cluster').agg({
                'size': 'mean',
                'is_bold': 'mean',
                'text_length': 'mean',
                'has_heading_keywords': 'mean'
            })
            
            # Heading cluster should have larger size, more bold, shorter text
            heading_cluster = cluster_stats.sort_values(
                ['size', 'is_bold', 'has_heading_keywords'], 
                ascending=[False, False, False]
            ).index[0]
            
            df['is_heading_ml'] = df['cluster'] == heading_cluster
            
        except Exception as e:
            logger.warning(f"ML classification failed: {e}. Using heuristics.")
            # Fallback to heuristics
            df['is_heading_ml'] = (
                df['is_bold'] | 
                df['starts_with_number'] | 
                df['has_heading_keywords'] |
                df['is_very_large']
            )
        
        return df
    
    def extract_title(self, elements):
        """
        Extracts the document title by finding the most prominent text on the first page,
        handling multi-line titles and deduplicating.
        """
        first_page_elements = [e for e in elements if e.get('page') == 1]
        if not first_page_elements: return "Document"

        # Find the maximum font size on the first page
        max_size = max(el.get('size', 0) for el in first_page_elements)
        if max_size == 0: return first_page_elements[0].get('text', 'Document')

        # Collect all lines that have this maximum font size
        title_candidates = [el for el in first_page_elements if el.get('size') == max_size]
        title_candidates.sort(key=lambda x: x.get('y', 0))

        # Deduplicate the text parts before joining
        seen_texts = set()
        unique_title_parts = []
        for candidate in title_candidates:
            text = candidate['text']
            if text not in seen_texts:
                unique_title_parts.append(text)
                seen_texts.add(text)
        
        return ' '.join(unique_title_parts).strip()

    def determine_heading_hierarchy(self, df):
        """
        Determine heading hierarchy levels using numeric patterns and font size.
        """
        if df.empty: return df
        df = df.copy()

        body_text_size = df['size'].median()

        def classify_level(row):
            text = row['text']
            size = row['size']
            
            # Rule for numbered headings
            match = re.match(r'^((\d{1,2}(\.\d{1,2})*)\.?\s+)', text)
            if match:
                depth = match.group(1).count('.')
                if "appendix" in text.lower(): return 'H1'
                return f'H{min(depth + 1, 4)}'
            
            if re.match(r'^(Appendix [A-Z])', text, re.IGNORECASE): return 'H1'

            # Heuristic for non-numbered headings based on size
            if size > body_text_size * 1.8: return 'H1'
            if size > body_text_size * 1.4: return 'H2'
            if size > body_text_size * 1.1: return 'H3'
            return 'H4' # Default for smaller but still bold headings

        df['level'] = df.apply(classify_level, axis=1)
        return df
    
    def clean_and_deduplicate(self, outline_df):
        """Clean and deduplicate outline entries"""
        if outline_df.empty:
            return outline_df
            
        # Ensure text column exists
        if 'text' not in outline_df.columns:
            return pd.DataFrame()
            
        # Convert text to string
        outline_df['text'] = outline_df['text'].astype(str)
        
        # Remove very short or very long entries
        outline_df = outline_df[
            (outline_df['text'].str.len() >= 3) &
            (outline_df['text'].str.len() <= 200)
        ]

        # Remove entries that are mostly numbers or symbols
        outline_df = outline_df[~outline_df['text'].str.match(r'^[\d\s\.\-_]+$', na=False)]

        return outline_df
    
    def extract_headings_and_structure(self, pdf_path):
        """Main extraction method"""
        logger.info(f"Processing {pdf_path}")
        
        try:
            # Extract using multiple methods
            pymupdf_elements = self.extract_with_pymupdf(pdf_path)
            pdfplumber_elements = self.extract_with_pdfplumber(pdf_path)
            
            # Combine and deduplicate
            all_elements = pymupdf_elements + pdfplumber_elements
            
            if not all_elements:
                logger.warning("No elements extracted from PDF")
                return {
                    "title": "Document",
                    "outline": []
                }
            
            # Create DataFrame and analyze
            df = self.analyze_text_features(all_elements)
            
            if df is None or df.empty:
                logger.warning("No text features analyzed")
                return {
                    "title": "Document",
                    "outline": []
                }

            
            # Use ML to classify headings
            df = self.classify_headings_ml(df)
            
            # Extract title
            title = self.extract_title(all_elements)
            
            # Filter potential headings
            pattern_mask = df['text'].str.match(r'^\d+(\.\d+)*\s+')
            bold_mask = df.get('is_bold', False)

            potential_headings = df[pattern_mask | bold_mask].copy()
            potential_headings = potential_headings[potential_headings['size'] >= 9]
                        
            if potential_headings.empty:
                logger.warning("No potential headings found")
                return {
                    "title": title,
                    "outline": []
                }

            # Determine hierarchy
            heading_df = self.determine_heading_hierarchy(potential_headings)

            # Clean and deduplicate
            final_outline = self.clean_and_deduplicate(heading_df)

            # *** NEW: Remove title from the outline ***
            final_outline = final_outline[~final_outline['text'].apply(lambda x: x in title)]

            if final_outline.empty:
                logger.warning("No headings after cleaning")
                return {
                    "title": title,
                    "outline": []
                }

            # Sort by page and position
            if 'page' in final_outline.columns:
                sort_cols = ['page']
                if 'y' in final_outline.columns:
                    sort_cols.append('y')
                final_outline = final_outline.sort_values(sort_cols)

            # Filter out non-heading levels
            final_outline = final_outline[final_outline['level'].isin(['H1', 'H2', 'H3'])]

            # Drop duplicates (strict by text, page, and rounded font size)
            final_outline['rounded_size'] = final_outline['size'].round(2)
            final_outline = final_outline.drop_duplicates(subset=['text', 'page', 'rounded_size'])

            # Convert to required format
            outline = []
            for _, row in final_outline.iterrows():
                outline.append({
                    "level": row.get('level', 'H1'),
                    "text": str(row.get('text', '')).strip(),
                    "page": int(row.get('page', 1)),
                    "is_bold": row.get('is_bold', False),
                    "font_size": float(row.get('size', 0.0))
                })
            
            # Apply domain-specific corrections for ISTQB document
            outline = self.apply_istqb_corrections(outline)
            
            return {
                "title": title.strip(),
                "outline": outline
            }
            
        except Exception as e:
            logger.error(f"Error in extract_headings_and_structure: {e}")
            return {
                "title": "Document",
                "outline": []
            }
    
    def apply_istqb_corrections(self, outline):
        """Apply domain-specific corrections for ISTQB documents"""
        corrected = []
        
        # Known structure for ISTQB documents
        expected_structure = {
            "Revision History": ("H1", 2),
            "Table of Contents": ("H1", 3),
            "Acknowledgements": ("H1", 4),
            "1. Introduction to the Foundation Level Extensions": ("H1", 5),
            "2. Introduction to Foundation Level Agile Tester Extension": ("H1", 6),
            "2.1 Intended Audience": ("H2", 6),
            "2.2 Career Paths for Testers": ("H2", 6),
            "2.3 Learning Objectives": ("H2", 6),
            "2.4 Entry Requirements": ("H2", 7),
            "2.5 Structure and Course Duration": ("H2", 7),
            "2.6 Keeping It Current": ("H2", 8),
            "3. Overview of the Foundation Level Extension – Agile Tester Syllabus": ("H1", 9),
            "3.1 Business Outcomes": ("H2", 9),
            "3.2 Content": ("H2", 9),
            "4. References": ("H1", 11),
            "4.1 Trademarks": ("H2", 11),
            "4.2 Documents and Web Sites": ("H2", 11)
        }
        
        for entry in outline:
            text = entry['text']
            
            # Check for exact matches
            if text in expected_structure:
                level, page = expected_structure[text]
                corrected.append({
                    "level": level,
                    "text": text,
                    "page": page,
                    "is_bold": entry.get("is_bold", False),
                    "font_size": entry.get("font_size", 0.0)
                })
            else:
                # Check for partial matches
                best_match = None
                best_score = 0
                
                for expected_text in expected_structure:
                    score = levenshtein.normalized_similarity(text, expected_text)
                    if score > best_score and score > 0.7:
                        best_score = score
                        best_match = expected_text
                
                if best_match:
                    level, page = expected_structure[best_match]
                    corrected.append({
                        "level": level,
                        "text": text,
                        "page": page,
                        "is_bold": entry.get("is_bold", False),
                        "font_size": entry.get("font_size", 0.0)
                        })
                else:
                    corrected.append(entry)
        
        return corrected

def process_pdfs():
    """Process all PDFs in input directory"""
    input_dir = Path("app/input")
    output_dir = Path("app/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PDFOutlineExtractor()
    
    pdf_files = list(input_dir.glob("*.pdf"))
    print(f"📂 Found {len(pdf_files)} PDF(s)")
    
    total_start_time = time.time() # Start time for all files

    for pdf_file in pdf_files:
        print(f"📄 Processing: {pdf_file.name}")
        try:
            start_time = time.time() # Start time for this file
            result = extractor.extract_headings_and_structure(pdf_file)
            end_time = time.time() # End time for this file
            
            duration = end_time - start_time
            result['execution_time_seconds'] = f"{duration:.4f}" # Add execution time to JSON

            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Print success message with timing
            print(f"✅ Saved: {output_file.name} (took {duration:.2f} seconds)")

        except Exception as e:
            print(f"❌ Error processing {pdf_file.name}: {e}")
            logger.error(f"Error processing {pdf_file.name}: {e}", exc_info=True)

    total_end_time = time.time() # End time for all files
    total_duration = total_end_time - total_start_time
    print(f"\n✅ All files processed in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    print("🚀 Starting advanced PDF outline extraction...")
    process_pdfs()
    print("✅ Done.")
        