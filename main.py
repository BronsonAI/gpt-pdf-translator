import os
import re
import uuid
import shutil
import html
import fitz  # PyMuPDF
import requests
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Request, WebSocket, BackgroundTasks, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import tempfile
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Dictionary to store translation progress
translation_progress = {}

# Create FastAPI app
app = FastAPI(
    title="PDF Translator",
    description="Translate PDF documents while preserving the original layout",
    version="1.0.0"
)

# Enable CORS for WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("converted", exist_ok=True)
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Language mapping
LANGUAGES = {
    "af": "Afrikaans",
    "sq": "Albanian",
    "am": "Amharic",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "ceb": "Cebuano",
    "zh": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "co": "Corsican",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "eo": "Esperanto",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "fy": "Frisian",
    "gl": "Galician",
    "ka": "Georgian",
    "de": "German",
    "el": "Greek",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "ha": "Hausa",
    "haw": "Hawaiian",
    "he": "Hebrew",
    "hi": "Hindi",
    "hmn": "Hmong",
    "hu": "Hungarian",
    "is": "Icelandic",
    "ig": "Igbo",
    "id": "Indonesian",
    "ga": "Irish",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "km": "Khmer",
    "rw": "Kinyarwanda",
    "ko": "Korean",
    "ku": "Kurdish",
    "ky": "Kyrgyz",
    "lo": "Lao",
    "la": "Latin",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "lb": "Luxembourgish",
    "mk": "Macedonian",
    "mg": "Malagasy",
    "ms": "Malay",
    "ml": "Malayalam",
    "mt": "Maltese",
    "mi": "Maori",
    "mr": "Marathi",
    "mn": "Mongolian",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "no": "Norwegian",
    "ny": "Nyanja (Chichewa)",
    "or": "Odia (Oriya)",
    "ps": "Pashto",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "pa": "Punjabi",
    "ro": "Romanian",
    "ru": "Russian",
    "sm": "Samoan",
    "gd": "Scots Gaelic",
    "sr": "Serbian",
    "st": "Sesotho",
    "sn": "Shona",
    "sd": "Sindhi",
    "si": "Sinhala (Sinhalese)",
    "sk": "Slovak",
    "sl": "Slovenian",
    "so": "Somali",
    "es": "Spanish",
    "su": "Sundanese",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog (Filipino)",
    "tg": "Tajik",
    "ta": "Tamil",
    "tt": "Tatar",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "tk": "Turkmen",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "ug": "Uyghur",
    "uz": "Uzbek",
    "vi": "Vietnamese",
    "cy": "Welsh",
    "xh": "Xhosa",
    "yi": "Yiddish",
    "yo": "Yoruba",
    "zu": "Zulu"
}

def translate_text(text, target_language):
    """
    Translate text using OpenAI API.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code
        
    Returns:
        str: Translated text
    """
    if not text or not text.strip():
        return text
    
    # Ensure API key is available
    if not OPENAI_API_KEY:
        print("Error: OpenAI API key is not set. Please check your .env file.")
        return text
    
    # Call OpenAI API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "system",
                "content": f"You are a professional translator. Translate the following text to {LANGUAGES.get(target_language, target_language)}. Preserve all formatting, line breaks, and special characters. Only respond with the translated text, nothing else."
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 0.3
    }
    
    try:
        print(f"Sending translation request for text: '{text[:30]}...' to {target_language}")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # Add timeout to prevent hanging requests
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"API error: Status code {response.status_code}, Response: {response.text}")
            return text
        
        response_data = response.json()
        
        # Check if the response contains the expected data
        if "choices" not in response_data or not response_data["choices"] or "message" not in response_data["choices"][0]:
            print(f"Unexpected API response format: {response_data}")
            return text
        
        translated_text = response_data["choices"][0]["message"]["content"]
        print(f"Translation successful. Result: '{translated_text[:30]}...'")
        return translated_text
    
    except requests.exceptions.Timeout:
        print("Translation request timed out. The API might be experiencing high load.")
        return text
    except requests.exceptions.RequestException as e:
        print(f"Request error during translation: {str(e)}")
        return text
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

def batch_translate(texts, target_language, batch_size=10, task_id=None):
    """
    Translate a list of texts in batches.
    
    Args:
        texts (list): List of texts to translate
        target_language (str): Target language code
        batch_size (int, optional): Batch size. Defaults to 10.
        task_id (str, optional): Task ID for progress tracking
        
    Returns:
        list: List of translated texts
    """
    translated_texts = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    print(f"\n[Translation] Starting translation of {len(texts)} text blocks to {LANGUAGES.get(target_language, target_language)}")
    print(f"[Translation] Processing in {total_batches} batches of {batch_size} texts each")
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_translated = []
        
        current_batch = i // batch_size + 1
        print(f"\n[Translation] Batch {current_batch}/{total_batches} ({(current_batch/total_batches)*100:.1f}%)")
        
        for j, text in enumerate(batch):
            print(f"[Translation] Translating text {j+1}/{len(batch)} in batch {current_batch}", end="\r")
            translated = translate_text(text, target_language)
            batch_translated.append(translated)
        
        translated_texts.extend(batch_translated)
        
        # Update progress if task_id is provided
        if task_id:
            progress = 30 + (current_batch / total_batches) * 40  # Translation is 30-70% of the process
            message = f"Translating batch {current_batch} of {total_batches} ({(current_batch/total_batches)*100:.1f}%)"
            print(f"\n[Progress] {message} - {progress:.1f}%")
            
            translation_progress[task_id].update({
                "progress": progress,
                "message": message
            })
        
    print(f"\n[Translation] Completed translation of {len(texts)} text blocks")
    return translated_texts

def translate_pdf_directly(input_pdf_path, target_language, task_id=None):
    """
    Translate a PDF document by preserving the original layout and replacing text.
    
    Args:
        input_pdf_path (str): Path to the input PDF file
        target_language (str): Target language code
        task_id (str, optional): Task ID for progress tracking
        
    Returns:
        str: Path to the translated PDF file
    """
    # Create output path for translated PDF
    output_pdf_path = f"{os.path.splitext(input_pdf_path)[0]}_{target_language}.pdf"
    
    print(f"\n[PDF Translation] Starting translation of {input_pdf_path} to {LANGUAGES.get(target_language, target_language)}")
    
    # Initialize progress if task_id is provided
    if task_id:
        translation_progress[task_id] = {
            "status": "extracting",
            "progress": 0,
            "message": "Extracting text from PDF..."
        }
        print(f"[Progress] Task ID: {task_id} - Extracting text from PDF...")
    
    # Open the PDF
    pdf_document = fitz.open(input_pdf_path)
    
    # Create a new PDF document for the translated version
    translated_pdf = fitz.open()
    
    # Ensure the standard fonts are available
    # PyMuPDF uses these built-in fonts by default
    standard_fonts = ["helv", "tiro", "zadb", "symb"]
    
    # Extract all text blocks from all pages
    all_text_blocks = []
    all_texts = []
    
    # First pass: extract all text
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        
        # Update progress if task_id is provided
        if task_id:
            progress = (page_num / pdf_document.page_count) * 30  # First 30% is extraction
            translation_progress[task_id].update({
                "progress": progress,
                "message": f"Extracting text from page {page_num + 1} of {pdf_document.page_count}..."
            })
            print(f"[Progress] Task ID: {task_id} - Extracting text from page {page_num + 1} of {pdf_document.page_count}...")
        
        # Extract text with positions
        page_blocks = []
        text_dict = page.get_text("dict")
        
        # Process text blocks
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if text:
                            # Get position and font info
                            x0, y0, x1, y1 = span["bbox"]
                            font_size = span["size"]
                            font_name = span["font"]
                            is_bold = "bold" in font_name.lower() or span["flags"] & 2 != 0
                            is_italic = "italic" in font_name.lower() or "oblique" in font_name.lower() or span["flags"] & 1 != 0
                            
                            # Create a text block
                            text_block = {
                                "text": text,
                                "bbox": (x0, y0, x1, y1),
                                "font_size": font_size,
                                "is_bold": is_bold,
                                "is_italic": is_italic,
                                "page": page_num
                            }
                            
                            page_blocks.append(text_block)
                            all_texts.append(text)
        
        all_text_blocks.extend(page_blocks)
    
    # Translate all texts at once
    if task_id:
        translation_progress[task_id].update({
            "status": "translating",
            "progress": 30,
            "message": "Translating text..."
        })
        print(f"[Progress] Task ID: {task_id} - Translating text...")
    
    translated_texts = batch_translate(all_texts, target_language, batch_size=20, task_id=task_id)
    
    # Second pass: create translated PDF
    if task_id:
        translation_progress[task_id].update({
            "status": "generating",
            "progress": 70,
            "message": "Generating translated PDF..."
        })
        print(f"[Progress] Task ID: {task_id} - Generating translated PDF...")
    
    for page_num in range(pdf_document.page_count):
        # Update progress if task_id is provided
        if task_id:
            progress = 70 + (page_num / pdf_document.page_count) * 30  # Last 30% is PDF generation
            translation_progress[task_id].update({
                "progress": progress,
                "message": f"Generating page {page_num + 1} of {pdf_document.page_count}..."
            })
            print(f"[Progress] Task ID: {task_id} - Generating page {page_num + 1} of {pdf_document.page_count}...")
        
        # Insert the original page into the new document
        translated_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
        translated_page = translated_pdf[page_num]
        
        # Get text blocks for this page
        page_blocks = [block for block in all_text_blocks if block["page"] == page_num]
        
        # Cover original text with white rectangles and add translated text
        for i, block in enumerate(page_blocks):
            block_index = all_text_blocks.index(block)
            if block_index < len(translated_texts):
                # Cover original text with white rectangle
                rect = fitz.Rect(block["bbox"])
                translated_page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                
                # Add translated text
                text = translated_texts[block_index]
                
                try:
                    # Create a TextWriter object for more reliable text insertion
                    tw = fitz.TextWriter(translated_page.rect)
                    
                    # Determine font properties
                    font_name = "helv"  # Use the default Helvetica font
                    if block["is_bold"] and block["is_italic"]:
                        font = fitz.Font("helv", weight=fitz.FONTWEIGHT_BOLD, is_italic=True)
                    elif block["is_bold"]:
                        font = fitz.Font("helv", weight=fitz.FONTWEIGHT_BOLD)
                    elif block["is_italic"]:
                        font = fitz.Font("helv", is_italic=True)
                    else:
                        font = fitz.Font("helv")
                    
                    # Calculate available width for text
                    available_width = rect.width
                    
                    # Handle text wrapping if needed
                    if len(text) > 5:  # Only process wrapping for non-trivial text
                        # Estimate character width (approximate)
                        char_width = block["font_size"] * 0.6
                        
                        # Calculate approximate characters per line
                        chars_per_line = int(available_width / char_width)
                        
                        if chars_per_line > 0:
                            # Simple word wrapping algorithm
                            words = text.split()
                            lines = []
                            current_line = []
                            current_length = 0
                            
                            for word in words:
                                # Add word length plus a space
                                word_length = len(word) + 1
                                
                                if current_length + word_length <= chars_per_line:
                                    current_line.append(word)
                                    current_length += word_length
                                else:
                                    if current_line:  # Add the current line if it's not empty
                                        lines.append(' '.join(current_line))
                                    current_line = [word]
                                    current_length = word_length
                            
                            # Add the last line
                            if current_line:
                                lines.append(' '.join(current_line))
                            
                            # Now add each line with proper positioning
                            for i, line in enumerate(lines):
                                line_y = rect.y0 + (i * block["font_size"] * 1.2)
                                
                                # Make sure we don't go beyond the bottom of the original text block
                                if line_y + block["font_size"] <= rect.y1:
                                    tw.append(
                                        fitz.Point(rect.x0, line_y + (block["font_size"] * 0.8)),
                                        line,
                                        font=font,
                                        fontsize=block["font_size"]
                                    )
                        else:
                            # Fallback for very narrow blocks
                            tw.append(
                                fitz.Point(rect.x0, rect.y0 + (block["font_size"] * 0.8)),
                                text,
                                font=font,
                                fontsize=block["font_size"]
                            )
                    else:
                        # For very short text, no need for wrapping
                        tw.append(
                            fitz.Point(rect.x0, rect.y0 + (block["font_size"] * 0.8)),
                            text,
                            font=font,
                            fontsize=block["font_size"]
                        )
                    
                    # Write the text to the page
                    tw.write_text(translated_page)
                    
                except Exception as e:
                    # Fallback method if TextWriter fails
                    print(f"Text insertion error: {str(e)}. Using fallback method.")
                    try:
                        # Simple text insertion as fallback
                        text_point = fitz.Point(rect.x0, rect.y0 + (block["font_size"] * 0.8))
                        translated_page.insert_text(
                            text_point,
                            text,
                            fontname="helv",  # Default font
                            fontsize=block["font_size"],
                            color=(0, 0, 0)  # Black text
                        )
                    except Exception as e2:
                        print(f"Fallback text insertion failed: {str(e2)}")
    
    # Save the translated PDF
    translated_pdf.save(output_pdf_path)
    translated_pdf.close()
    pdf_document.close()
    
    # Update progress if task_id is provided
    if task_id:
        translation_progress[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Translation completed!",
            "pdf_path": output_pdf_path
        })
        print(f"[Progress] Task ID: {task_id} - Translation completed!")
    
    return output_pdf_path

def extract_text_blocks(page):
    """
    Extract text blocks from a PDF page.
    
    Args:
        page (fitz.Page): PDF page object
    
    Returns:
        list: List of text blocks with their positions
    """
    text_dict = page.get_text("dict")
    text_blocks = []
    
    for block in text_dict["blocks"]:
        if block["type"] == 0:  # Text block
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        # Get position and font info
                        x0, y0, x1, y1 = span["bbox"]
                        font_size = span["size"]
                        font_name = span["font"]
                        is_bold = "bold" in font_name.lower() or span["flags"] & 2 != 0
                        is_italic = "italic" in font_name.lower() or "oblique" in font_name.lower() or span["flags"] & 1 != 0
                        
                        # Create a text block
                        text_block = {
                            "text": text,
                            "bbox": (x0, y0, x1, y1),
                            "font_size": font_size,
                            "is_bold": is_bold,
                            "is_italic": is_italic
                        }
                        
                        text_blocks.append(text_block)
    
    return text_blocks

def create_translated_pdf(input_pdf_path, output_pdf_path, text_blocks, translated_texts):
    """
    Create a translated PDF by replacing text in the original PDF.
    
    Args:
        input_pdf_path (str): Path to the input PDF file
        output_pdf_path (str): Path to the output translated PDF file
        text_blocks (list): List of text blocks with their positions
        translated_texts (list): List of translated texts
    """
    # Make sure PIL is installed for background color detection
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pillow"])
        from PIL import Image
    
    # Open the input PDF
    pdf_document = fitz.open(input_pdf_path)
    
    # Create a new PDF document for the translated version
    translated_pdf = fitz.open()
    
    # Ensure the standard fonts are available
    # PyMuPDF uses these built-in fonts by default
    standard_fonts = ["helv", "tiro", "zadb", "symb"]
    
    # Create a mapping of text blocks to translations
    translation_map = {}
    text_only_blocks = [block for block in text_blocks if block["text"].strip()]
    
    # Make sure we have translations for all text blocks
    for i, block in enumerate(text_only_blocks):
        if i < len(translated_texts):
            translation_map[block["text"]] = translated_texts[i]
        else:
            # If we somehow have fewer translations than blocks, use original text
            translation_map[block["text"]] = block["text"]
            print(f"Warning: Missing translation for block: {block['text']}")
    
    # Process each page
    for page_num in range(pdf_document.page_count):
        # Insert the original page into the new document
        translated_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
        translated_page = translated_pdf[page_num]
        original_page = pdf_document[page_num]
        
        # Get text blocks for this page
        page_blocks = [block for block in text_blocks if block.get("page") == page_num]
        
        # Cover original text with background-colored rectangles and add translated text
        for block in page_blocks:
            if not block["text"].strip():
                continue  # Skip empty blocks
                
            # Get the rectangle for this text block
            rect = fitz.Rect(block["bbox"])
            
            # Detect the background color for this text block
            bg_color = detect_background_color(original_page, rect)
            print(f"Detected background color for text '{block['text'][:20]}...': {bg_color}")
            
            # Cover original text with detected background color rectangle
            translated_page.draw_rect(rect, color=bg_color, fill=bg_color)
            
            # Add translated text if available
            text = translation_map.get(block["text"], block["text"])
            
            try:
                # Create a TextWriter object for more reliable text insertion
                tw = fitz.TextWriter(translated_page.rect)
                
                # Determine font properties
                font_name = "helv"  # Use the default Helvetica font
                if block["is_bold"] and block["is_italic"]:
                    font = fitz.Font("helv", weight=fitz.FONTWEIGHT_BOLD, is_italic=True)
                elif block["is_bold"]:
                    font = fitz.Font("helv", weight=fitz.FONTWEIGHT_BOLD)
                elif block["is_italic"]:
                    font = fitz.Font("helv", is_italic=True)
                else:
                    font = fitz.Font("helv")
                
                # Calculate available width for text
                available_width = rect.width
                
                # Handle text wrapping if needed
                if len(text) > 5:  # Only process wrapping for non-trivial text
                    # Estimate character width (approximate)
                    char_width = block["font_size"] * 0.6
                    
                    # Calculate approximate characters per line
                    chars_per_line = int(available_width / char_width)
                    
                    if chars_per_line > 0:
                        # Simple word wrapping algorithm
                        words = text.split()
                        lines = []
                        current_line = []
                        current_length = 0
                        
                        for word in words:
                            # Add word length plus a space
                            word_length = len(word) + 1
                            
                            if current_length + word_length <= chars_per_line:
                                current_line.append(word)
                                current_length += word_length
                            else:
                                if current_line:  # Add the current line if it's not empty
                                    lines.append(' '.join(current_line))
                                current_line = [word]
                                current_length = word_length
                        
                        # Add the last line
                        if current_line:
                            lines.append(' '.join(current_line))
                        
                        # Now add each line with proper positioning
                        for i, line in enumerate(lines):
                            line_y = rect.y0 + (i * block["font_size"] * 1.2)
                            
                            # Make sure we don't go beyond the bottom of the original text block
                            if line_y + block["font_size"] <= rect.y1:
                                tw.append(
                                    fitz.Point(rect.x0, line_y + (block["font_size"] * 0.8)),
                                    line,
                                    font=font,
                                    fontsize=block["font_size"]
                                )
                    else:
                        # Fallback for very narrow blocks
                        tw.append(
                            fitz.Point(rect.x0, rect.y0 + (block["font_size"] * 0.8)),
                            text,
                            font=font,
                            fontsize=block["font_size"]
                        )
                else:
                    # For very short text, no need for wrapping
                    tw.append(
                        fitz.Point(rect.x0, rect.y0 + (block["font_size"] * 0.8)),
                        text,
                        font=font,
                        fontsize=block["font_size"]
                    )
                
                # Write the text to the page
                tw.write_text(translated_page)
                
            except Exception as e:
                # Fallback method if TextWriter fails
                print(f"Text insertion error: {str(e)}. Using fallback method.")
                try:
                    # Simple text insertion as fallback
                    text_point = fitz.Point(rect.x0, rect.y0 + (block["font_size"] * 0.8))
                    translated_page.insert_text(
                        text_point,
                        text,
                        fontname="helv",  # Default font
                        fontsize=block["font_size"],
                        color=(0, 0, 0)  # Black text
                    )
                except Exception as e2:
                    print(f"Fallback text insertion failed: {str(e2)}")
    
    # Save the translated PDF
    translated_pdf.save(output_pdf_path)
    translated_pdf.close()
    pdf_document.close()

def detect_background_color(page, rect):
    """
    Detect the background color of a text block in a PDF page.
    
    Args:
        page (fitz.Page): PDF page
        rect (fitz.Rect): Rectangle area to check
        
    Returns:
        tuple: RGB color tuple (r, g, b) where each value is between 0 and 1
    """
    try:
        # Slightly expand the rectangle to ensure we capture the background
        expanded_rect = fitz.Rect(
            rect.x0 - 1, 
            rect.y0 - 1, 
            rect.x1 + 1, 
            rect.y1 + 1
        )
        
        # Render the area to a pixmap
        pix = page.get_pixmap(clip=expanded_rect, alpha=False)
        
        # Convert to PIL Image for easier color analysis
        from PIL import Image
        img_data = pix.samples
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        
        # Get the most common color in the image (likely the background)
        colors = img.getcolors(pix.width * pix.height)
        if not colors:
            return (1, 1, 1)  # Default to white if no colors found
            
        # Sort by count (first element of tuple) in descending order
        colors.sort(reverse=True, key=lambda x: x[0])
        
        # Get the most common color (RGB)
        most_common = colors[0][1]
        
        # Convert from 0-255 range to 0-1 range for PyMuPDF
        r, g, b = most_common
        return (r/255, g/255, b/255)
        
    except Exception as e:
        print(f"Error detecting background color: {str(e)}")
        return (1, 1, 1)  # Default to white on error

@app.post("/translate-pdf")
async def translate_pdf(request: Request, file: UploadFile = File(...), target_language: str = Form(...), background_tasks: BackgroundTasks = None):
    """
    Endpoint to translate a PDF file.
    """
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Initialize progress tracking for this task
    translation_progress[task_id] = {
        "status": "waiting",
        "progress": 0,
        "message": "Preparing to translate your PDF..."
    }
    
    print(f"[API] Starting PDF translation task with ID: {task_id}")
    print(f"[API] File: {file_path}, Target language: {target_language}")
    
    # Start background task for translation
    background_tasks.add_task(translate_pdf_background, file_path, target_language, task_id)
    
    # Return task ID for tracking progress
    return {"task_id": task_id}

@app.websocket("/ws/progress/{task_id}")
async def websocket_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint to send real-time progress updates for a specific task.
    """
    await websocket.accept()
    print(f"[WebSocket] Client connected for task ID: {task_id}")
    
    # Initialize connection status
    connected = True
    
    # Keep track of last sent progress to avoid duplicate updates
    last_sent_progress = None
    
    try:
        # Send initial progress data if available
        if task_id in translation_progress:
            await websocket.send_json(translation_progress[task_id])
            last_sent_progress = translation_progress[task_id].copy()
            print(f"[WebSocket] Sent initial progress for task {task_id}: {translation_progress[task_id]}")
        
        # Keep connection open and periodically send updates
        while connected:
            if task_id in translation_progress:
                progress_data = translation_progress[task_id]
                
                # Only send update if progress has changed
                if last_sent_progress is None or progress_data != last_sent_progress:
                    # Send progress update
                    await websocket.send_json(progress_data)
                    last_sent_progress = progress_data.copy()
                    print(f"[WebSocket] Sent progress update for task {task_id}: {progress_data}")
                
                # If translation is completed or failed, close the connection after sending final update
                if progress_data["status"] in ["completed", "error"]:
                    print(f"[WebSocket] Task {task_id} {progress_data['status']}, closing connection")
                    await websocket.close()
                    connected = False
                    break
            
            # Wait before checking for the next update (shorter interval for more responsive updates)
            await asyncio.sleep(0.5)
    
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected for task ID: {task_id}")
        connected = False
    except Exception as e:
        print(f"[WebSocket] Error for task ID {task_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        connected = False
        try:
            await websocket.close()
        except:
            pass

@app.get("/progress/{task_id}")
async def get_progress(task_id: str):
    """
    REST API endpoint to get progress for a specific task.
    This serves as a fallback if WebSocket connection fails.
    """
    if task_id in translation_progress:
        print(f"[API] Progress request for task {task_id}: {translation_progress[task_id]}")
        return translation_progress[task_id]
    else:
        return {"status": "unknown", "progress": 0, "message": "Task not found"}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Root endpoint that serves the main application page.
    """
    return templates.TemplateResponse("pdf_translator.html", {
        "request": request,
        "languages": LANGUAGES
    })

async def translate_pdf_background(file_path, target_language, task_id):
    """
    Background task to translate a PDF file.
    Updates the progress in the global translation_progress dictionary.
    """
    try:
        print(f"[Background] Starting PDF translation for task ID: {task_id}")
        
        # Update status to extracting
        translation_progress[task_id] = {
            "status": "extracting",
            "progress": 5,
            "message": "Extracting text from PDF..."
        }
        print(f"[Background] Progress updated: {translation_progress[task_id]}")
        
        # Extract text blocks from the PDF
        pdf_doc = fitz.open(file_path)
        total_pages = len(pdf_doc)
        
        # Get all text blocks with their positions
        all_blocks = []
        for page_num in range(total_pages):
            # Update progress for extraction phase more frequently
            extraction_progress = 5 + int((page_num / total_pages) * 20)  # 5-25% for extraction
            translation_progress[task_id] = {
                "status": "extracting",
                "progress": extraction_progress,
                "message": f"Extracting text from page {page_num + 1} of {total_pages}..."
            }
            print(f"[Background] Progress updated: {translation_progress[task_id]}")
            
            # Add a small delay to ensure WebSocket has time to send the update
            await asyncio.sleep(0.1)
            
            page = pdf_doc[page_num]
            blocks = extract_text_blocks(page)
            
            # Add page number to each block
            for block in blocks:
                block["page"] = page_num
            
            all_blocks.extend(blocks)
        
        # Filter out empty blocks
        text_blocks = [block for block in all_blocks if block["text"].strip()]
        
        # Check if we have any text to translate
        if not text_blocks:
            print(f"[Background] No text found to translate in the PDF")
            translation_progress[task_id] = {
                "status": "error",
                "progress": 0,
                "message": "No text found to translate in the PDF. The document may be image-based or contain no extractable text."
            }
            return None
        
        # Update status to translating
        translation_progress[task_id] = {
            "status": "translating",
            "progress": 25,
            "message": "Translating text content..."
        }
        print(f"[Background] Progress updated: {translation_progress[task_id]}")
        
        # Extract all text for translation
        all_texts = [block["text"] for block in text_blocks]
        total_texts = len(all_texts)
        
        # Translate texts in batches
        translated_texts = []
        batch_size = 10  # Adjust based on API limits and performance
        
        for i in range(0, total_texts, batch_size):
            batch = all_texts[i:i+batch_size]
            batch_translated = []
            
            # Update progress before starting batch translation
            translation_batch_progress = 25 + int((i / total_texts) * 50)
            translation_progress[task_id] = {
                "status": "translating",
                "progress": translation_batch_progress,
                "message": f"Translating text ({i} of {total_texts} blocks)..."
            }
            print(f"[Background] Progress updated: {translation_progress[task_id]}")
            
            # Add a small delay to ensure WebSocket has time to send the update
            await asyncio.sleep(0.1)
            
            for j, text in enumerate(batch):
                print(f"[Translation] Translating text {j+1}/{len(batch)} in batch {i//batch_size + 1}")
                translated = translate_text(text, target_language)
                batch_translated.append(translated)
                
                # Update progress for each text within the batch
                text_progress = translation_batch_progress + int((j / len(batch)) * (50 / (total_texts / batch_size)))
                translation_progress[task_id] = {
                    "status": "translating",
                    "progress": min(text_progress, 74),  # Cap at 74% to leave room for PDF generation
                    "message": f"Translating text ({i + j + 1} of {total_texts} blocks)..."
                }
                print(f"[Background] Progress updated: {translation_progress[task_id]}")
                
                # Add a small delay to ensure WebSocket has time to send the update
                await asyncio.sleep(0.1)
            
            translated_texts.extend(batch_translated)
        
        # Update status to generating PDF
        translation_progress[task_id] = {
            "status": "generating",
            "progress": 75,
            "message": "Generating translated PDF..."
        }
        print(f"[Background] Progress updated: {translation_progress[task_id]}")
        
        # Create translated PDF
        output_pdf_path = os.path.join("static", f"translated_{task_id}.pdf")
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
        
        # Update progress before starting PDF creation
        translation_progress[task_id] = {
            "status": "generating",
            "progress": 80,
            "message": "Creating PDF with translated text..."
        }
        print(f"[Background] Progress updated: {translation_progress[task_id]}")
        
        # Add a small delay to ensure WebSocket has time to send the update
        await asyncio.sleep(0.1)
        
        # Create the translated PDF
        create_translated_pdf(file_path, output_pdf_path, all_blocks, translated_texts)
        
        # Update progress after PDF creation
        translation_progress[task_id] = {
            "status": "generating",
            "progress": 95,
            "message": "Finalizing translated document..."
        }
        print(f"[Background] Progress updated: {translation_progress[task_id]}")
        
        # Add a small delay to ensure WebSocket has time to send the update
        await asyncio.sleep(0.1)
        
        # Check if the file was created successfully
        if not os.path.exists(output_pdf_path):
            print(f"[Background] Error: Translated PDF file was not created at {output_pdf_path}")
            translation_progress[task_id] = {
                "status": "error",
                "progress": 0,
                "message": "Error creating translated PDF file"
            }
            return None
        
        # Update status to completed
        translation_progress[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Translation completed successfully!"
        }
        print(f"[Background] Task completed: {translation_progress[task_id]}")
        
        return output_pdf_path
        
    except Exception as e:
        print(f"[Background] Error in translation task: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Update status to error
        translation_progress[task_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}"
        }
        print(f"[Background] Task failed: {translation_progress[task_id]}")

@app.get("/download-pdf/{task_id}")
async def download_pdf(task_id: str):
    """
    Endpoint to download the translated PDF.
    """
    # Check if the task is completed
    if task_id not in translation_progress or translation_progress[task_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Translated PDF not found or translation not completed")
    
    # Path to the translated PDF
    pdf_path = os.path.join("static", f"translated_{task_id}.pdf")
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        # Try alternative path in case it was saved there
        pdf_path = os.path.join("translated", f"translated_{task_id}.pdf")
        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail="Translated PDF file not found")
    
    # Return the file as a response
    return FileResponse(
        path=pdf_path,
        filename=f"translated_document.pdf",
        media_type="application/pdf"
    )

# Mount the converted directory to serve the generated files
app.mount("/converted", StaticFiles(directory="converted"), name="converted")

# Mount the uploads directory to serve the uploaded files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
