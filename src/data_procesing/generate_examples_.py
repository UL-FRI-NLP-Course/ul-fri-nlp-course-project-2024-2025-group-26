

#V poročilu se pojavijo naslednji html tagi:  ['li', 'u', 'strong', 'a', 'p', 'br', 'ul', 'em']
import polars as pl
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from typing import List, Dict, Tuple, Optional

# Assuming create_train_dev_test_split.py contains get_split function
from create_train_dev_test_split import get_split

# --- Configuration Constants ---
SBERT_MODEL_NAME: str = 'EMBEDDIA/sloberta'
SIMILARITY_THRESHOLD: float = 0.99
K_ROWS_BACK: int = 1000  # Number of historical traffic reports to consider
EXAMPLES_TO_PROCESS = 3  # Number of RTF examples to process

# File Paths
RTF_DATA_PATH: str = r".\Data\rtf_data.json"
PROMETNO_POROCILO_PATHS: List[str] = [
    r".\Data\PrometnoPorocilo_2022.csv",
    r".\Data\PrometnoPorocilo_2023.csv",
    r".\Data\PrometnoPorocilo_2024.csv"
]
OUTPUT_FILENAME: str = f".\\Data\\examples_{K_ROWS_BACK}_vrstic_{SBERT_MODEL_NAME.split('/')[1]}{int(SIMILARITY_THRESHOLD*100)}_for_finetune.json"

# Column Names
# Columns to select from traffic reports and also used as keys for HTML content
HTML_PROCESSING_COLUMNS: List[str] = ['LegacyId',
    'A1', 'ContentPomembnoSLO', 'ContentNesreceSLO', 'ContentZastojiSLO',
    'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO',
    'ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO'
]
# Headers for the final formatted input, maintaining order
# Typically the same as HTML_PROCESSING_COLUMNS or a subset/reordered version
FINAL_INPUT_HEADERS: List[str] = HTML_PROCESSING_COLUMNS[:]

# CSV Parsing Options
NULL_VALUES_FOR_CSV: List[str] = ["", "NULL", "NaN", "N/A", "NA", "null", "nan", "null"]

# Text Processing
# Simple, exact strings to ignore after stripping whitespace from paragraphs
STRINGS_TO_IGNORE_IN_PARAGRAPHS: set[str] = {".", "!", ""}


# --- Global Model Initialization ---
try:
    sbert_model: Optional[SentenceTransformer] = SentenceTransformer(SBERT_MODEL_NAME)
    print(f"SentenceTransformer model '{SBERT_MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Error loading SentenceTransformer model '{SBERT_MODEL_NAME}': {e}")
    sbert_model = None

# --- Helper Functions ---

def load_traffic_report_csv(path: str) -> pl.DataFrame:
    """Loads a traffic report CSV file into a Polars DataFrame."""
    df = pl.read_csv(
        path,
        has_header=True,
        separator=";",
        encoding="utf8",
        null_values=NULL_VALUES_FOR_CSV,
        # try_parse_dates=True, # We will parse manually for robustness
        ignore_errors=False
    )
    
    df = df.with_columns(
        pl.col("Datum").str.strptime(pl.Datetime, format="%d/%m/%Y %H:%M", strict=False).alias("Datum")
    )
    return df.select(['Datum'] + HTML_PROCESSING_COLUMNS)



def deduplicate_sentences_sbert(
    sentences: List[str],
    model: Optional[SentenceTransformer],
    threshold: float = SIMILARITY_THRESHOLD
) -> Tuple[List[str], List[str]]:
    """
    Deduplicates a list of sentences based on semantic similarity using SBERT.
    Returns a tuple of (unique_sentences, removed_sentences).
    """
    if not sentences:
        return [], []

    processed_sentences = [s for s in sentences if isinstance(s, str) and s.strip()]
    if not processed_sentences:
        return [], []
    if len(processed_sentences) == 1:
        return processed_sentences, []

    if model is None:
        print("Warning: SBERT model not available. Skipping deduplication, returning all processed sentences.")
        return processed_sentences, []

    embeddings = model.encode(processed_sentences, convert_to_tensor=False, show_progress_bar=False)

    unique_indices: List[int] = []
    removed_indices: List[int] = []
    is_duplicate: List[bool] = [False] * len(processed_sentences)

    for i in range(len(processed_sentences)):
        if is_duplicate[i]:
            removed_indices.append(i)
            continue
        unique_indices.append(i)
        for j in range(i + 1, len(processed_sentences)):
            if is_duplicate[j]:
                continue
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]
            if sim > threshold:
                is_duplicate[j] = True
                
    unique_sents = [processed_sentences[i] for i in unique_indices]
    removed_sents = [processed_sentences[i] for i in removed_indices]
    return unique_sents, removed_sents


def parse_html_columns_to_soup(
    eager_df: pl.DataFrame,
    columns_to_parse: List[str]
) -> Dict[str, List[Optional[BeautifulSoup]]]:
    """
    Parses HTML content from specified columns of an eager Polars DataFrame.
    Cleans <a>, <strong>, and <u> tags.
    """
    soup_documents: Dict[str, List[Optional[BeautifulSoup]]] = {col: [] for col in columns_to_parse}

    
    for col_name in columns_to_parse:
        if col_name in eager_df.columns:
            for cell_content in eager_df[col_name]:
                if cell_content is not None and isinstance(cell_content, str):
                    soup = BeautifulSoup(cell_content, "html.parser")
                    for tag_type, action in [('a', 'decompose'), ('strong', 'unwrap'), ('u', 'unwrap')]:
                        for tag in soup.find_all(tag_type):
                            getattr(tag, action)()
                    soup_documents[col_name].append(soup)
                else:
                    soup_documents[col_name].append(None)
        else:
            print(f"Warning: Column '{col_name}' not found in DataFrame for HTML parsing.")
            soup_documents[col_name] = [None] * eager_df.height # Maintain structure         
    
    return soup_documents


def create_structured_input_from_soup(
    html_docs_by_column: Dict[str, List[Optional[BeautifulSoup]]],
    sbert_model_instance: Optional[SentenceTransformer],
    ordered_headers: List[str]
) -> str:
    """
    Creates a structured text input from parsed HTML documents.
    Extracts paragraphs, deduplicates them, and formats with headers in specified order.
    """
    extracted_paragraphs_map: Dict[str, List[str]] = {header: [] for header in ordered_headers}

    for header, soup_list in html_docs_by_column.items():
        if header not in ordered_headers: # Process only relevant headers
            continue
        if soup_list:
            for soup in soup_list:
                if soup:
                    for p_tag in soup.find_all('p'):
                        # Use separator=" " to ensure spaces between text from different inner tags
                        text = p_tag.get_text(separator=" ", strip=True) 
                        if text and text not in STRINGS_TO_IGNORE_IN_PARAGRAPHS:
                            extracted_paragraphs_map[header].append(text)
    
    final_paragraphs_map: Dict[str, List[str]] = {}
    for header, p_list in extracted_paragraphs_map.items():
        if len(p_list) > 1:
            p_list = list(set(p_list))
            unique_paragraphs, _ = deduplicate_sentences_sbert(p_list, sbert_model_instance)
            final_paragraphs_map[header] = unique_paragraphs
        else: # 0 or 1 paragraph, already unique or empty
            final_paragraphs_map[header] = p_list
            
    output_lines: List[str] = []
    for header_key in ordered_headers:
        p_list_final = final_paragraphs_map.get(header_key, [])
        if p_list_final:
            display_header = header_key.removeprefix("Content").removesuffix("SLO")
            # Keep A1, B1 as is, otherwise use the cleaned name
            display_header = header_key if header_key in ["A1", "B1"] else display_header
            output_lines.append(f"#{display_header}")
            output_lines.extend(p_list_final)
            
    return "\n".join(output_lines)

# --- Main Execution Logic ---

def process_data():
    """Main script execution flow."""
    if sbert_model is None:
        print("SBERT model could not be loaded. Aborting processing.")
        return

    # Load and combine traffic reports
    data_frames = [load_traffic_report_csv(path) for path in PROMETNO_POROCILO_PATHS]
    all_traffic_reports = pl.concat(
        data_frames, how="vertical_relaxed"
    ).sort('Datum', descending=True)
    
    try:
        train_data, _, _ = get_split(RTF_DATA_PATH) # Assuming dev and test are not used here
    except Exception as e:
        print(f"Error loading or splitting RTF data from '{RTF_DATA_PATH}': {e}")
        return
        
    # Process a subset or all of the RTF data
    rtf_df = pl.DataFrame(train_data).slice(100,EXAMPLES_TO_PROCESS)
    
    rtf_df = rtf_df.with_columns(
        pl.col('date').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S", strict=False)
    )
    
    results: List[Dict[str, str]] = []

    for row_index, rtf_row in enumerate(rtf_df.iter_rows(named=True)):
        current_date = rtf_row['date']
        print(f"Processing RTF example {len(results)}/{EXAMPLES_TO_PROCESS}, from {current_date}", end = "\r")

        
        reports_filtered = all_traffic_reports.filter(
            pl.col('Datum') <= current_date + pl.duration(minutes=1)
        ).head(K_ROWS_BACK + 1)
        
        
        input_text = ""
        if reports_filtered.height > 0:
            bs_docs = parse_html_columns_to_soup(reports_filtered, HTML_PROCESSING_COLUMNS)
            input_text = create_structured_input_from_soup(bs_docs, sbert_model, FINAL_INPUT_HEADERS)
            input_text = current_date.strftime("%d.%m.%Y %H:%M") + "\n" + input_text
        else:
            print(f"  No traffic reports found up to date {current_date}.")
        
            
        #results.append({"Input": input_text, "GroundTruth": rtf_row['markdown'], "Date": str(current_date), "AllReportsFromCSV": reports_filtered.write_csv()})
        results.append({"Input": input_text, "GroundTruth": rtf_row['markdown'], "Date": str(current_date)})
    
    print(f"Processing RTF example {len(results)}/{EXAMPLES_TO_PROCESS}, from {current_date}")   
    # Save the results
    try:
        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"Processing complete. Output saved to {OUTPUT_FILENAME}")
    except IOError as e:
        print(f"Error saving results to '{OUTPUT_FILENAME}': {e}")

# --- Script Entry Point ---

if __name__ == "__main__":
    process_data()
