import os
import glob
import re
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import win32com.client
from contextlib import contextmanager
from bs4 import BeautifulSoup



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@contextmanager
def word_application():
    """Context manager for Word application to ensure proper resource cleanup."""
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    try:
        yield word
    finally:
        try:
            word.Quit()
        except:
            logger.warning("Error while closing Word application")


def load_rtf_file(word_app, file_path: str) -> Optional[str]:
    """Load a single RTF file using Word application."""
    try:
        doc = word_app.Documents.Open(file_path)
        content = doc.Content.Text
        doc.Close(False)
        return content
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def load_files(promet_folder: str, promet_subfolders: List[str], limit: Optional[int] = None) -> List[str]:
    """
    Load RTF files from specified directories and return their contents as a list of strings.
    
    Args:
        promet_folder: Base directory containing RTF files
        promet_subfolders: List of subdirectories to search within promet_folder
        limit: Maximum number of files to load (None for all files)
        
    Returns:
        List of strings containing file contents
    """
    rtf_contents = []
    count = 0
    
    with word_application() as word:
        for subfolder in promet_subfolders:
            folder_path = os.path.join(promet_folder, subfolder)
            if not os.path.exists(folder_path):
                logger.warning(f"Folder {folder_path} does not exist.")
                continue
        
            rtf_files = glob.glob(os.path.join(folder_path, "**", "*.rtf"), recursive=True)
            logger.info(f"Found {len(rtf_files)} RTF files in {subfolder}")

            for i, rtf_file in enumerate(rtf_files):
                logger.info(f"Loading file {i+1}/{len(rtf_files)} in {subfolder}")
                content = load_rtf_file(word, rtf_file)
                
                if content:
                    rtf_contents.append(content)
                    count += 1
                    
                if limit is not None and count >= limit:
                    logger.info(f"Reached file limit of {limit}")
                    break

    logger.info(f"Successfully loaded {len(rtf_contents)} files")
    return rtf_contents


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def extract_date(line: str) -> Optional[str]:
    match = re.search(r'\d{1,2}\. \d{1,2}\. \d{2,4} \d{1,2}\.\d{1,2}', line)
    return match.group(0) if match else None


def files_to_dataframe(files: List[str]) -> pd.DataFrame:
    """
    Convert a list of RTF file contents to a DataFrame with 'Datum' and 'GroundTruth' columns.
    """
    data = []
    
    for i, ground_truth in enumerate(files):
        logger.info(f"Processing file {i+1}/{len(files)}")
        
        # Split by line breaks and clean text
        lines = [clean_text(line) for line in re.split(r'\r', ground_truth)]
        # Remove empty lines
        lines = [line for line in lines if line]
        
        if not lines:
            logger.warning(f"No content found in file {i+1}")
            continue
            
        line_with_date = lines[0]
        result_date_string = extract_date(line_with_date)
        
        if not result_date_string:
            logger.warning(f"Date not found in line: {line_with_date}")
            continue
        
        # Join remaining lines for ground truth text
        result = ' '.join(lines[1:])
        
        # Convert date string to datetime
        try:
            result_date = pd.to_datetime(result_date_string, format='%d. %m. %Y %H.%M')
            data.append({'Datum': result_date, 'GroundTruth': result})
        except ValueError:
            logger.warning(f"Date conversion error for line: {line_with_date}")
    
    df = pd.DataFrame(data)
    logger.info(f"Created DataFrame with {len(df)} entries")
    return df


def html_to_plain_text(html_content):
    """Convert HTML to plain text while preserving structure."""
    if not html_content or not isinstance(html_content, str) or not html_content.strip():
        return ""
    
    # Only process content that might have HTML
    if '<' not in html_content and '>' not in html_content:
        return html_content
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Handle lists specifically
    for ul in soup.find_all('ul'):
        for li in ul.find_all('li'):
            li.insert_before('• ')
            li.append('\n')
    
    # Get text while preserving whitespace structure
    text = soup.get_text(separator=' ')
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text



def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess CSV data file.
    """
    logger.info(f"Loading data from {file_path}")

    # Read CSV file
    df = pd.read_csv(file_path, sep=';', low_memory=False)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    
    columns_to_drop = [ 'LegacyId', 'Operater','TitlePomembnoSLO', 'TitleNesreceSLO', 'TitleZastojiSLO',
       'TitleVremeSLO', 'TitleOvireSLO', 'TitleDeloNaCestiSLO',
       'TitleOpozorilaSLO', 'TitleMednarodneInformacijeSLO',
       'TitleSplosnoSLO']
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # cast date to date
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y %H:%M', errors='coerce')
    
    # Handle missing values
    df = df.fillna('')

    #parse html content to plain text
    text_columns = ['A1', 'B1', 'C1', 'A2', 'B2', 'C2', 'ContentPomembnoSLO', 'ContentNesreceSLO', 'ContentZastojiSLO', 'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO', 'ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO']
    
    for column in text_columns:
        df[column] = df[column].apply(html_to_plain_text)
    
    # Clean whitespace
    df[text_columns] = df[text_columns].replace(r'^\s*$', '', regex=True)
    df[text_columns] = df[text_columns].apply(
        lambda x: x.str.replace(r'\s+', ' ', regex=True).str.strip()
    )
    
    # Specific content cleaning for ContentOvireSLO column
    if 'ContentOvireSLO' in df.columns:
        df['ContentOvireSLO'] = (df['ContentOvireSLO']
                              .str.replace('- na ', 'Na ')
                              .str.replace(r'; - (\w)', lambda m: '. ' + m.group(1).upper(), regex=True)
                              .str.replace('; ', '. '))
    
    # Remove cells with just punctuation
    df[text_columns] = df[text_columns].replace(r'^[!.]$', '', regex=True)
    
    logger.info(f"Finished preprocessing {len(df)} rows")
    return df


def select_matching_lines(ground_truth_df: pd.DataFrame, 
                         data_df: pd.DataFrame, 
                         min_size: int, 
                         sentences_offset: pd.Timedelta) -> List[Dict[str, str]]:
    """
    Match ground truth data with input data based on timestamps.
    
    Args:
        ground_truth_df: DataFrame with ground truth data
        data_df: DataFrame with input data
        min_size: Minimum number of rows to include
        sentences_offset: Time window to look back for matching data
        
    Returns:
        List of dictionaries with matched data
    """
    results = []
    logger.info(f"Matching {len(ground_truth_df)} ground truth entries")
    
    for i, (datum, ground_truth) in enumerate(ground_truth_df.itertuples(False)):
        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(ground_truth_df)} entries")
            
        # Get rows within time window
        offset_rows = data_df[data_df['Datum'].between(datum - sentences_offset, datum)]
        
        # If not enough rows, take the last min_size rows before datum
        if len(offset_rows) < min_size:
            offset_rows = data_df[data_df['Datum'] <= datum].tail(min_size)
            print(f"selecting last {min_size} rows")
        
        # Select text columns
        string_columns = offset_rows.select_dtypes(include=['object']).columns
        
        # Replace duplicate cells with empty strings (keeping only the last occurrence)
        deduped = offset_rows.copy()
        for col in string_columns:
            # Find rows with duplicate values (keeping the last occurrence)
            duplicated = deduped[col].duplicated(keep='last')
            # Replace duplicates with empty string
            deduped.loc[duplicated, col] = ''
        
        # Concatenate text columns
        deduped['Concated'] = deduped[string_columns].apply(
            lambda row: ' '.join(value for value in row.values.astype(str) if value.strip()), 
            axis=1
        )
        
        # Join all rows
        inputs_for_llm = '\n'.join(deduped['Concated'].tolist())
        
        results.append({
            'Input': inputs_for_llm,
            'GroundTruth': ground_truth
        })
    
    logger.info(f"Created {len(results)} matched entries")
    return results


def save_to_json(data: List[Dict], output_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved data to {output_path}")


def main(config: Dict[str, Any]):
    """Main pipeline function."""
    logger.info("Starting data processing pipeline")
    
    # Load and process RTF files
    files = load_files(
        config['promet_folder'], 
        config['promet_subfolders'], 
        limit=config['files_to_load']
    )
    ground_truth_df = files_to_dataframe(files)
    
    # Load and process CSV data
    data_df = load_data(config['csv_file'])
    
    # Match ground truth with input data
    examples = select_matching_lines(
        ground_truth_df, 
        data_df, 
        config['min_size'], 
        config['sentences_offset']
    )
    
    # Save results
    save_to_json(examples, config['output_file'])
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    # Configuration
    config = {
        'promet_folder': r"C:\Users\rozma\Downloads\RTVSlo\RTVSlo\Podatki",
        'promet_subfolders': [r'Promet 2022'],
        'csv_file': r'C:\Users\rozma\Downloads\Podatki - PrometnoPorocilo_2022_2023_2024.csv',
        'save_folder': r".\Data",
        'files_to_load': 100,
        'min_size': 5,
        'sentences_offset': pd.Timedelta(minutes=0),
        'output_file': os.path.join(r".\Data", 'examples.json')
    }
    
    main(config)