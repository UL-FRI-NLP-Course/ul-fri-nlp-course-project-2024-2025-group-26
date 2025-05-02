import pandas as pd
import logging
from bs4 import BeautifulSoup
import re
import json


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def clean_csv_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess CSV data file.
    """
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    
    
    columns_to_drop = [ 'LegacyId', 'Operater','TitlePomembnoSLO', 'TitleNesreceSLO', 'TitleZastojiSLO',
       'TitleVremeSLO', 'TitleOvireSLO', 'TitleDeloNaCestiSLO',
       'TitleOpozorilaSLO', 'TitleMednarodneInformacijeSLO',
       'TitleSplosnoSLO']
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    df['datetime_iso'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y %H:%M')

    # Convert to ISO 8601 string format
    df['Datum'] = df['datetime_iso'].dt.strftime('%Y-%m-%dT%H:%M:%S')
    
    df.drop(columns=['datetime_iso'], inplace=True, errors='ignore')

    
    # Handle missing values
    df = df.fillna('')
    
    logger.info(f"Finished preprocessing {len(df)} rows")
    return df

def csv_files_to_dataframe(csv2022,csv2023,csv2024) -> pd.DataFrame:

    df = pd.read_csv(csv2022, sep=';', low_memory=False)
    df1 = pd.read_csv(csv2023, sep=';', low_memory=False)
    df2 = pd.read_csv(csv2024, sep=';', low_memory=False)
    
    combined_df = pd.concat([df, df1, df2], ignore_index=True)

    return combined_df

def load_csv_files(csv2022,csv2023,csv2024) -> pd.DataFrame:
    """
    Load CSV files and preprocess the data.
    """
    logger.info("Loading CSV files...")
    
    df = csv_files_to_dataframe(csv2022,csv2023,csv2024)
    
    unique_tags = set()
    
    tag_pattern = re.compile(r'<\s*([a-zA-Z][\w:-]*)')

    for xml in df.values.flatten():
        try:
            matches = tag_pattern.findall(str(xml))
            unique_tags.update(matches)
        except Exception:
            continue
        
    print(f"Unique tags found: {unique_tags}")
    
    
    # Clean the data
    df = clean_csv_data(df)
    
    #save the cleaned data to a new json file
    output_file = "csv_data.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict(orient='records'), f, ensure_ascii=False, indent=4)
        
    logger.info("CSV files loaded and preprocessed successfully.")
    
if __name__ == "__main__":
    csv_file_22 = r'C:\Faks\NLP\ul-fri-nlp-course-project-2024-2025-group-26\Data\PrometnoPorocilo_2022.csv'
    csv_file_23 = r'C:\Faks\NLP\ul-fri-nlp-course-project-2024-2025-group-26\Data\PrometnoPorocilo_2023.csv'
    csv_file_24 = r'C:\Faks\NLP\ul-fri-nlp-course-project-2024-2025-group-26\Data\PrometnoPorocilo_2024.csv'
    
    load_csv_files(csv_file_22, csv_file_23, csv_file_24)