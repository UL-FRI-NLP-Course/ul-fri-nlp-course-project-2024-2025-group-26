import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any
from create_train_dev_test_split import get_split, rtf_data_to_df
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def rtf_files_to_dataframe(rtf_data:str) -> pd.DataFrame:

    train, dev, test = get_split(rtf_data)
        
    train_df = rtf_data_to_df(train)
    dev_df = rtf_data_to_df(dev)

    return train_df, dev_df

def csv_files_to_dataframe(csv_data:str) -> pd.DataFrame:

    with open(csv_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        df = pd.DataFrame(data)
        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%dT%H:%M:%S')

    return df


def select_matching_lines(ground_truth_df: pd.DataFrame, 
                         data_df: pd.DataFrame, 
                         min_size: int, 
                         sentences_offset: pd.Timedelta,
                         deduplicate_sentances:bool,
                         remove_english:bool) -> List[Dict[str, str]]:
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
    
    if remove_english:
        data_df = data_df.drop(columns=['C1','C2'],inplace=False)
    
    for i in range(len(ground_truth_df)):
        ground_truth = ground_truth_df.iloc[i]['markdown']
        datum = ground_truth_df.iloc[i]['Datum']
        
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
        
        if deduplicate_sentances:
            sentances = []
            for cell in deduped[string_columns].values.flatten():
                sentances_of_cell = re.split(r'(?<=[a-z])\. (?=[A-Z])', cell)
                sentances.extend(sentances_of_cell)
        
            # Remove empty strings and duplicates
            sentances = [s.strip() for s in sentances if s.strip()]
        
            deduped = list(set(sentances))
            
            results.append({
                'Input': ' '.join(sentances),
                'GroundTruth': ground_truth
            })
        
        else:
            # Concatenate text columns
            deduped['Concated'] = deduped[string_columns].apply(
                lambda row: ' '.join(value for value in row.values.astype(str) if value.strip()), 
                axis=1
            )
            
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
    
    
    ground_truth_df,_ = rtf_files_to_dataframe(config['rtf_file'])
    csv_files = csv_files_to_dataframe(config['csv_file'])
    
    ground_truth_df = ground_truth_df.head(config['files_to_load'])
    
    
    # Match ground truth with input data
    examples = select_matching_lines(
        ground_truth_df, 
        csv_files, 
        config['min_size'], 
        config['sentences_offset'],
        True,
        True
    )
    
    # Save results
    save_to_json(examples, config['output_file'])
    
    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    # Configuration
    config = {
        'csv_file': r'C:\Faks\NLP\ul-fri-nlp-course-project-2024-2025-group-26\Data\csv_data.json',
        'rtf_file': r'C:\Faks\NLP\ul-fri-nlp-course-project-2024-2025-group-26\Data\rtf_data.json',
        'save_folder': r".\Data",
        'files_to_load': 100,
        'min_size': 5,
        'sentences_offset': pd.Timedelta(minutes=0),
        'output_file': os.path.join(r".\Data", 'examples.json')
    }
    
    main(config)