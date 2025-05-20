import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any
from create_train_dev_test_split import get_split, rtf_data_to_df
import re
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def rtf_files_to_dataframe(rtf_data: str) -> pd.DataFrame:
    train, dev, test = get_split(rtf_data)
    train_df = rtf_data_to_df(train)
    dev_df = rtf_data_to_df(dev)
    return train_df, dev_df


def csv_files_to_dataframe(csv_data: str) -> pd.DataFrame:
    with open(csv_data, 'r', encoding='utf-8') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        df['Datum'] = pd.to_datetime(df['Datum'], format='%Y-%m-%dT%H:%M:%S')
    return df


def parse_html_to_sections(html: str) -> Dict[str, str]:
    """
    Parse HTML snippet into a mapping: section -> combined text.
    Automatically detect sections by:
      1. Standalone <p><strong>Header</strong></p>
      2. Inline 'Header: content' at start of paragraph
    """
    soup = BeautifulSoup(html, 'html.parser')
    section_texts: Dict[str, List[str]] = {}
    current_section = None

    for p in soup.find_all('p'):
        raw = p.get_text(' ', strip=True)
        # Detect standalone header tag
        strong = p.find('strong')
        if strong and raw == strong.get_text(strip=True):
            current_section = raw
            continue
        # Detect inline header before colon
        m = re.match(r'^(.+?):\s*(.*)$', raw)
        if m:
            # treat group1 as new section
            current_section = m.group(1)
            text = m.group(2)
        else:
            text = raw

        section = current_section or 'General'
        if text:
            section_texts.setdefault(section, []).append(text)

    # Combine texts per section
    combined: Dict[str, str] = {}
    for sec, texts in section_texts.items():
        combined[sec] = ' '.join(texts)
    return combined


def select_matching_lines(ground_truth_df: pd.DataFrame,
                         data_df: pd.DataFrame,
                         min_size: int,
                         sentences_offset: pd.Timedelta,
                         deduplicate_sentences: bool,
                         remove_english: bool) -> List[Dict[str, str]]:
    """
    Match ground truth data with input data based on timestamps,
    parsing HTML content and deduplication without fuzzy matching.
    """
    results = []
    seen_records = set()

    text_columns = data_df.select_dtypes(include=['object']).columns.tolist()

    logger.info(f"Matching {len(ground_truth_df)} ground truth entries")

    if remove_english:
        data_df = data_df.drop(columns=['C1', 'C2'], errors='ignore')

    for i, gt_row in ground_truth_df.iterrows():
        ground_truth = gt_row['markdown']
        datum = gt_row['Datum']
        if i % 10 == 0:
            logger.info(f"Processed {i}/{len(ground_truth_df)} entries")

        # select rows in time window
        window = data_df[data_df['Datum'].between(datum - sentences_offset, datum)]
        if len(window) < min_size:
            window = data_df[data_df['Datum'] <= datum].tail(min_size)

        # Gather section texts over the window
        aggregated_sections: Dict[str, List[str]] = {}
        for _, row in window.iterrows():
            for col in text_columns:
                html = row.get(col, '')
                if isinstance(html, str) and '<p>' in html:
                    sec_map = parse_html_to_sections(html)
                    for sec, txt in sec_map.items():
                        if txt and (sec, txt) not in seen_records:
                            seen_records.add((sec, txt))
                            aggregated_sections.setdefault(sec, []).append(txt)

        # Build unique combined records
        records = []
        for sec, texts in aggregated_sections.items():
            combined_text = ' '.join(texts)
            records.append({'section': sec, 'text': combined_text})

        # Format input string with newline per section
        parts = [f"{r['section']}: {r['text']}" for r in records]
        input_str = '\n'.join(parts)

        results.append({'Input': input_str, 'GroundTruth': ground_truth})

    logger.info(f"Created {len(results)} matched entries")
    return results


def save_to_json(data: List[Dict], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved data to {output_path}")


def main(config: Dict[str, Any]):
    logger.info("Starting data processing pipeline")
    ground_truth_df, _ = rtf_files_to_dataframe(config['rtf_file'])
    csv_df = csv_files_to_dataframe(config['csv_file'])
    ground_truth_df = ground_truth_df.head(config.get('files_to_load', len(ground_truth_df)))

    examples = select_matching_lines(
        ground_truth_df,
        csv_df,
        config['min_size'],
        config['sentences_offset'],
        deduplicate_sentences=True,
        remove_english=True
    )

    save_to_json(examples, config['output_file'])
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    config = {
        'csv_file': r'C:\\Faks\\NLP\\ul-fri-nlp-course-project-2024-2025-group-26\\Data\\csv_data.json',
        'rtf_file': r'C:\\Faks\\NLP\\ul-fri-nlp-course-project-2024-2025-group-26\\Data\\rtf_data.json',
        'files_to_load': 3,
        'min_size': 5,
        'sentences_offset': pd.Timedelta(minutes=0),
        'output_file': os.path.join(r".\\Data", 'examples.json')
    }
    main(config)
