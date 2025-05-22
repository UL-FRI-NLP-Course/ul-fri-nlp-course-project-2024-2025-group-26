

#V poročilu se pojavijo naslednji html tagi:  ['li', 'u', 'strong', 'a', 'p', 'br', 'ul', 'em']
import polars as pl
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from create_train_dev_test_split import get_split
import cProfile
import pstats
from pstats import SortKey




def load_porocilo(path: str) -> pl.LazyFrame: # Return LazyFrame
    return pl.scan_csv(
        path,
        has_header=True,
        separator=";",
        encoding="utf8",
        null_values=["", "NULL", "NaN", "N/A", "NA", "null", "nan", "null"],
        try_parse_dates=True,
        ignore_errors=False
    ).select([
        'Datum', 'A1', 'B1', 'ContentPomembnoSLO', 'ContentNesreceSLO', 'ContentZastojiSLO',
        'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO',
        'ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO'
    ]) # Removed .collect()
    


def deduplicate_using_SloVo(sentances:list[str], model_name:str = 'paraphrase-multilingual-MiniLM-L12-v2', similarity_threshold:float = 0.90) -> tuple[list[str], list[str]]: # Returns tuple
    if not sentances:
        return [], []

    processed_sentances = [s for s in sentances if isinstance(s, str) and s.strip()]
    if not processed_sentances:
        return [], []

    model = SentenceTransformer(model_name)
    embeddings = model.encode(processed_sentances, convert_to_tensor=False, show_progress_bar=False)

    unique_indices = []
    duplicate_indices = [] # To store indices of removed sentences
    is_duplicate = [False] * len(processed_sentances)

    for i in range(len(processed_sentances)):
        if is_duplicate[i]:
            duplicate_indices.append(i) # Mark as duplicate
            continue

        unique_indices.append(i) 

        for j in range(i + 1, len(processed_sentances)):
            if is_duplicate[j]:
                continue
            
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]

            if sim > similarity_threshold:
                is_duplicate[j] = True
                
    unique_sents = [processed_sentances[i] for i in unique_indices]
    removed_sents = [processed_sentances[i] for i in duplicate_indices] # Collect removed sentences
                
    return unique_sents, removed_sents
    
    
def create_input(csv_documents: dict[str, list[BeautifulSoup]]) -> str: # Corrected type hint
    headers = ['A1','ContentPomembnoSLO', 'ContentNesreceSLO', 'ContentZastojiSLO',
        'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO',
        'ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO']
    
    paragraphs = {header : set() for header in headers}
    
    if csv_documents: # Ensure csv_documents is not None and is a dict
        for header, soup_list in csv_documents.items():
            if soup_list: # Ensure soup_list is not None
                for soup in soup_list:
                    if soup:
                        for p_tag in soup.find_all('p'):
                            paragraphs[header].add(p_tag.get_text(strip=False))
                    
    for header, p_set in paragraphs.items():
        if len(list(p_set)) > 0: 
            unique_paragraphs, _ = deduplicate_using_SloVo(list(p_set)) 
            paragraphs[header] = unique_paragraphs 
        else:
            paragraphs[header] = [] 
        
    output_lines = []
    for header, p_list in paragraphs.items(): 
        if len(p_list) > 0:
            output_lines.append("#" + header.removeprefix("Content").removesuffix("SLO"))
            for p in p_list:
                if p not in [".", " ", "", "\n", "\r\n", "\r", ". ", " .", " . "]:
                    output_lines.append(p)
    return "\n".join(output_lines)   

def get_bs_dict(df:pl.DataFrame) -> dict[str, list[BeautifulSoup]]: # df is an eager DataFrame
    
    string_cols = ['A1', 'ContentPomembnoSLO','ContentNesreceSLO', 'ContentZastojiSLO', 'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO','ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO'
    ]
    
    print(f"Parsing {df.height} rows and {len(string_cols)} columns for HTML content")
    
    soup_documents:dict[str,list[BeautifulSoup]] = {}

    for col in string_cols:
        soup_documents[col] = []
        if col in df.columns: # Check if column exists in the dataframe
            for cell in df[col]: # Iterate over the passed DataFrame 'df'
                if cell is not None:
                    # Ensure cell is a string before passing to BeautifulSoup
                    soup = BeautifulSoup(str(cell), "html.parser") 
                    soup_documents[col].append(soup)
                else:
                    soup_documents[col].append(None)
        else:
            # Handle case where a column might be missing in the filtered df
            # Or ensure all selected columns in load_porocilo are always present
            print(f"Warning: Column {col} not found in DataFrame for HTML parsing.")


    for col, soup_list in soup_documents.items():
        for i, soup in enumerate(soup_list):
            if soup:
                for a_tag in soup.find_all('a'):
                    a_tag.decompose()
                for strong_tag in soup.find_all('strong'):
                    strong_tag.unwrap() # unwrap is generally preferred over replace_with get_text
                for u_tag in soup.find_all('u'):
                    u_tag.unwrap() # unwrap is generally preferred
                    
    return soup_documents


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Load data lazily
    pp2022_lazy = load_porocilo(r".\Data\PrometnoPorocilo_2022.csv")
    pp2023_lazy = load_porocilo(r".\Data\PrometnoPorocilo_2023.csv")
    pp2024_lazy = load_porocilo(r".\Data\PrometnoPorocilo_2024.csv")

    # Combine lazy frames
    # Use how="vertical_relaxed" if schemas might slightly differ, or "vertical" if identical
    prometna_porocila_lazy = pl.concat(
        [pp2022_lazy, pp2023_lazy, pp2024_lazy], 
        how="vertical_relaxed" 
    )
    
    rtf_path = r".\Data\rtf_data.json"
    train, dev, test = get_split(rtf_path)
    
    # Using a small subset for rtf_data as in original code
    rtf_data = pl.DataFrame(train).head(3) 
    rtf_data = rtf_data.with_columns(pl.col('date').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S"))
    
    k_rows_back = 5
    res = []
    for row_index, row in enumerate(rtf_data.iter_rows(named=True)): # Added index for logging
        print(f"Processing RTF entry {row_index + 1}/{rtf_data.height} with date: {row['date']}")
        date = row['date']
        
        reports_lazy = prometna_porocila_lazy.filter(
            pl.col('Datum') <= date
        ).sort(
            'Datum', descending=True
        ).head(k_rows_back + 1)
        
        
        # Collect only the required subset of data
        print(f"Collecting filtered traffic reports for date {date}...")
        reports_eager = reports_lazy.collect() 
        print(f"Collected {reports_eager.height} traffic reports.")
        
        if reports_eager.height > 0:
            bs_dict = get_bs_dict(reports_eager) 
            input_text = create_input(bs_dict) # Renamed variable
            
            res.append({"Input": input_text, "GroundTruth": row['markdown']})
        else:
            print(f"No traffic reports found for date {date} in the specified range.")
            
    # Save the results to a JSON file
    output_filename = r".\Data\examples_3vrstive_brez.json" # Using a distinct name
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
          
    print(f"Processing complete. Output saved to {output_filename}")
    
    stats = pstats.Stats(profiler).sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20) # Print top 20 functions by cumulative time
    