

#V poročilu se pojavijo naslednji html tagi:  ['li', 'u', 'strong', 'a', 'p', 'br', 'ul', 'em']

import polars as pl
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from create_train_dev_test_split import get_split



def load_porocilo(path: str):
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
    ]).collect()
    


def deduplicate_using_SloVo(sentances:list[str], model_name:str = 'paraphrase-multilingual-MiniLM-L12-v2', similarity_threshold:float = 0.90) -> list[str]:
    if not sentances:
        return []

    # Filter out None values or empty strings if they might be present
    # and ensure all elements are strings
    processed_sentances = [s for s in sentances if isinstance(s, str) and s.strip()]
    if not processed_sentances:
        return []

    # Store original indices to map back if needed, or just work with processed_sentances
    # For simplicity, this implementation returns a subset of the processed_sentances

    model = SentenceTransformer(model_name)
    
    # Encode sentences
    embeddings = model.encode(processed_sentances, convert_to_tensor=False, show_progress_bar=False)

    unique_indices = []
    is_duplicate = [False] * len(processed_sentances)

    for i in range(len(processed_sentances)):
        if is_duplicate[i]:
            continue

        unique_indices.append(i) # Add sentence i as unique

        # Compare sentence i with all subsequent sentences j
        for j in range(i + 1, len(processed_sentances)):
            if is_duplicate[j]:
                continue
            
            # Calculate cosine similarity between embeddings[i] and embeddings[j]
            sim = cosine_similarity(embeddings[i].reshape(1, -1), embeddings[j].reshape(1, -1))[0][0]

            if sim > similarity_threshold:
                is_duplicate[j] = True
                
    return [processed_sentances[i] for i in unique_indices]
    
    
def create_input(csv_documents:list[BeautifulSoup]) -> str:
    headers = ['A1','ContentPomembnoSLO', 'ContentNesreceSLO', 'ContentZastojiSLO',
        'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO',
        'ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO']
    
    paragraphs = {header : set() for header in headers}
    
    for header, soup_list in csv_documents.items():
        for soup in soup_list:
            if soup:
                # Find all <p> tags and add their text to the set
                for p_tag in soup.find_all('p'):
                    paragraphs[header].add(p_tag.get_text(strip=False))
                    
    #deduplicate each set by findig sentance embedding for items and removing duplicates

    for header, p_set in paragraphs.items():
        if len(list(p_set)) > 0: # Convert set to list for deduplicate_using_SloVo
            unique_paragraphs, _ = deduplicate_using_SloVo(list(p_set)) # Unpack returned tuple
            paragraphs[header] = unique_paragraphs # Store unique paragraphs
        else:
            paragraphs[header] = [] # Ensure it's an empty list if no paragraphs
        
    
    output_lines = []
    for header, p_list in paragraphs.items(): # p_list is now a list of unique paragraphs
        if len(p_list) > 0:
            output_lines.append("#" + header.removeprefix("Content").removesuffix("SLO"))
            for p in p_list:
                output_lines.append(p)
    return "\n".join(output_lines)   

def get_bs_dict(df:pl.DataFrame) -> dict[str, list[BeautifulSoup]]:
    
    string_cols = ['A1', 'ContentPomembnoSLO','ContentNesreceSLO', 'ContentZastojiSLO', 'ContentVremeSLO', 'ContentOvireSLO', 'ContentDeloNaCestiSLO', 'ContentOpozorilaSLO','ContentMednarodneInformacijeSLO', 'ContentSplosnoSLO'
    ]
    
    print(f"Parsing {df.height} rows and {len(string_cols)} columns")
    
    # Parse each page once and store the BeautifulSoup objects in a dictionary keyed by column name and row index.
    soup_documents:dict[str,BeautifulSoup] = {}

    for col in string_cols:
        soup_documents[col] = []
        for i, cell in enumerate(prometna_porocila[col]):
            if cell is not None:
                soup = BeautifulSoup(cell, "html.parser")
                soup_documents[col].append(soup)
            else:
                soup_documents[col].append(None)

    #remove tags a and unwrap tags strong and u
    for col, soup_list in soup_documents.items():
        for i, soup in enumerate(soup_list):
            if soup:
                # Remove <a> tags
                for a_tag in soup.find_all('a'):
                    a_tag.decompose()
            #find </strong> and <strong> and replace with ''
                for strong_tag in soup.find_all('strong'):
                    strong_tag.replace_with(strong_tag.get_text(strip=False))
                
                for u_tag in soup.find_all('u'):
                    u_tag.replace_with(u_tag.get_text(strip=False))
                    
    return soup_documents


if __name__ == "__main__":
    
    pp2022 = load_porocilo(r".\Data\PrometnoPorocilo_2022.csv")
    pp2023 = load_porocilo(r".\Data\PrometnoPorocilo_2023.csv")
    pp2024 = load_porocilo(r".\Data\PrometnoPorocilo_2024.csv")

    prometna_porocila = pp2022.vstack(pp2023).vstack(pp2024).rechunk()
    
    
    
    
    rtf_path = r".\Data\rtf_data.json"
    train, dev, test = get_split(rtf_path)
    rtf_data = pl.DataFrame(train).head(3)
    rtf_data = rtf_data.with_columns(pl.col('date').str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S"))
    
    
    #for each file in train, extract the date and find the row in prometna_porocila with the same date and rows 1h before
    offset = pl.duration(hours=1)
    res = []
    for row in rtf_data.iter_rows(named=True):
        date = row['date']
        reports = prometna_porocila.filter(pl.col('Datum').is_between(date - offset, date + pl.duration(minutes=1)))
        
        if reports.height > 0:
            bs_dict = get_bs_dict(reports)
            input = create_input(bs_dict)
            
            res.append({"Input": input, "GroundTruth": row['markdown']})
            
    # Save the results to a JSON file
    with open(r".\Data\examples_1h.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
          
    #create_input(soup_documents)
                    
    