import pandas as pd
import re
import os
import glob
import re
import win32com.client  # Required for RTF reading via Word
import json




def load_files(promet_folder:str, promet_subfolders:list[str], limit:int=None)->list[str]:
    """
    Load RTF files from a specified directory and return their contents as a list of strings."""
    
    
    rtf_contents = []

    # Start Word
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    
    count = 0
    for subfolder in promet_subfolders:
        
        folder_path = os.path.join(promet_folder, subfolder)
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist.")
            return []
    
        rtf_files = glob.glob(os.path.join(folder_path, "**", "*.rtf"), recursive=True)

        for i, rtf_file in enumerate(rtf_files):
            count += 1
            print(f"Loading file {i+1}/{len(rtf_files)} in {subfolder}", end='\r')
            try:
                doc = word.Documents.Open(rtf_file)
                doc_content = doc.Content.Text
                doc.Close(False)
                rtf_contents.append(doc_content)
                
                if limit is not None and count >= limit:
                    break
            except Exception as e:
                print(f"Error reading file {rtf_file}: {e}")

    # Quit Word
    word.Quit()
    print("")
    print("Loading done!")
    return rtf_contents



def files_to_dataframe(files:list[str])->pd.DataFrame:
    """
    Convert a list of RTF file contents to a DataFrame with 'Datum' and 'GroundTruth' columns.
    Each file's content is split into lines, and the first line is used to extract the date.
    The rest of the lines are concatenated to form the 'GroundTruth' text. Removes new lines, tabs, multiple spaces..."""
    
    gt = pd.DataFrame(columns=['Datum', 'GroundTruth'])
    
    count = 0
    files_count = len(files)

    for ground_truth in files:
        count += 1
        print(f"Processing file {count}/{files_count} ", end='\r')  
        
        #split by line breaks, remove tabs, new lines, multispaces...
        lines = re.split(r'\r', ground_truth)
        lines = map(lambda x: re.sub(r'\s+', ' ', x), lines)
        lines = list(map(lambda x: re.sub(r' +', ' ', x), lines))
        
        #remove empty lines
        lines = list(filter(lambda x: x.strip() != '', lines))
        
        line_with_date = lines[0]
        
        relavant_lines = lines[1:]
        
        result_date_string = re.search(r'\d{1,2}\. \d{1,2}\. \d{2,4} \d{1,2}\.\d{1,2}', line_with_date)

        if result_date_string is None:
            print(f"Date not found in line: {line_with_date}")
            continue
        
        result_date_string = result_date_string.group(0)
        
        result =  ' '.join(relavant_lines)
        #try converting the date to datetime
        try:
            result_date = pd.to_datetime(result_date_string, format='%d. %m. %Y %H.%M')
            gt.loc[len(gt)] = [result_date, result]
        except ValueError:
            print(f"Date conversion error for line: {line_with_date}")
            continue
    print("")
    print(f"Processing done!")
    return gt

def load_data(file_path:str)->pd.DataFrame:
    
    #read csv file, Datum column is in format 01/01/2022 00:07 others are in utf-8
    df = pd.read_csv(file_path, sep=';', low_memory=False)

    #replace html tags with space
    df = df.replace(r'<.*?>', ' ', regex=True) 

    #remove all columns that start with Title
    df = df[df.columns.drop(list(df.filter(regex='Title')))]

    #remove columns 'LegacyId', and 'Operater'
    df = df.drop(columns=['LegacyId', 'Operater'])

    #convert columns 'Datum' from 01/01/2022 00:07 to datetime
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d/%m/%Y %H:%M')

    # #replace all nan values with empty string
    df = df.replace('nan', '')

    # #replace all NULL values with empty string
    df = df.replace('NULL', '')

    #replace all None values with empty string
    df = df.fillna('')

    column_names_without_datum_column = df.columns.drop('Datum')

    #replace all cells with only whitespace with empty string
    df[column_names_without_datum_column] = df[column_names_without_datum_column].replace(r'^\s*$', '', regex=True)

    #remove all double triple spaces
    df[column_names_without_datum_column] = df.select_dtypes(include=['object']).apply(lambda x: x.str.replace(r'\s+', ' ', regex=True).str.strip())

    #replace "- na " in ContentOvireSLO with "Na "
    #velike začetnice za npr Na štajerski avtocesti
    df['ContentOvireSLO'] = df['ContentOvireSLO'].str.replace('- na ', 'Na ')

    #replace "; - " and any one letter after it in ContentOvireSLO with ". " and that letter in uppercase
    #nekje se pojavljajo stavki kot naprimer "...v Ljubljani; - na štajerski avtocesti..."
    df['ContentOvireSLO'] = df['ContentOvireSLO'].str.replace(r'; - (\w)', lambda m: '. ' + m.group(1).upper(), regex=True)

    #replace "; " in ContentOvireSLO with ". "
    #še vedno se pojavljajo stavki kot naprimer "...v Ljubljani; Na štajerski avtocesti..."
    df['ContentOvireSLO'] = df['ContentOvireSLO'].str.replace('; ', '. ')

    #nekatere celice imajo samo klicaj ali samo piko, zamenjamo jih z prazno celico
    df[column_names_without_datum_column] = df[column_names_without_datum_column].replace(r'^[!.]$', '', regex=True)
    
    return df


def select_matching_lines(ground_truth_df:pd.DataFrame, data_df:pd.DataFrame, min_size:int, sentances_offset:pd.Timedelta)->list[dict]:
    results = []
   
    for datum, ground_truth in ground_truth_df.itertuples(False):
        
        offset_rows = data_df[data_df['Datum'].between(datum - sentances_offset, datum)]
        
        if offset_rows.shape[0] < min_size:
            offset_rows = (data_df[data_df['Datum'] <= datum]).tail(min_size)
        
        string_column_names = list(offset_rows.select_dtypes(include=['object']).columns)
        
        removed_duplicates = offset_rows.copy()
        for col_nam in string_column_names:
            removed_duplicates.drop_duplicates(subset=[col_nam],keep='last',inplace=True)
            
        removed_duplicates['Concated'] = removed_duplicates[string_column_names].apply(lambda row: ' '.join(value for value in row.values.astype(str) if value.strip()), axis=1)
        
        inputs_for_llm = '\n'.join(removed_duplicates['Concated'].tolist())
        
        results.append({
            'GroundTruth': ground_truth,
            'Input': inputs_for_llm
        })
        
    return results


if __name__ == "__main__":
    
    PROMET_FOLDER = r"C:\Users\rozma\Downloads\RTVSlo\RTVSlo\Podatki"
    PROMET_SUBFOLDERS = [r'Promet 2022']
    CSV_FILE = r'C:\Users\rozma\Downloads\Podatki - PrometnoPorocilo_2022_2023_2024.csv'
    SAVE_FOLDER = r".\Data"

    files_to_load = 100
    min_size = 1
    sentances_offset = pd.Timedelta(minutes=30.0)

    files = load_files(PROMET_FOLDER, PROMET_SUBFOLDERS, limit=files_to_load)
    ground_truth_df = files_to_dataframe(files)

    #load sentances for files
    data_df = load_data(CSV_FILE)
    
    testing_examples = select_matching_lines(ground_truth_df, data_df, min_size, sentances_offset)
    
    #save to json file
    output_file = os.path.join(SAVE_FOLDER, 'examples.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(testing_examples, f, ensure_ascii=False, indent=4)
            
        

    