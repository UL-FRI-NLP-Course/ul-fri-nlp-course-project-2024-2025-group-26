#Creates a train/dev/test split of the data
#The data is split into 80% train, 10% dev, and 10% test
import random
import json
import pandas as pd
random.seed(42)



def get_split(rtf_data:str,train=0.8, dev=0.1, test=0.1):

    with open(rtf_data, 'r', encoding="utf-8") as f:
        all_files = json.load(f)
        random.shuffle(all_files)
    
        # Split the files into train, dev, and test sets
        train_files = all_files[:int(len(all_files)*train)]
        dev_files = all_files[int(len(all_files)*train):int(len(all_files)*(train+dev))]
        test_files = all_files[int(len(all_files)*(train+dev)):]
        
        return train_files, dev_files, test_files

def rtf_data_to_df(rtf_data:list):
    df = pd.DataFrame(rtf_data)
    df['Datum'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')
    df.drop(columns=['date'], inplace=True, errors='ignore')
    return df