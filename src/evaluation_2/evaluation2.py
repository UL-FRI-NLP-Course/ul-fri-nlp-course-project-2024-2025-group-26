import os
import json
import argparse
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from load_experiments import load_experiments_v1, load_experiments_v2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Experiment:
    """Class to hold the data for each experiment.
    Attributes:
        ground_truth (list): List of ground truth sentences.
        model_output (list): List of model output sentences.
        input (list): List of input sentences.
        few_shot_count (int): Number of few-shot examples used.
        model_type (str): Type of model used (e.g., "base", "finetuned").
        excel_rows (int): Number of rows in the Excel file, that were used for generation of input.
        sloberta_treshold (float): Threshold for Sloberta sentance deduplication.
    """
    def __init__(self, data, few_shot_count, model_type, excel_rows, sloberta_treshold):
        self.ground_truth = list(map(lambda x: x["GroundTruth"], data))
        self.model_output = list(map(lambda x: x["ModelOutput"], data))
        self.input = list(map(lambda x: x["Input"], data))
        self.few_shot_count = few_shot_count
        self.model_type = model_type
        self.excel_rows = excel_rows
        self.sloberta_treshold = sloberta_treshold


# Function to compute BLEU score between ground truth and generated output
def compute_bleu(gt, hyp):
    # tokenize (for Slovenian you may need a custom tokenizer; for quick tests whitespace split also works)
    ref_tokens = word_tokenize(gt, language="slovene")
    hyp_tokens = word_tokenize(hyp, language="slovene")

    # compute sentence‑level BLEU with smoothing
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
    return bleu


    

    
    
def get_results(onlyv2=False)->pd.DataFrame:
    if onlyv2:
        experiments = load_experiments_v2()
    else:
        experiments = load_experiments_v1() + load_experiments_v2()

    # Prepare data for Polars DataFrame
    records = []
    for i, exp in enumerate(experiments):
        
        for gt, hyp, inp in zip(exp.ground_truth, exp.model_output, exp.input):
            if gt and hyp:  # Ensure both ground truth and hypothesis are not empty
                records.append({
                    "model_type": exp.model_type,
                    "few_shot_count": exp.few_shot_count,
                    "excel_rows": exp.excel_rows,
                    "sloberta_treshold": exp.sloberta_treshold,
                    "ground_truth": gt,
                    "model_output": hyp,
                    "input_sentence": inp,
                })

    if not records:
        print("No data to process after loading experiments.")
        return

    # Create Polars DataFrame
    df = pd.DataFrame(records)
    
    bleu_scores_list = []
    for row in df.itertuples(index=False):
        bleu_score = compute_bleu(row.ground_truth, row.model_output)
        bleu_scores_list.append(bleu_score)
    
    # Add the BLEU scores as a new column to the DataFrame
    df = df.assign(bleu_score=bleu_scores_list)
    
    #calculate lengths
    df['input_length'] = df['input_sentence'].apply(lambda x: len(x))
    df['ground_truth_length'] = df['ground_truth'].apply(lambda x: len(x))
    df['model_output_length'] = df['model_output'].apply(lambda x: len(x))
    
    df['deviation'] = df['model_output_length'] - df['ground_truth_length']
    
    return df

if __name__ == "__main__":
    nltk.download("punkt")