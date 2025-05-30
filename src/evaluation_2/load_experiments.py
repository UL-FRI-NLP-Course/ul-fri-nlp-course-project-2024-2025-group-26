import json
import os

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
        results: dictionary to hold the results of the evaluation.
    """
    def __init__(self, data, few_shot_count, model_type, excel_rows, sloberta_treshold):
        self.ground_truth = list(map(lambda x: x["GroundTruth"], data))
        self.model_output = list(map(lambda x: x["ModelOutput"], data))
        self.input = list(map(lambda x: x["Input"], data))
        self.few_shot_count = few_shot_count
        self.model_type = model_type
        self.excel_rows = excel_rows
        self.sloberta_treshold = sloberta_treshold

def load_experiments_v1() -> list[Experiment]:
    
    experiments = []
    
    #v1 of experiments has two files with results and two files with examples, we load them and combine
    r1 = json.load(open(r"./arnes/v1/results/results_1.json", "r", encoding="utf-8"))
    r2 = json.load(open(r"./arnes/v1/results/results_1.json", "r", encoding="utf-8"))
    i1 = json.load(open(r".\Data\examples_1_vrstica.json", "r", encoding="utf-8"))
    i2 = json.load(open(r".\Data\examples_1_vrstica.json", "r", encoding="utf-8"))
    
    #stupid but works
    experiment1_data = []
    for i in range(len(r1)):
    
        input_found = list(filter(lambda x: x["GroundTruth"] == r1[i]["GroundTruth"], i1))[0]
        assert input_found is not None
        experiment1_data.append({
            "Input": input_found["Input"],
            "GroundTruth": r1[i]["GroundTruth"],
            "ModelOutput": r1[i]["ModelOutput"],
        })
    experiments.append(Experiment(
        data=experiment1_data,
        few_shot_count=3,
        model_type="base",
        excel_rows=1,
        sloberta_treshold=1
    ))
        
    experiment2_data = []
    for i in range(len(r2)):
    
        input_found = list(filter(lambda x: x["GroundTruth"] == r2[i]["GroundTruth"], i2))[0]
        assert input_found is not None
        experiment2_data.append({
            "Input": input_found["Input"],
            "GroundTruth": r2[i]["GroundTruth"],
            "ModelOutput": r2[i]["ModelOutput"],
        })
        
    experiments.append(Experiment(
        data=experiment1_data,
        few_shot_count=3,
        model_type="base",
        excel_rows=3,
        sloberta_treshold=1
    ))
    
    return experiments
    
def load_experiments_v2() -> list[Experiment]:
    
    experiments = []
    base_path = r"./arnes/v2/outputs/"

    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_base_results_examples_10_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="base", excel_rows=10, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_base_results_examples_10_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="base", excel_rows=10, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_base_results_examples_10_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="base", excel_rows=10, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_base_results_examples_5_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="base", excel_rows=5, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_base_results_examples_5_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="base", excel_rows=5, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_base_results_examples_5_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="base", excel_rows=5, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_finetuned_results_examples_10_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="finetuned", excel_rows=10, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_finetuned_results_examples_10_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="finetuned", excel_rows=10, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_finetuned_results_examples_10_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="finetuned", excel_rows=10, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_finetuned_results_examples_5_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="finetuned", excel_rows=5, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_finetuned_results_examples_5_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="finetuned", excel_rows=5, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '2_finetuned_results_examples_5_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=2, model_type="finetuned", excel_rows=5, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_base_results_examples_10_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="base", excel_rows=10, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_base_results_examples_10_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="base", excel_rows=10, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_base_results_examples_10_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="base", excel_rows=10, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_base_results_examples_5_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="base", excel_rows=5, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_base_results_examples_5_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="base", excel_rows=5, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_base_results_examples_5_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="base", excel_rows=5, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_finetuned_results_examples_10_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="finetuned", excel_rows=10, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_finetuned_results_examples_10_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="finetuned", excel_rows=10, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_finetuned_results_examples_10_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="finetuned", excel_rows=10, sloberta_treshold=0.99
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_finetuned_results_examples_5_vrstic_sloberta90.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="finetuned", excel_rows=5, sloberta_treshold=0.90
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_finetuned_results_examples_5_vrstic_sloberta95.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="finetuned", excel_rows=5, sloberta_treshold=0.95
    ))
    experiments.append(Experiment(
        data=json.load(open(os.path.join(base_path, '8_finetuned_results_examples_5_vrstic_sloberta99.json'), "r", encoding="utf-8")),
        few_shot_count=8, model_type="finetuned", excel_rows=5, sloberta_treshold=0.99
    ))
    
    return experiments