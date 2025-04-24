import os
import json
import argparse
import nltk
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# Function to compute BLEU score between ground truth and generated output
def compute_bleu(gt, hyp):
    # tokenize (for Slovenian you may need a custom tokenizer; for quick tests whitespace split also works)
    ref_tokens = word_tokenize(gt, language="slovene")
    hyp_tokens = word_tokenize(hyp, language="slovene")

    # compute sentence‑level BLEU with smoothing
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth)
    return bleu


# Main function to handle reading the file, comparing and averaging the BLEU scores
def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Compare Model Output with Ground Truth using BLEU Score."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the JSON file containing 'ModelOutput' and 'GroundTruth' fields.",
    )
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: The file {args.input_file} does not exist.")
        return

    # Load the data from the provided JSON file
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize BLEU score accumulator
    total_bleu = 0
    count = 0

    # Iterate through each entry in the JSON file
    for entry in data:
        # Ensure required fields are present
        if "ModelOutput" in entry and "GroundTruth" in entry:
            model_output = entry["ModelOutput"]
            ground_truth = entry["GroundTruth"]

            # Compute BLEU score for the current pair
            bleu_score = compute_bleu(ground_truth, model_output)
            print(bleu_score)
            total_bleu += bleu_score
            count += 1
        else:
            print(
                "Warning: Missing 'ModelOutput' or 'GroundTruth' in one of the entries."
            )

    # Compute the average BLEU score
    if count > 0:
        avg_bleu = total_bleu / count
        print(f"Average BLEU score: {avg_bleu:.4f}")
    else:
        print("No valid pairs found to calculate BLEU score.")


if __name__ == "__main__":
    nltk.download("punkt_tab")
    main()
