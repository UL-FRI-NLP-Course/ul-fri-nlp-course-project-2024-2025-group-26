import os
import json
import random
import argparse

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# === Load the model pipeline based on the mode (base or finetuned) ===
def load_pipeline(mode: str):
    base_model_id = "cjvt/GaMS-9B-Instruct"

    if mode == "base":
        return pipeline(
            "text-generation",
            model=base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    elif mode == "finetuned":
        print("🔧 Nalaganje fine-tuned modela ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        adapter_path = os.getenv("ADAPTER_PATH", "/models/finetuned")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        merged_model = model.merge_and_unload()

        return pipeline(
            "text-generation",
            model=merged_model,
            tokenizer=AutoTokenizer.from_pretrained(base_model_id),
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        raise ValueError("⚠️ 'mode' must be either 'base' or 'finetuned'")


def main():
    # === Parse command-line arguments ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "finetuned"], required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    # === Load the model pipeline ===
    pline = load_pipeline(args.mode)

    # === Load input examples from file ===
    with open(args.input_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    few_shot_count = 2
    num_trials = 32

    if len(examples) < few_shot_count + 1:
        raise ValueError("⚠️ Potrebujemo vsaj 3 primere v datoteki!")

    results = []

    # === Main loop: Run multiple trials ===
    for trial_num in range(num_trials):
        print("=" * 100)
        print(f"🔁 Trial {trial_num + 1}/{num_trials} — Few-shot: {few_shot_count}")

        # Randomly sample few-shot examples and a test example
        selected_examples = random.sample(examples, k=few_shot_count + 1)
        few_shot_examples = selected_examples[:-1]
        test_example = selected_examples[-1]

        # Build the prompt from few-shot examples and test input
        prompt_parts = [
            f"Vhod:\n{ex['Input']}\nIzhod:\n{ex['GroundTruth']}\n"
            for ex in few_shot_examples
        ]
        prompt_parts.append(f"Vhod:\n{test_example['Input']}\nIzhod:\n")
        full_prompt = "\n".join(prompt_parts)

        # Construct chat-style message
        message = [{"role": "user", "content": full_prompt}]

        # Generate output
        with torch.no_grad():
            response = pline(
                message,
                max_new_tokens=1024
            )

        # Extract output
        generated_text = response[0]["generated_text"]
        model_output = (
            generated_text[-1]["content"]
            if isinstance(generated_text, list)
            else generated_text
        )

        # Print comparison
        print("\n✅ Pričakovani izhod:")
        print(test_example['GroundTruth'].strip())

        print("\n🤖 Modelov izhod:")
        print(model_output.strip())

        # Save to results
        results.append({
            "ModelOutput": model_output.strip(),
            "GroundTruth": test_example['GroundTruth'].strip(),
            "Input": test_example['Input'].strip()
        })

        print("=" * 100 + "\n")

    # === Save results to output file ===
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
