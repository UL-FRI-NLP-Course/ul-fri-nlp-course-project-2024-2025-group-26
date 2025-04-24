# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

This project aims to automate the generation of concise, accurate traffic news for RTV Slovenija. The current process relies on manual reporting every 30 minutes, which is both time-consuming and prone to error. By leveraging state-of-the-art large language models, prompt engineering techniques, and parameter-efficient fine-tuning methods, the project seeks to deliver real-time, high-quality traffic updates.

## Project Folder Structure

```none
.
├── arnes/      # Everything related to running LLMs on ARNES
├── Articles/   # Related works
├── Data/       # Data files
├── Notebooks/  # Relevant notebooks for our project
├── report/
└── src/        # Preprocessing & evaluation code
```

## Evaluation

Once you obtain results by prompting a model on ARNES, you can evaluate the results like so:

```bash
cd src
python -m venv metrics
source metrics/bin/activate  # On Windows: .\metrics\Scripts\activate
pip install nltk
python evaluation.py ../Data/results.json
```

### 📊 BLEU Score Results (GaMS-9B-Instruct)

| Excel Rows per Input | Few-Shot Count | Test Example   | Trials | Avg. BLEU Score |
|----------------------|----------------|----------------|--------|-----------------|
| 1                    | 8              | predict 9th    | 32     | **0.1942**      |
| 3                    | 2              | predict 3rd    | 32     | **0.1576**      |
