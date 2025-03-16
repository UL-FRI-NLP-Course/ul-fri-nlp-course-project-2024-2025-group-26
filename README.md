# Natural language processing course: `Automatic generation of Slovenian traffic news for RTV Slovenija`

This project aims to automate the generation of concise, accurate traffic news for RTV Slovenija. The current process relies on manual reporting every 30 minutes, which is both time-consuming and prone to error. By leveraging state-of-the-art large language models, prompt engineering techniques, and parameter-efficient fine-tuning methods, the project seeks to deliver real-time, high-quality traffic updates.

## 1. Introduction

The project "Automatic Generation of Slovenian Traffic News for RTV Slovenija (Slavko)" focuses on replacing the manual production of traffic reports with an automated system. Traffic data from the promet.si portal (contained in Podatki - PrometnoPorocilo_2022_2023_2024.xlsx) and detailed guidelines provided in documents like PROMET.docx and PROMET, osnove.docx serve as the backbone for this system. These sources define the precise language, naming conventions, and event priorities required for the output.

### Project Context and Motivation

- **Timely Information**: Reliable traffic updates are critical for public safety and efficient transportation management.
- **Manual Process Limitations**: Currently, students manually verify and type these reports every 30 minutes, leading to potential delays and inaccuracies.
- **Automated Solution**: By utilizing LLMs and advanced prompt engineering (inspired by works on news headline generation), we aim to generate clear, concise, and contextually accurate traffic news.

### Methodological Approach
- **Initial Prompt Engineering**: Experiments will start by generating news texts directly from the traffic data using well-crafted prompts, as demonstrated in recent research.
- **Enhanced Generation through Fine-Tuning**: The project will then incorporate parameter-efficient fine-tuning (e.g., LoRA) and retrieval techniques to further refine the generated content, ensuring correct road naming and event descriptions.
- **Evaluation**: A robust evaluation framework will be established using both automatic metrics (such as ROUGE, precision, recall, and F1) and human judgment to verify that the generated texts meet RTV Slovenija’s standards.
  
## 2. Related work and initial ideas

### Related Work
Recent studies in neural summarization and headline generation provide valuable insights for our project:

- **HG-News: News Headline Generation Based on a Generative Pre-Training Model**: This paper presents a decoder-only architecture that incorporates pointer mechanisms and n-gram language features. Its focus on generating succinct news headlines while accurately handling out-of-vocabulary terms (such as specific road names) can be directly applicable to our task.
- **Algorithm for Automatic Abstract Generation of Russian Text under ChatGPT System**: This work leverages the ChatGPT system to preprocess and generate concise summaries of Russian texts. Although in Russian, Its detailed data preprocessing pipeline (including tokenization, removal of stop words, and punctuation handling) and the integration of pointer mechanisms to capture key information offer a robust framework that can be maybe adapted to Slovene to ensure the fluency and accuracy of traffic news.

### Initial Ideas

Building on these insights, our initial ideas include:

- **Decoder-Only Architecture**:
  Adopt a decoder-only approach to directly generate text from preprocessed traffic data. This strategy simplifies the model while focusing on producing clear and concise outputs.

- **Pointer Mechanisms**:
  Integrate pointer mechanisms to ensure that domain-specific terminology (e.g., road names and traffic event descriptors) is accurately reproduced, minimizing the risk of omitting critical details.

- **Advanced Preprocessing**:
Apply rigorous preprocessing techniques—such as tokenization, normalization, and filtering—to both the traffic data and guideline documents. This will help standardize the input and ensure adherence to established formats.

- **N-gram Language Features**:
Incorporate n-gram features to improve language fluency and ensure that generated sentences are coherent and stylistically consistent with existing RTV Slovenija news.

- **Robust Evaluation Framework**:
Develop an evaluation strategy that combines ROUGE metrics with custom checks (precision, recall, F1 for key elements) to ensure that the generated reports are both informative and accurate.

These ideas draw upon state-of-the-art approaches and are well-suited to addressing the unique challenges of automating Slovenian traffic news generation.

## 3. Proposed Project Dataset and Repository Organization
### 3.1 Proposed Project Dataset / Corpus 

**Data Sources and Content**
**Traffic Data**:
The primary dataset is the Excel file Podatki - PrometnoPorocilo_2022_2023_2024.xlsx, which contains detailed real-time and historical traffic information from the promet.si portal.

**News Texts and Guidelines**:

**RTV Slo News Texts**: These texts serve as style and format references for the traffic reports.
**Instructional Documents**: Files such as PROMET.docx and PROMET, osnove.docx provide the rules for road naming, event hierarchy, and news formulation.
**Supplementary Material**:
Notebooks like Prompting_and_lora_fine-tuning.ipynb, Traditional language modelling and knowledge bases.ipynb, and Retrieval_augmented_generation.ipynb offer insights into prompt engineering and fine-tuning, which will guide our approach.

**Corpus Preparation**
**Data Integration**:
Combine the traffic data with news texts and guidelines into a unified corpus.

**Preprocessing**:
Normalize the data by enforcing standard naming conventions, filtering out irrelevant details, and tagging urgent events (e.g., nujna prometna informacija).

**Quality Assurance**:
Apply rule-based checks to verify that the generated texts meet the required standards for accuracy and style.




