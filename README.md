# SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection

> This repo contains all the codes and data used in our paper. [TO DO: add paper link]

## Introduction

![slm_llm_hd_new drawio](https://github.com/user-attachments/assets/0d019a45-6a81-42ab-b352-df60c47b8aec)

Large language models (LLMs) are highly capable but face latency challenges in real-time applications, such as conducting online hallucination detection. To overcome this issue, we propose a novel framework that leverages a small language model  (SLM)  classifier for initial detection, followed by a LLM as constrained reasoner to generate detailed explanations for detected hallucinated content. This study optimizes the real-time interpretable hallucination detection by introducing effective prompting techniques that align LLM-generated explanations with SLM decisions. Empirical experiment results demonstrate its effectiveness, thereby enhancing the overall  user experience.

## Requirements
1. a requirement.txt - [ToDo: Ray]

2. GPT-4 turbo is called from Azure OpenAI Service (AOAI), so there are implementation of securally calling GPT via AOAI 
 with the key of the resource saved in a key vault. So the user should specify the resource details in aoai_config.json:  resource url as "OPENAI_API_BASE",  the key vault url as "OPENAI_API_KEY_VAULT". It will work after user login in with az login. For audiance calling GPT via other methods, please revise the code, mostly aoaiutil.py.

## Main Script

1. **Run Constrained Reasoner**:
   - Execute the `reason_analysis.sh` bash script.
   - Specify the paths for grounding sources, the hypothesis file, a data name for tracking, and the test mode (set to `0` to run all data, or `n > 0` to sample `n` hypotheses).
   - This will run all three approaches and save the results in the `results` folder.

2. **Human Review**:
   - Review the output reasons and determine whether each reason explains hallucination or not.

3. **Run Analysis**:
   - [ToDo: Ray]

## Data

- **Original Data**:
  - Located in the `data` folder.
  - Each dataset contains a `groundingsources` folder and a hypothesis file.
  - The `groundingsources` folder includes all grounding source files, named as `EncounterID.txt`.
  - The hypothesis file contains the following columns:
    - `EncounterID`: It is used to match the grounding source files in the folder to the corresponding hypotheses in the hypothesis file.
    - `SentenceID`: Index of the hypotheses within the same encounter.
    - `Sentence`: The hypotheses to be judged.
    - `IsHallucination`: Ground-truth indicating whether the hypothesis is hallucinated (`1` for hallucination).

- **Results**:
  - Algorithm results are saved as TSV files in the `results` folder. Column "GPTreason" is the generated reasons. 
  - Human reviewers judge the real intention of the generated reasons in the `GPTJudgement` column. "1" means the output really explains hallucination, while "0" means the constrained reasoner disagrees with the upstream decision and gives reasons why the text is a non-hallucination.
  - Final labeled files are located in the `results/labelled` folder.

## Citation
[TO Do]
