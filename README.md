# SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection

> This repo contains all the codes and data used in our paper.

## Introduction

![slm_llm_hd_new drawio](https://github.com/user-attachments/assets/0d019a45-6a81-42ab-b352-df60c47b8aec)

Large language models (LLMs) are highly capable but face latency challenges in real-time applications, such as conducting online hallucination detection. To overcome this issue, we propose a novel framework that leverages a small language model  (SLM)  classifier for initial detection, followed by a LLM as constrained reasoner to generate detailed explanations for detected hallucinated content. This study optimizes the real-time interpretable hallucination detection by introducing effective prompting techniques that align LLM-generated explanations with SLM decisions. Empirical experiment results demonstrate its effectiveness, thereby enhancing the overall  user experience.

## Requirements

GPT-4 turbo is called from Azure OpenAI Service (AOAI), so there are implementation of securally calling GPT via AOAI 
 with the key of the resource saved in a key vault. So the user should specify the resource details in aoai_config.json:  resource url as "OPENAI_API_BASE",  the key vault url as "OPENAI_API_KEY_VAULT". It will work after user login in with az login. For audiance calling GPT via other methods, please revise the code, mostly aoaiutil.py.

## main script
1. Run Constrained Reasoner:
Execute the reason_analysis.sh bash script.
Specify the paths for grounding sources, the hypothesis file, a data name for tracking, and the test mode (set to 0 to run all data, or n > 0 to sample n hypotheses).
This will run all three approaches and save the results in the results folder.
2. Human Review:
Review the output reasons and determine whether each reason explains hallucination or not.
3. Run Analysis:
[ToDo: Ray]

## data
1. Original Data:
Located in the data folder.
Each dataset contains a groundingsources folder and a hypothesis file.
The groundingsources folder includes all grounding source files, named as EncounterID.txt.
The hypothesis file contains the following columns:
EncounterID: Matches the grounding source files in the folder.
SentenceID: Index of the hypothesis.
Sentence: The hypotheses to be judged.
IsHallucination: Ground-truth indicating whether the hypothesis is hallucinated (1 for hallucination).
2. Results:
Algorithm results are saved as TSV files in the results folder.
Human reviewers judge whether the generated reasons explain hallucination or non-hallucination in the GPTJudgement column.
Final labeled files are located in the results/labelled folder.
