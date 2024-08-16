# SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection

> This repo contains all the codes and data used in our paper.

## Introduction

![slm_llm_hd_new drawio](https://github.com/user-attachments/assets/0d019a45-6a81-42ab-b352-df60c47b8aec)

Large language models (LLMs) are highly capable but face latency challenges in real-time applications, such as conducting online hallucination detection. To overcome this issue, we propose a novel framework that leverages a small language model  (SLM)  classifier for initial detection, followed by a LLM as constrained reasoner to generate detailed explanations for detected hallucinated content. This study optimizes the real-time interpretable hallucination detection by introducing effective prompting techniques that align LLM-generated explanations with SLM decisions. Empirical experiment results demonstrate its effectiveness, thereby enhancing the overall  user experience.

## Requirements

GPT-4 turbo is called from Azure OpenAI Service (AOAI), so there are implementation of securally calling GPT via AOAI 
 with the key of the resource saved in a key vault. So the user should specify the resource details in aoai_config.json:  resource url as "OPENAI_API_BASE",  the key vault url as "OPENAI_API_KEY_VAULT". It will work after user login in with az login. For audiance calling GPT via other methods, please revise the code, mostly aoaiutil.py.

## main script
1. Run Constrained Reasoner: You can run the bash file reason_analysis.sh and specify the path for grounding sources, the hypthesis file, a data name to track, testmode ( if 0 to run all data,  n>0 to sample n hypothses to run). In this way, it will ran all the 3 approaches, and save the reasults in the results folder.

2. Human review the output reasons and judge whether each reason is explaning hallucination or not.

3. Run Analysis:

## data
Orignal data are in data folder. Under each data set, there is a folder groundingsources folder and a hypothesis file. groundingsources folder contains all the grounding source files, and the file name is EncounterID.txt. The hypothesis file has columns: EncounterID, which is to match the grounding source files in the folder; SentenceID, which is the index of hypothesis; Sentence are the hypotheses to be judged. Column "IsHallucination" is the ground-truth of whether the hypothesis is hallucinated. Value 1 is hallucination. 
file name folder name.
The result from the algorithm and then added human label in result/labelled. Column "IsHallucination" is the ground-truth of whether the hypothesis is hallucinated. Value 1 is hallucination.
 Column "GPTJudgement"  We asking annotators to careful read148
tk and mark sk whether the reason is explaining149
the hypothesis is hallucination. GPTreason, GPTReasonCategoryLast,GPTReasonCategoryAll,
GPTUnknown
