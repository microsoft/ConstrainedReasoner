# SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection

> This repo contains all the codes and data used in our paper.

## Introduction
[slm_llm_hd_new.drawio.pdf](https://github.com/user-attachments/files/16640335/slm_llm_hd_new.drawio.pdf)

Large language models (LLMs) are highly capable but face latency challenges in real-time applications, such as conducting online hallucination detection. To overcome this issue, we propose a novel framework that leverages a small language model  (SLM)  classifier for initial detection, followed by a LLM as constrained reasoner to generate detailed explanations for detected hallucinated content. This study optimizes the real-time interpretable hallucination detection by introducing effective prompting techniques that align LLM-generated explanations with SLM decisions. Empirical experiment results demonstrate its effectiveness, thereby enhancing the overall  user experience.

## Requirements

GPT-4 turbo is called from Azure OpenAI Service (AOAI), so there are implementation of securally calling GPT via AOAI 
 with the key of the resource saved in a key vault. So the user should specify the resource details in aoai_config.json:  resource url as "OPENAI_API_BASE",  the key vault url as "OPENAI_API_KEY_VAULT". It will work after user login in with az login. For audiance calling GPT via other methods, please revise the code, mostly aoaiutil.py.

 

