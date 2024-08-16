# SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection

> This repo contains all the code and data used in our paper.



## Introduction

Large language models (LLMs) are highly capable but face latency challenges in real-time applications, such as conducting online hallucination detection. To overcome this issue, we propose a novel framework that leverages a small language model  (SLM)  classifier for initial detection, followed by a LLM as constrained reasoner to generate detailed explanations for detected hallucinated content. This study optimizes the real-time interpretable hallucination detection by introducing effective prompting techniques that align LLM-generated explanations with SLM decisions. Empirical experiment results demonstrate its effectiveness, thereby enhancing the overall  user experience.

## Requirements

GPT-4 turbo is called from Azure OpenAI Service. 

