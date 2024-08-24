# SLM Meets LLM: Balancing Latency, Interpretability and Consistency in Hallucination Detection

> This repo contains all the codes and data used in our paper. [TO DO: add paper link]

## Introduction

![slm_llm_hd_new drawio](https://github.com/user-attachments/assets/0d019a45-6a81-42ab-b352-df60c47b8aec)

Large language models (LLMs) are highly capable but face latency challenges in real-time applications, such as conducting online hallucination detection. To overcome this issue, we propose a novel framework that leverages a small language model  (SLM)  classifier for initial detection, followed by a LLM as constrained reasoner to generate detailed explanations for detected hallucinated content. This study optimizes the real-time interpretable hallucination detection by introducing effective prompting techniques that align LLM-generated explanations with SLM decisions. Empirical experiment results demonstrate its effectiveness, thereby enhancing the overall  user experience.


### Environment Requirements
Please use the following command install all the required python packages:

```
pip install -r requirements.txt
```

### GPT Resource Requirements:
We leverage [Azure OpenAI Service](https://azure.microsoft.com/en-in/products/ai-services/openai-service/) to conduct the experiment. We use `GPT-4 turbo` as our model deployment, set `temperature=0` and `top_p=0.6`. To Avoid sharing key in the repository, we read the key securally from [Azure Key Vault](https://azure.microsoft.com/en-us/products/key-vault/?ef_id=_k_CjwKCAjw8fu1BhBsEiwAwDrsjBxgU5RzyeDoQQ0Vn6S6vs-3NJCbkwX7BfsoRT-tZrpAyXipuEXU7hoCsXIQAvD_BwE_k_&OCID=AIDcmm5edswduu_SEM__k_CjwKCAjw8fu1BhBsEiwAwDrsjBxgU5RzyeDoQQ0Vn6S6vs-3NJCbkwX7BfsoRT-tZrpAyXipuEXU7hoCsXIQAvD_BwE_k_&gad_source=1&gclid=CjwKCAjw8fu1BhBsEiwAwDrsjBxgU5RzyeDoQQ0Vn6S6vs-3NJCbkwX7BfsoRT-tZrpAyXipuEXU7hoCsXIQAvD_BwE), please make sure to assign yourself a role assignment of `Key Vault Secrets User` in IAM, and save your key into the Key Vault Secrets with a `SECRET_NAME`. Then the user should specify the resource details in [aoai_config.json](configs/aoai_config.json) :  
- resource url as "OPENAI_API_BASE"
- the key vault url as "OPENAI_API_KEY_VAULT". It will work after user login in with az login. 
- key vault secret_name as: "OPENAI_API_KEY_SECRET"
 
For audiance calling GPT via other methods, please revise the code, mostly [aoaiutil.py](modules/aoaiutil.py).


## Main Script

1. **Run Constrained Reasoner**:
   - Execute the `reason_analysis.sh` bash script.
   - Specify the paths for grounding sources, the hypothesis file, a data name for tracking, and the test mode (set to `0` to run all data, or `n > 0` to sample `n` hypotheses).
   - This will run all three approaches and save the results in the `results` folder.

2. **Human Review**:
   - Review the output reasons and determine whether each reason explains hallucination or not.

3. **Run Analysis**:
   - To reproduce the results in the paper, follow the `analyze_reasoning_results.ipynb` notebook.

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
