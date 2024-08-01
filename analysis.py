import pandas as pd
import os
import matplotlib.pyplot as plt
dataset_name = "NHNETtest"

exp_name="_reason_nocategory"
results_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'HM', dataset_name, exp_name)
file = os.path.join(results_folder, "reason_" + dataset_name + exp_name+"_result.tsv")
df_nocat = pd.read_csv(file, sep='\t')

exp_name="_reason_category"
results_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'HM', dataset_name, exp_name)
file = os.path.join(results_folder, "reason_" + dataset_name + exp_name+"_Categoryresult.tsv")
df_cat = pd.read_csv(file, sep='\t')
# for the prompt given a Other category option
# for df_cat, I want to see the distribution of the categories "GPTReasonCategoryLast" (there are 1-12 different categories) for 0, 1 values in "IsHallucination" column
# Create a figure and axis with a larger size
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the data with larger font sizes
df_cat.groupby(["IsHallucination", "GPTReasonCategoryLast"]).size().unstack().plot(kind='bar', stacked=True, ax=ax, fontsize=14)
plt.xticks(rotation=45, ha='right')
# Adjust the legend font size
ax.legend(fontsize=14)
#save plot
plt.savefig(os.path.join(results_folder, "reason_Categoryresult.png"))

'''
both dataframes, I want to add a conlumn to see it it conflict with the hallucination label
 so here, the contradictary cases are ["is supported by", "is actually supported by","is not a hallucination","no hallucination","does not appear to be a hallucination"]
if the "GPTreason", contains none of these words, then GPTjudgedasHallu = 1     
'''
contradictory_cases = ["is supported by", "is actually supported by","is not a hallucination","no hallucination","does not appear to be a hallucination"]
df_nocat["GPTjudgedasHallu"] = df_nocat["GPTreason"].apply(lambda x: 0 if any(word in x for word in contradictory_cases) else 1)
df_cat["GPTjudgedasHallu"] = df_cat["GPTreason"].apply(lambda x: 0 if any(word in x for word in contradictory_cases) else 1)
# split the dataframe into different groups based on "GPTjudgedasCorrect" and "IsHallucination"
# I want to have a stacked bar plot to show the distribution of the categories "GPTReasonCategoryLast" the different groups
# Create a figure and axis with a larger size
fig, ax = plt.subplots(figsize=(12, 8))

# Grouping the DataFrame
grouped = df_cat.groupby(['GPTjudgedasHallu', 'IsHallucination', 'GPTReasonCategoryLast']).size().unstack(fill_value=0)
# Plotting the stacked bar plot
grouped.plot(kind='bar', stacked=True)
plt.xticks(rotation=45, ha='right')
# Adding labels and title
plt.xlabel('Groups')
plt.ylabel('Count')
plt.title('Distribution of GPTReasonCategoryLast in Different Groups')
plt.legend(title='GPTReasonCategoryLast')
plt.savefig(os.path.join(results_folder, "reason_Categoryresult_withGPTjudgeasHallu.png"))
# also save the percentage of the distribution for each group as a table
grouped = grouped.div(grouped.sum(axis=1), axis=0)
grouped.to_csv(os.path.join(results_folder, "reason_Categoryresult_withGPTjudgeasHallu_percentage.tsv"), sep='\t')

# for df_nocat, I want to see the distribution of GPTUnknown 
# group by GPTjudgedasHallu and IsHallucination
fig, ax = plt.subplots(figsize=(12, 8))

# Grouping the DataFrame
grouped = df_nocat.groupby(['GPTjudgedasHallu', 'IsHallucination', 'GPTUnknown']).size().unstack(fill_value=0)
# Plotting the stacked bar plot
grouped.plot(kind='bar', stacked=True)
plt.xticks(rotation=45, ha='right')
# Adding labels and title
plt.xlabel('Groups')
plt.ylabel('Count')
plt.title('Distribution of GPTUnknown in Different Groups')
plt.legend(title='GPTUnknown')
plt.savefig(os.path.join(results_folder, "reason_noCategoryresult_withGPTjudgeasHallu.png"))
# also save the percentage of the distribution for each group as a table
grouped = grouped.div(grouped.sum(axis=1), axis=0)
grouped.to_csv(os.path.join(results_folder, "reason_noCategoryresult_withGPTjudgeasHallu_percentage.tsv"), sep='\t')
