from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import logging
import re
import os
import time
from glob import glob
from typing import Dict, List
from sklearn.metrics import classification_report
from pathlib import Path
import pandas as pd
from modules.arguments import OpenaiArguments, create_openai_arguments, MitigationArguments 
from modules.hallucination_mitigator import HdResult
from modules.hd_constants import AllHallucinations, FieldName, HM_constants
from modules.conversion_utils import str2bool
from modules.encounter_loader import EncounterLoader
from modules.hallucination_mitigation_prompt import hallucination_reasoning_prompt
from modules.aoaiutil import AOAIUtil
import http.client
import numpy as np
import ast
from collections import Counter
import json
import logging
import nltk
import os
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from tqdm import tqdm
from typing import Dict, List
from dataclasses import dataclass
import time
import evaluate
from evaluate import load
@dataclass
class ReasonResult:
    sentence_id: str # sentence id in the transcript
    reason: str # why the sentence is hallucination or not
class evaluate_nlg():
    def __init__(self):
        self.meteor = evaluate.load('meteor')
        self.bertscore = load("bertscore")
        self.bleu = evaluate.load("bleu")

    def evaluate(self, predictions, references):
        '''
        predictions: a list of predictions to score. Each prediction should be a string with tokens separated by spaces.    
        references: a list of references (in the case of one reference per prediction), or a list of lists of references (in the case of multiple references per prediction. Each reference should be a string with tokens separated by spaces.
        METEOR, an automatic metric for machine translation evaluation that is based on a generalized concept of unigram matching between the machine-produced translation and human-produced reference translations. Unigrams can be matched based on their surface forms, stemmed forms, and meanings;
        '''
        results = dict()
        results.update(self.meteor.compute(predictions=predictions, references=references))# returns like {'meteor': 0.6368421052631579}
        berts = self.bertscore.compute(predictions=predictions, references=references, lang="en")
        bleus = self.bleu.compute(predictions=predictions, references=references)
        results['BLEU'] = bleus['bleu']
        results['BERT_precision_avg'] = sum(berts['precision'])/len(berts['precision'])
        results['BERT_recall_avg'] = sum(berts['recall'])/len(berts['recall'])
        results['BERT_f1_avg'] = sum(berts['f1'])/len(berts['f1'])
        return results

@dataclass
class HRResult:
    encounter_id: str
    sen_reasons: List[ReasonResult] 

class HDreasoning():
    def __init__(self, config_setting='gpt-4-turbo', category = False):
        ''' category: whether we need to use the category to constrain the reason of hallucination. the parsing and evaluation will be different.'''
        config_file="configs/aoai_config.json"
        openai_args = create_openai_arguments( config_setting, 1, config_file= config_file)
        self._openai_args = openai_args 
        self.aoaiUtil = AOAIUtil(
            config_setting=openai_args.config_setting,
            api_key=self._openai_args.api_key,
            config_file=config_file,
            )
        self._mitigation_args =  MitigationArguments()
        self._prompt_util = hallucination_reasoning_prompt(use_chat_completions = openai_args.use_chat_completions,  category= category)
        self._evaluator = evaluate_nlg()
        self._category = category 
        if self._category:
            # mapping: a dict of category label as key and the category label from human label. We desinged this is to avoid mixed together with other normal text during parsing
            self._category_mapping = {"Hallu_" + str(i+1):  str(i+1) for i in range(12)}
    @staticmethod
    def clean_span(x):
        return " ".join(re.sub("[\\<].*?[\\>]", "", x).split())
           
    @staticmethod
    def get_indexed_sens(hd_results = List[HdResult]) -> str:
        # dedup sentence
        hd_results_dedup = []
        hd_sentences = set()
        for hd_result in hd_results:
            sentence = HDreasoning.clean_span(hd_result.hallucinated_sentence)
            if sentence not in hd_sentences:
                hd_sentences.add(sentence)
                hd_results_dedup.append(hd_result)

        # get gpt instructions 
        final_instructions = ""
        index2senid = {}
        n_item = len(hd_results_dedup)

        for i in range( n_item):
            final_instructions += \
                f"({i}). <<Sentence>>: {hd_results[i].hallucinated_sentence}\n" 
            index2senid[i] = hd_results[i].sentence_id
        return final_instructions, index2senid, n_item

    @staticmethod
    def create_payload(item, promptUtil) -> Dict:
        GPT_OUTPUT_LENGTH_EXPECTATION = 4096
        prompt_to_send_to_gpt = promptUtil.create_prompt(
            transcript = item['transcript'], 
            sentences = item['sentences'], 
            max_tokens = GPT_OUTPUT_LENGTH_EXPECTATION,
            )
        return {'prompt': prompt_to_send_to_gpt, 'item': item}
    # send payload to GPT endpoint and get back the results
    @staticmethod
    def process_payload_by_GPT(payload, aoaiUtil : AOAIUtil, openai_args : OpenaiArguments, mitigation_args : MitigationArguments) -> Dict:

        outputs = []
        try:
            logging.info(f"Start to call GPT to process the encounter")
            if openai_args.use_chat_completions:
                gpt_response = aoaiUtil.get_chat_completion(
                    messages = payload['prompt'],
                    temperature = mitigation_args.temp,
                    top_p = mitigation_args.top_p, 
                    max_tokens = mitigation_args.max_tokens,
                    frequency_penalty = mitigation_args.freq_penalty,
                    presence_penalty = mitigation_args.presence_penalty,
                    generations=mitigation_args.generations)
                choices = gpt_response['choices']
                for choice in choices:
                    outputs.append(choice['message']['content'])
                payload['gpt_raw_output'] = outputs
            else:
                gpt_response = aoaiUtil.get_completion(
                    prompt = payload['prompt'],
                    max_tokens = mitigation_args.max_tokens,
                    temperature = mitigation_args.temp,
                    top_p = mitigation_args.top_p,
                    frequency_penalty = mitigation_args.freq_penalty,
                    presence_penalty = mitigation_args.presence_penalty,
                    logprobs = mitigation_args.log_prob,
                    generations=mitigation_args.generations)
                choices = gpt_response['choices']
                for choice in choices:
                    outputs.append(choice['text'])
                payload['gpt_raw_output'] = outputs
            logging.info(f"Completed calling GPT to process the encounter")
        except Exception as exc:
            logging.warning(f"Failed to call GPT: output format wrong!")
            logging.warning(f'Exception: {exc}')
            payload['gpt_raw_output'] = [ 'the format of gpt output is wrong' ]

        return payload

    def find_reasons(self,
            encounter_ids: List[str],
            transcripts: Dict[str, str],
            hd_results: Dict[str, List[HdResult]],
            ) -> List[HRResult]:
        max_parallelism = self._openai_args.max_parallelism
        items, results = [], []
        first_key = next(iter(transcripts))
        if isinstance(first_key, int):
            if isinstance(encounter_ids[0], str):
                # we need to convert the encid from hyp to int
                change = 1
        elif isinstance(first_key, str): # trans key is string
            if isinstance(encounter_ids[0], int):
                change = 2 # we need to convert the encid from hyp to str  
            elif isinstance(encounter_ids[0], str):
                change = 0 # no action needed
    
        for encounter_id in encounter_ids:
            try:
                hd_result = hd_results[encounter_id]
                transcript = transcripts[encounter_id]               
            except KeyError:
                import pdb; pdb.set_trace()

            if len(hd_result) == 0:
                results.append(HRResult(encounter_id, []))
                continue
            
            sentences, index2senid, n_item = HDreasoning.get_indexed_sens(hd_result)
            # sending one request per encounter
            request = {
                'encounter_id': encounter_id,
                'transcript': transcript,
                'sentences': sentences,
                'index2senid': index2senid,
                'n_item': n_item
                }
            items.append(request)
            
        gpt_request_payloads = [
            self.create_payload(
                item = items[i],
                promptUtil = self._prompt_util,
            )
            for i in range(len(items))
        ]
        if len(gpt_request_payloads) > 0:
            gpt_results_raw = list()
            max_workers = min(max(max_parallelism, 1), len(gpt_request_payloads))
            with tqdm(total=len(gpt_request_payloads)) as pbar:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            HDreasoning.process_payload_by_GPT,
                            payload,
                            self.aoaiUtil,
                            self._openai_args,
                            self._mitigation_args): payload
                        for payload in gpt_request_payloads
                    }
                    
                    for future in as_completed(futures):
                        gpt_results_raw.append(future.result())
                        pbar.update(1)
            results += HDreasoning.parse_gpt_result(gpt_results_raw )
        return results

    @staticmethod   
    def parse_gpt_items(gpt_out: str, n_item: int, index2senid: Dict[int, int] ):
        gpt_out_noprefix =  gpt_out
        ans = []
        if n_item == 1:
            try:
                # sometimes it will forget the index when there is only one item
                no = '(0).'
                sen_id = index2senid[0] 
                q_out = gpt_out_noprefix.replace(no,'').strip()
                item_result= ReasonResult(sen_id, q_out)
            except BaseException:
                logging.error(f'Unexpected parsing error seen !! ExpectedItemCount:{n_item}\nIter:{i}\n<GPT_OUTPUT>\n{gpt_out}\n</GPT_OUTPUT>')
                item_result=ReasonResult(sen_id, f'PARSE ERROR SEEN!!! {gpt_out}' )
            ans.append(item_result)
        else:
            for i in range(n_item):
                no = '(' + str(i) + ').'
                next_no = '(' + str(i + 1) + ').'
                sen_id = index2senid[i]   
                parse_successful = True
                q_out = ''
                try:
                    if i != (n_item - 1):  # not the last
                        q_out = gpt_out_noprefix.split(no)[1].strip().split(next_no)[0].strip()
                    else:
                        q_out = gpt_out_noprefix.split(no)[1].strip()
                except BaseException:
                    parse_successful = False

                if parse_successful:
                    item_result= ReasonResult(sen_id, q_out)
                else:
                    logging.error(f'Unexpected parsing error seen !! ExpectedItemCount:{n_item}\nIter:{i}\n<GPT_OUTPUT>\n{gpt_out}\n</GPT_OUTPUT>')
                    item_result=ReasonResult(sen_id, f'PARSE ERROR SEEN!!! {gpt_out}' )
                ans.append(item_result)
        return ans

    # parse gpt result per encounter
    @staticmethod
    def parse_gpt_result(gpt_results_raw, outputreason = False) -> List[HRResult]:
        results = []
        for gpt_result_raw in gpt_results_raw:
            gpt_raw_output = gpt_result_raw["gpt_raw_output"][0]
            results.append(
                HRResult(
                    encounter_id = gpt_result_raw['item']['encounter_id'],
                    sen_reasons = HDreasoning.parse_gpt_items(gpt_raw_output,gpt_result_raw['item']['n_item'], gpt_result_raw['item']['index2senid'] ),
                )
            )
        return results

    @staticmethod
    def parse_IsNeutral( reason):
        if "PARTIAL NEUTRAL" in reason:
            return 0
        elif "NEUTRAL" in reason:
            return 1
        else:
            return 0
    @staticmethod
    def parse_unknown( reason):
        if "UNKNOWN" in reason:
            return 1
        else:
            return 0
    @staticmethod
    def parse_HC( reason, categories):
        # Regex pattern to match 'Hallu' followed by one or two digits, but not more
        # need to use Regex rather than string match because Hallu12 will be matched with Hallu1 too
        pattern = r'Hallu_\d{1,2}\b'
        # Find all matches in the text
        all_categories = re.findall(pattern, reason)
        allcat = [    categories[category]   for category in categories.keys() if category in all_categories]
        cat = allcat[0]
        if len(allcat) == 0:
            import pdb; pdb.set_trace()
        return cat, allcat
    @staticmethod
    def match_categories( GTs, Preds):
        '''the GTs is a list of lists of the hallucination categories from different annotators, the preds is the list of lists of the predicted categories
        check the ratio of preds in GTs
        Then take the average for the whole data set.
        '''
        pers = []
        for i, GTlist in enumerate(GTs):
            Predlist = Preds[i]
            # Calculate the percentage of values in Predlist that are in GTlist
            matching_values_count = sum(1 for value in Predlist if value in GTlist)
            percentage_matching = (matching_values_count / len(Predlist))
            pers.append(percentage_matching)
        return np.mean(pers)
    @staticmethod
    def get_first_majority_vote(values_list):
        # Count the occurrences of each value in the list
        counts = Counter(values_list)
        # Find the maximum occurrence
        max_count = max(counts.values())
        # Get all values with the maximum occurrence
        top_votes = [value for value, count in counts.items() if count == max_count]
        # Return the first one from the top votes in case of a tie
        return  top_votes[0]
    
    @staticmethod
    def convert_df( df) -> Dict[str, List[HdResult]]:
        hd_results = {}
        for index, row in df.iterrows():
            encounter_id = row['EncounterID']
            if encounter_id not in hd_results:
                hd_results[encounter_id] = []
            hd_results[encounter_id].append(HdResult(
                sentence_id = row['SentenceID'],
                hallucinated_sentence = row['Sentence']
            ))

        return hd_results
              
    def reason(self, hd_results_file: str, transcript_path: str, dataset_name : str, exp_name : str, onlygt_label = 1, testmode=0  ):
        '''
        hd_results_file: tsv file. we load the hallucinated sentences and the explanation if it has in this file
        transcript_path： where the grounding sources are.
        onlygt_label: if we only load the part of the gt in the hypothesis. 
        If 1, we only load the hallucinated hypothesis. 
        if 0, we load correct hypothesis.
        if 2, we load all hypothesis
        exp_name: the experiment name, used to save the result
        '''
        results_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results', 'HM', dataset_name, exp_name)
        os.makedirs(results_folder, exist_ok=True)
        
        # load hd result
        df = pd.read_csv(hd_results_file, sep='\t')
        if onlygt_label == 1:
            df = df[df['IsHallucination'] == 1]
        elif onlygt_label == 0:
            df = df[df['IsHallucination'] == 0]
        if testmode >0 :
            df = df.head(testmode)
        df['EncounterID'] = df['EncounterID'].astype(str)# in case the encounter id is int when it is all numbers, this will mismatch the str type in the encounter
        hd_results = HDreasoning.convert_df(df)#load to payloads
        # load encounter transcripts and sentences need to be judged.
        encounterloader = EncounterLoader( transcriptfolder=transcript_path)

        start_time = time.time()
        results = self.find_reasons(
        encounter_ids = list(hd_results.keys()),
        transcripts = encounterloader._transcripts,
        hd_results = hd_results, 
        )
        
        end_time = time.time() - start_time
        print("time used for reasoning:" + str(end_time) + "s")

        enc_res = []
        
        if not self._category: # the reason are pure NL.
            for result in results: # the result is a list [ {encounter_id: xx, sen_reasons: [{sentence_id:yy, reason:zz }] ...}  ...]
                for hr_res in result.sen_reasons:
                    res = {}
                    res["EncounterID"]=result.encounter_id
                    res["SentenceID"]=hr_res.sentence_id
                    res["GPTreason"] = hr_res.reason
                    res["GPTNeutral"] = HDreasoning.parse_IsNeutral(hr_res.reason)
                    res["GPTUnknown"] = HDreasoning.parse_unknown(hr_res.reason)
                    enc_res.append(res)
            df_res = pd.DataFrame(enc_res)
            # merge df_res back to df
            df = pd.merge(df, df_res, on=["EncounterID", "SentenceID"], how="left")
            file = os.path.join(results_folder, "reason_" + dataset_name + exp_name+"_result.tsv")
            df.to_csv(file, sep='\t', index=False)
        else:
            for result in results: # the result is a list [ {encounter_id: xx, sen_reasons: [{sentence_id:yy, reason:zz }] ...}  ...]
                for hr_res in result.sen_reasons:
                    res = {}
                    res["EncounterID"]=result.encounter_id
                    res["SentenceID"]=hr_res.sentence_id
                    res["GPTreason"] = hr_res.reason
                    cat, allcat = HDreasoning.parse_HC(hr_res.reason, self._category_mapping)
                    res["GPTReasonCategoryLast"] = cat
                    res["GPTReasonCategoryAll"] = allcat
                    enc_res.append(res)
            df_res = pd.DataFrame(enc_res)
            df = pd.merge(df, df_res, on=["EncounterID", "SentenceID"], how="left")
            file = os.path.join(results_folder, "reason_" + dataset_name + exp_name+"_Categoryresult.tsv")
            df.to_csv(file, sep='\t', index=False)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='downstream reason')
    parser.add_argument('--groundingsource',  type=str, help='the folder where the grounding sources are')
    parser.add_argument('--hyp' ,  type=str, help='the tsv file contains "EncounterID", "SentenceID", "Sentence","IsHallucination"')  
    parser.add_argument('--dataname' ,  type=str,  help='data name for this run')
    parser.add_argument('--category' ,  type=str2bool, default=False, help='For the two reason prompt, whether we need to use the one with detailed categories')
    parser.add_argument('--testmode' ,  type=int, default=0, help='If 0, we load all the data. If >0, we only load the first testmode number of data')
    args = parser.parse_args()
    Hreasonor = HDreasoning(config_setting='gpt-4-turbo', category = args.category)
    result_df = []
    if args.category:
        name = ""
    else:
        name ="no"
    Hreasonor.reason(args.hyp, args.groundingsource, args.dataname ,"_reason_" + name+ "category", onlygt_label = 2, testmode=args.testmode)


  


