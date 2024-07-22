from glob import glob
import os
from typing import Dict
import pandas as pd
from modules.summarypreprocess import SummaryPreprocess


class EncounterLoader:

    def __init__(self,
                 hypothesis: str,
                 transcriptfolder: str,
                 testmode: int = 0) -> None:
        self._hypothesis = {}
        self._transcripts = {}
        self._hypothesis_preproc_sentences = {}
        self._encounter_ids = list()

        if os.path.isdir(hypothesis):
            self._hypothesis = self.__load_file_inputs(hypothesis, "*.txt")
            self._hypothesis_preproc_sentences = self.hypothesis_preprocess_into_sentences(
                self._hypothesis)
        elif os.path.isfile(hypothesis) and hypothesis.endswith('.tsv'):
            self._hypothesis_preproc_sentences = self.__load_sentencelevel_file(
                hypothesis)  # hypothesis here is sentences
        else:
            raise ValueError(
                '--inputhypothesis is incorrect. not a valid path or not a tsv file.')

        print(f'Hypotheses Found: {len(self._hypothesis_preproc_sentences)}')

        self._transcripts = self.__load_file_inputs(transcriptfolder, "*.txt")
        print(f'Transcript Files Found: {len(self._transcripts)}')

        self._encounter_ids = list(set(self._hypothesis_preproc_sentences.keys(
        )).intersection(set(self._transcripts.keys())))
        print(f'Total Unique Encounters Found: {len(self._encounter_ids)}')

        self.__validate()
        self._encounter_ids = sorted(self._encounter_ids)

        if testmode > 0:
            self._encounter_ids = self._encounter_ids[0:testmode]
            print(
                f'Running reduced dataset of {testmode} encounters: {self._encounter_ids} ...')

    def __load_file_inputs(self, folderpath: str,
                           searchpattern: str = "") -> Dict[str, str]:
        retval = {}
        for fname in glob(os.path.join(folderpath, searchpattern)):
            with open(fname, "r", encoding="utf-8") as f:
                fnamenoext = os.path.splitext(os.path.basename(fname))[0]
                retval[fnamenoext] = f.read()
        return retval

    def __load_sentencelevel_file(self, hypothesisfile) -> dict:
        hypdf = pd.read_csv(hypothesisfile, sep='\t', header=0)
        hypsens = {}
        important_columns = hypdf[["EncounterID", "SentenceID", "Sentence"]]
        important_columns = important_columns.drop_duplicates()
        for index, row in important_columns.iterrows():
            # because the loaded result IDs are int, but transcripts IDs are
            # string
            En_Id = str(row["EncounterID"])
            if not hypsens.__contains__(En_Id):
                hypsens[En_Id] = []
            sen_dict = {
                'sentence_id': row["SentenceID"],
                'text': row["Sentence"]
            }
            hypsens[En_Id].append(sen_dict)
        return hypsens

    def __validate(self) -> None:
        if len(self._encounter_ids) == 0:
            print('=============\n!! ERROR !!\n=============\nThere are no matching encounters.\nTerminating the run..\n=============')
            raise ValueError(
                '--inputhypothesis or --inputtranscripts is incorrect.  We found no matching encounter ids.')

        if len(
            self._encounter_ids) != len(
            self._hypothesis_preproc_sentences) or len(
            self._encounter_ids) != len(
                self._transcripts):
            print('=============\n!! WARNING !!\n=============\nThe number of unique encounter ids is different between your transcript and hypothesis folders.\nPlease confirm your inputs are correct before proceeding...\n=============')

    def hypothesis_preprocess_into_sentences(self, hypothesis) -> dict:
        hyp_sentences_preproc = {}
        # Configure summary preprocessing module
        # There is an expectation that the input data is cleaned before running Hallucination Detection
        # Hallucination detection is run at the sentence level of the summary, against the source transcript
        # Given that the summary has headers and other templatized information, we want to omit that information prior to
        # sentence breaking and Hallucination Detection
        # , 'Constitutional: ','HEENT: ','Neck: ','Lungs: ','Heart: ','Abdomen: ','Musculoskeletal: ','Back: ','Skin: ','Neurologic: ','Psychiatric: '])
        preprocess_skipStartsWithSet = set(['#'])
        # ['HISTORY OF PRESENT ILLNESS', 'SOCIAL HISTORY', 'FAMILY HISTORY', 'ALLERGIES', 'MEDICATIONS', 'IMMUNIZATIONS', 'REVIEW OF SYSTEMS', 'PHYSICAL EXAM', 'RESULTS', 'ASSESSMENT AND PLAN', 'PROCEDURE'])
        preprocess_replaceWordSet = set([])
        summarypreprocess = SummaryPreprocess(
            skipStartsWithSet=preprocess_skipStartsWithSet,
            replaceSet=preprocess_replaceWordSet)
        for enc_id in hypothesis.keys():
            hyp_sentences_preproc[enc_id] = summarypreprocess.Preprocess(
                hypothesis[enc_id])
        return hyp_sentences_preproc
