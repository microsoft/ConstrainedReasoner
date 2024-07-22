import nltk
from typing import Dict, List


class SummaryPreprocess:

    def __init__(self, skipStartsWithSet, replaceSet) -> None:
        nltk.download('punkt')
        self._break_sentence = nltk.data.load(
            "tokenizers/punkt/english.pickle")
        self._skipStartsWithSet = skipStartsWithSet
        self._replaceSet = replaceSet

    def Preprocess(self, text: str) -> List[Dict[str,str]]:
        retval = list()
        shouldSkip = False
        sentencesRemoved = 0
        sentenceId = 0
        for line in text.split('\n'):
            for sent in self._break_sentence.tokenize(line):
                sent = sent.replace('__lf1__', '').replace('__lf2__', '')
                sent = sent.strip()

                if (sent.startswith('#') and sent.endswith(
                        '#')) or (sent.isupper()):
                    sentencesRemoved += 1
                    continue

                for swWord in self._skipStartsWithSet:
                    if sent.startswith(swWord):
                        shouldSkip = True
                        continue

                if shouldSkip:
                    shouldSkip = False
                    sentencesRemoved += 1
                    continue

                for rWord in self._replaceSet:
                    sent = sent.replace(rWord, '')

                sent = sent.replace('<|im_end|>','').strip()

                if len(sent) <= 3:
                    sentencesRemoved += 1
                    continue
                sentenceId += 1
                retval.append({'sentence_id': sentenceId, 'text': sent})
        # print(f'Sentences Removed: {sentencesRemoved}')
        return retval
