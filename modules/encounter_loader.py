from glob import glob
import os
from typing import Dict
import pandas as pd
from modules.summarypreprocess import SummaryPreprocess


class EncounterLoader:
    def __init__(self,
                 transcriptfolder: str) -> None:
        self._transcripts = self.__load_file_inputs(transcriptfolder, "*.txt")
        print(f'Transcript Files Found: {len(self._transcripts)}')

    def __load_file_inputs(self, folderpath: str,
                           searchpattern: str = "") -> Dict[str, str]:
        retval = {}
        for fname in glob(os.path.join(folderpath, searchpattern)):
            with open(fname, "r", encoding="utf-8") as f:
                fnamenoext = os.path.splitext(os.path.basename(fname))[0]
                retval[fnamenoext] = f.read()
        return retval
