from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
import re
from modules.aoaiutil import AOAIUtil
from modules.hallucination_mitigation_prompt import hallucination_mitigation_prompt
from modules.arguments import OpenaiArguments, MitigationArguments
import modules.gpt_output_utils

@dataclass
class HdResult:
    sentence_id: str # sentence id in the transcript
    hallucinated_sentence: str # original sentence found in summary