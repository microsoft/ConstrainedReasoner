from dataclasses import dataclass

@dataclass
class HdResult:
    sentence_id: str # sentence id in the transcript
    hallucinated_sentence: str # original sentence found in summary