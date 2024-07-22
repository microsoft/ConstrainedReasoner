# field name constant
class FieldName:
    ENCOUNTER_ID = 'encounter_id'
    SENTENCE_ID = 'sentence_id'
    SENTENCE_TEXT = 'text'
    IS_HALLUCINATED = 'is_hallucinated'
    DETECTION_TYPE = 'detection_type'
    HD_ENTITY = 'entity'
    SPAN = 'span'
    NAME = 'name'
    TYPE = 'type'
    REASON = 'reason'

class AllHallucinations:
    ENCOUNTER_ID = 'enc_id'
    HALLUCINATED = 'hallucinated'
    HALLUCINATION_SCORE = 'hallucination_score'
    HALLUCINATIONS = 'hallucinations'
    NUM_TOTAL_SENTENCES = 'num_total_sentences'
    NUM_TOTAL_HALLUCINATIONS = 'num_total_hallucinations'

class OverallDetectionFeilds:
    ENCOUNTER_ID = 'encounter_id'
    ENCOUNTER_TEXT = 'summary'
    DOCUMENT = 'src'
    IS_HALLUCINATED = 'is_hallucination'
    GROUNDING_SCORE = 'score'
    DETECTION_TYPE = 'detection_type'
    
class HallucinationDebugMetadata:
    ENCOUNTER_ID = 'EncounterId'
    SENTENCE_ID = 'SentenceId'
    DETECTION_TYPE = 'DetectionType'
    HAS_ENTITY = 'HasEntity'
    SHOULD_KEEP_ENTITY = 'ShouldKeepEntity'
    IS_HALLUCINATED_ENTITY = 'IsHallucinatedEntity'
    SENTENCE = 'Sentence'
    DETECTED_ENTITY_CLEANED = 'DetectedEntityCleaned'
    DETECTED_ENTITY_TYPE = 'DetectedEntityType'
    SENTENCE_WITH_SPAN = 'SentenceWithSpan'
    RAW_OUTPUT = 'GPTrawOutput'
    IS_HALLUCINATED_JUDGEMENT = 'isHallucinatedGptJudgement'
    IS_HALLUCINATION_FINAL = 'IsHallucination_Final'
    
class HallucinationPerfMetadata:
    ENCOUNTER_ID = 'EncounterId'
    GPT_REQUESTS = 'GptRequests'
    GPT_CALLS = 'GptCalls'
    TRANSCRIPT_TOKENS = 'TranscriptTokens'
    SENTENCES = 'Sentences'
    CONTENT_TOKENS = 'ContentTokens'
    HD_ROUND_1 = 'HDRound1'
    HD_ROUND_2 = 'HDRound2'
    HD_TOTAL = 'HDTotal'
    ED_TIME = 'EDTime'
    ENTITIES = 'Entities'
    SENTENCE_FILTER_TIME = 'SentenceFilterTime'
    SENTENCE_FILTER_CPU_MAX = 'SentenceFilterCpuMax'
    SENTENCE_FILTER_MEM_MAX = 'SentenceFilterMemMax'
    SENTENCE_FILTER_MEM_MAX_FIRST = 'SentenceFilterMemMaxFirst'
    SENTENCE_FILTER_MEM_MAX_PEAK = 'SentenceFilterMemMaxPeak'

# maybe the mark used by human and instructed in the prompt is different
class HM_constants:
    REMOVED_prompt = "||REMOVE||"
    REMOVED_human = ""