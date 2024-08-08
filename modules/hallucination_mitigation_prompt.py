import copy
import os
import yaml
from transformers import GPT2TokenizerFast
class hallucination_reasoning_prompt :
    def __init__(self, use_chat_completions : bool, prompt_resource_root_folder : str = None, category = False, simple = False) -> None:
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._prompt_resource_root = prompt_resource_root_folder if prompt_resource_root_folder is not None else os.path.join(os.path.dirname(__file__), "..")
        self._maxPromptTokens = 32000
        self._use_chat_completions = use_chat_completions
        if category:
            self._prompt = self._loadPrompt('Reasoning_Category', use_chat_completions) 
        elif simple: #simple prompt
            self._prompt = self._loadPrompt('Simplebasic', use_chat_completions)
        else:
            self._prompt = self._loadPrompt('Reasoning', use_chat_completions) 

    def resolve_file_path(self, file_path : str) -> str:
        # if file_path is not a full path, then resolve it to the full path
        if not os.path.isabs(file_path) :
            file_path = os.path.join(self._prompt_resource_root, file_path)
        
        return file_path
    
    def load_file_content(self, file_path : str) -> str:
        file_path = self.resolve_file_path(file_path)
        return open(file_path, 'r', encoding="utf8").read()

    def _loadPrompt(self, filename : str, useChatCompletions : bool = False) :
        yamlfile = self.resolve_file_path(f'prompts/chat_completions/{filename}.yaml')
        if useChatCompletions:
            return yaml.safe_load(self.load_file_content(yamlfile))
        elif os.path.exists(yamlfile):
            yaml_config = yaml.safe_load(self.load_file_content(yamlfile))
            content = ''
            for item in yaml_config:
                content += item['content'] + '\n'
            return content
        raise FileNotFoundError(f'{yamlfile} was not found')

    def _validate_prompt(self, prompt, max_tokens) :
        if not self._use_chat_completions :
            if len(self._tokenizer(prompt, truncation=True, max_length=32000)['input_ids']) + max_tokens > self._maxPromptTokens :
                raise ValueError(f'len(prompt) ({len(prompt)}) + max_tokens ({max_tokens}) must be less than {self._maxPromptTokens}') 
        return prompt

    def _replace(self, promptOrMessageObj, before : str, after : str, simpleReplaceOverride : bool = False) :
        if self._use_chat_completions and not simpleReplaceOverride:
            retval = copy.deepcopy(promptOrMessageObj)
            for item in retval :
                item['content'] = item['content'].replace(before, after)
            return retval
        return promptOrMessageObj.replace(before, after)
    

    def create_prompt(self, transcript: str, sentences: str, max_tokens: int) -> str:
        # sentences is indexed sentences.
        prompt = copy.deepcopy(self._prompt)
        prompt = self._replace(prompt, '{{transcript}}', transcript)
        prompt = self._replace(prompt, '{{sentences}}', sentences)
        self._validate_prompt(prompt, max_tokens)
        return prompt
