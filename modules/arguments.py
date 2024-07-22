from dataclasses import dataclass, field
import json
import logging
from typing import Optional, List
import os
from pathlib import Path

@dataclass
class OpenaiArguments:
    """
    Configuration for OpenAI engine
    """
    config_setting: Optional[str] = field(
        default='gpt-4-32k', metadata={"help": "The pre-defined AOAI config settings to"}
    )
    api_key: Optional[str] = field(
        default=os.environ.get('GPT_API_KEY', None), metadata={"help": "API key to call openai gpt"}
    )
    use_chat_completions: Optional[bool] = field(
        default=True, metadata={"help": "Use chat completion or completion api"}
    )
    max_parallelism: Optional[int] = field(
        default=2, metadata={"help": "maximum parallelism to use per data"}
    )

def create_openai_arguments(config_setting_key : str, max_parallelism : int, config_file : str = None ) -> OpenaiArguments:
    if config_file is None:
        # use default config file. This seems a bit strange - assume some file outside of current package folder
        config_file = Path(__file__).absolute().parent / 'configs' / 'aoai_config.json'
    with open(config_file, "r") as config_file:
        config = json.load(config_file)
    
    if config_setting_key is None:
        logging.warning(f"AOAI config setting key is None, using default config setting key")
        if len(config) > 1:
            raise ValueError(f"AOAI config setting key is None, but config file has more than 1 setting. Please specify the config setting key")
        config_setting_key = list(config.keys())[0]

    if config_setting_key not in config:
        raise ValueError(f"AOAI config setting {config_setting_key} not found in {config_file}")

    settings = config[config_setting_key]
    openai_args = OpenaiArguments()
    openai_args.config_setting = config_setting_key
    openai_args.use_chat_completions = bool(settings['USE_CHAT_COMPLETIONS'])
    openai_args.max_parallelism = max_parallelism
    return openai_args

@dataclass
class MitigationArguments:
    """
    Configuration for OpenAI GPT Model
    """
    temp: Optional[float] = field(
        default=0, metadata={"help": "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."}
    )
    top_p: Optional[float] = field(
        default=0.6, metadata={"help": "An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered."}
    )
    max_tokens: Optional[int] = field(
        default=1024, metadata={"help": "The maximum number of tokens to generate in the completion."}
    )
    freq_penalty: Optional[float] = field(
        default=0, metadata={"help": "Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim."}
    )
    presence_penalty: Optional[float] = field(
        default=0, metadata={"help": "Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics."}
    )
    log_prob: Optional[int] = field(
        default=0, metadata={"help": "Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens. For example, if logprobs is 5, the API will return a list of the 5 most likely tokens. The API will always return the logprob of the sampled token, so there may be up to logprobs+1 elements in the response."}
    )
    batch_size: Optional[int] = field(
        default=10, metadata={"help": "Batched request to send in a single prompt"}
    )
    generations: Optional[int] = field(
        default=1, metadata={"help": "Number of generations (outputs) to produce"}
    )
