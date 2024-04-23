from copy import deepcopy
import torch
from torch.nn import functional
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import Any, Dict, List, Union, Optional

from ..common.cache import Cache, CacheConfig, NoCache
from ..common.hierarchical_logger import htrack_block, hlog
from ..common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from ..common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
from .client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from .huggingface_model_registry import HuggingfaceModelQuantizationConfig
from .huggingface_tokenizer import HuggingFaceTokenizers
from .huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceModelConfig,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
    ModelLoader,
    WeightType
)
from threading import Lock

from ..common.request import ErrorFlags


class StopOnToken(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generation results encounter the token(s) specified in
    `stop_token_ids`. This is particularly useful for chat models with multiple stopping criteria
    """
    def __init__(self, stop_token_id: Union[int, torch.LongTensor]):
        if isinstance(stop_token_id, torch.LongTensor):
            stop_token_id = stop_token_id.item()
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1] == self.stop_token_id


class HuggingFaceServer:
    def __init__(self, model_config: HuggingFaceModelConfig):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        model_kwargs = {}
        # If the HuggingFace model is stored locally, it will have a path defined and we should load it from there.
        # Otherwise, download it from the HuggingFace hub by passing in its identifier.
        if isinstance(model_config, HuggingFaceLocalModelConfig):
            model_name = model_config.path
        elif isinstance(model_config, HuggingFaceHubModelConfig):
            model_name = model_config.model_id
            if model_config.revision:
                model_kwargs["revision"] = model_config.revision
        else:
            raise Exception(f"Unknown type of model_config: {model_config}")

        if model_config.model_dtype == WeightType.FLOAT16:
            model_kwargs["torch_dtype"] = torch.float16
        elif model_config.model_dtype == WeightType.BFLOAT16:
            model_kwargs["torch_dtype"] = torch.bfloat16

        model_kwargs["device_map"] = model_config.device_map

        with htrack_block(f"Loading Hugging Face model for config {model_config}"):
            # WARNING this may fail if your GPU does not have enough memory
            quantization_config: Optional[HuggingfaceModelQuantizationConfig] = model_config.quantization_config
            if quantization_config and quantization_config.model_loader == ModelLoader.AWQ:
                from awq import AutoAWQForCausalLM
                self.model = AutoAWQForCausalLM.from_quantized(
                    model_name, model_config.quantization_config.quant_file, fuse_layers=True
                )
            elif quantization_config and quantization_config.model_loader == ModelLoader.GPTQ:
                from auto_gptq import AutoGPTQForCausalLM
                self.model = AutoGPTQForCausalLM.from_quantized(
                    model_name, trust_remote_code=True,
                    disable_exllama=model_config.quantization_config.disable_exllama,
                    inject_fused_attention=model_config.quantization_config.inject_fused_attention,
                    **model_kwargs
                )
                hlog(self.model.hf_device_map)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=True, **model_kwargs
                )
                hlog(self.model.hf_device_map)

        with htrack_block(f"Loading Hugging Face tokenizer model for config {model_config}"):
            # When the quantized model has uses a different tokenizer than its moddel name
            if model_config.tokenizer_name:
                tokenizer_name: str = model_config.tokenizer_name
                if "revision" in model_kwargs:
                    model_kwargs.pop("revision")
            else:
                tokenizer_name: str = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **model_kwargs)

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
            self.device
        )
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:
            stop_sequence_ids = self.tokenizer(
                raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
            )
            for stop_id in stop_sequence_ids.input_ids:
                assert len(stop_id) == 1, "Total number of tokens in each stop word should be 1."
            del raw_request["stop_sequences"]
            raw_request["stopping_criteria"] = StoppingCriteriaList(
                [StopOnToken(stop_token_id=token) for token in stop_sequence_ids]
            )
            # TODO: Support stop words that could include multiple tokens. (Useful for chat models)
        elif len(raw_request["stop_token_ids"]) > 0:
            eos_token_id: int = self.tokenizer.eos_token_id
            if eos_token_id not in raw_request["stop_token_ids"]:
                raw_request["stop_token_ids"].append(eos_token_id)
            raw_request["stopping_criteria"] = StoppingCriteriaList(
                [StopOnToken(stop_token_id=token) for token in raw_request["stop_token_ids"]]
            )

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences", "stop_token_ids"]
        }

        # Use HuggingFace's `generate` method.
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]):] for sequence in sequences]

        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
                all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


_servers_lock: Lock = Lock()
_servers: Dict[str, HuggingFaceServer] = {}


def _get_singleton_server(model_config: HuggingFaceModelConfig) -> HuggingFaceServer:
    """Lookup or create a new HuggingFaceServer that will be shared among all threads.

    When --num-threads > 1, multiple threads will attempt to instantiate
    `HuggingFaceServer`s simultaneously. Since we have limited GPU memory, we want to
    just share a single copy of each model we are using. So, this function uses a lock
    to make sure that for each model, only one thread creates a HuggingFaceServer.
    The other threads can share that same server in the global _servers dictionary."""
    global _servers_lock
    global _servers
    with _servers_lock:
        if model_config.model_id not in _servers:
            _servers[model_config.model_id] = HuggingFaceServer(model_config)
    return _servers[model_config.model_id]


class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}

    def get_model_server_instance(self, model) -> HuggingFaceServer:
        model_config = get_huggingface_model_config(model)
        # Special-case some models in so that users don't have to enable them with --enable-huggingface-models
        if not model_config:
            # Other HuggingFace hub models that we'll look up for you even if you didn't enable them via the flag
            if model == "EleutherAI/gpt-j-6B":
                model_config = HuggingFaceHubModelConfig.from_string("EleutherAI/gpt-j-6B")
            elif model == "huggingface/gpt2":
                model_config = HuggingFaceHubModelConfig.from_string("gpt2")
            elif model == "bigcode/santacoder":
                model_config = HuggingFaceHubModelConfig.from_string("bigcode/santacoder")
            elif model == "huggingface/starcoder":
                model_config = HuggingFaceHubModelConfig.from_string("bigcode/starcoder")
            elif model == "mosaicml/mpt-7b":
                model_config = HuggingFaceHubModelConfig.from_string("mosaicml/mpt-7b")
            else:
                raise Exception(f"Unknown HuggingFace model: {model}")
        return _get_singleton_server(model_config)

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
            "stop_token_ids": request.stop_token_ids
        }

        # if request.temperature > 0:
        #     raw_request["temperature"] = request.temperature
        # elif request.temperature == 0:
        #     raw_request["top_k_per_token"] = 1
        # else:
        #     raise ValueError(f"Invalid temperature {request.temperature}!")

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        try:

            def do_it():
                return model_server_instance.serve_request(raw_request)

            cache_key = Client.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:  # Do something if error is encountered.
            raise e
            # error: str = f"HuggingFace error: {e}"
            # return RequestResult(
            #     success=False, cached=False, error=error, completions=[], embedding=[],
            #     error_flags=ErrorFlags(is_retriable=False, is_fatal=True),
            # )

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"]:]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                    generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        # These models already handle the "▁" character correctly with the
                        # convert_tokens_to_string method. We prefer to use this method instead
                        # of the hacky cleanup_tokens method below as it might handle cases
                        # we haven't thought of in cleanup_tokens.
                        tokens = [
                            tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request.text)
                        ]
                    else:
                        # Tokenizes the text and returns the tokens as a list of strings,
                        # not a list of token objects (otherwise "Hello world" would be"
                        # ["Hello", "▁world"] and not ["Hello", " world"])
                        # We could do this with a simple replace like this:
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                        # But this replaces all the "▁" characters by "", which is not what we want.
                        # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                        # Just like tokenize("Hello", encode=False) would return ["Hello"].
                        tokens = tokenizer.tokenize(request.text)
                        tokens = cleanup_tokens(tokens, request.tokenizer)
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            # error: str = f"HuggingFace error: {e}"
            # return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])
            raise e

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )