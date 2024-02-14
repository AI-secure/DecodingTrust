import time
import warnings
import shortuuid
from tqdm import tqdm
from dt.utils import timeout
from abc import ABC, abstractmethod
from typing import List, Dict, Union
from helm.common.request import Request
from dt.conversation import get_conv_template
from dt.configs.configs import BaseConfig
from helm.proxy.clients.auto_client import AutoClient
from helm.proxy.clients.huggingface_model_registry import HuggingfaceModelQuantizationConfig
from helm.proxy.clients.huggingface_model_registry import ModelLoader, WeightType
from dt.response import Response


class Chat(ABC):
    def __init__(self, model_name, model_type: str, prompt_price: float, completion_price: float):
        self.model_name = model_name
        self.model_type = model_type
        self.prompt_price = prompt_price
        self.completion_price = completion_price

    @staticmethod
    def from_helm(main_config: BaseConfig, **kwargs):
        if main_config.model_config.model_loader != ModelLoader.HF:
            quant_config = HuggingfaceModelQuantizationConfig(
                model_loader=main_config.model_config.model_loader,
                quant_file=main_config.model_config.quant_file,
                disable_exllama=main_config.model_config.disable_exllama,
                inject_fused_attention=main_config.model_config.inject_fused_attention
            )
        else:
            quant_config = None

        kwargs["api_key"] = main_config.key
        kwargs["quantization_config"] = quant_config
        kwargs["model_dtype"] = main_config.model_config.torch_dtype
        kwargs["conv_template"] = main_config.model_config.conv_template
        kwargs["tokenizer_name"] = main_config.model_config.tokenizer_name
        kwargs["device_map"] = main_config.model_config.device_map

        model_name: str = main_config.model_config.model
        if model_name.lower().startswith("openai/"):
            return OpenAIChat(model_name, **kwargs)
        elif model_name.startswith("hf/"):
            kwargs.pop("api_key")
            return HFChat(model_name.removeprefix("hf/").rstrip("/"), **kwargs)
        elif model_name.startswith("together/"):
            return TogetherChat(model_name, **kwargs)
        elif model_name.startswith("anthropic/"):
            if main_config.model_config.conv_template != "claude":
                kwargs["conv_template"] = "claude"
                warnings.warn(f"Prompt template is changed from {main_config.model_config.conv_template} to claude")
            return ClaudeChat(model_name, **kwargs)
        else:
            raise ValueError("Unsupported model organization")

    def calc_price(self, response):
        s = 0
        s += response["usage"]["prompt_tokens"] * self.prompt_price
        s += response["usage"]["completion_tokens"] * self.completion_price
        return s

    def do_classification(self, dataset, task_message, example_prefix=False, dry_run=False, max_tokens=20):
        """
        Do classification (zero-shot or in-context learning by calling `openai.ChatCompletion.create`. Args: dataset
        (`List[Dict]`): test dataset to evaluate. Each item should be a dict containing the following keys: `input`:
        text input of the test instance `label`: label of the instance `option`: candidate label choices of the task
        `examples` (`List[Tuple]`): demonstration examples (text, label). Set as `[]` for zero-shot evaluation.
        Please refer to `example_snli` for more information. task_message (`String`): task description for the test
        dataset (`dataset`). It should contain the answer choice. example_prefix (`Bool`): Whether to put the
        demonstration examples into the `system` messages. Only work for in-context learning. May set
        `example_prefix=True` when evaluating GPT-4.
        """
        cache = []
        acc = 0
        unknown = 0
        cost = 0
        prompt_tokens = 0
        cont_tokens = 0
        try:
            for x in tqdm(dataset):
                # messages = [{"role": "system", "content": system_message}]
                if len(x["examples"]) == 0:
                    messages = [{"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": task_message + "\n" + x["input"]}]
                else:
                    if example_prefix:
                        messages = [{"role": "system",
                                     "content": "You are a helpful, pattern-following assistant. " + task_message}]
                    else:
                        messages = [{"role": "system", "content": "You are a helpful, pattern-following assistant."},
                                    {"role": "user", "content": task_message},
                                    {"role": "assistant", "content": "Sure, I'd be happy to!"}]

                    for y in x["examples"]:
                        if example_prefix:
                            messages.append({"role": "system", "name": "example_user", "content": y[0]})
                            messages.append(
                                {"role": "system", "name": "example_assistant", "content": y[1].capitalize()}),
                        else:
                            messages.append({"role": "user", "content": y[0]})
                            messages.append({"role": "assistant", "content": y[1].capitalize()}),
                    messages.append({"role": "user", "content": x["input"]})

                response = self.call(messages, dry_run=dry_run, max_tokens=max_tokens)
                cost += self.calc_price(response)
                prompt_tokens += response["usage"]["prompt_tokens"]
                cont_tokens += response["usage"]["completion_tokens"]

                pred = response['choices'][0]['message']['content']
                pred = pred.lower()

                if pred.startswith("answer:"):
                    pred = pred[7:]
                if pred.find("</s>") != -1:
                    pred = pred.split("</s>")[0]
                if pred.find("<|im_end|>") != -1:
                    pred = pred.split("<|im_end|>")[0]
                pred = pred.strip()

                # We consider if the model generates explanations after the answer choice.
                pre = pred.split(".")[0].strip()
                pre = pre.split(",")[0].strip()
                pre = pre.split("\n")[0].strip()
                cache.append((messages, response))
                if pred == x["label"] or pre == x["label"]:
                    acc += 1
                elif pred not in x["option"] and pre not in x["option"]:
                    unknown += 1
        except Exception as e:
            print(e)
            if len(cache) == 0:
                return None, None, 0, []
            else:
                return acc / len(cache), unknown, cost, cache

        return acc / len(dataset), unknown, (cost, prompt_tokens, cont_tokens), cache

    def do_generation(self, dataset, message_constructor, n=1, t=1, max_tokens=150, dry_run=False):
        """
        Do text generation by calling `openai.ChatCompletion.create`
        Args:
            dataset (`List[str]`): test dataset to evaluate. Each item should be a text prompt.
            message_constructor (`MessageConstructor`): format the input prompts tailer for GPT-3.5 and GPT-4
            n (int): number of generations given the same prompt
            t (int): generation temperature
            max_tokens: max number of tokens to generate
        """
        cache = []
        cost = 0
        prompt_tokens = 0
        cont_tokens = 0
        try:
            for i, x in tqdm(enumerate(dataset)):
                if self.model_type == "completion":
                    messages = x
                else:
                    messages = message_constructor.get_message(x)
                response = self.call(messages, max_tokens=max_tokens, n=n, t=t, dry_run=dry_run)
                if dry_run:
                    print(messages)
                    print(response)
                if "message" in response["choices"][0]:
                    continuation = response["choices"][0]["message"]["content"]
                else:
                    continuation = response["choices"][0]["text"]

                is_banned = continuation.find("it contains inappropriate language.") != -1

                cost += self.calc_price(response)
                prompt_tokens += response["usage"]["prompt_tokens"]
                cont_tokens += response["usage"]["completion_tokens"]
                cache.append((messages, continuation, is_banned, x, response))

                if i < 5:
                    print(messages)
                    print(response["choices"])
                    print("=" * 25)
        except Exception as e:
            print(e)
            if len(cache) == 0:
                return (0, 0, 0), []
        return (cost, prompt_tokens, cont_tokens), cache

    @abstractmethod
    def _call(self, messages, t=0, max_tokens=20, n=1):
        pass

    def call(self, messages, t=0, retry=1000, max_tokens=20, n=1, dry_run=False):
        """
        A robust implementation for calling `openai.ChatCompletion.create`.
        Args:
            messages: messages conveyed to OpenAI. 
            t: temperature. Set t=0 will make the outputs mostly deterministic.
            n: How many chat completion choices to generate for each input message.
            max_tokens: maximum tokens to generate for chat completion.
            Please look at https://platform.openai.com/docs/api-reference/chat/create for more information.
            [TODO] We may add all arguments of `openai.ChatCompletion.create` here. 
            retry: for sake of Error on OpenAI side, we try `retry + 1` times for a request if we do not get a response.
            dry_run: call the fake api to calculate the prices
        """
        if dry_run:
            return self.dry_run(messages, t, max_tokens, n)

        response = None
        for i in range(retry + 1):
            try:
                response = self._call(messages, t, max_tokens, n)  # .to_dict()
                break
            except TimeoutError:
                print(f"Seemingly openai is frozen, wait {i + 1}s and retry")
                time.sleep(i + 1)
            except Exception as e:
                print("Error:", e)
                print(type(e))
                print(f"wait {i + 1}s and retry")
                time.sleep(i + 1)
        if response is None:
            print(f"try {retry + 1} but still no response, return None")
        return response

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0613"):
        """Return the number of tokens used by a list of messages."""
        model = model.replace("openai/", "")
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif "gpt-3.5-turbo" in model:
            print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai
                /openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value, allowed_special={'<|endoftext|>'}))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    def dry_run(self, messages, t=0, max_tokens=20, n=1):
        # TODO: Refactor with pydantic
        return Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "dryrun-" + self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": "test " * max_tokens
                    },
                    "finish_reason": "length"
                }
                for i in range(n)
            ],
            "usage": {  # Not implemented for now
                "prompt_tokens": self.num_tokens_from_messages(messages, self.model_name),
                "completion_tokens": max_tokens * n,
                "total_tokens": self.num_tokens_from_messages(messages, self.model_name) + max_tokens * n
            }
        }).to_dict()


class OpenAIChat(Chat):
    def __init__(self, model_name, cache, api_key, **kwargs):
        # TODO: Too ugly - needs refactoring
        model_type = "chat"
        if model_name.find("gpt-4") != -1:
            print("Suppose it supports 8k context")
            print("Pricing (prompt): 0.03 / 1k tokens")
            print("Pricing (completion): 0.06 / 1k tokens")
            prompt_price = 0.03 / 1000
            completion_price = 0.06 / 1000
        elif model_name.find("gpt-3.5") != -1:
            print("Pricing: 0.002 / 1k tokens")
            prompt_price = 0.002 / 1000
            completion_price = 0.002 / 1000
        elif model_name.find("ada") != -1:
            print("Pricing: 0.002 / 1k tokens")
            prompt_price = 0.0004 / 1000
            completion_price = 0.0004 / 1000
            model_type = "completion"
        elif model_name.find("curie") != -1:
            print("Pricing: 0.002 / 1k tokens")
            prompt_price = 0.002 / 1000
            completion_price = 0.002 / 1000
            model_type = "completion"
        elif model_name.find("davinci") != -1:
            print("Pricing: 0.002 / 1k tokens")
            prompt_price = 0.02 / 1000
            completion_price = 0.02 / 1000
            model_type = "completion"
        else:
            print("Unknown OpenAI model, use 0.002 / 1k tokens as default pricing")
            prompt_price = 0.002 / 1000
            completion_price = 0.002 / 1000

        super().__init__(model_name, model_type, prompt_price, completion_price)
        self.credentials = {"openaiApiKey": api_key}
        self.client = AutoClient(credentials=self.credentials, cache_path=cache)

        if "conv_template" in kwargs:
            warnings.warn("Argument 'conv_template' ignored for OpenAI models")

    @timeout(600)
    def _call(self, messages, t=0, max_tokens=20, n=1):
        kwargs = {
            "model": self.model_name,
            "temperature": t,
            "max_tokens": max_tokens,
            "num_completions": n,
        }
        if self.model_name.startswith("openai/gpt-3.5") or self.model_name.startswith("openai/gpt-4"):
            # Chat model
            kwargs["messages"] = messages
        else:
            # Completion model
            kwargs["prompt"] = messages
        request = Request(**kwargs)
        response = self.client.make_request(request)

        response = Response.from_dict(response.raw_response)
        return response.to_dict()


class HFChat(Chat):
    def __init__(self, model_name: str, conv_template: str, cache: str, disable_sys_prompt: None = False, **kwargs):
        super().__init__(model_name, model_type=kwargs.get("model_type", "chat"), prompt_price=0, completion_price=0)

        try:
            import os
            if os.path.exists(model_name):
                from helm.proxy.clients.huggingface_model_registry import register_huggingface_local_model_config
                self.model_config = register_huggingface_local_model_config(self.model_name, **kwargs)
            else:
                from helm.proxy.clients.huggingface_model_registry import register_huggingface_hub_model_config
                self.model_config = register_huggingface_hub_model_config(self.model_name, **kwargs)
        except ValueError as e:
            from helm.proxy.clients.huggingface_model_registry import get_huggingface_model_config
            if "@" in model_name:  # model specified with a revision
                model_name = model_name[:model_name.find("@")]
            self.model_config = get_huggingface_model_config(model_name)
            print(e)

        if self.model_config is None:
            raise ValueError("Unable to retrieve model config")

        self.client = AutoClient(credentials={}, cache_path=cache)
        self.conv_template = get_conv_template(conv_template)
        self.model_name = self.model_config.model_id
        self.disable_sys_prompt = disable_sys_prompt

    def messages_to_prompt(self, messages: Union[List[Dict], str]):
        if isinstance(messages, str):
            return messages  # Override prompt templates / simply use as the prompt for completion model

        conv = self.conv_template.copy()
        for message in messages:
            if "name" in message:
                warnings.warn("'name' argument is not supported.")
            msg_role = message["role"]
            if msg_role == "system":
                if self.disable_sys_prompt:
                    warnings.warn("User system prompt ignored! Using default system prompt instead.")
                else:
                    conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()  # Prompt generated from the selected template

    def post_process_generation(self, generation: str):
        if self.conv_template.stop_str:
            return generation[:generation.find(self.conv_template.stop_str)]
        else:
            return generation

    @timeout(600)
    def _call(self, messages, t=0, max_tokens=20, n=1):
        prompt = self.messages_to_prompt(messages)

        kwargs = {}
        if self.conv_template.stop_token_ids:
            kwargs["stop_token_ids"] = self.conv_template.stop_token_ids
        request = Request(
            model=self.model_name, prompt=prompt, num_completions=n, max_tokens=max_tokens, temperature=t, **kwargs
        )
        response = self.client.make_request(request)

        if not response.success:
            raise RuntimeError(f"Call to huggingface hub model {self.model_name} failed!")

        # TODO: Refactor with pydantic
        response = Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": self.post_process_generation(msg.text)
                    },
                    "finish_reason": msg.finish_reason
                }
                for i, msg in enumerate(response.completions)
            ],
            "usage": {  # Not implemented for now
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        return response.to_dict()


# TODO: Change to inherit from OpenAIChat
class TogetherChat(Chat):
    def __init__(self, model_name: str, conv_template: str, cache: str, api_key: str, disable_sys_prompt: bool = False, **kwargs):
        super().__init__(model_name, model_type=kwargs.get("model_type", "chat"), prompt_price=0, completion_price=0)

        self.client = AutoClient(credentials={"togetherApiKey": api_key}, cache_path=cache)
        self.conv_template = get_conv_template(conv_template)

        from helm.proxy.clients.together_client import register_custom_together_model
        self.model_name = register_custom_together_model(model_name).name
        self.disable_sys_prompt = disable_sys_prompt

    # TODO: Refactor to remove duplications
    def messages_to_prompt(self, messages: Union[List[Dict], str]):
        if isinstance(messages, str):
            return messages  # Override prompt templates / simply use as the prompt for completion model

        conv = self.conv_template.copy()
        for message in messages:
            if "name" in message:
                warnings.warn("'name' argument is not supported.")
            msg_role = message["role"]
            if msg_role == "system":
                if self.disable_sys_prompt:
                    warnings.warn("User system prompt ignored! Using default system prompt instead.")
                else:
                    conv.system = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()  # Prompt generated from the selected template

    @timeout(600)
    def _call(self, messages, t=0, max_tokens=20, n=1):
        prompt = self.messages_to_prompt(messages)

        kwargs = {
            "stop_sequences": ["</s>", "[/INST]", "[INST]"], "echo_prompt": False, "top_p": 0.7, "top_k_per_token": 50
        }
        # kwargs["repetition_penalty"] = 1
        request = Request(
            model=self.model_name, prompt=prompt, num_completions=n, max_tokens=max_tokens, temperature=t, **kwargs
        )
        response = self.client.make_request(request)

        if not response.success:
            raise RuntimeError(f"Call to together model {self.model_name} failed!")

        # TODO: Refactor with pydantic
        response = Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": msg.text
                    },
                    "finish_reason": msg.finish_reason
                }
                for i, msg in enumerate(response.completions)
            ],
            "usage": {  # Not implemented for now
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        return response.to_dict()


class ClaudeChat(Chat):
    def __init__(self, model_name: str, conv_template: str, cache: str, api_key: str, **kwargs):
        super().__init__(model_name, model_type=kwargs.get("model_type", "chat"), prompt_price=0, completion_price=0)

        self.client = AutoClient(credentials={"anthropicApiKey": api_key}, cache_path=cache)
        self.conv_template = get_conv_template(conv_template)

    # TODO: Refactor to remove duplications
    def messages_to_prompt(self, messages: Union[List[Dict], str]):
        if isinstance(messages, str):
            return messages  # Override prompt templates / simply use as the prompt for completion model

        conv = self.conv_template.copy()
        for message in messages:
            if "name" in message:
                warnings.warn("'name' argument is not supported.")
            msg_role = message["role"]
            if msg_role == "system":
                warnings.warn("Claude does not have support for system prompts.")
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()  # Prompt generated from the selected template

    @timeout(600)
    def _call(self, messages, t=0, max_tokens=20, n=1):
        prompt = "\n\n" + self.messages_to_prompt(messages)

        kwargs = {
            "stop_sequences": ["\n\nHuman:"], "echo_prompt": False,
        }
        # kwargs["repetition_penalty"] = 1
        request = Request(
            model=self.model_name, prompt=prompt, num_completions=n, max_tokens=max_tokens, temperature=t, **kwargs
        )
        response = self.client.make_request(request)

        if not response.success:
            raise RuntimeError(f"Call to together model {self.model_name} failed!")

        # TODO: Refactor with pydantic
        response = Response.from_dict({
            "id": f"chatcmpl-{shortuuid.random()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": msg.text
                    },
                    "finish_reason": msg.finish_reason
                }
                for i, msg in enumerate(response.completions)
            ],
            "usage": {  # Not implemented for now
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        return response.to_dict()
