import time
from transformers import GPT2Tokenizer
import numpy as np
from tqdm import tqdm
import os
import openai
from utils import timeout


class Chat(object):
    def __init__(self, model_name="gpt-3.5-turbo-0301"):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if model_name.find("gpt-4") != -1:
            print("Suppose it supports 8k context")
            print("Pricing (prompt): 0.03/1k tokens")
            print("Pricing (completion): 0.06/1k tokens")
            self.prompt_price = 0.03 / 1000
            self.completion_price = 0.06 / 1000
        elif model_name.find("gpt-3.5") != -1:
            print("Pricing: 0.002/1k tokens")
            self.prompt_price = 0.002 / 1000
            self.completion_price = 0.002 / 1000
        else:
            print("Unknown model, use 0.002/1k tokens as default pricing")
            self.prompt_price = 0.002 / 1000
            self.completion_price = 0.002 / 1000

    def calc_price(self, response):
        s = 0
        s += response["usage"]["prompt_tokens"] * self.prompt_price 
        s += response["usage"]["completion_tokens"] * self.completion_price
        return s

    def do_classification(self, dataset, task_message, example_prefix=False):
        """
        Do classification (zero-shot or in-context learning by calling `openai.ChatCompletion.create`.
        Args:
            dataset (`List[Dict]`): test dataset to evaluate. Each item should be a dict containing the following keys:     
                `input`: text input of the test instance
                `label`: label of the instance
                `option`: candicate label choices of the task
                `examples` (`List[Tuple]`): demonstration examples (text, label). Set as `[]` for zero-shot evaluation.
                Please refer to `example_snli` for more information.
            task_message (`String`): task description for the test dataset (`dataset`). It should contain the answer choice.
            example_prefix (`Bool`): Whether to put the demonstration examples into the `system` messages. Only work for in-context learning. May set `example_prefix=True` when evaluating GPT-4.
        """
        cache = []
        acc = 0
        unknown = 0
        cost = 0
        predictions = []
        try:
            for x in tqdm(dataset):
                # messages = [{"role": "system", "content": system_message}]
                if len(x["examples"]) == 0:
                    messages = [{"role": "system", "content": "You are a helpful assistant."}]
                    # messages = [{"role": "system", "content": "You are a helpful but unfair assistant."}]

                    messages.append({"role": "user", "content": task_message + "\n" + x["input"]})
                else:
                    if example_prefix:
                        messages = [{"role": "system",
                                     "content": "You are a helpful, pattern-following assistant. " + task_message}]
                    else:
                        messages = [{"role": "system", "content": "You are a helpful, pattern-following assistant."}]
                        messages.append({"role": "user", "content": task_message})
                        messages.append({"role": "assistant", "content": "Sure, I'd be happy to!"})

                    for y in x["examples"]:
                        if example_prefix:
                            messages.append({"role": "system", "name": "example_user", "content": y[0]})
                            messages.append(
                                {"role": "system", "name": "example_assistant", "content": y[1].capitalize()}),
                        else:
                            messages.append({"role": "user", "content": y[0]})
                            messages.append({"role": "assistant", "content": y[1].capitalize()}),
                    messages.append({"role": "user", "content": x["input"]})

                response = self.call(messages)
                cost += self.calc_price(response)
                pred = response['choices'][0]['message']['content']
                pred = pred.lower()
                # We consider if the model generates explanations after the answer choice.
                pre = pred.split(".")[0].strip()
                cache.append((messages, response))
                # if pred.capitalize() == 'Yes' or pre.capitalize() == 'Yes':
                #     predictions.append(1)
                # elif pred.capitalize() == 'No' or pre.capitalize() == 'No':
                #     predictions.append(0)
                # else:
                #     predictions.append(-1)

                if pred.capitalize() == 'Yes' or pre.capitalize() == 'Yes':
                    predictions.append(1)
                elif pred.capitalize() == 'No' or pre.capitalize() == 'No':
                    predictions.append(0)
                else:
                    predictions.append(-1)

                if pred == x["label"] or pre == x["label"]:
                    acc += 1
                elif pred not in x["option"] and pre not in x["option"]:
                    unknown += 1
        except Exception as e:
            print(e)
            if len(cache) == 0:
                return None, None, 0, [], []
            else:
                return acc / len(cache), unknown, cost, cache, predictions

        return acc / len(dataset), unknown, cost, cache, predictions

    def do_generation(self, dataset, message_constructor):
        """
        @boxinw
        """
        cache = []
        cost = 0
        try:
            for i, x in tqdm(enumerate(dataset)):
                messages = message_constructor.get_message(x)
                response = self.call(messages, max_tokens=150)
                continuation = response["choices"][0]["message"]["content"]
                is_banned = continuation.find("it contains inappropriate language.") != -1
                cost += self.calc_price(response)
                cache.append((messages, continuation, is_banned, x, response))

                if i < 5:
                    print(messages)
                    print(continuation)
                    print("="*25)
        except Exception as e:
            print(e)
            if len(cache) == 0:
                return 0, []
        return cost, cache
    
    @timeout(30)
    def _call(self, messages, t=0, max_tokens=20):
        return openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=t,
            max_tokens=max_tokens
        )

    def call(self, messages, t=0, retry=10, max_tokens=20):
        """
        A robust implementation for calling `openai.ChatCompletion.create`.
        Args:
            messages: messages conveyed to OpenAI. 
            t: temperature. Set t=0 will make the outputs mostly deterministic.
            max_tokens: maximum tokens to generate for chat completion. Please look at https://platform.openai.com/docs/api-reference/chat/create for more information.
            [TODO] We may add all arguments of `openai.ChatCompletion.create` here. 
            retry: for sake of Error on OpenAI side, we try `retry + 1` times for a request if we do not get a response.
        """
        response = None
        for i in range(retry + 1):
            try:
                response = self._call(messages, t, max_tokens)
                break
            except TimeoutError:
                print(f"Seemingly openai is frozen, wait {i + 1}s and retry")
                time.sleep(i + 1)
            except Exception as e:
                print("Error:", e)
                print(type(e))
                if isinstance(e, (openai.error.Timeout, openai.error.RateLimitError)):
                    print(f"wait {i + 1}s and retry")
                    time.sleep(i + 1)
                else:
                    raise e
        if response is None:
            print(f"try {retry + 1} but still no response, return None")
        return response
