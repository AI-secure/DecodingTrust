from dataclasses import dataclass, field, fields, asdict
from typing import List, Dict, Any
import shortuuid
import time


@dataclass
class Message:
    role: str
    # The role associated with the message (e.g., "assistant")

    content: str
    # The content of the message

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __getitem__(self, item):
        return getattr(self, item)

@dataclass
class Choice:
    index: int
    # The index of the choice in the list of choices

    message: Message
    # The role associated with the message (e.g., "assistant") with its corresponding content of the message

    finish_reason: str

    # The reason this choice was finished/selected

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Choice":
        data["message"] = Message.from_dict(data["message"])
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        return {"index": self.index, "message": self.message.to_dict(), "finish_reason": self.finish_reason}

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class Response:
    model: str
    # The name of the model used to generate the completions

    choices: List[Choice]
    # A list of Choice objects representing the possible choices for this completion

    usage: Dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    # A dictionary containing usage statistics for the completion

    id: str = field(default_factory=lambda: f"chatcmpl-{shortuuid.uuid()}")
    # A unique identifier for the chat completion, generated using a random UUID

    object: str = "chat.completion"
    # The type of object (in this case, "chat.completion")

    created: int = field(default_factory=lambda: int(time.time()))

    # The creation time of the object, generated using the current time

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        # print(data)
        # print(data["choices"])
        # for choice in data["choices"]:
            # print(choice)
            # print(Choice.from_dict(choice))
        data["choices"] = [Choice.from_dict(choice) for choice in data["choices"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        choices = [choice.to_dict() for choice in self.choices]
        return {"id": self.id, "object": self.object, "created": self.created, "model": self.model, "choices": choices,
                "usage": self.usage}

    def __getitem__(self, item):
        return getattr(self, item)


if __name__ == '__main__':
    # Example dictionary representing a chat completion object
    chat_completion_dict = {
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Sure, here is an example message."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }

    # Initialize a ChatCompletion object from the dictionary
    chat_completion_obj = Response.from_dict(chat_completion_dict)

    # Print the object to verify that it has been correctly initialized
    print(chat_completion_obj)
    print(chat_completion_obj.to_dict())
