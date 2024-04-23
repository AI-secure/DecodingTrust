import os
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

from retrying import RetryError, Attempt

from ..common.model_deployment_registry import get_model_deployment
from ..common.cache import CacheConfig, MongoCacheConfig, SqliteCacheConfig
from ..common.hierarchical_logger import hlog
from ..common.object_spec import create_object
from ..common.request import Request, RequestResult
from ..common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
)
from ..common.retry import retry_request, NonRetriableException
from .client import Client
from .http_model_client import HTTPModelClient
from .huggingface_model_registry import get_huggingface_model_config

if TYPE_CHECKING:
    import huggingface_client


class AuthenticationError(NonRetriableException):
    pass


class AutoClient(Client):
    """Automatically dispatch to the proper `Client` based on the organization.

    The modules for each client are lazily imported when the respective client is created.
    This greatly speeds up the import time of this module, and allows the client modules to
    use optional dependencies."""

    def __init__(self, credentials: Mapping[str, Any], cache_path: str, mongo_uri: str = ""):
        self.credentials = credentials
        self.cache_path = cache_path
        self.mongo_uri = mongo_uri
        self.clients: Dict[str, Client] = {}
        self.tokenizer_clients: Dict[str, Client] = {}
        # self._huggingface_client is lazily instantiated by get_huggingface_client()
        self._huggingface_client: Optional[".huggingface_client.HuggingFaceClient"] = None
        hlog(f"AutoClient: cache_path = {cache_path}")
        hlog(f"AutoClient: mongo_uri = {mongo_uri}")

    def _build_cache_config(self, organization: str) -> CacheConfig:
        if self.mongo_uri:
            return MongoCacheConfig(self.mongo_uri, collection_name=organization)

        client_cache_path: str = os.path.join(self.cache_path, f"{organization}.sqlite")
        # TODO: Allow setting CacheConfig.follower_cache_path from a command line flag.
        return SqliteCacheConfig(client_cache_path)

    def _get_client(self, model: str) -> Client:
        """Return a client based on the model, creating it if necessary."""
        client: Optional[Client] = self.clients.get(model)

        if client is None:
            organization: str = model.split("/")[0]
            cache_config: CacheConfig = self._build_cache_config(organization)

            # TODO: Migrate all clients to use model deployments
            model_deployment = get_model_deployment(model)
            if model_deployment:
                api_key = None
                if "deployments" not in self.credentials:
                    raise AuthenticationError("Could not find key 'deployments' in credentials.conf")
                deployment_api_keys = self.credentials["deployments"]
                if model not in deployment_api_keys:
                    raise AuthenticationError(
                        f"Could not find key '{model}' under key 'deployments' in credentials.conf"
                    )
                api_key = deployment_api_keys[model]
                client = create_object(
                    model_deployment.client_spec, additional_args={"cache_config": cache_config, "api_key": api_key}
                )

            elif get_huggingface_model_config(model):
                from .huggingface_client import HuggingFaceClient

                client = HuggingFaceClient(cache_config=cache_config)
            elif organization == "neurips":
                client = HTTPModelClient(cache_config=cache_config)
            elif organization == "openai":
                from .openai_client import OpenAIClient

                org_id = self.credentials.get("openaiOrgId", None)
                api_key = self.credentials.get("openaiApiKey", None)
                client = OpenAIClient(
                    cache_config=cache_config,
                    api_key=api_key,
                    org_id=org_id,
                )
            elif organization == "AlephAlpha":
                from .aleph_alpha_client import AlephAlphaClient

                client = AlephAlphaClient(api_key=self.credentials["alephAlphaKey"], cache_config=cache_config)
            elif organization == "ai21":
                from .ai21_client import AI21Client

                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_config=cache_config)
            elif organization == "cohere":
                from .cohere_client import CohereClient

                client = CohereClient(api_key=self.credentials["cohereApiKey"], cache_config=cache_config)
            elif organization == "gooseai":
                from .goose_ai_client import GooseAIClient

                org_id = self.credentials.get("gooseaiOrgId", None)
                client = GooseAIClient(
                    api_key=self.credentials["gooseaiApiKey"], cache_config=cache_config, org_id=org_id
                )
            elif organization == "huggingface" or organization == "mosaicml":
                from .huggingface_client import HuggingFaceClient

                client = HuggingFaceClient(cache_config)
            elif organization == "anthropic":
                from .anthropic_client import AnthropicClient

                client = AnthropicClient(
                    api_key=self.credentials.get("anthropicApiKey", None),
                    cache_config=cache_config,
                )
            elif organization == "microsoft":
                from .microsoft_client import MicrosoftClient

                org_id = self.credentials.get("microsoftOrgId", None)
                lock_file_path: str = os.path.join(self.cache_path, f"{organization}.lock")
                client = MicrosoftClient(
                    api_key=self.credentials.get("microsoftApiKey", None),
                    lock_file_path=lock_file_path,
                    cache_config=cache_config,
                    org_id=org_id,
                )
            elif organization == "google":
                from .google_client import GoogleClient

                client = GoogleClient(cache_config=cache_config)
            elif organization == "vertexai":
                from .vertexai_client import VertexAIChatClient
                client = VertexAIChatClient(cache_config=cache_config)
            elif organization in ["together", "databricks", "eleutherai", "meta", "stabilityai"]:
                from .together_client import TogetherClient

                client = TogetherClient(api_key=self.credentials.get("togetherApiKey", None), cache_config=cache_config)
            elif organization == "simple":
                from .simple_client import SimpleClient

                client = SimpleClient(cache_config=cache_config)
            elif organization == "nvidia":
                from .megatron_client import MegatronClient

                client = MegatronClient(cache_config=cache_config)
            else:
                raise ValueError(f"Could not find client for model: {model}")
            self.clients[model] = client
        return client

    def make_request(self, request: Request) -> RequestResult:
        """
        Dispatch based on the the name of the model (e.g., openai/davinci).
        Retries if request fails.
        """

        # TODO: need to revisit this because this swallows up any exceptions that are raised.
        @retry_request
        def make_request_with_retry(client: Client, request: Request) -> RequestResult:
            return client.make_request(request)

        client: Client = self._get_client(request.model)

        try:
            return make_request_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = (
                f"Failed to make request to {request.model} after retrying {last_attempt.attempt_number} times"
            )
            hlog(retry_error)

            # Notify our user that we failed to make the request even after retrying.
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def _get_tokenizer_client(self, tokenizer: str) -> Client:
        """Return a client based on the tokenizer, creating it if necessary."""
        organization: str = tokenizer.split("/")[0]
        client: Optional[Client] = self.tokenizer_clients.get(tokenizer)

        if client is None:
            cache_config: CacheConfig = self._build_cache_config(organization)
            if get_huggingface_model_config(tokenizer):
                from .huggingface_client import HuggingFaceClient

                client = HuggingFaceClient(cache_config=cache_config)
            elif organization == "neurips":
                client = HTTPModelClient(cache_config=cache_config)
            elif organization in [
                "bigscience",
                "bigcode",
                "EleutherAI",
                "facebook",
                "google",
                "gooseai",
                "huggingface",
                "meta-llama",
                "microsoft",
                "hf-internal-testing",
            ]:
                from .huggingface_client import HuggingFaceClient

                client = HuggingFaceClient(cache_config=cache_config)
            elif organization == "openai":
                from .openai_client import OpenAIClient

                client = OpenAIClient(
                    cache_config=cache_config,
                )
            elif organization == "AlephAlpha":
                from .aleph_alpha_client import AlephAlphaClient

                client = AlephAlphaClient(api_key=self.credentials["alephAlphaKey"], cache_config=cache_config)
            elif organization == "anthropic":
                from .anthropic_client import AnthropicClient

                client = AnthropicClient(
                    api_key=self.credentials.get("anthropicApiKey", None), cache_config=cache_config
                )
            elif organization == "TsinghuaKEG":
                from .ice_tokenizer_client import ICETokenizerClient

                client = ICETokenizerClient(cache_config=cache_config)
            elif organization == "Yandex":
                from .yalm_tokenizer_client import YaLMTokenizerClient

                client = YaLMTokenizerClient(cache_config=cache_config)
            elif organization == "ai21":
                from .ai21_client import AI21Client

                client = AI21Client(api_key=self.credentials["ai21ApiKey"], cache_config=cache_config)
            elif organization == "cohere":
                from .cohere_client import CohereClient

                client = CohereClient(api_key=self.credentials["cohereApiKey"], cache_config=cache_config)
            elif organization == "simple":
                from .simple_client import SimpleClient

                client = SimpleClient(cache_config=cache_config)
            elif organization == "nvidia":
                from .megatron_client import MegatronClient

                client = MegatronClient(cache_config=cache_config)
            else:
                raise ValueError(f"Could not find tokenizer client for model: {tokenizer}")
            self.tokenizer_clients[tokenizer] = client
        return client

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        """Tokenizes based on the name of the tokenizer (e.g., huggingface/gpt2)."""

        def tokenize_with_retry(client: Client, request: TokenizationRequest) -> TokenizationRequestResult:
            return client.tokenize(request)

        client: Client = self._get_tokenizer_client(request.tokenizer)

        try:
            return tokenize_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to tokenize after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        """Decodes based on the the name of the tokenizer (e.g., huggingface/gpt2)."""

        def decode_with_retry(client: Client, request: DecodeRequest) -> DecodeRequestResult:
            return client.decode(request)

        client: Client = self._get_tokenizer_client(request.tokenizer)

        try:
            return decode_with_retry(client=client, request=request)
        except RetryError as e:
            last_attempt: Attempt = e.last_attempt
            retry_error: str = f"Failed to decode after retrying {last_attempt.attempt_number} times"
            hlog(retry_error)
            return replace(last_attempt.value, error=f"{retry_error}. Error: {last_attempt.value.error}")
        
    def get_huggingface_client(self) -> ".huggingface_client.HuggingFaceClient":
        """Get the Hugging Face client."""
        from .huggingface_client import HuggingFaceClient

        if self._huggingface_client:
            assert isinstance(self._huggingface_client, HuggingFaceClient)
            return self._huggingface_client
        self._huggingface_client = HuggingFaceClient(self._build_cache_config("huggingface"))
        return self._huggingface_client
