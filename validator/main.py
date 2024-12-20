import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.stores.context import get_call_kwarg
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import pipeline


@register_validator(
    name="tryolabs/restricttotopic", data_type="string", has_guardrails_endpoint=True
)
class RestrictToTopic(Validator):
    """Checks if text's main topic is specified within a list of valid topics
    and ensures that the text is not about any of the invalid topics.

    This validator accepts at least one valid topic and an optional list of
    invalid topics.

    Default behavior first runs a Zero-Shot model, and then falls back to
    ask OpenAI's `gpt-3.5-turbo` if the Zero-Shot model is not confident
    in the topic classification (score < 0.5).

    In our experiments this LLM fallback increases accuracy by 15% but also
    increases latency (more than doubles the latency in the worst case).

    Both the Zero-Shot classification and the GPT classification may be toggled.

    **Key Properties**

    | Property                      | Description                              |
    | ----------------------------- | ---------------------------------------- |
    | Name for `format` attribute   | `tryolabs/restricttotopic`               |
    | Supported data types          | `string`                                 |
    | Programmatic fix              | Removes lines with off-topic information |

    Args:
        valid_topics (List[str]): topics that the text should be about
            (one or many).
        invalid_topics (List[str], Optional, defaults to []): topics that the
            text cannot be about.
        device (Optional[Union[str, int]], Optional, defaults to -1): Device ordinal for
            CPU/GPU supports for Zero-Shot classifier. Setting this to -1 will leverage
            CPU, a positive will run the Zero-Shot model on the associated CUDA
            device id.
        model (str, Optional, defaults to 'facebook/bart-large-mnli'): The
            Zero-Shot model that will be used to classify the topic. See a
            list of all models here:
            https://huggingface.co/models?pipeline_tag=zero-shot-classification
        llm_callable (Union[str, Callable, None], Optional, defaults to
            'gpt-4o'): Either the name of the OpenAI model, or a callable
            that takes a prompt and returns a response.
        disable_classifier (bool, Optional, defaults to False): controls whether
            to use the Zero-Shot model. At least one of disable_classifier and
            disable_llm must be False.
        classifier_api_endpoint (str, Optional, defaults to None): An API endpoint
            to recieve post requests that will be used when provided. If not provided, a
            local model will be initialized.
        disable_llm (bool, Optional, defaults to False): controls whether to use
            the LLM fallback. At least one of disable_classifier and
            disable_llm must be False.
        zero_shot_threshold (float, Optional, defaults to 0.5): The threshold used to
            determine whether to accept a topic from the Zero-Shot model. Must be
            a number between 0 and 1.
        llm_threshold (int, Optional, defaults to 3): The threshold used to determine
        if a topic exists based on the provided llm api. Must be between 0 and 5.
    """

    def __init__(
        self,
        valid_topics: Optional[List[str]] = None,
        invalid_topics: Optional[List[str]] = None,
        device: Optional[Union[str, int]] = -1,
        model: Optional[str] = "facebook/bart-large-mnli",
        llm_callable: Union[str, Callable, None] = None,
        disable_classifier: Optional[bool] = False,
        classifier_api_endpoint: Optional[str] = None,
        disable_llm: Optional[bool] = False,
        on_fail: Optional[Callable[..., Any]] = None,
        zero_shot_threshold: Optional[float] = 0.5,
        llm_threshold: Optional[int] = 3,
        **kwargs,
    ):
        super().__init__(
            valid_topics=valid_topics,
            invalid_topics=invalid_topics,
            device=device,
            model=model,
            disable_classifier=disable_classifier,
            classifier_api_endpoint=classifier_api_endpoint,
            disable_llm=disable_llm,
            llm_callable=llm_callable,
            on_fail=on_fail,
            zero_shot_threshold=zero_shot_threshold,
            llm_threshold=llm_threshold,
            **kwargs,
        )
        if valid_topics is None:
            self._valid_topics = []
        else:
            self._valid_topics = valid_topics
        if invalid_topics is None:
            self._invalid_topics = []
        else:
            self._invalid_topics = invalid_topics

        if not (self._valid_topics or self._invalid_topics):
            raise ValueError("Either valid topics or invalid topics must be specified.")

        self._device = (
            str(device).lower()
            if str(device).lower() in ["cpu", "mps"]
            else int(device)
        )
        self._model = model
        self._disable_classifier = disable_classifier
        self._disable_llm = disable_llm
        self._classifier_api_endpoint = classifier_api_endpoint

        self._zero_shot_threshold = zero_shot_threshold
        if self._zero_shot_threshold < 0 or self._zero_shot_threshold > 1:
            raise ValueError("zero_shot_threshold must be a number between 0 and 1")

        self._llm_threshold = llm_threshold
        if self._llm_threshold < 0 or self._llm_threshold > 5:
            raise ValueError("llm_threshold must be a number between 0 and 5")
        self.set_callable(llm_callable)

        if self._classifier_api_endpoint is None and self.use_local:
            self._classifier = pipeline(
                "zero-shot-classification",
                model=self._model,
                device=self._device,
                hypothesis_template="This example has to do with topic {}.",
                multi_label=True,
            )
        else:
            # TODO api endpoint
            ...

    def get_topics_ensemble(self, text: str, candidate_topics: List[str]) -> List[str]:
        """Finds the topics in the input text based on if it is determined by the zero
        shot model or the llm.

        Args:
            text (str): The input text to find categories from
            candidate_topics (List[str]): The topics to search for in the input text

        Returns:
            List[str]: The found topics
        """
        # Find topics based on zero shot model
        zero_shot_topics = self._inference({"text": text, "valid_topics": candidate_topics, "invalid_topics": []})

        # Find topics based on llm
        llm_topics = self.get_topics_llm(text, candidate_topics)

        return list(set(zero_shot_topics + llm_topics))

    def get_topics_llm(self, text: str, candidate_topics: List[str]) -> List[str]:
        """Returns a list of the topics identified in the given text using an LLM
        callable

        Args:
            text (str): The input text to classify topics.
            candidate_topics (List[str]): The topics to identify if present in the text.

        Returns:
            List[str]: The topics found in the input text.
        """
        llm_topics = self.call_llm(text, candidate_topics)
        found_topics = []
        for llm_topic in llm_topics:
            if llm_topic in candidate_topics:
                found_topics.append(llm_topic)
        return found_topics

    def get_client_args(self) -> Tuple[Optional[str], Optional[str]]:
        """Returns neccessary data for api calls.

        Returns:
            str: api key
        """

        load_dotenv()
            
        api_key = get_call_kwarg("api_key") or os.environ.get("OPENAI_API_KEY")
        api_base = get_call_kwarg("api_base") or os.environ.get("OPENAI_API_BASE")

        return (api_key, api_base)

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def call_llm(self, text: str, topics: List[str]) -> str:
        """Call the LLM with the given prompt.

        Expects a function that takes a string and returns a string.
        Args:
            text (str): The input text to classify using the LLM.
            topics (List[str]): The list of candidate topics.
        Returns:
            response (str): String representing the LLM response.
        """
        return self._llm_callable(text, topics)

    def set_callable(self, llm_callable: Union[str, Callable, None]) -> None:
        """Set the LLM callable.

        Args:
            llm_callable: Either the name of the OpenAI model, or a callable that takes
                a prompt and returns a response.
        """

        if llm_callable is None:
            llm_callable = "gpt-4o"

        if isinstance(llm_callable, str):
            if llm_callable not in ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]:
                raise ValueError(
                    "llm_callable must be one of 'gpt-3.5-turbo', 'gpt-4', or 'gpt-4o'"
                    "If you want to use a custom LLM, please provide a callable."
                    "Check out ProvenanceV1 documentation for an example."
                )

            def openai_callable(text: str, topics: List[str]) -> str:
                api_key, api_base = self.get_client_args()
                client = OpenAI(api_key=api_key, base_url=api_base)
                response = client.chat.completions.create(
                    model=llm_callable,
                    response_format={"type": "json_object"},
                    messages=[
                        {
                            "role": "user",
                            "content": f"""
                                Given a text and a list of topics, return a valid json list of which topics are present in the text. If none, just return an empty list.

                                Output Format:
                                -------------
                                "topics_present": []

                                Text:
                                ----
                                "{text}"

                                Topics: 
                                ------
                                {topics}

                                Result:
                                ------ """,
                        },
                    ],
                )
                return json.loads(response.choices[0].message.content)["topics_present"]

            self._llm_callable = openai_callable
        elif isinstance(llm_callable, Callable):
            self._llm_callable = llm_callable
        else:
            raise ValueError("llm_callable must be a string or a Callable")

    def validate(
        self, value: str, metadata: Optional[Dict[str, Any]] = {}
    ) -> ValidationResult:
        """Validates that a string contains at least one valid topic and no invalid topics.

        Args:
            value (str): The given string to classify
            metadata (Optional[Dict[str, Any]], optional): Dictionary containing valid and invalid topics. Defaults to {}.


        Raises:
            ValueError: If a topic is invalid and valid
            ValueError: If no valid topics are set
            ValueError: If there is no llm or zero shot classifier set

        Returns:
            ValidationResult: PassResult if a topic is restricted and valid,
            FailResult otherwise
        """
        valid_topics = set(metadata.get('valid_topics', self._valid_topics))
        invalid_topics = set(metadata.get('invalid_topics', self._invalid_topics))
        all_topics = list(valid_topics | invalid_topics)

        # If there aren't any valid or invalid topics...
        if len(all_topics) == 0:
            raise ValueError(
                "There must be at least one valid or invalid topic."
            )
        
        # Throw if valid and invalid topics are not disjoint:
        if bool(valid_topics.intersection(invalid_topics)):
            raise ValueError("A topic cannot be valid and invalid at the same time.")

        model_input = {
            "text": value,
            "valid_topics": valid_topics,
            "invalid_topics": invalid_topics
        }
        
        # Ensemble method
        if not self._disable_classifier and not self._disable_llm:
            found_topics = self.get_topics_ensemble(value, all_topics)
        # LLM Classifier Only
        elif self._disable_classifier and not self._disable_llm:
            found_topics = self.get_topics_llm(value, all_topics)
        # Zero Shot Classifier Only
        elif not self._disable_classifier and self._disable_llm:
            found_topics = self._inference(model_input)
        else:
            raise ValueError("Either classifier or llm must be enabled.")
    
        # Determine if valid or invalid topics were found
        invalid_topics_found = []
        valid_topics_found = []
        for topic in found_topics:
            if topic in valid_topics:
                valid_topics_found.append(topic)
            elif topic in invalid_topics:
                invalid_topics_found.append(topic)

        error_spans = []
        
        # Require no invalid topics and, if present, at least one valid topic.
        if invalid_topics_found:
            for topic in invalid_topics_found:
                error_spans.append(
                    ErrorSpan(
                        start=value.find(topic),
                        end=value.find(topic) + len(topic),
                        reason=f"Text contains invalid topic: {topic}",
                    )
                )
            return FailResult(
                error_message=f"Invalid topics found: {invalid_topics_found}",
                error_spans=error_spans
            )
        if len(valid_topics) > 0 and not valid_topics_found:
            return FailResult(
                error_message="No valid topic was found.",
                error_spans=[ErrorSpan(
                    start=0,
                    end=len(value),
                    reason="No valid topic was found."
                )]
            )
        return PassResult()
    
    def _inference_local(self, model_input: Any) -> Any:
        """Local inference method for the restrict-to-topic validator."""
        text = model_input["text"]

        valid_topics = model_input["valid_topics"]
        invalid_topics = model_input["invalid_topics"]

        # There's a chance that valid topics will be passed as a plain string or a set.
        # If that happens the '+ action might fail and the call to _classifier will
        # not behave as expected.
        if isinstance(valid_topics, str):
            valid_topics = [valid_topics, ]
        elif isinstance(valid_topics, set):
            valid_topics = list(valid_topics)
        if isinstance(invalid_topics, str):
            invalid_topics = [invalid_topics, ]
        elif isinstance(invalid_topics, set):
            invalid_topics = list(invalid_topics)
        candidate_topics = valid_topics + invalid_topics

        result = self._classifier(text, candidate_topics)
        topics = result["labels"]
        scores = result["scores"]
        found_topics = []
        for topic, score in zip(topics, scores):
            if score > self._zero_shot_threshold:
                found_topics.append(topic)
        return found_topics

    
    def _inference_remote(self, model_input: Any) -> Any:
        """Remote inference method for the restrict-to-topic validator."""
        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [1],
                    "data": [model_input["text"]],
                    "datatype": "BYTES"
                },
                {
                    "name": "candidate_topics",
                    "shape": [len(model_input["valid_topics"]) + len(model_input["invalid_topics"])],
                    "data": list(model_input["valid_topics"]) + list(model_input["invalid_topics"]),
                    "datatype": "BYTES"
                },
                {
                    "name": "zero_shot_threshold",
                    "shape": [1],
                    "data": [self._zero_shot_threshold],
                    "datatype": "FP32"
                }
            ]
        }
        
        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)
        
        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)
        
        return response["outputs"][0]["data"]