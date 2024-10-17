from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union, cast
import torch
from models_host.base_inference_spec import BaseInferenceSpec

from transformers import pipeline

import os

class InferenceData(BaseModel):
    name: str
    shape: List[int]
    data: Union[List[str], List[float]]
    datatype: str


class InputRequest(BaseModel):
    inputs: List[InferenceData]


class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[InferenceData]


class InferenceSpec(BaseInferenceSpec):
    model = None
    model_name = "facebook/bart-large-mnli"

    @property
    def torch_device(self):
        env = os.environ.get("env", "dev")
        torch_device = "cuda" if env == "prod" else "cpu"
        return torch_device

    def load(self):
        model_name = self.model_name
        torch_device = self.torch_device
        print(f"Loading model {model_name} using device {torch_device}...")
        self.model = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=torch.device(torch_device),
            hypothesis_template="This example has to do with topic {}.",
            multi_label=True,
        )

    def process_request(self, input_request: InputRequest):
        # get args, kwargs to pass to the infer method
        text_vals = None
        candidate_topics = None
        zero_shot_threshold = 0.5

        for inp in input_request.inputs:
            if inp.name == "text":
                text_vals = inp.data
            elif inp.name == "candidate_topics":
                candidate_topics = inp.data
            elif inp.name == "zero_shot_threshold":
                zero_shot_threshold = float(inp.data[0])

        if text_vals is None or candidate_topics is None:
            raise HTTPException(status_code=400, detail="Invalid input format")

        args = (text_vals, candidate_topics, zero_shot_threshold)
        kwargs = {}
        return args, kwargs

    def infer(self, text_vals, candidate_topics, threshold) -> OutputResponse:
        outputs = []
        for idx, text in enumerate(text_vals):
            results = self.model(text, candidate_topics) # type: ignore
            pred_labels = [
                label for label, score in zip(results["labels"], results["scores"]) if score > threshold 
            ]

            if not pred_labels:
                pred_labels = ["No valid topic found."]

            outputs.append(
                InferenceData(
                    name=f"result{idx}",
                    datatype="BYTES",
                    shape=[len(pred_labels)],
                    data=pred_labels,
                )
            )

        output_data = OutputResponse(
            modelname="RestrictToTopicModel", modelversion="1", outputs=outputs
        )

        return output_data