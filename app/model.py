import torch
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np
from config import CONFIG


class Model:
    """
    Model wrapper for BERTRegressor
    """
    def __init__(self, model_path: str = CONFIG['MODEL_PATH']):
        """
        Initialize the model wrapper

        :param model_path: str, path to the ONNX model
        """

        self.model_path = model_path

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['BERT_MODEL'])
        
        self.session = ort.InferenceSession(self.model_path)

    def _preprocess(self, text: str):
        """
        Preprocess the input text for the model

        :param text: str, input text to be preprocessed
        :return: dict, preprocessed input for the model
        """

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=CONFIG.get('MAX_LEN', 2048),
            return_tensors="np"
        )

        return {
            'input_ids': enc['input_ids'].astype(np.int64),
            'attention_mask': enc['attention_mask'].astype(np.int64)
        }

    def predict(self, text: str) -> float:
        """
        Predict the rating for the input text
        :param text: str, input text to be predicted
        :return: float, predicted rating
        """

        inputs = self._preprocess(text)

        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }

        outputs = self.session.run(None, ort_inputs)

        result = outputs[0]

        if isinstance(result, np.ndarray):
            return float(result.flatten()[0])

        return float(result)