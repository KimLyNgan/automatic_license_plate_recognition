#onnx_handler.py
import numpy as np
import onnxruntime
import os
from typing import Dict, List

class ONNXModelHandler:
    def __init__(self, model_path: str, providers: List[str] | None = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'ONNX model not found at: {model_path}')
        try:
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)
            self.input_name: str = self.session.get_inputs()[0].name
            self.output_names: List[str] = [output.name for output in self.session.get_outputs()]
        except Exception as e:
            print(f'Error loading ONNX model {model_path}: {e}')
            raise

    def run(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        input_feed: Dict[str, np.ndarray] = {self.input_name: input_tensor}
        try:
            output_tensors: List[np.ndarray] = self.session.run(self.output_names, input_feed=input_feed)
            return output_tensors
        except Exception as e:
            print(f'Error running ONNX model: {e}')
            raise