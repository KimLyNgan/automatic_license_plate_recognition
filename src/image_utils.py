#image_utils.py
import os
from typing import Tuple
import cv2
import numpy as np

def load_image(image_path: str) -> np.ndarray | None:
    if not os.path.exists(image_path):
        print(f'Lỗi ko tìm thấy ảnh ở đường dẫn {image_path}')
        return None
    image_np: np.ndarray = cv2.imread(image_path)
    return image_np

def preprocess_for_onnx(image_np: np.ndarray, is_fix_target_size: bool = True, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    if image_np is None:
        raise
    if is_fix_target_size:
        image_processed = image_np
    else:
        image_processed = cv2.resize(image_np, target_size, interpolation=cv2.INTER_LINEAR)
    image_normalized = image_processed.astype(np.float32) / 255
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_tensor = np.expand_dims(image_transposed, axis=0)
    return image_tensor
