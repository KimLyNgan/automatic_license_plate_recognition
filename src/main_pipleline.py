import cv2
import numpy as np
import os
from src.onnx_handler import ONNXModelHandler
from src.image_utils import  preprocess_for_onnx
from src.postprocess_output import postprocess_output, display_character, draw_bbox, cropped_box, draw_bbox_character, \
    lookup_province_by_plate

plate_model_handler = None
char_model_handler = None

def initialize_models(base_path_to_models: str) -> bool: # Hàm này cần nhận đường dẫn
    global plate_model_handler, char_model_handler
    try:
        # Sử dụng os.path.join để xây dựng đường dẫn an toàn
        plate_model_path = os.path.join(base_path_to_models, 'best_new.onnx')
        char_model_path = os.path.join(base_path_to_models, 'kytubiensoxe.onnx')

        plate_model_handler = ONNXModelHandler(plate_model_path, ['CPUExecutionProvider'])
        char_model_handler = ONNXModelHandler(char_model_path, ['CPUExecutionProvider'])
        print("ONNX models loaded successfully in main_pipeline.")
        return True
    except Exception as e:
        print(f'Failed to load ONNX model in main_pipeline: {e}')
        plate_model_handler = None
        char_model_handler = None
        return False

def run_anpr_pipeline(original_image_np: np.ndarray):
    if plate_model_handler is None or char_model_handler is None:
        return {
            'plate_number': 'MODELS_NOT_LOADED',
            'original_image_with_bbox': original_image_np,
            'cropped_plate_image_with_char_bbox': None,
            'message': 'AI models are not loaded. Please restart the server.'
        }
    image_for_display = original_image_np.copy()
    # --- Stage 1: Phát hiện biển số ---
    st1_image_input_tensor = preprocess_for_onnx(
        image_np=image_for_display,
        is_fix_target_size=False,
        target_size=(640, 640)
    )
    st1_raw_output = plate_model_handler.run(st1_image_input_tensor)
    st1_detections = postprocess_output(
        raw_output=st1_raw_output,
        confidence_threshold=0.65
    )
    drawn_original_image = draw_bbox(
        detections=st1_detections,
        image_np=image_for_display,
    )
    if not st1_detections:
        return {
            'plate_number': 'NOT_DETECTED',
            'original_image_with_bbox': drawn_original_image,
            'cropped_plate_image_with_char_bbox': None,
            'message': 'No license plate detected in the image.'
        }
    st1_cropped_images = cropped_box(
        original_image_np=original_image_np,
        detections=st1_detections,
    )
    if not st1_cropped_images:
        return {
            'plate_number': 'CROP_FAILED',
            'original_image_with_bbox': drawn_original_image,
            'cropped_plate_image_with_char_bbox': None,
            'message': 'Could not crop license plate image.'
        }
    cropped_plate_image = st1_cropped_images[0]
    # --- Stage 2: Nhận diện ký tự ---
    st2_input_image = cropped_plate_image.copy()
    st2_input_image_tensor = preprocess_for_onnx(
        image_np=st2_input_image,
        is_fix_target_size=False,
        target_size=(640, 640)
    )
    st2_raw_output = char_model_handler.run(st2_input_image_tensor)
    st2_detections_for_char_display = postprocess_output(
        raw_output=st2_raw_output,
        confidence_threshold=0.5
    )
    drawn_cropped_plate_image = draw_bbox_character(
        detections=st2_detections_for_char_display,
        image_np=st2_input_image,
    )
    recognized_plate_string = display_character(st2_detections_for_char_display)
    if not recognized_plate_string:
        recognized_plate_string = "NO_CHARS_DETECTED_OR_ERROR"

    province_name = lookup_province_by_plate(recognized_plate_string)

    return {
        'plate_number': recognized_plate_string,
        'plate_province': province_name,
        'original_image_with_bbox': drawn_original_image,
        'cropped_plate_image_with_char_bbox': drawn_cropped_plate_image,
        'message': 'License plate recognized successfully.'
    }

# if __name__ == '__main__':
#     BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
#     if initialize_models(BASE_MODEL_DIR):
#         test_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'biensoxe.PNG')
#         if os.path.exists(test_image_path):
#             test_image = cv2.imread(test_image_path)
#             if test_image is not None:
#                 results = run_anpr_pipeline(test_image)
#                 print(f"Recognized Plate: {results['plate_number']}")
#                 if results['cropped_plate_image_with_char_bbox'] is not None:
#                     cv2.imshow("Cropped Plate with Char BBox", results['cropped_plate_image_with_char_bbox'])
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()



