import os.path
import requests
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import base64
from flask_cors import CORS
from src.main_pipleline import initialize_models, run_anpr_pipeline

app = Flask(__name__)
CORS(app)

BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
print('Intializing AI models...')
models_loaded_successfully = initialize_models(BASE_MODEL_DIR)
if not models_loaded_successfully:
    print('WARNING: Models cloud not be loaded. ANPR functionality will be limited or unavailable.')
else:
    print("AI models initialized successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    try:
        image_np_raw = np.frombuffer(image_file.read(), np.uint8)
        original_image = cv2.imdecode(image_np_raw, cv2.IMREAD_COLOR)
        if original_image is None:
            return jsonify({'error': 'Could not decode image, check file format'}), 400

    except Exception as e:
        print(f"Error reading or decoding image: {e}")
        return jsonify({'error': f'Error reading or decoding image: {e}'}), 500

    anpr_results = run_anpr_pipeline(original_image_np=original_image)

    processed_image_base64 = None
    original_image_bbox_base64 = None

    if anpr_results['original_image_with_bbox'] is not None:
        try:
            is_success, buffer = cv2.imencode(".jpg", anpr_results['original_image_with_bbox'])
            if is_success:
                original_image_bbox_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding original_image_with_bbox to Base64: {e}")
            original_image_bbox_base64 = None  # Đặt lại None nếu lỗi

            # Mã hóa ảnh biển số đã crop với bbox ký tự
        if anpr_results['cropped_plate_image_with_char_bbox'] is not None:
            try:
                is_success, buffer = cv2.imencode(".jpg", anpr_results['cropped_plate_image_with_char_bbox'])
                if is_success:
                    processed_image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
            except Exception as e:
                print(f"Error encoding cropped_plate_image_with_char_bbox to Base64: {e}")
                processed_image_base64 = None  # Đặt lại None nếu lỗi
    response_data = {
        'message': anpr_results.get('message', 'Processing complete.'),
        'plate_number': anpr_results.get('plate_number', 'N/A'),
        'original_image_with_bbox_base64': original_image_bbox_base64,
        'cropped_plate_image_with_char_bbox_base64': processed_image_base64,
        'plate_province': anpr_results.get('plate_province', 'N/A'),
        # Các thông tin khác nếu có
        'original_image_shape': original_image.shape[:2] if original_image is not None else [0, 0],
        'processed_image_shape': anpr_results['cropped_plate_image_with_char_bbox'].shape[:2] if anpr_results[
                                                                                                     'cropped_plate_image_with_char_bbox'] is not None else [
            0, 0]
    }

    # Xử lý các trường hợp lỗi từ run_anpr_pipeline
    if anpr_results.get('plate_number') in ['MODELS_NOT_LOADED', 'NOT_DETECTED', 'CROP_FAILED',
                                            'NO_CHARS_DETECTED_OR_ERROR']:
        return jsonify(response_data), 500  # Hoặc 400 tùy mức độ lỗi

    return jsonify(response_data), 200


# --- API Endpoint MỚI để xử lý ảnh từ URL ---
@app.route('/process_image_from_url', methods=['POST'])
def process_image_from_url():
    # 1. Nhận URL từ JSON body
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({'error': 'No image_url provided in JSON body'}), 400

    image_url = data['image_url']

    # 2. Tải ảnh từ URL sử dụng requests
    try:
        response = requests.get(image_url, stream=True, timeout=10)  # Thêm timeout
        response.raise_for_status()  # Kiểm tra lỗi HTTP (4xx, 5xx)
        image_np_raw = np.frombuffer(response.content, np.uint8)
        original_image = cv2.imdecode(image_np_raw, cv2.IMREAD_COLOR)

        if original_image is None:
            return jsonify({'error': 'Could not decode image from URL, check image format or URL validity'}), 400

    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from URL: {e}")
        return jsonify({'error': f'Error fetching image from URL: {e}'}), 500
    except Exception as e:
        print(f"Error processing image from URL: {e}")
        return jsonify({'error': f'Error processing image from URL: {e}'}), 500

    # 3. Gọi pipeline ANPR (logic này giống hệt process_image)
    anpr_results = run_anpr_pipeline(original_image)

    processed_image_base64 = None
    original_image_bbox_base64 = None

    if anpr_results['original_image_with_bbox'] is not None:
        try:
            is_success, buffer = cv2.imencode(".jpg", anpr_results['original_image_with_bbox'])
            if is_success:
                original_image_bbox_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding original_image_with_bbox to Base64: {e}")
            original_image_bbox_base64 = None

    if anpr_results['cropped_plate_image_with_char_bbox'] is not None:
        try:
            is_success, buffer = cv2.imencode(".jpg", anpr_results['cropped_plate_image_with_char_bbox'])
            if is_success:
                processed_image_base64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        except Exception as e:
            print(f"Error encoding cropped_plate_image_with_char_bbox to Base64: {e}")
            processed_image_base64 = None

    response_data = {
        'message': anpr_results.get('message', 'Processing complete.'),
        'plate_number': anpr_results.get('plate_number', 'N/A'),
        'plate_province': anpr_results.get('plate_province', 'N/A'),
        'original_image_with_bbox_base64': original_image_bbox_base64,
        'cropped_plate_image_with_char_bbox_base64': processed_image_base64,
        'original_image_shape': original_image.shape[:2] if original_image is not None else [0, 0],
        'processed_image_shape': anpr_results['cropped_plate_image_with_char_bbox'].shape[:2] if anpr_results[
                                                                                                     'cropped_plate_image_with_char_bbox'] is not None else [
            0, 0]
    }

    if anpr_results.get('plate_number') in ['MODELS_NOT_LOADED', 'NOT_DETECTED', 'CROP_FAILED',
                                            'NO_CHARS_DETECTED_OR_ERROR']:
        return jsonify(response_data), 500

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port= 5050)


