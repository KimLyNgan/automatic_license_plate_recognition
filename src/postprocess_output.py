#postprocess_output.py
from typing import List, Dict, Any
import cv2
import numpy as np

CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
VIETNAM_PLATE_CODES_MAP = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "13": "Quảng Ninh", "14": "Quảng Ninh",
    "15": "Hải Phòng", "16": "Hải Phòng", "17": "Thái Bình", "18": "Nam Định",
    "19": "Phú Thọ", "20": "Thái Nguyên", "21": "Yên Bái", "22": "Tuyên Quang",
    "23": "Hà Giang", "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La",
    "27": "Điện Biên", "28": "Hòa Bình",
    # Hà Nội: 29-33 và 40
    "29": "Hà Nội", "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội", "33": "Hà Nội", "40": "Hà Nội",
    "34": "Hải Dương", "35": "Ninh Bình", "36": "Thanh Hóa", "37": "Nghệ An",
    "38": "Hà Tĩnh", "39": "Đồng Nai", "43": "Đà Nẵng", "47": "Đắk Lắk",
    "48": "Đắk Nông", "49": "Lâm Đồng",
    # TP. Hồ Chí Minh: 41, 50-59
    "41": "TP. Hồ Chí Minh", "50": "TP. Hồ Chí Minh", "51": "TP. Hồ Chí Minh", "52": "TP. Hồ Chí Minh",
    "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh", "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh",
    "57": "TP. Hồ Chí Minh", "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh",
    "60": "Đồng Nai", "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang",
    "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp", "67": "An Giang",
    "68": "Kiên Giang", "69": "Cà Mau", "70": "Tây Ninh", "71": "Bến Tre",
    "72": "Bà Rịa-Vũng Tàu", "73": "Quảng Bình", "74": "Quảng Trị", "75": "Thừa Thiên-Huế",
    "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên", "79": "Khánh Hòa",
    "81": "Gia Lai", "82": "Kon Tum", "83": "Sóc Trăng", "84": "Trà Vinh",
    "85": "Ninh Thuận", "86": "Bình Thuận", "88": "Vĩnh Phúc", "89": "Hưng Yên",
    "90": "Hà Nam", "92": "Quảng Nam", "93": "Bình Phước", "94": "Bạc Liêu",
    "95": "Hậu Giang", "97": "Bắc Kạn", "98": "Bắc Giang", "99": "Bắc Ninh"
}

def postprocess_output(raw_output: List[np.ndarray], confidence_threshold: float = 0.65) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    detections_tensor: np.ndarray | None = raw_output[0][0]
    if detections_tensor is not None and detections_tensor.shape[0] > 0:
        valid_idx = detections_tensor[:, 4] > confidence_threshold
        valid_boxes = detections_tensor[valid_idx]
        for box in valid_boxes:
            bbox = box[:4].tolist()
            confidence = box[4]
            cls_id = box[5]
            height = bbox[3] - bbox[1]
            y_min = bbox[1]
            x_min = bbox[0]
            box_info = {
                'bbox': bbox,
                'confidence': confidence,
                'cls_id': cls_id,
                'height': height,
                'y_min': y_min,
                'x_min': x_min
            }
            detections.append(box_info)
    else:
        if detections_tensor is None:
            print('No detections found in the detections tensor.')
    return detections

def cropped_box(original_image_np: np.ndarray, detections: List[Dict[str, Any]]):
    image_to_crop: np.ndarray = original_image_np.copy()
    ratio_w = image_to_crop.shape[1] / 640
    ratio_h = image_to_crop.shape[0] / 640
    array_image = []
    for detect in detections:
        bbox = detect['bbox']
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min, x_max = int(x_min * ratio_w), int(x_max * ratio_w)
        y_min, y_max = int(y_min * ratio_h), int(y_max * ratio_h)
        image_to_crop = image_to_crop[y_min:y_max, x_min:x_max]
        array_image.append(image_to_crop)
    return array_image

def draw_bbox(detections: List[Dict[str, Any]], image_np: np.ndarray):
    image_drawn: np.ndarray = image_np.copy()
    ratio_w = image_np.shape[1] / 640
    ratio_h = image_np.shape[0] / 640
    for detect in detections:
        bbox = detect['bbox']
        confidence = detect['confidence']
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min, x_max = int(x_min*ratio_w), int(x_max*ratio_w)
        y_min, y_max = int(y_min*ratio_h), int(y_max*ratio_h)
        label = f'{confidence:.2f}'
        cv2.putText(image_drawn, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.rectangle(image_drawn, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    #cv2.imshow('Detection Plate Number', image_drawn)
    return image_drawn

def draw_bbox_character(detections: List[Dict[str, Any]], image_np: np.ndarray):
    image_drawn: np.ndarray = image_np.copy()
    ratio_w = image_np.shape[1] / 640
    ratio_h = image_np.shape[0] / 640
    for detect in detections:
        bbox = detect['bbox']
        cls_id = detect['cls_id']
        x_min, y_min, x_max, y_max = map(int, bbox)
        x_min, x_max = int(x_min * ratio_w), int(x_max * ratio_w)
        y_min, y_max = int(y_min * ratio_h), int(y_max * ratio_h)
        cls_name = CHARACTERS[int(cls_id)]
        label = f'{cls_name}'
        cv2.putText(image_drawn, label, (int((x_max + x_min) / 2), y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        cv2.rectangle(image_drawn, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    #cv2.imshow('Bbox Character', image_drawn)
    return image_drawn

def display_character(detections: List[Dict[str, Any]]):
    list_dict_output = detections
    y_min_array = [det['y_min'] for det in list_dict_output if 'y_min' in det]
    heights = [det['height'] for det in list_dict_output if 'height' in det]
    height_average = np.mean(heights)
    minimum_y = np.min(y_min_array)
    threshold_y = int(minimum_y + 0.85*height_average)
    line1_chars = []
    line2_chars = []

    for det in list_dict_output:
        if det['y_min'] < threshold_y:
            line1_chars.append(det)
        else:
            line2_chars.append(det)
    if len(line1_chars) > 0:
        line1_chars = sorted(line1_chars, key=lambda x: x['x_min'])
    if len(line2_chars) > 0:
        line2_chars = sorted(line2_chars, key=lambda x: x['x_min'])

    plate_number = ''
    for char_info in line1_chars:
        idx = char_info['cls_id']
        plate_number += CHARACTERS[int(idx)]

    for char_info in line2_chars:
        idx = char_info['cls_id']
        plate_number += CHARACTERS[int(idx)]
    return plate_number
    #print(plate_number)

def lookup_province_by_plate(plate_number: str) -> str:
    province_code = plate_number[:2]
    province_name = VIETNAM_PLATE_CODES_MAP.get(province_code, 'Unknown')
    return province_name