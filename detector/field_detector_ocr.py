from ultralytics import YOLO # type: ignore
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import cv2
from PIL import Image
import traceback
import random 
import os
import json
from datetime import datetime

class FieldDetectorOCR:
    FIELD_NAMES = {
        0: 'current_place',
        1: 'dob',
        2: 'expire_date',
        3: 'gender',
        4: 'id',
        5: 'name',
        6: 'nationality',
        7: 'origin_place',
    }

    def __init__(self, model_path: str, conf_threshold: float=0.4 , ocr_model: str = 'vgg_transformer'):
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            print(f"Loaded field detection model from {model_path}")

            self._init_vietocr(ocr_model)

        except Exception as e:
            print(f" Error when loading models: {e}")
            raise

    def _init_vietocr(self, model_type='vgg_transformer'):
        try:
            config = Cfg.load_config_from_name(model_type)
            config['cnn']['pretrained'] = False
            config['device'] = 'cuda:0' # cpu
            config['predictor']['beamsearch'] = False

            self.ocr_predictor = Predictor(config)
            print(f"Loaded VietOCR model: {model_type}")
        except Exception as e:
            print(f"Error initializing VietOCR: {e}")
            raise
        
    def detect_and_recognize(self, card_image, padding=2) -> dict:
        """
        Detect fields and perform OCR in one pass
        
        Args:
            card_image: Preprocessed card image
            padding: Padding around field when cropping (pixels)
            
        Returns:
            dict: {
                'success': True/False,
                'fields': [
                    {
                        'field_name': 'name',
                        'class_id': 1,
                        'bbox': (x1, y1, x2, y2),
                        'bbox_confidence': 0.95,
                        'crop': cropped image,
                        'ocr_text': 'NGUYEN VAN A',
                        'ocr_confidence': 0.88
                    },
                    ...
                ],
                'total_fields': 9
            }
        """
        #----------- FIELD DETECTION -----------
        try:
            results = self.model(card_image, conf=self.conf_threshold, verbose=False)
            if results[0].boxes is None or len(results[0].boxes) == 0:
                return {
                    'success': False,
                    'message': 'No field detected',
                    'fields': [],
                    'total_field': 0
                }
            
            fields = []
            h, w = card_image.shape[:2]

            print(f"Detected {len(results[0].boxes)} fields, performing OCR...")

            #---------------PROCESS EACH FIELD--------------------
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                bbox_conf = float(box.conf[0])
                field_name = self.FIELD_NAMES.get(class_id, '')

                # Add padding
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)

                field_crop = card_image[y1:y2, x1:x2]

                # ---------- OCR ON FIELD -----------
                ocr_text, ocr_conf = self._recognize_field(field_crop, field_name)
                fields.append({
                    'field_name': field_name,
                    'class_id': class_id,
                    'bbox': (x1, y1, x2, y2),
                    'bbox_confidence': bbox_conf,
                    'crop': field_crop,
                    'ocr_text': ocr_text,
                    'ocr_confidence': ocr_conf
                })

                print(f" [{idx+1}] {field_name:12s}: '{ocr_text}' (det: {bbox_conf:3f}, ocr: {ocr_conf:3f}) ")
            fields.sort(key=lambda x: x['class_id'])
            return {
                'success': True,
                'fields': fields,
                'total_fields': len(fields)
            }
        
        except Exception as e:
            print(f"Detection/OCR error: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'message': str(e),
                'fields': [],
                'total_fields': 0
            }
    
    def _recognize_field(self, field_crop, field_name) -> tuple:
        """
        Args:
            field_crop: Cropped field image (BGR)
            field_name: name of the field
        Returns:
            tuple: (text, conf)
        """
        try:
            processed = self._preprocess_for_ocr(field_crop, field_name)

            # BGR -> RGB for VietOCR requirement
            rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            ocr_result = self.ocr_predictor.predict(pil_image, return_prob=True)

            if isinstance(ocr_result, tuple):
                text, conf = ocr_result
            else:
                text = ocr_result
                conf = 0
            
            # Post-process text
            text = self._postprocess_text(text, field_name)
            return text, conf
        except Exception as e:
            print(f"OCR error for {field_name}: {e}")
            return "", 0.0

    def _preprocess_for_ocr(self, field_crop, field_name):
        """
        Args:
            field_crop: Original field crop
            field_name: Field name for custom processing
            
        Returns:
            Preprocessed image
        """

        gray = cv2.cvtColor(field_crop, cv2.COLOR_BGR2GRAY)
        if field_name in ['id', 'dob', 'expire_date']:
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, blockSize=11, C=2 
            )            

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Resized if too small 
        h, w = denoised.shape[:2]
        if h < 32:
            scale = 32/h
            new_w = int(w * scale)
            denoised = cv2.resize(denoised, (new_w, 32), interpolation=cv2.INTER_CUBIC)
        
        # Convert back to BGR
        bgr = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        return bgr

    def _postprocess_text(self, text, field_name):
        """        
        Args:
            text: Raw OCR text
            field_name: Field name
            
        Returns:
            Cleaned text
        """

        text = ' '.join(text.split())
        if field_name == 'id':
            # ID number: keep only digits
            text = ''.join(c for c in text if c.isdigit())
        
        elif field_name in ['dob', 'expire_date']:
            # Date: format as DD/MM/YYYY
            digits = ''.join(c for c in text if c.isdigit())
            if len(digits) == 8:
                text = f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
        
        elif field_name == 'gender':
            text = text.upper()
            if 'NAM' in text in text:
                text = 'Nam'
            elif 'NỮ' in text or 'NU' in text:
                text = 'Nữ'
        
        elif field_name == 'name':
            text = text.upper()
        
        return text

    def visualize(self, card_image, detection_result, save_path=None):
        if not detection_result['success']:
            print(f"{detection_result['message']}")
            return
        vis_image = card_image.copy()
        colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) for _ in range(9)]
        for field in detection_result['fields']:
            x1, y1, x2, y2 = field['bbox']
            color = colors[field['class_id'] % len(colors)]
            cv2.rectangle(vis_image, (x1,y1), (x2,y2), color, 2)

            # Draw label with OCR text
            label = f"{field['field_name']}"

            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            
            # Text
            cv2.putText(vis_image, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"Saved visualization: {save_path}")
        
        return vis_image
    
    def save_fields_crops(self, detection_result, output_dir='field_crops'):
        if not detection_result['success']:
            print(f"{detection_result['message']}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        for idx, field in enumerate(detection_result['fields']):
            filename = f"{idx:02d}_{field['field_name']}.jpg"
            # Clean filename
            filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, field['crop'])
        
        print(f"Saved {len(detection_result['fields'])} field crops to: {output_dir}")
    
    def export_to_json(self, detection_result, output_path=None, include_metadata=True):
        """
        Returns:
            {
                "id": "001234567890",
                "name": "Pham Minh Tu",
                "dob": 24/02/2004,
                ...
            }
        """
        if not detection_result['success']:
            print(f"Cannot export: {detection_result['message']}")
            return None
        
        json_data = {
            'sucess': True,
            'total_fields': detection_result['total_fields'],
            'timestamp': datetime.now().isoformat(),
            'fields': {}
        }

        for field in detection_result['fields']:
            field_info = {
                'text': field['ocr_text'],
            }

            if include_metadata:
                field_info['detection_confidence'] = round(field['bbox_confidence'], 4)
                field_info['ocr_confidence'] = round(field['ocr_confidence'], 4)
                #field_info['bbox'] = field['bbox']
            json_data['fields'][field['field_name']] = field_info
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ocr_result_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
                return json_data
        except Exception as e:
            print(f"Error when saving JSON: {e}")
            return None

# ============= TEST MODULE =============

def test_field_detector_ocr():
    """
    Test Combined Field Detector + OCR
    """
    print("="*60)
    print("TEST MODULE: FIELD DETECTOR + OCR")
    print("="*60)
    
    # Load preprocessed card image
    card_image_path = 'debug_steps/preprocessed_result_after.jpg'
    
    card_image = cv2.imread(card_image_path)
    if card_image is None:
        print(f" Cannot read image: {card_image_path}")
        return None
    
    print(f"\nInput: {card_image_path}")
    print(f"Size: {card_image.shape[1]}x{card_image.shape[0]}")
    
    detector = FieldDetectorOCR(
        model_path='model/Text_field_detect_best_03_12_25.pt',
        conf_threshold=0.4,
        ocr_model='vgg_transformer'
    )
    
    # Detect + OCR
    print("\nDetecting fields and performing OCR...")
    result = detector.detect_and_recognize(card_image, padding=2)
    
    if result['success']:
        print(f"\n SUCCESS!")
        print(f"  Total fields: {result['total_fields']}/10")
        
        print("\n" + "="*60)
        print("EXTRACTED DATA")
        print("="*60)
        for field in result['fields']:
            print(f"{field['field_name']:12s}: {field['ocr_text']}")
        
        detector.visualize(card_image, result, save_path='debug_steps/field_ocr_result.jpg')
        detector.save_fields_crops(result, output_dir='field_crops_ocr')
        detector.export_to_json(result, output_path='results/id_card_data.json')
    else:
        print(f"\n FAILED: {result['message']}")
    
    return result


if __name__ == "__main__":
    test_field_detector_ocr()