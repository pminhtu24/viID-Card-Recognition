from ultralytics import YOLO # type: ignore
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor
import torch
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

    def __init__(self, model_path: str, 
                 conf_threshold: float=0.4, 
                 ocr_model: str = 'vgg_transformer',
                 custom_ocr_weights = None):
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            print(f"Loaded field detection model from {model_path}")

            self._init_vietocr(ocr_model, custom_ocr_weights)

        except Exception as e:
            print(f"Error when loading models: {e}")
            raise

    def _init_vietocr(self, model_type='vgg_transformer', custom_weights_path=None):
        try:
            config = Cfg.load_config_from_name(model_type)
            
            config['cnn']['pretrained'] = False  
            if custom_weights_path and os.path.exists(custom_weights_path):
                print(f"Loading custom weights: {custom_weights_path}")
                config['weights'] = custom_weights_path
            else:
                print(f"No custom weights provided, using pretrained: {model_type}")
            
            config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            config['predictor']['beamsearch'] = False
            
            self.ocr_predictor = Predictor(config)
            print(f"VietOCR initialized successfully!")
            print(f"Device: {config['device']}")
            print(f"Weights: {config.get('weights', 'default pretrained')}")

            
        except Exception as e:
            print(f"Error initializing VietOCR: {e}")
            traceback.print_exc()
            raise

    def detect_and_recognize(self, card_image, padding=2) -> dict:
        """
        Detect fields and perform OCR in one pass
        
        Args:
            card_image: Preprocessed card image (BGR format from cv2)
            padding: Padding around field when cropping (pixels)
            
        Returns:
            dict: {
                'success': True/False,
                'fields': [...],
                'total_fields': int
            }
        """
        try:
            # ========== FIELD DETECTION ==========
            results = self.model(card_image, conf=self.conf_threshold, verbose=False)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                return {
                    'success': False,
                    'message': 'No field detected',
                    'fields': [],
                    'total_fields': 0
                }
            
            fields = []
            h, w = card_image.shape[:2]

            print(f"\nDetected {len(results[0].boxes)} fields, performing OCR...")

            # ========== PROCESS EACH FIELD ==========
            for idx, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                bbox_conf = float(box.conf[0])
                field_name = self.FIELD_NAMES.get(class_id, f'unknown_{class_id}')

                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(w, x2 + padding)
                y2 = min(h, y2 + padding)
                field_crop = card_image[y1:y2, x1:x2]

                # ========== OCR ON FIELD ==========
                ocr_text, ocr_conf = self._recognize_field(field_crop, field_name)
                
                if ocr_conf is None:
                    ocr_conf = 0.0
                
                fields.append({
                    'field_name': field_name,
                    'class_id': class_id,
                    'bbox': (x1, y1, x2, y2),
                    'bbox_confidence': bbox_conf,
                    'crop': field_crop,
                    'ocr_text': ocr_text,
                    'ocr_confidence': ocr_conf
                })

                print(f"[{idx+1:2d}] {field_name:15s}: '{ocr_text:30s}' (det: {bbox_conf:.3f}, ocr: {ocr_conf:.3f})")
            
            # Sort by class_id for consistent output
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
        Perform OCR on a field crop
        
        Args:
            field_crop: Cropped field image (BGR format)
            field_name: Name of the field
            
        Returns:
            tuple: (text, confidence)
        """
        try:
            # ========== PREPROCESSING ==========
            processed = self._preprocess_for_ocr(field_crop, field_name)
            
            # ========== BGR -> RGB for VietOCR ==========
            rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # ========== RUN OCR ==========
            ocr_result = self.ocr_predictor.predict(pil_image, return_prob=True)

            if isinstance(ocr_result, tuple):
                text, conf = ocr_result
            else:
                text = ocr_result
                conf = 0.0
            
            # ========== POST-PROCESS TEXT ==========
            text = self._postprocess_text(text, field_name)
            
            return text, conf
            
        except Exception as e:
            print(f" OCR error for '{field_name}': {e}")
            traceback.print_exc()
            return "", 0.0

    def _preprocess_for_ocr(self, field_crop, field_name):
        return field_crop

    def _postprocess_text(self, text, field_name):
        text = ' '.join(text.split())
        if field_name == 'id':
            text = ''.join(c for c in text if c.isdigit())
        
        elif field_name in ['dob', 'expire_date']:
            # Date: Format as DD/MM/YYYY
            digits = ''.join(c for c in text if c.isdigit())
            if len(digits) >= 8:
                text = f"{digits[0:2]}/{digits[2:4]}/{digits[4:8]}"
            else:
                # Keep original if not enough digits
                pass
        
        elif field_name == 'gender':
            text = text.upper()
            if 'NAM' in text:
                text = 'Nam'
            elif 'NỮ' in text or 'NU':
                text = 'Nữ'
        
        elif field_name == 'name':
            text = text.upper()
        elif field_name in ['origin_place', 'current_place']:
            text = text.strip()
            if text.lower().startswith('tỉnh '):
                text = text[5:].strip() 
            elif text.lower().startswith('tinh '):
                text = text[5:].strip()
            text = text.replace('Tỉnh ', '').replace('tỉnh ', '')
            text = text.replace('Tinh ', '').replace('tinh ', '')
        elif field_name == 'nationality':
            if not text.strip():
                text = 'Việt Nam'
        return text

    def visualize(self, card_image, detection_result, save_path=None):
        if not detection_result['success']:
            print(f"Cannot visualize: {detection_result['message']}")
            return
        
        vis_image = card_image.copy()
        
        # Generate colors for each field type
        colors = [(random.randint(0,255), random.randint(0,255), random.randint(0,255)) 
                  for _ in range(len(self.FIELD_NAMES))]
        
        for field in detection_result['fields']:
            x1, y1, x2, y2 = field['bbox']
            color = colors[field['class_id'] % len(colors)]
            
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            label = f"{field['field_name']}"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x1, y1 - text_h - 6), (x1 + text_w + 4, y1), color, -1)
            
            cv2.putText(vis_image, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
            print(f"Saved visualization: {save_path}")
        
        return vis_image
    
    def save_fields_crops(self, detection_result, output_dir='field_crops'):
        if not detection_result['success']:
            print(f"Cannot save crops: {detection_result['message']}")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        for idx, field in enumerate(detection_result['fields']):
            filename = f"{idx:02d}_{field['field_name']}.jpg"
            filename = "".join(c for c in filename if c.isalnum() or c in '._- ')
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, field['crop'])
        
        print(f"Saved {len(detection_result['fields'])} field crops to: {output_dir}")
    
    def export_to_json(self, detection_result, output_path=None, include_metadata=True):
        if not detection_result['success']:
            print(f"Cannot export: {detection_result['message']}")
            return None
        
        json_data = {
            'success': True,
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
            
            json_data['fields'][field['field_name']] = field_info
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ocr_result_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"Saved JSON: {output_path}")
            return json_data
        except Exception as e:
            print(f"Error saving JSON: {e}")
            return None


# ============= TEST MODULE =============

def test_field_detector_ocr():
    """
    Test Combined Field Detector + OCR
    """
    print("="*70)
    print("TEST MODULE: FIELD DETECTOR + OCR (FIXED VERSION)")
    print("="*70)
    
    card_image_path = './debug_steps/preprocessed_color_after.jpg'
    card_image = cv2.imread(card_image_path)
    if card_image is None:
        print(f"Cannot read image: {card_image_path}")
        return None
    
    print(f"\nInput: {card_image_path}")
    print(f"Size: {card_image.shape[1]}x{card_image.shape[0]}")
    
    # Initialize detector with custom weights
    detector = FieldDetectorOCR(
        model_path='model/Text_field_detect_best_03_12_25.pt',
        conf_threshold=0.4,
        custom_ocr_weights="./model/vgg_transformer_ocr_cccd_v3.pth"
    )
    
    # Detect + OCR
    print("\nDetecting fields and performing OCR...")
    result = detector.detect_and_recognize(card_image, padding=5)
    
    if result['success']:
        print(f"\nSUCCESS!")
        print(f"   Total fields detected: {result['total_fields']}")
        
        print("\n" + "="*70)
        print("EXTRACTED DATA")
        print("="*70)
        for field in result['fields']:
            conf_str = f"(det:{field['bbox_confidence']:.3f}, ocr:{field['ocr_confidence']:.3f})"
            print(f"  {field['field_name']:15s}: {field['ocr_text']:30s} {conf_str}")
        
        # Save outputs
        print("\n" + "="*70)
        print("SAVING OUTPUTS")
        print("="*70)
        detector.visualize(card_image, result, save_path='results/field_ocr_visualization.jpg')
        detector.save_fields_crops(result, output_dir='results/field_crops')
        detector.export_to_json(result, output_path='results/id_card_data.json')
        print("="*70 + "\n")
        
    else:
        print(f"\nFAILED: {result['message']}")
    
    return result


if __name__ == "__main__":
    test_field_detector_ocr()