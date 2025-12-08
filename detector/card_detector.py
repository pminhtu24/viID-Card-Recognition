from ultralytics import YOLO # type: ignore
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  

class CardDetector: 
    # Khoi tao model, nguong tin cay
    def __init__(self, model_path, conf_threshold=0.5):
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            print(f"Loaded card detection model from: {model_path}")
        except Exception as e:
            print(f"Error when loading model: {e}")
            raise

    def detect(self, image: np.ndarray, padding: int=15, padding_percent: float=0.08) -> dict:
        """
        Args:
            image: (BGR format - from cv2.imread)
            padding: Number of padding pixels around the card when cropping (fixed value)
            padding_percent: Padding as percentage of bbox size (e.g, 0.05 = 5% padding)
            
        Returns:
            dict: {
                'success': True/False,
                'cropped_card_image': cropped card image (if success),
                'bbox': (x1, y1, x2, y2) bounding box,
                'confidence': confidence score
            }
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        if len(results[0].boxes) == 0:
            return {
                'success': False,
                'message': 'No card detected in image!'
            }
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf[0])

        # Calculate padding
        if padding_percent is not None:
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            pad_x = int(bbox_width * padding_percent)
            pad_y = int(bbox_height * padding_percent)
        else:
            # Use fixed padding (pixel)
            pad_x = padding
            pad_y = padding
        
        # Add padding
        h, w = image.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        card_image = image[y1:y2, x1:x2]
        
        print(f" Card detected with confidence: {confidence:.3f}")
        print(f" Position: ({x1}, {y1}) -> ({x2}, {y2})")
        print(f" Padding applied: {pad_x}px (x), {pad_y}px (y)")
        print(f" Crop size: {card_image.shape[1]}x{card_image.shape[0]}")

        return {
            'success': True,
            'cropped_card_image': card_image,
            'bbox': (x1, y1, x2, y2),
            'confidence': confidence,
            'original_image': image
        }
    
    def detect_from_path(self, image_path: str, padding: int=10, padding_percent: float=0.08) -> dict:
        """        
        Args:
            image_path: str
            padding: pixel padding (fixed)
            padding_percent: Padding in % of bbox (e.g, 0.05 = 5%)
            
        Returns:
            dict
        """
        image = cv2.imread(image_path) # -> numpy , RGB -> BGR
        if image is None:
            return {
                'success': False,
                'message': f'Can not read: {image_path}'
            }
        
        return self.detect(image, padding, padding_percent)
    
    
    def visualize_simple(self, result, save_prefix='result') -> None:
        """    
        Args:
            result: Output from detect()
            save_prefix: Prefix for file
        """
        if not result['success']:
            print(f" {result['message']}")
            return
        
        # save og pic with bbox
        img_with_box = result['original_image'].copy()
        x1, y1, x2, y2 = result['bbox']
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        text = f"Card: {result['confidence']:.3f}"
        cv2.putText(img_with_box, text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        output_dir = "output_module1"
        os.makedirs(output_dir,exist_ok=True)

        bbox_path = f'{output_dir}/{save_prefix}_bbox.jpg'
        cv2.imwrite(bbox_path, img_with_box)
        print(f" Saved bbox image: {bbox_path}")
        
        crop_path = f'{output_dir}/{save_prefix}_crop.jpg'
        cv2.imwrite(crop_path, result['cropped_card_image'])
        print(f" Saved cropped card: {crop_path}")


# ============= TEST MODULE =============

def test_card_detector():
    """
    Test function for Card Detector
    """
    print("="*60)
    print("TEST MODULE 1: CARD DETECTOR")
    print("="*60)
    
    detector = CardDetector(
        model_path='model/card_detect_best.pt',
        conf_threshold=0.5
    )
    
    test_image_path = 'samples/cccd_2.jpeg'
    
    print(f"\ntest with: {test_image_path}")
    
    # use fixed padding (pixel)
    # result = detector.detect_from_path(test_image_path, padding=20)
    
    # Use padding in % 
    result = detector.detect_from_path(test_image_path, padding_percent=0.05)
    
    if result['success']:
        print("\n DETECTION Sucess!")
        print(f" >>> Confidence: {result['confidence']:.3f}")
        print(f" >>> Bbox: {result['bbox']}")
        
        # Visualize 
        detector.visualize_simple(result, save_prefix='test_result')
                
    else:
        print(f"\n DETECTION FAILED {result['message']}")
    
    return result


if __name__ == "__main__":
    test_card_detector()
