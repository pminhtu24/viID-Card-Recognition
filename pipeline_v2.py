from detector.card_detector import CardDetector
from detector.field_detector_ocr import FieldDetectorOCR
from card_preprocessor import CardPreprocessor
from pathlib import Path
from datetime import datetime
import json
class IDCardPipeline:
    def __init__(self, 
                 card_model_path = "model/card_detect_best.pt", 
                 field_model_path = "model/Text_field_detect_best_03_12_25.pt",
                 card_conf: float = 0.5,
                 field_conf: float =0.4,
                 min_fields_required = 7,
                 custom_ocr_weights = "./model/vgg_transformer_ocr_cccd_v3.pth"
                 ):
        self.min_fields_required = min_fields_required
        self.card_detector = CardDetector(
            model_path=card_model_path,
            conf_threshold=card_conf
        )
        
        self.field_detector = FieldDetectorOCR(
            model_path=field_model_path,
            conf_threshold=field_conf,
            ocr_model='vgg_transformer',
            custom_ocr_weights=custom_ocr_weights,
        )
        self.preprocessor = CardPreprocessor(
            resize_ratio=0.5,
            enable_perspective=True,
            enable_enhance=True,
            fixed_output_size=(800,500)
        )
    
    def process_image(self, image_path, output_dir='results', save_debug=True):
        Path(output_dir).mkdir(exist_ok=True)
        if save_debug:
            Path(f"{output_dir}/debug").mkdir(exist_ok=True)

        base_name = Path(image_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        #-------------Detect and Crop Card----------------
        card_result = self.card_detector.detect_from_path(
            image_path=image_path,
            padding_percent=0.25
        )

        if not card_result['success']:
            print(f"Card detection failed: {card_result['message']}")
            return self._create_failure_response(
                stage='card_detection',
                message=card_result['message'],
                request_retake=True,
                reason="No card detected in image",
            )
        
        print(f"Card detected (confidence: {card_result['confidence']:.3f})")
        if save_debug:
            self.card_detector.visualize_simple(
                card_result,
                save_prefix=f"{base_name}_01_card"
            )
        cropped_card_image = card_result['cropped_card_image']

        #-------------Preprocessing and Enhance-------------------
        preprocess_result = self.preprocessor.preprocess(cropped_card_image)
        if not preprocess_result['success']:
            print("Preprocessing failed !")
            return self._create_failure_response(
                stage='preprocessing',
                message='Failed to preprocess card',
                request_retake=True,
                reason='Could not find 4 corners for perspective correction'
            )
        print("Preprocessing successful !")
        processed_card = preprocess_result['processed_image']
        if save_debug:
            self.preprocessor.visualize_simple(
                preprocess_result,
                original_img=cropped_card_image,
                save_prefix=f"{base_name}_02_preprocess",
            )
        #--------------Field Detection and OCR----------------------
        print("\nDetecting fields (After preprocessing)")
        ocr_result = self.field_detector.detect_and_recognize(
            processed_card,
            padding = 10
        )
        if not ocr_result['success']:
            print("Field detection failed even after preprocessing")
            return self._create_failure_response(
                stage='fied_detection',
                message="Failed to detect fields after preprocessing",
                request_retake=True,
                reason = "Insufficient fields detected even after image enhancement"
            )
        fields_found = ocr_result['total_fields']
        print(f">>> Found {fields_found} fields after preprocessing")
        if fields_found >= self.min_fields_required:
            print(f"Success! {fields_found} >= {self.min_fields_required} fields detected")
            return self._save_results(
                ocr_result=ocr_result,
                card_result=card_result,
                processed_card=processed_card,
                base_name=base_name,
                output_dir=output_dir,
                save_debug=save_debug,
                preprocessing_applied=True,
                attempt_number=1,
                improvement_note=None

            )
        else:
            print(f"Still insufficient fields: {fields_found} < {self.min_fields_required}")
            return self._create_failure_response(
                stage='insufficient_fields',
                message=f'Only {fields_found} fields detected (minimum: {self.min_fields_required})',
                request_retake=True,
                reason='Card quality too poor or obscured',
            )

    def _create_failure_response(self, stage, message, request_retake, reason):
        print("\n" + "="*70)
        print("PIPELINE FAILED")
        print("="*70)
        print(f"Stage: {stage}")
        print(f"Reason: {reason}")
        print(f"\nACTION REQUIRED: Please retake the photo")
        print("\nSuggestions:")
        print("  > Ensure good lighting")
        print("  > Keep card flat and fully visible")
        print("  > Avoid shadows and glare")
        print("  > Make sure all text is clear")
        print("="*70 + "\n")

        response = {
            'success': False,
            'request_retake': request_retake,
            'stage': stage,
            'message': message,
            'reason': reason,
            'suggestions': [
                'Ensure good lighting',
                'Keep card flat and fully visible',
                'Avoid shadows and glare',
                'Make sure background clean'
            ]
        }

        
        return response

    def _save_results(self, ocr_result, card_result, processed_card,
                      base_name, output_dir, save_debug,
                      preprocessing_applied, attempt_number, improvement_note=None):
        # Save visualization
        vis_path = f"{output_dir}/{base_name}_result.jpg"
        self.field_detector.visualize(
            processed_card, 
            ocr_result, 
            save_path=vis_path
        )
        
        # Save field crops
        if save_debug:
            crops_dir = f"{output_dir}/debug/{base_name}_field_crops"
            self.field_detector.save_fields_crops(ocr_result, output_dir=crops_dir)
        
        # Save JSON
        json_path = f"{output_dir}/{base_name}_data.json"
        json_data = self.field_detector.export_to_json(
            ocr_result, 
            output_path=json_path,
            include_metadata=True
        )
        
        # Add pipeline metadata to JSON
        if json_data:
            json_data['pipeline_info'] = {
                'preprocessing_applied': preprocessing_applied,
                'attempt_number': attempt_number,
                'card_confidence': round(card_result['confidence'], 4)
            }
            if improvement_note:
                json_data['pipeline_info']['note'] = improvement_note
            
            # Re-save with metadata
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print("\n" + "="*70)
        print("EXTRACTED DATA")
        print("="*70)
        for field in ocr_result['fields']:
            print(f"{field['field_name']:15s}: {field['ocr_text']}")
        
        print("\n" + "="*70)
        print("PIPELINE INFO")
        print("="*70)
        print(f"Preprocessing applied: {'YES' if preprocessing_applied else 'NO'}")
        print(f"Detection attempts: {attempt_number}")
        print(f"Total fields extracted: {ocr_result['total_fields']}")
        if improvement_note:
            print(f"Note: {improvement_note}")
        
        print("\n" + "="*70)
        print("OUTPUT FILES")
        print("="*70)
        print(f"Visualization: {vis_path}")
        print(f"JSON data: {json_path}")
        if save_debug:
            print(f"Debug files: {output_dir}/debug/")
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70 + "\n")
        
        return {
            'success': True,
            'request_retake': False,
            'preprocessing_applied': preprocessing_applied,
            'attempt_number': attempt_number,
            'card_detection': card_result,
            'ocr_result': ocr_result,
            'json_data': json_data,
            'output_files': {
                'visualization': vis_path,
                'json': json_path
            }
        }

def main():

    pipeline = IDCardPipeline(
        card_model_path='model/card_detect_best.pt',
        field_model_path='model/Text_field_detect_best_03_12_25.pt',
        card_conf=0.5,
        field_conf=0.4,
        min_fields_required=7,
        custom_ocr_weights="./model/vgg_transformer_ocr_cccd_v3.pth" 
    )
    
    result = pipeline.process_image(
        image_path='samples/cccd_2.jpeg',
        output_dir='results',
        save_debug=True
    )                                                                 
    
    # Check result
    if result['success']:
        print("\nImage processed successfully!")
        print(f"Extracted {result['ocr_result']['total_fields']} fields")
    else:
        print("\nProcessing failed !")
        print(f" Reason: {result['reason']}")
        if result['request_retake']:
            print("Please retake the photo")


if __name__ == "__main__":
    main()