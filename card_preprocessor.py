import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')

class CardPreprocessor:
    def __init__(self, 
                 resize_ratio=0.5,
                 enable_perspective=True,
                 enable_enhance=True,
                 fixed_output_size=(800,500)):
        self.resize_ratio = resize_ratio
        self.enable_perspective = enable_perspective
        self.enable_enhance = enable_enhance
        self.fixed_output_size = fixed_output_size
    
    def preprocess(self, card_image) -> dict:
        """        
        Args:
            card_image: cropped card
            
        Returns:
            dict: {
                'success': True/False,
                'processed_image': preprocessed card image
            }
        """
        
        original = card_image.copy()
        perspective_applied = False
        
        try:   
            # Perspective transform 
            if self.enable_perspective:
                perspective_result = self._apply_perspective_transform(card_image)
                if perspective_result is not None:
                    card_image = perspective_result
                    perspective_applied = True
                    print("Applied perspective transform")
                else:
                    print("Skipped perspective transform (Not found 4 corners)")
            
            # Enhance 
            if self.enable_enhance:
                enhanced = self._enhance_card(card_image)
                card_image = enhanced
                print("Applied enhancement")
            
            return {
                'success': True,
                'processed_image': card_image,
            }
            
        except Exception as e:
            print(f" Preprocessing error: {e}")
            return {
                'success': False,
                'processed_image': original,
                'error': str(e)
            }
    
    def _apply_perspective_transform(self, image):
        """
        Find 4 corners and transform back to rectangle
        """
        # Resize 
        original = image.copy()
        resized = self._opencv_resize(image, self.resize_ratio)
        
        # Preprocessing to find contour
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Morphology to clarify edges
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        dilated = cv2.dilate(blurred, rectKernel)
        
        # Edge detection
        edged = cv2.Canny(dilated, 100, 250, apertureSize=3)
        cv2.imwrite('debug_steps/debug_edged.jpg', edged) # debug
        contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
        
        # Get largest contour
        largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        image_with_contours = cv2.drawContours(resized.copy(), largest_contours, -1, (0,255,0), 2)
        cv2.imwrite('debug_steps/debug_contours.jpg', image_with_contours)

        result = self._get_card_contour(largest_contours)
        if result is None:
            return None
        
        card_contour, used_bounding_rect = result
        
        if card_contour is None:
            return None
        
        # Convert contour to rectangle and resize back to original
        rect = self._contour_to_rect(card_contour, self.resize_ratio)
        
        # Apply perspective transform
        warped = self._wrap_perspective(original, rect)
        
        if used_bounding_rect:
            print("  Used bounding rectangle for better coverage")
        
        return warped
    
    def _get_card_contour(self, contours):
        """        
        Returns:
            tuple: (contour, use_bounding_rect)
            - contour: Contour 
            - use_bounding_rect: True if using bounding rect 
        """
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                return approx, False
            
            # Nếu có 5-8 góc dùng bounding rectangle
            if 5 <= len(approx) <= 8:
                print(f"  Found {len(approx)} corners (rounded), using bounding rectangle")
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.array(box, dtype=np.int32)
                return box.reshape(-1, 1, 2), True
        
        return None, False
    
    def _contour_to_rect(self, contour, resize_ratio):
        """
        Convert contour -> 4 corners of rectangle
        Order: top-left, top-right, bottom-right, bottom-left
        """
        pts = contour.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        
        # Top-left: smallest sum x + y
        # Bottom-right: largest sum x+y 
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Top-right: smallest difference x - y
        # Bottom-left: largest difference x - y
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        # Scale back
        return rect / resize_ratio
    
    def _wrap_perspective(self, img, rect, fixed_size=(800,500)):
        """
        Apply perspective transform 
        """
        if fixed_size is None:
            # Auto-calculate
            (tl, tr, br, bl) = rect
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width = max(int(widthA), int(widthB))
            
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            height = max(int(heightA), int(heightB))
        else:
            # Fixed size
            width, height = fixed_size

        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (width, height))
        
        return warped

    def _enhance_card(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21) # switch gray <=> image ( optional)
        scale = 2.8
        h, w = denoised.shape
        resized = cv2.resize(denoised, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)) 
        enhanced = clahe.apply(resized)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)

        gaussian = cv2.GaussianBlur(enhanced, (9, 9), 10.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)

        # convert back to BGR for YOLO
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR) 

        return enhanced_bgr
    
    def _opencv_resize(self, image, ratio):
        width = int(image.shape[1] * ratio)
        height = int(image.shape[0] * ratio)
        dim = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
    def visualize_simple(self, result, original_img, save_prefix='preprocessed') -> None:
        """
        Visualize - save before/after image
        """
        if not result['success']:
            print(f"Cannot visualize: {result.get('error', 'Unknown error')}")
            return
                
        # Before
        before_path = f'debug_steps/{save_prefix}_before.jpg'
        cv2.imwrite(before_path, original_img)
        print(f"Saved before: {before_path}")
        
        # After
        after_path = f'debug_steps/{save_prefix}_after.jpg'
        cv2.imwrite(after_path, result['processed_image'])
        print(f"Saved after: {after_path}")


# ============= TEST MODULE =============

def test_preprocessor():
    print("="*60)
    print("TEST MODULE 2: CARD PREPROCESSOR")
    print("="*60)
    
    card_image_path = 'output_module1/test_result_crop.jpg'
    
    card_image = cv2.imread(card_image_path)
    if card_image is None:
        print(f" Cant read {card_image_path}")
        return None
    original_img = card_image.copy() 

    
    print(f"\nInput: {card_image_path}")
    print(f"  Size: {card_image.shape[1]}x{card_image.shape[0]}")
    
    preprocessor = CardPreprocessor(
        resize_ratio=0.5,
        enable_perspective=True,
        enable_enhance=True
    )
    
    # Preprocess
    print("\n Processing...")
    result = preprocessor.preprocess(card_image)
    
    if result['success']:
        print("\nPREPROCESSING SUCESS!")
        processed = result['processed_image']
        print(f"  Output size: {processed.shape[1]}x{processed.shape[0]}")
        
        # Visualize
        preprocessor.visualize_simple(result, save_prefix='preprocessed_result', original_img=original_img)
        
    else:
        print(f"\nFAILED WHEN PREPROCESSING : {result.get('error', 'Unknown')}")
    
    return result


if __name__ == "__main__":
    test_preprocessor()
