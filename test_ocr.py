import torch
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from PIL import Image
import os

WEIGHT_PATH = './model/vgg_transformer_ocr_cccd_v3.pth'

config = Cfg.load_config_from_name('vgg_transformer')
config['weights'] = WEIGHT_PATH
config['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
config['predictor']['beamsearch'] = False
config['cnn']['pretrained'] = False

predictor = Predictor(config)

def test_single_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError
    
    img = Image.open(image_path).convert('RGB')
    text = predictor.predict(img)
    print(f"Results: {text}")
    print("-" * 50)

def test_folder(folder_path):
    if not os.path.isdir(folder_path):
        raise NotADirectoryError
    
    supported = ['.jpg', '.jpeg', '.png']
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) 
              if os.path.splitext(f.lower())[1] in supported]    
    for img_path in images:
        test_single_image(img_path)

if __name__ == "__main__":
    # Input is cropped field
    test_single_image("./results/debug/cccd_1_field_crops/05_id.jpg")
    