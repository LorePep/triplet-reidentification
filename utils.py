from PIL import Image

def load_img(img_path, bounding_box=None):
    img = Image.open(img_path).convert("RGB")    
    
    return img