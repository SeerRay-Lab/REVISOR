import os
from PIL import Image, ImageDraw, ImageFont


def add_watermark_to_pil_image(pil_image, 
                              text, 
                              font_size_ratio=0.1, 
                              color=(255, 0, 0), 
                              margin_ratio=0.05) -> Image:
    """
    Add watermark to a PIL Image in memory.
    
    :param pil_image: Input PIL Image
    :param text: Watermark text to add
    :param font_size_ratio: Font size relative to image height
    :param color: Watermark color (R, G, B)
    :param margin_ratio: Margin relative to image height
    :return: Watermarked PIL Image
    """
    def find_font_file():
        # ... (keep the same font finding logic as original) ...
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        ]
        for path in font_paths:
            if os.path.exists(path):
                return path
        return None

    try:
        # Convert to RGB if necessary
        image = pil_image.convert("RGB")
        draw = ImageDraw.Draw(image)
        img_width, img_height = image.size
        
        # Set font and size
        font_file = find_font_file()
        font_size = int(img_height * font_size_ratio)
        
        if font_file:
            font = ImageFont.truetype(font_file, font_size)
        else:
            font = ImageFont.load_default()

        # Get text bounding box
        if hasattr(draw, 'textbbox'):
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
        else:
            text_width, text_height = draw.textsize(text, font=font)
            
        # Calculate position (lower-left corner)
        margin = int(img_height * margin_ratio)
        x = margin
        y = img_height - text_height - margin
        
        # Add watermark
        draw.text((x, y), text, font=font, fill=color)
        return image
        
    except Exception as e:
        print(f"Error adding watermark: {e}")
        return pil_image  # Return original image on error
