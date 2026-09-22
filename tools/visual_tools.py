import re
from typing import Tuple, List, Optional, Dict
from PIL import Image, ImageDraw, ImageFont

def parse_grounding_tag(text: str) -> Optional[Dict]:
    """
    Parses <grounding>{"bbox_2d": [x0, y0, x1, y1], "source": "original_image"}</grounding>
    from model response.
    """
    pattern = r'<grounding>\{"bbox_2d":\s*(\[[^\]]+\]),\s*"source":\s*["\']([^"\']+)["\']\}</grounding>'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        # Fallback looser regex in case whitespace or quoting varies
        pattern_loose = r'<grounding>.*?bbox_2d.*?(\[[0-9\.,\s]+\]).*?source.*?["\']([^"\']+)["\'].*?</grounding>'
        match = re.search(pattern_loose, text, re.DOTALL)
    
    if match:
        try:
            bbox = eval(match.group(1).strip())
            source = match.group(2).strip()
            if isinstance(bbox, list) and len(bbox) == 4:
                return {"bbox_2d": bbox, "source": source}
        except Exception:
            pass
    return None

def crop_image(
    image: Image.Image,
    bbox: List[float],
    relative: bool = True,
    resize_multiplier: float = 2.0,
    min_dim: int = 56
) -> Image.Image:
    """
    Crops an image given [x0, y0, x1, y1] coordinates and applies high-quality Lanczos resampling.
    """
    w, h = image.size
    if relative:
        x0, y0, x1, y1 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
    else:
        x0, y0, x1, y1 = bbox

    # Normalize bounding box
    left = max(0, int(min(x0, x1)))
    top = max(0, int(min(y0, y1)))
    right = min(w, int(max(x0, x1)))
    bottom = min(h, int(max(y0, y1)))

    if right - left < 5:
        right = min(w, left + 10)
    if bottom - top < 5:
        bottom = min(h, top + 10)

    cropped = image.crop((left, top, right, bottom))
    crop_w, crop_h = cropped.size

    # Lanczos super-sampling for clearer reading
    target_w = max(min_dim, int(crop_w * resize_multiplier))
    target_h = max(min_dim, int(crop_h * resize_multiplier))
    target_w = min(1500, target_w)
    target_h = min(1500, target_h)

    return cropped.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

def draw_bounding_boxes(
    original_image: Image.Image,
    box_records: List[Dict]
) -> Image.Image:
    """
    Draws neat highlighted bounding boxes with turn labels on the original image.
    box_records format: [{'bbox': [x0,y0,x1,y1], 'label': 'Turn 1: Watch Face', 'color': '#00E5FF'}]
    """
    canvas = original_image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size

    # Preset vibrant colors for different turns
    colors = ["#00E5FF", "#FF3D00", "#76FF03", "#FFD600", "#E040FB", "#00B0FF"]

    for idx, record in enumerate(box_records):
        bbox = record["bbox"]
        color = record.get("color", colors[idx % len(colors)])
        label = record.get("label", f"Turn {idx + 1}")

        x0, y0, x1, y1 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        left, top = max(0, int(min(x0, x1))), max(0, int(min(y0, y1)))
        right, bottom = min(w, int(max(x0, x1))), min(h, int(max(y0, y1)))

        # Draw thick rectangle
        line_width = max(3, int(min(w, h) * 0.005))
        draw.rectangle([left, top, right, bottom], outline=color, width=line_width)

        # Draw label background tag
        label_text = f" {label} "
        font = None
        try:
            # Try to load default or truetype font if available
            font = ImageFont.load_default()
        except Exception:
            pass
        
        # Estimate text bbox
        text_bbox = draw.textbbox((left, max(0, top - 20)), label_text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((left, max(0, top - 20)), label_text, fill="black", font=font)

    return canvas
