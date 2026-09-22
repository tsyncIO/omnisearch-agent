import os
import re
from typing import Tuple, List, Optional, Dict
from PIL import Image, ImageDraw, ImageFont

def parse_grounding_tags(text: str) -> List[Dict]:
    """
    Parses all <grounding>{"bbox_2d": [...], "source": "..."} tags from model output.
    Supports both single and multiple tags per turn.
    """
    results = []
    pattern = r'<grounding>\{"bbox_2d":\s*(\[[^\]]+\])(?:,\s*"source":\s*["\']([^"\']+)["\'])?'
    matches = re.finditer(pattern, text)
    for m in matches:
        try:
            bbox = eval(m.group(1).strip())
            source = m.group(2).strip() if m.group(2) else "original_image"
            if isinstance(bbox, list) and len(bbox) == 4:
                results.append({"bbox_2d": bbox, "source": source})
        except Exception:
            continue

    # Fallback to single match if regex was slightly off
    if not results:
        pattern_loose = r'<grounding>.*?bbox_2d.*?(\[[0-9\.,\s]+\])'
        matches_loose = re.finditer(pattern_loose, text)
        for ml in matches_loose:
            try:
                bbox = eval(ml.group(1).strip())
                if isinstance(bbox, list) and len(bbox) == 4:
                    results.append({"bbox_2d": bbox, "source": "original_image"})
            except Exception:
                continue

    return results

def parse_grounding_tag(text: str) -> Optional[Dict]:
    tags = parse_grounding_tags(text)
    return tags[0] if tags else None

def crop_image(
    image: Image.Image,
    bbox: List[float],
    relative: Optional[bool] = None,
    resize_multiplier: float = 2.0,
    min_dim: int = 56
) -> Image.Image:
    """
    Crops an image given [x0, y0, x1, y1] coordinates and applies high-quality Lanczos resampling.
    Automatically detects if coordinates are relative (0-1) or absolute pixels.
    """
    w, h = image.size
    
    # Auto-detect relative vs absolute
    if relative is None:
        relative = max(bbox) <= 1.0

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

        is_rel = max(bbox) <= 1.0
        if is_rel:
            x0, y0, x1, y1 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        else:
            x0, y0, x1, y1 = bbox
        left, top = max(0, int(min(x0, x1))), max(0, int(min(y0, y1)))
        right, bottom = min(w, int(max(x0, x1))), min(h, int(max(y0, y1)))

        # Scalable font sizing
        font_size = max(16, int(min(w, h) * 0.032))
        font = None
        for font_path in [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"
        ]:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, size=font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        # Draw thick rectangle
        line_width = max(4, int(min(w, h) * 0.006))
        draw.rectangle([left, top, right, bottom], outline=color, width=line_width)

        # Label tag positioning
        label_text = f" {label} "
        tag_y = top - font_size - 6
        if tag_y < 0:
            tag_y = top + line_width + 2

        text_bbox = draw.textbbox((left, tag_y), label_text, font=font)
        # Pad background tag
        pad_bbox = (text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2)
        draw.rectangle(pad_bbox, fill=color)
        draw.text((left, tag_y), label_text, fill="black", font=font)

    return canvas
