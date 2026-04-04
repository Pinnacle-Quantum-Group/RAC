#!/usr/bin/env python3
"""
RAC Engine Sprite Pipeline — Background Removal & Sprite Sheet Generator
Pinnacle Quantum Group — April 2026

Processes a character image:
1. Removes background using edge-detection + flood-fill (pure Pillow)
2. Crops to bounding box with padding
3. Generates sprite sheet with animation frames (idle, walk, shoot)
4. Exports as C header with embedded pixel data for the engine

Usage:
    python3 sprite_pipeline.py input.png [--output assets/] [--size 64]
"""

import sys
import os
from PIL import Image, ImageFilter, ImageDraw

def remove_background(img, threshold=30):
    """Remove background using corner-sampling + color distance.
    Samples corner pixels to estimate background color, then removes
    similar pixels via color distance threshold."""

    rgba = img.convert("RGBA")
    pixels = rgba.load()
    w, h = rgba.size

    # Sample corners for background color (average of corner 5x5 blocks)
    corners = []
    for cy, cx in [(0,0), (0,w-1), (h-1,0), (h-1,w-1)]:
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                py, px = cy+dy, cx+dx
                if 0 <= py < h and 0 <= px < w:
                    corners.append(pixels[px, py][:3])

    bg_r = sum(c[0] for c in corners) // len(corners)
    bg_g = sum(c[1] for c in corners) // len(corners)
    bg_b = sum(c[2] for c in corners) // len(corners)

    print(f"  Estimated background color: ({bg_r}, {bg_g}, {bg_b})")

    # Create mask: pixels close to background become transparent
    for y in range(h):
        for x in range(w):
            r, g, b, a = pixels[x, y]
            dist = abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b)
            if dist < threshold:
                pixels[x, y] = (r, g, b, 0)  # transparent

    # Erode edges slightly to clean up fringing
    # Convert alpha to mask, apply min filter, then restore
    alpha = rgba.split()[3]
    alpha = alpha.filter(ImageFilter.MinFilter(3))
    rgba.putalpha(alpha)

    return rgba

def crop_to_content(img, padding=4):
    """Crop image to non-transparent bounding box with padding."""
    bbox = img.getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img.width, x1 + padding)
    y1 = min(img.height, y1 + padding)
    return img.crop((x0, y0, x1, y1))

def resize_sprite(img, target_size):
    """Resize to target size maintaining aspect ratio, centered."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Center on target_size x target_size canvas
    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas

def generate_animation_frames(base_sprite, num_frames=8):
    """Generate animation frames from a single sprite.
    Creates idle (bob), walk (shift+bob), and shoot (recoil) animations."""

    size = base_sprite.width
    frames = []

    # Idle animation: gentle bob (4 frames)
    for i in range(4):
        frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        import math
        bob_y = int(2 * math.sin(2 * math.pi * i / 4))
        frame.paste(base_sprite, (0, bob_y), base_sprite)
        frames.append(("idle", i, frame))

    # Walk animation: shift + bob (4 frames)
    for i in range(4):
        frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        bob_y = int(3 * math.sin(2 * math.pi * i / 4))
        shift_x = int(2 * math.sin(2 * math.pi * i / 4))
        frame.paste(base_sprite, (shift_x, bob_y), base_sprite)
        frames.append(("walk", i, frame))

    # Shoot animation: recoil (4 frames)
    for i in range(4):
        frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        recoil = int(3 * math.exp(-i) * (1 if i == 0 else -1))
        frame.paste(base_sprite, (recoil, 0), base_sprite)
        frames.append(("shoot", i, frame))

    return frames

def build_sprite_sheet(frames, sprite_size):
    """Arrange frames into a horizontal sprite sheet."""
    n = len(frames)
    sheet = Image.new("RGBA", (sprite_size * n, sprite_size), (0, 0, 0, 0))
    for i, (anim, idx, frame) in enumerate(frames):
        sheet.paste(frame, (i * sprite_size, 0))
    return sheet

def export_c_header(sprite, name, output_path):
    """Export sprite as C header with embedded RGBA pixel data."""
    w, h = sprite.size
    pixels = list(sprite.getdata())

    with open(output_path, 'w') as f:
        f.write(f"/* Auto-generated sprite data: {name} */\n")
        f.write(f"/* {w}x{h} RGBA8888 */\n\n")
        f.write(f"#ifndef RAC_SPRITE_{name.upper()}_H\n")
        f.write(f"#define RAC_SPRITE_{name.upper()}_H\n\n")
        f.write(f"#define RAC_SPRITE_{name.upper()}_W {w}\n")
        f.write(f"#define RAC_SPRITE_{name.upper()}_H {h}\n\n")
        f.write(f"static const unsigned char rac_sprite_{name}_data[] = {{\n")

        for i, (r, g, b, a) in enumerate(pixels):
            if i % 8 == 0:
                f.write("    ")
            f.write(f"0x{r:02x},0x{g:02x},0x{b:02x},0x{a:02x},")
            if i % 8 == 7:
                f.write("\n")

        f.write("\n};\n\n")
        f.write(f"#endif /* RAC_SPRITE_{name.upper()}_H */\n")

    print(f"  Exported {output_path}: {w}x{h}, {len(pixels)*4} bytes")

def export_rgb_raw(sprite, output_path):
    """Export as raw RGB888 for the engine's framebuffer format."""
    rgb = sprite.convert("RGB")
    w, h = rgb.size
    data = rgb.tobytes()
    with open(output_path, 'wb') as f:
        # Simple header: width(4), height(4), then RGB data
        import struct
        f.write(struct.pack('<II', w, h))
        f.write(data)
    print(f"  Exported {output_path}: {w}x{h} RGB888, {len(data)} bytes")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sprite_pipeline.py input.png [--output dir] [--size N]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = "assets"
    sprite_size = 64

    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]
        elif arg == "--size" and i + 1 < len(sys.argv):
            sprite_size = int(sys.argv[i + 1])

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Sprite Pipeline] Loading {input_path}...")
    img = Image.open(input_path)
    print(f"  Input: {img.size[0]}x{img.size[1]}, mode={img.mode}")

    # Step 1: Remove background
    print("[Step 1] Removing background...")
    nobg = remove_background(img, threshold=40)
    nobg_path = os.path.join(output_dir, "character_nobg.png")
    nobg.save(nobg_path)
    print(f"  Saved: {nobg_path}")

    # Step 2: Crop to content
    print("[Step 2] Cropping to content...")
    cropped = crop_to_content(nobg, padding=8)
    cropped_path = os.path.join(output_dir, "character_cropped.png")
    cropped.save(cropped_path)
    print(f"  Cropped to {cropped.size[0]}x{cropped.size[1]}")

    # Step 3: Resize to sprite size
    print(f"[Step 3] Resizing to {sprite_size}x{sprite_size}...")
    sprite = resize_sprite(cropped, sprite_size)
    sprite_path = os.path.join(output_dir, "character_sprite.png")
    sprite.save(sprite_path)

    # Step 4: Generate animation frames
    print("[Step 4] Generating animation frames...")
    frames = generate_animation_frames(sprite, 4)
    print(f"  Generated {len(frames)} frames: idle(4) + walk(4) + shoot(4)")

    # Step 5: Build sprite sheet
    print("[Step 5] Building sprite sheet...")
    sheet = build_sprite_sheet(frames, sprite_size)
    sheet_path = os.path.join(output_dir, "character_sheet.png")
    sheet.save(sheet_path)
    print(f"  Sheet: {sheet.size[0]}x{sheet.size[1]}")

    # Step 6: Export for engine
    print("[Step 6] Exporting for RAC engine...")
    export_c_header(sprite, "badger", os.path.join(output_dir, "sprite_badger.h"))

    # Export sprite sheet as raw RGB for engine
    export_rgb_raw(sheet, os.path.join(output_dir, "character_sheet.raw"))

    # Also save individual frames as PPM for verification
    for anim, idx, frame in frames:
        rgb = Image.new("RGB", frame.size, (20, 20, 30))  # engine bg color
        rgb.paste(frame, mask=frame.split()[3])
        ppm_path = os.path.join(output_dir, f"frame_{anim}_{idx}.ppm")
        rgb.save(ppm_path)

    print(f"\n[Done] Sprite pipeline complete!")
    print(f"  Character sprite: {sprite_path}")
    print(f"  Sprite sheet: {sheet_path}")
    print(f"  Animation frames: idle(4), walk(4), shoot(4)")

if __name__ == "__main__":
    main()
