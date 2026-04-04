#!/usr/bin/env python3
"""
RAC Engine Sprite Pipeline — Process character art into engine sprite sheet.

Takes a character image (with black or solid background), removes the
background, crops to content, and generates an animated sprite sheet
for the RAC engine.

Usage:
    python3 sprite_pipeline.py badger.png
    python3 sprite_pipeline.py badger.png --size 256
    python3 sprite_pipeline.py badger.png --size 512 --output assets/

The script handles:
  - Black background removal (threshold-based alpha masking)
  - Auto-crop to character bounding box
  - Duplicate detection (if image has character + reflection, takes top half)
  - Resize to target sprite size preserving aspect ratio
  - 12-frame animation: idle(4), walk(4), shoot(4)
  - Export to RAC engine raw format + PNG sprite sheet
"""

import sys
import os
import math
import struct
from PIL import Image, ImageFilter, ImageDraw

def remove_black_background(img, threshold=35):
    """Remove black/near-black background pixels."""
    rgba = img.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size

    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            # Black background: all channels low
            if r < threshold and g < threshold and b < threshold:
                px[x, y] = (r, g, b, 0)

    return rgba


def remove_background_auto(img, threshold=40):
    """Auto-detect background color from corners and remove it."""
    rgba = img.convert("RGBA")
    px = rgba.load()
    w, h = rgba.size

    # Sample 10x10 corner blocks
    samples = []
    for cy, cx in [(5, 5), (5, w-5), (h-5, 5), (h-5, w-5)]:
        for dy in range(-5, 5):
            for dx in range(-5, 5):
                py, ppx = max(0, min(h-1, cy+dy)), max(0, min(w-1, cx+dx))
                samples.append(px[ppx, py][:3])

    bg_r = sum(c[0] for c in samples) // len(samples)
    bg_g = sum(c[1] for c in samples) // len(samples)
    bg_b = sum(c[2] for c in samples) // len(samples)
    print(f"  Background color: ({bg_r}, {bg_g}, {bg_b})")

    # If background is very dark, use black removal
    if bg_r < 30 and bg_g < 30 and bg_b < 30:
        return remove_black_background(img, threshold)

    # Otherwise remove by color distance
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            dist = abs(r - bg_r) + abs(g - bg_g) + abs(b - bg_b)
            if dist < threshold:
                px[x, y] = (r, g, b, 0)

    return rgba


def crop_top_character(img):
    """If image has a character + reflection (doubled), crop to top half."""
    rgba = img.convert("RGBA")
    w, h = rgba.size

    # Scan vertical center column for content gaps
    # If there's a transparent/black gap in the middle, take the top portion
    center_x = w // 2
    px = rgba.load()

    # Find content bands
    content_rows = []
    for y in range(h):
        row_alpha = sum(px[x, y][3] for x in range(w // 4, 3 * w // 4)) / (w // 2)
        content_rows.append(row_alpha > 20)

    # Find gaps (runs of empty rows)
    in_content = False
    sections = []
    start = 0
    for y in range(h):
        if content_rows[y] and not in_content:
            start = y
            in_content = True
        elif not content_rows[y] and in_content:
            sections.append((start, y))
            in_content = False
    if in_content:
        sections.append((start, h))

    if len(sections) >= 2:
        # Take the largest section (main character)
        largest = max(sections, key=lambda s: s[1] - s[0])
        print(f"  Found {len(sections)} sections, using largest: rows {largest[0]}-{largest[1]}")
        return rgba.crop((0, largest[0], w, largest[1]))

    return rgba


def crop_to_content(img, padding=8):
    """Crop to non-transparent bounding box."""
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
    """Resize maintaining aspect ratio, centered on target canvas."""
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    ox = (target_size - new_w) // 2
    oy = target_size - new_h  # align to bottom
    canvas.paste(resized, (ox, oy))
    return canvas


def generate_frames(sprite):
    """Generate 12 animation frames: idle(4), walk(4), shoot(4)."""
    size = sprite.width
    frames = []

    for anim in ['idle', 'walk', 'shoot']:
        for i in range(4):
            frame = Image.new('RGBA', (size, size), (0, 0, 0, 0))
            if anim == 'idle':
                bob = int(size * 0.015 * math.sin(2 * math.pi * i / 4))
                frame.paste(sprite, (0, bob), sprite)
            elif anim == 'walk':
                bob = int(size * 0.025 * math.sin(2 * math.pi * i / 4))
                shift = int(size * 0.015 * math.sin(2 * math.pi * i / 4))
                frame.paste(sprite, (shift, bob), sprite)
            elif anim == 'shoot':
                recoil = int(size * 0.03 * max(0, 1.0 - i * 0.4))
                frame.paste(sprite, (recoil, -1 if i == 0 else 0), sprite)
            frames.append(frame)

    return frames


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sprite_pipeline.py <image.png> [--size N] [--output dir]")
        print("  Default size: 256 (good for 4K). Use 128 for 1080p.")
        sys.exit(1)

    input_path = sys.argv[1]
    target_size = 256
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    for i, arg in enumerate(sys.argv):
        if arg == "--size" and i + 1 < len(sys.argv):
            target_size = int(sys.argv[i + 1])
        elif arg == "--output" and i + 1 < len(sys.argv):
            output_dir = sys.argv[i + 1]

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Sprite Pipeline] Loading {input_path}...")
    img = Image.open(input_path)
    print(f"  Input: {img.size[0]}x{img.size[1]} {img.mode}")

    # Step 1: Remove background
    print("[1] Removing background...")
    nobg = remove_background_auto(img)
    nobg.save(os.path.join(output_dir, "character_nobg.png"))

    # Step 2: Handle duplicate (character + reflection)
    print("[2] Detecting character region...")
    character = crop_top_character(nobg)

    # Step 3: Crop to content
    print("[3] Cropping to content...")
    cropped = crop_to_content(character)
    print(f"  Cropped: {cropped.size[0]}x{cropped.size[1]}")
    cropped.save(os.path.join(output_dir, "character_cropped.png"))

    # Step 4: Resize to sprite size
    print(f"[4] Resizing to {target_size}x{target_size}...")
    sprite = resize_sprite(cropped, target_size)
    sprite.save(os.path.join(output_dir, "character_sprite.png"))

    # Step 5: Generate animation frames
    print("[5] Generating 12 animation frames...")
    frames = generate_frames(sprite)

    # Step 6: Build sprite sheet
    print("[6] Building sprite sheet...")
    sheet = Image.new('RGBA', (target_size * 12, target_size), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        sheet.paste(f, (i * target_size, 0))
    sheet.save(os.path.join(output_dir, "character_sheet.png"))

    # Step 7: Export raw for engine
    print("[7] Exporting for RAC engine...")
    raw = sheet.convert('RGBA').tobytes()
    raw_path = os.path.join(output_dir, "character_sheet.raw")
    with open(raw_path, 'wb') as f:
        f.write(struct.pack('<III', target_size, target_size, 12))
        f.write(raw)

    print(f"\n[Done!] Sprite pipeline complete")
    print(f"  Sprite: {target_size}x{target_size}")
    print(f"  Sheet:  {target_size*12}x{target_size} ({len(raw)+12} bytes)")
    print(f"  Output: {output_dir}/")
    print(f"\nTo use in demo:")
    print(f"  cd engine/build && rm -rf * && cmake .. -DCMAKE_BUILD_TYPE=Release")
    print(f"  make -j$(nproc) && ./rac_engine_demo --4k --frames 150 --output")
    print(f"  ffmpeg -framerate 30 -i frame_%04d.ppm -c:v libx264 -pix_fmt yuv420p demo.mp4")


if __name__ == "__main__":
    main()
