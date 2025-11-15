"""Generate app icon with AI theme"""
from PIL import Image, ImageDraw, ImageFont
import os

# Create a 256x256 icon with gradient background and AI symbol
size = 256
img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Gradient background - modern teal to purple
for y in range(size):
    ratio = y / size
    r = int(30 + (120 - 30) * ratio)
    g = int(144 + (81 - 144) * ratio)
    b = int(255 + (169 - 255) * ratio)
    draw.rectangle([(0, y), (size, y+1)], fill=(r, g, b, 255))

# Round the corners
mask = Image.new('L', (size, size), 0)
mask_draw = ImageDraw.Draw(mask)
mask_draw.rounded_rectangle([(0, 0), (size, size)], radius=40, fill=255)
img.putalpha(mask)

# Draw AI circuit pattern
circuit_color = (255, 255, 255, 180)
center_x, center_y = size // 2, size // 2

# Central AI brain node
draw.ellipse([(center_x-30, center_y-30), (center_x+30, center_y+30)], 
             fill=(255, 255, 255, 220), outline=circuit_color, width=3)

# Neural connection nodes
nodes = [
    (center_x-70, center_y-70, 15),
    (center_x+70, center_y-70, 15),
    (center_x-70, center_y+70, 15),
    (center_x+70, center_y+70, 15),
    (center_x, center_y-90, 12),
    (center_x, center_y+90, 12),
]

for x, y, r in nodes:
    draw.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255, 255, 255, 200))
    # Connect to center
    draw.line([(center_x, center_y), (x, y)], fill=circuit_color, width=2)

# Add sparkle effect for AI magic
sparkles = [(80, 70), (180, 80), (70, 180), (190, 170)]
for sx, sy in sparkles:
    draw.line([(sx-8, sy), (sx+8, sy)], fill=(255, 255, 100, 200), width=2)
    draw.line([(sx, sy-8), (sx, sy+8)], fill=(255, 255, 100, 200), width=2)

# Save icon in multiple sizes
img.save('app_icon.png')
img.resize((128, 128), Image.Resampling.LANCZOS).save('app_icon_128.png')
img.resize((64, 64), Image.Resampling.LANCZOS).save('app_icon_64.png')
img.resize((32, 32), Image.Resampling.LANCZOS).save('app_icon_32.png')
img.resize((16, 16), Image.Resampling.LANCZOS).save('app_icon_16.png')

print("✓ Icon files created successfully!")
