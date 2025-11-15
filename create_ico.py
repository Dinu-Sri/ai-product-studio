"""Convert PNG icon to ICO format for Windows executable"""
from PIL import Image
import os

# Open the main icon
img = Image.open('app_icon.png')

# Create ICO with multiple sizes
icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
img.save('app_icon.ico', format='ICO', sizes=icon_sizes)

print("✓ Windows ICO file created: app_icon.ico")
