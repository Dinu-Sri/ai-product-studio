# 🎨 AI Product Studio - Professional Background Removal & Enhancement

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

**AI Product Studio** is a powerful, professional-grade desktop application for product photography enhancement. Leveraging cutting-edge AI models and GPU acceleration, it provides studio-quality background removal, advanced shadow generation, and comprehensive image editing tools - all in an intuitive interface.

![AI Product Studio](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)

---

## ✨ Key Features

### 🤖 **AI-Powered Background Removal**
- **14 State-of-the-Art AI Models**:
  - **U²-Net**: General purpose, fast and accurate
  - **U²-Net Human**: Optimized for people and portraits
  - **U²-Net Cloth**: Specialized for clothing and fashion
  - **SILUETA**: High-quality silhouette extraction
  - **ISNET (General & Anime)**: Advanced segmentation
  - **SAM**: Segment Anything Model for complex scenes
  - **BiRefNet Series** (General, Portrait, DIS, HRSOD, COD, Massive): Latest state-of-the-art models for various scenarios
- **GPU Acceleration**: Automatic CUDA/DirectML/CPU detection
- **Batch Processing**: Process multiple images in parallel with progress tracking
- **Advanced Edge Refinement**: 
  - Alpha matting for smooth edges
  - Edge softness control (0-10 levels)
  - Uncertainty detection for smart edge refinement
  - Post-processing mask options

### 🌑 **Professional Shadow System**
Three shadow types with real-time preview and 300ms debouncing for smooth performance:

#### 1. **Drop Shadow (Amazon-style)**
- Perfect for e-commerce product listings
- Adjustable opacity, blur, distance, angle
- Vertical compression for perspective effect
- **Unique Gap Control** (-50% to +100%): Create floating effects or ground-contact shadows
- Soft edge feathering

#### 2. **Natural Shadow (Side Light)**
- Realistic directional lighting simulation
- 360° angle control for any light direction
- Variable distance and compression
- Professional studio lighting effects

#### 3. **Reflection Shadow (Mirror/Glossy)**
- Mirror reflection for glossy surfaces
- Adjustable reflection height (10%-200%)
- Gap control for surface distance
- Fade control for realistic glass/water reflections
- Perfect for luxury product photography

**Shadow Features**:
- ⚡ Real-time preview with debouncing (no lag)
- 🎨 Custom shadow color picker
- 💾 **5 Profile Presets per Shadow Type**: Save and load your favorite settings
- 🔄 Auto-reset parameters when switching shadow types

### 🖼️ **Background Management**
- **Transparent**: Maintain alpha channel for web use
- **Solid Colors**: Custom color picker + 5 quick presets (White, Black, Gray, Blue, Green)
- **Image Backgrounds**: 
  - Load custom background images
  - 4 Fit Modes: Stretch, Fit, Fill (crop), Tile
  - Supports PNG, JPG, BMP, GIF, WebP

### 🎯 **Transform & Positioning**
- **Scale**: 1% - 2000% (20x zoom) with 5% steps
- **Position**: Precise X/Y coordinate control (-5000 to +5000)
- **Rotation**: -180° to +180° with degree precision
- **Quick Scale Presets**: 50%, 100%, 150%, 200% buttons
- **Auto-Center**: Intelligent product centering
- **100ms Debouncing**: Smooth, lag-free adjustments
- 💾 **5 Transform Presets**: Save favorite transform configurations

### 🎨 **Image Adjustments**
- **Brightness**: 0-200% control with fine-tuning
- **Contrast**: 0-200% enhancement
- **Saturation**: 0-200% color intensity
- **Sharpness**: 0-200% detail enhancement
- **Blur**: 0-10 levels for artistic effects
- Real-time preview for all adjustments

### 🖥️ **Canvas Management**
- **Auto-Resize**: Smart canvas sizing based on content
- **Target Size Presets**:
  - Square: 512×512, 1024×1024, 2048×2048, 4096×4096
  - Web: 800×600, 1920×1080
  - Print: A4 (3508×2480), A3 (4961×3508)
  - Custom dimensions with aspect ratio lock
- Maintain aspect ratio or force exact dimensions

### 💾 **Export & Batch Processing**
- **Format Options**: PNG, JPG, WEBP, BMP, TIFF
- **Quality Control**: Adjustable compression (1-100)
- **Batch Export**:
  - Process entire folders
  - Custom filename prefixes/suffixes
  - Progress tracking with ETA
  - Auto-save mode
  - Parallel processing for speed
- **Undo/Redo**: 20 levels of history

### 🎯 **Professional UI/UX**
- **Dark Theme**: Eye-friendly Fusion dark theme
- **Drag & Drop**: Quick image loading
- **Live Preview**: Real-time adjustments
- **Thumbnail List**: Quick navigation between batch images
- **Keyboard Shortcuts**: Ctrl+Z (Undo), Ctrl+Y (Redo), etc.
- **Processing Indicators**: Visual feedback for all operations
- **Windows Taskbar Integration**: Custom icon display

---

## 🚀 Installation

### Prerequisites
- **Python 3.8 or higher**
- **CUDA-capable GPU** (optional, for GPU acceleration)
- **8GB RAM minimum** (16GB recommended for batch processing)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Dinu-Sri/ai-product-studio.git
cd ai-product-studio
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: On first run, the application will automatically download required AI models (~200-500MB depending on selected models). This is a one-time download.

### Step 3: Run the Application
```bash
python main.py
```

Or use the provided batch file:
```bash
run.bat
```

---

## 📦 Requirements

```
PyQt6==6.10.0
Pillow>=10.0.0
rembg==2.0.56
onnxruntime-gpu==1.17.1  # or onnxruntime for CPU-only
opencv-python>=4.8.0
numpy>=1.24.0
scipy>=1.11.0
```

Full requirements in `requirements.txt`.

---

## 🎮 Usage Guide

### Basic Workflow
1. **Load Image**: Drag & drop or click "📂 Load Image"
2. **Select AI Model**: Choose from 14 models based on your subject
3. **Remove Background**: Click "🔮 Remove Background"
4. **Apply Shadow**: Go to Shadow tab, select type, adjust parameters
5. **Transform**: Scale, position, rotate as needed
6. **Add Background**: (Optional) Add solid color or image background
7. **Adjust**: Fine-tune brightness, contrast, etc.
8. **Export**: Save as PNG/JPG with desired settings

### Batch Processing
1. Click "📁 Load Batch"
2. Select multiple images
3. Configure settings in Basic/Shadow/Adjustments tabs
4. Enable "Auto-save after processing" in Export tab
5. Set output folder and filename format
6. Click "▶️ Start Batch" or "⏩ Start Parallel Batch"

### Profile System
- **Shadow Profiles**: Right-click any Profile 1-5 button to save current shadow settings
- **Transform Presets**: Right-click Transform P1-P5 to save position/scale/rotation
- Profiles are saved per shadow type and persist between sessions
- Left-click to load a saved profile instantly

---

## 🧠 AI Models Overview

| Model | Best For | Speed | Quality | Size |
|-------|----------|-------|---------|------|
| **u2net** | General products | ⚡⚡⚡ | ⭐⭐⭐⭐ | 176MB |
| **u2net_human_seg** | People, portraits | ⚡⚡⚡ | ⭐⭐⭐⭐ | 176MB |
| **u2netp** | Fast processing | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 4.7MB |
| **silueta** | High-quality edges | ⚡⚡ | ⭐⭐⭐⭐⭐ | 43MB |
| **isnet-general-use** | Complex scenes | ⚡⚡ | ⭐⭐⭐⭐⭐ | 175MB |
| **sam** | Segment Anything | ⚡ | ⭐⭐⭐⭐⭐ | 358MB |
| **birefnet-general** | Latest SOTA | ⚡⚡ | ⭐⭐⭐⭐⭐ | 512MB |
| **birefnet-portrait** | Professional portraits | ⚡⚡ | ⭐⭐⭐⭐⭐ | 512MB |

*SOTA = State of the Art*

---

## 🛠️ Technical Details

### Architecture
- **GUI Framework**: PyQt6 for modern, responsive UI
- **Image Processing**: Pillow (PIL) + OpenCV + NumPy
- **AI Backend**: ONNX Runtime with GPU acceleration
- **Database**: SQLite for profile persistence
- **Multi-threading**: Parallel batch processing with ThreadPoolExecutor

### Performance Optimizations
- **Debouncing**: 300ms for shadows, 100ms for transforms (prevents UI lag)
- **Vectorized Operations**: NumPy array operations for 100x faster shadow rendering
- **GPU Acceleration**: Automatic CUDA/DirectML detection
- **Memory Management**: Smart image resizing (max 8192px) to prevent OOM errors
- **Parallel Processing**: Multi-core batch processing

### Data Storage
- Profile presets stored in `presets.db` (SQLite)
- Shadow profiles: Separated by type (drop/natural/reflection)
- Transform presets: Global 5-slot storage
- Settings persist across sessions

---

## 🎨 Use Cases

### E-commerce
- Product photography for Amazon, eBay, Shopify
- Consistent white/colored backgrounds
- Professional drop shadows for depth
- Batch processing for entire catalogs

### Marketing & Design
- Social media content creation
- Advertisement graphics
- Website hero images
- Professional presentations

### Fashion & Apparel
- Clothing product shots
- Model photography enhancement
- Ghost mannequin effect (with BiRefNet-cloth)
- Catalog preparation

### Photography Studios
- Portrait background replacement
- Product photography enhancement
- Real estate image editing
- Batch client photo processing

---

## 🐛 Known Issues & Limitations

- Very large images (>8192px) are automatically resized to prevent memory issues
- First run requires model downloads (internet connection needed)
- GPU acceleration requires compatible NVIDIA GPU or DirectML support
- Some complex transparent objects (glass, hair) may require manual refinement

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via GitHub Issues
- Suggest features or improvements
- Submit pull requests
- Share your use cases and results

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**TL;DR**: Free to use, modify, and distribute. Just give credit to the original author (Dinu-Sri).

---

## 🙏 Acknowledgments

- **rembg**: For the excellent background removal library
- **ONNX Runtime**: For GPU-accelerated AI inference
- **PyQt6**: For the powerful GUI framework
- **U²-Net, BiRefNet, SAM**: For state-of-the-art AI models
- All open-source contributors whose libraries made this possible

---

## 📧 Contact & Support

- **GitHub**: [@Dinu-Sri](https://github.com/Dinu-Sri)
- **Issues**: [Report a bug](https://github.com/Dinu-Sri/ai-product-studio/issues)

---

## ⭐ Show Your Support

If you find this project useful, please consider:
- ⭐ **Starring** the repository
- 🐛 **Reporting** bugs and issues
- 💡 **Suggesting** new features
- 🔗 **Sharing** with others who might benefit

---

**Built with ❤️ by Dinu-Sri**

*Making professional product photography accessible to everyone.*
