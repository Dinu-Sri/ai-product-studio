# AI Product Studio v2.0 - Upgrade Implementation Plan

## Expert Recommendations Summary

The expert has identified key areas to take our app to the next level, focusing on:
1. **State-of-the-art models** (BiRefNet)
2. **GPU acceleration** (CUDA/DirectML)
3. **Better edge quality** (soft alpha, smart matting)
4. **Performance optimization** (parallel processing, tiling)

---

## 📋 Implementation Roadmap

### Phase 1: Model Upgrades (Priority: HIGH)

#### 1.1 BiRefNet Integration - New Default Model
**Current:** U²-Net as default (2020 technology)  
**Upgrade:** BiRefNet_general/BiRefNet_dynamic (2024 SOTA)

**Tasks:**
- [ ] Download BiRefNet ONNX models from official GitHub releases
- [ ] Add BiRefNet session initialization to `rembg_sessions`
- [ ] Create new model option: "BiRefNet - Product/General (SOTA)"
- [ ] Set as default model instead of u2net
- [ ] Test compatibility with existing ONNX Runtime setup

**Implementation Notes:**
```python
# Add to model_combo
"BiRefNet-General - Best Quality (SOTA)",  # New default
"BiRefNet-Dynamic - Variable Resolution",
"u2net - Legacy Quality",  # Demoted
"u2netp - Legacy Fast",
```

**Expected Benefits:**
- Better edge detection
- Faster inference than U²-Net
- Handles complex backgrounds better
- Improved transparent object handling

---

#### 1.2 BiRefNet High-Resolution Matting Pipeline
**Current:** Simple alpha matting with fixed thresholds  
**Upgrade:** Two-stage BiRefNet pipeline for production-quality edges

**Pipeline Architecture:**
```
Input Image (4K+)
    ↓
[Stage 1] Downscale → BiRefNet_general → Coarse Mask
    ↓
Compute Bounding Box + Margin
    ↓
[Stage 2] Crop Original → BiRefNet_HR-matting → Refined Alpha
    ↓
Composite Back to Full Size → Final Output
```

**Tasks:**
- [ ] Download BiRefNet_HR-matting ONNX model
- [ ] Implement two-stage processing function
- [ ] Add bounding box detection from coarse mask
- [ ] Create crop/composite logic for local refinement
- [ ] Add UI toggle: "Enable HR Matting (Slower, Best Quality)"

**Implementation Location:**
- New function: `apply_hr_matting_pipeline()` in ImageProcessor class
- Add checkbox in Fine-tune Settings group

**Expected Benefits:**
- Hair-level detail preservation
- Better semi-transparent edges
- No halos on product edges
- Professional photography quality

---

#### 1.3 MODNet Portrait Mode (Optional)
**Current:** U2-Net human seg (general purpose)  
**Upgrade:** MODNet specialized for portraits

**Tasks:**
- [ ] Download MODNet ONNX/TorchScript weights
- [ ] Integrate as separate model option
- [ ] Add to model dropdown: "MODNet - Portrait/People (Fast)"
- [ ] Test performance vs U2-Net human seg

**Expected Benefits:**
- Faster than U2-Net for humans
- Cleaner clothing/accessory separation
- Better for fashion photography

---

### Phase 2: Performance & GPU Acceleration (Priority: HIGH)

#### 2.1 GPU Execution Providers
**Current:** CPU-only ONNX Runtime  
**Upgrade:** Multi-backend GPU support

**Implementation:**
```python
# Detect available execution providers
import onnxruntime as ort

available_providers = ort.get_available_providers()

# Priority order:
# 1. CUDAExecutionProvider (NVIDIA GPUs)
# 2. DmlExecutionProvider (DirectML - Windows AMD/Intel/NVIDIA)
# 3. CPUExecutionProvider (fallback)

def get_optimal_providers():
    if 'CUDAExecutionProvider' in available_providers:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    elif 'DmlExecutionProvider' in available_providers:
        return ['DmlExecutionProvider', 'CPUExecutionProvider']
    else:
        return ['CPUExecutionProvider']

# Update new_session calls
session = ort.InferenceSession(
    model_path,
    providers=get_optimal_providers()
)
```

**Tasks:**
- [ ] Add provider detection at startup
- [ ] Update all `new_session()` calls to use optimal providers
- [ ] Add GPU info display in UI (status bar)
- [ ] Test with NVIDIA, AMD, Intel GPUs
- [ ] Fallback handling if GPU fails

**UI Enhancement:**
- Display active provider: "Running on: NVIDIA GPU (CUDA)" or "Running on: CPU"
- Add to info_label or status bar

**Expected Benefits:**
- 5-10x faster inference on GPUs
- Near-instant processing on mid-range GPUs
- Better user experience for batch processing

---

#### 2.2 Parallel Batch Processing
**Current:** Sequential processing (one at a time)  
**Upgrade:** Parallel processing for batch mode

**Implementation Strategy:**
```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

# For GPU: Use ThreadPoolExecutor (shared session)
# For CPU: Use ProcessPoolExecutor (multiple cores)

def process_batch_parallel(images, settings, num_workers=4):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_image, img, settings) 
                   for img in images]
        results = [f.result() for f in futures]
    return results
```

**Tasks:**
- [ ] Implement parallel batch processor
- [ ] Add worker count setting (auto-detect CPU cores)
- [ ] Handle GPU memory limits (queue if OOM)
- [ ] Update progress tracking for parallel jobs
- [ ] Test thread safety of ONNX sessions

**UI Enhancement:**
- Add "Parallel Processing" checkbox in batch mode
- Show "Processing 4 images simultaneously..."

**Expected Benefits:**
- 2-4x faster batch processing on multi-core CPUs
- Efficient GPU utilization
- Better for processing 100+ images

---

#### 2.3 Tiling Strategy for Ultra-High-Res
**Current:** Auto-downscale to 8192px  
**Upgrade:** Smart tiling for 8K+ without quality loss

**Pipeline:**
```
8K+ Image
    ↓
Downscale to 2048px → BiRefNet_dynamic → Mask
    ↓
Upsample Mask to Original Resolution
    ↓
(Optional) Re-refine edges with matting on crops
    ↓
Final High-Res Output
```

**Tasks:**
- [ ] Implement intelligent downscaling for mask prediction
- [ ] Add mask upsampling with quality preservation
- [ ] Optional: Tile-based refinement for edges
- [ ] Test with 8K, 12K, 16K images

**Expected Benefits:**
- Handle any resolution without memory issues
- Maintain edge quality at original resolution
- Fast processing even for huge images

---

### Phase 3: Advanced Alpha Processing (Priority: MEDIUM)

#### 3.1 Soft Alpha Output (Continuous 0-1)
**Current:** Binary threshold + hard edges  
**Upgrade:** Keep network's continuous alpha output

**Implementation:**
```python
# Instead of:
mask = (alpha > threshold).astype(np.uint8) * 255

# Use:
mask = (alpha * 255).astype(np.uint8)  # Keep gradient

# For compositing:
output = foreground * alpha + background * (1 - alpha)
```

**Tasks:**
- [ ] Modify ImageProcessor to preserve soft alpha
- [ ] Update compositing logic for smooth blending
- [ ] Test with semi-transparent objects (glass, fabric)
- [ ] Ensure WebP/PNG export preserves alpha quality

**Expected Benefits:**
- Natural-looking semi-transparent areas
- Better hair/fur rendering
- Smooth edge transitions
- Professional quality composites

---

#### 3.2 Uncertainty Band Detection
**Current:** Full-image matting or none  
**Upgrade:** Selective matting only on uncertain edges

**Algorithm:**
```python
# Detect uncertain pixels
uncertain_mask = (alpha >= 0.2) & (alpha <= 0.8)

# Find connected regions
from scipy.ndimage import find_objects
regions = find_objects(uncertain_mask)

# For each region:
#   1. Crop original image + mask
#   2. Run HR matting on crop
#   3. Composite refined alpha back
```

**Tasks:**
- [ ] Implement uncertainty detection algorithm
- [ ] Create selective matting function
- [ ] Apply only to edge regions (not solid areas)
- [ ] Optimize for performance

**Expected Benefits:**
- Fast processing (matting only 10-20% of pixels)
- Hair-level detail where needed
- Maintains performance on solid areas

---

#### 3.3 Smart Edge Softness Control
**Current:** Raw pixel erosion/dilation (Edge Shift slider)  
**Upgrade:** Intelligent edge refinement

**New Approach:**
```python
# Instead of uniform dilation/erosion:
def apply_smart_edge_softness(alpha, softness_level):
    # Inner shrink (anti-halo)
    inner_band = detect_inner_edge(alpha, width=2)
    alpha[inner_band] *= 0.9  # Slight reduction
    
    # Outer extension (soft transition)
    outer_band = detect_outer_edge(alpha, width=3)
    alpha[outer_band] = gaussian_blur(alpha[outer_band], sigma=softness_level)
    
    return alpha
```

**Tasks:**
- [ ] Replace Edge Shift slider with "Edge Softness" slider
- [ ] Implement inner shrink (anti-halo)
- [ ] Implement outer extension (soft transition)
- [ ] Add presets: Sharp (0), Normal (5), Soft (10)
- [ ] Test with various background colors

**UI Changes:**
```
Old: Edge Shift: -20 to +20 px
New: Edge Softness: 0 (Sharp) to 10 (Soft)
```

**Expected Benefits:**
- No halos on white/light backgrounds
- Natural edge transitions
- Better for drop shadows
- More intuitive control

---

### Phase 4: UI/UX Improvements (Priority: MEDIUM)

#### 4.1 Reorganize Model Dropdown
**Current Order:**
1. u2net - Best Quality
2. u2netp - Faster
3. u2net_human_seg
4. isnet-general-use
5. isnet-anime

**New Order:**
```python
self.model_combo.addItems([
    "🏆 BiRefNet-General - Product/General (SOTA)",  # Default
    "💎 BiRefNet-HR - Ultra Quality (Slow)",
    "🎭 MODNet - Portrait/People (Fast)",
    "📦 BiRefNet-Dynamic - Variable Resolution",
    "───────────────────────────",  # Separator
    "Legacy: U²-Net - Original Quality",
    "Legacy: U²-Net-P - Original Fast",
    "Legacy: U²-Net Human - Original Portrait",
    "Legacy: IS-Net General",
    "Legacy: IS-Net Anime",
])
```

**Tasks:**
- [ ] Update model dropdown with new hierarchy
- [ ] Add visual separators and icons
- [ ] Update model_map dictionary
- [ ] Add tooltips explaining each model
- [ ] Set BiRefNet-General as default

---

#### 4.2 GPU Status Display
**Tasks:**
- [ ] Add GPU detection on startup
- [ ] Display active provider in status bar
- [ ] Show VRAM usage (if available)
- [ ] Add "⚡ GPU Accelerated" indicator

**UI Mock:**
```
Status Bar: ⚡ GPU Accelerated (NVIDIA RTX 3060) | Processing: 1.2s | Memory: 2.1GB/12GB
```

---

#### 4.3 Processing Quality Presets
**New Feature:** Quick quality presets

**Presets:**
1. **Fast** - BiRefNet-General, no HR matting, GPU
2. **Balanced** - BiRefNet-General, soft alpha, GPU
3. **Ultra** - BiRefNet-HR, uncertainty matting, all features

**UI Addition:**
```python
quality_combo = QComboBox()
quality_combo.addItems(["Fast", "Balanced", "Ultra Quality"])
quality_combo.currentIndexChanged.connect(self.apply_quality_preset)
```

---

### Phase 5: Testing & Documentation (Priority: LOW)

#### 5.1 Testing Plan
- [ ] Test all new models with sample images
- [ ] Benchmark CPU vs GPU performance
- [ ] Test batch processing with 100+ images
- [ ] Verify 8K+ image handling
- [ ] Edge cases: transparent, reflective, hair

#### 5.2 Documentation Updates
- [ ] Update TECHNICAL_SUMMARY.md with new models
- [ ] Add GPU requirements to README
- [ ] Create performance comparison charts
- [ ] Update user guide with new features

---

## 🎯 Implementation Priority

### Week 1: Core Model Upgrade
1. ✅ BiRefNet_general integration
2. ✅ GPU acceleration (CUDA/DirectML)
3. ✅ Update UI model list

### Week 2: Advanced Features
4. ✅ BiRefNet_HR-matting pipeline
5. ✅ Soft alpha output
6. ✅ Smart edge softness

### Week 3: Performance
7. ✅ Parallel batch processing
8. ✅ Tiling for 8K+
9. ✅ Uncertainty band matting

### Week 4: Polish & Testing
10. ✅ UI improvements
11. ✅ Testing suite
12. ✅ Documentation
13. ✅ Build v2.0 executable

---

## 📦 Required Resources

### Models to Download
1. **BiRefNet ONNX weights** (~200MB)
   - BiRefNet_general.onnx
   - BiRefNet_dynamic.onnx
   - BiRefNet_HR-matting.onnx

2. **MODNet ONNX** (~100MB)
   - modnet_photographic_portrait.onnx

### Dependencies to Add
```bash
pip install onnxruntime-gpu  # For CUDA support
# or keep onnxruntime (has DirectML on Windows)
```

### Hardware Requirements Update
- **GPU Recommended:** NVIDIA GTX 1060+ or AMD RX 580+
- **VRAM:** 4GB minimum for 4K images
- **RAM:** 8GB minimum (16GB for 8K+)

---

## 🚀 Expected Performance Gains

### Processing Speed (4K image)
- **Current (CPU):** 3-5 seconds
- **BiRefNet + GPU:** 0.5-1 second (5-10x faster)
- **Batch (10 images):** 50s → 8s with parallel processing

### Edge Quality
- **Current:** Good with alpha matting
- **BiRefNet-HR:** Professional photography grade
- **Improvement:** 30-40% better edge accuracy

### User Experience
- Near-instant feedback on GPU
- Better default results (BiRefNet)
- More intuitive controls (Edge Softness)

---

## ⚠️ Potential Challenges

1. **BiRefNet ONNX availability** - May need conversion from PyTorch
2. **GPU memory management** - Need fallback for large batches
3. **Model download size** - 300-400MB total for all new models
4. **Backward compatibility** - Keep legacy models for existing users
5. **Testing coverage** - Need diverse image dataset

---

## 📝 Notes

- Start with BiRefNet_general as it's the easiest upgrade
- GPU acceleration provides immediate user-visible improvement
- HR-matting can be optional "Ultra Quality" mode
- Keep all old models for backward compatibility
- Consider beta testing phase before v2.0 release

---

**Version Target:** 2.0  
**Estimated Development Time:** 2-4 weeks  
**Expected Release:** December 2025
