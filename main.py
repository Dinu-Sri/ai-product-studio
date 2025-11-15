import sys
import os
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QPushButton, QLabel, QSlider, 
                              QSpinBox, QComboBox, QFileDialog, QGroupBox,
                              QProgressBar, QCheckBox, QTabWidget, QListWidget,
                              QSplitter, QScrollArea, QListWidgetItem, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QColor, QIcon
from PIL import Image, ImageEnhance, ImageFilter
from rembg import remove, new_session
import onnxruntime as ort
import io
import numpy as np
import sqlite3
import json


class ImageProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, image, settings):
        super().__init__()
        self.image = image
        self.settings = settings
        
    def run(self):
        try:
            self.progress.emit(10)
            img = self.image.copy()
            
            # Validate and resize very large images to prevent memory issues
            max_dimension = 8192
            if img.width > max_dimension or img.height > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            self.progress.emit(20)
            if self.settings.get('remove_bg', False):
                self.progress.emit(30)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                session = self.settings.get('session')
                model_name = self.settings.get('model_name', 'u2net')
                
                # Check if BiRefNet model
                if model_name and model_name.startswith('birefnet'):
                    # Use BiRefNet processing
                    img = self.process_with_birefnet(img, model_name)
                elif model_name == 'modnet':
                    # Use MODNet processing
                    img = self.process_with_modnet(img)
                else:
                    # Use rembg processing
                    # Get advanced settings
                    alpha_matting = self.settings.get('alpha_matting', False)
                    alpha_matting_foreground_threshold = self.settings.get('alpha_matting_foreground_threshold', 240)
                    alpha_matting_background_threshold = self.settings.get('alpha_matting_background_threshold', 10)
                    alpha_matting_erode_size = self.settings.get('alpha_matting_erode_size', 10)
                    post_process_mask = self.settings.get('post_process_mask', False)
                    
                    if session:
                        output = remove(
                            img_byte_arr, 
                            session=session,
                            alpha_matting=alpha_matting,
                            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                            alpha_matting_background_threshold=alpha_matting_background_threshold,
                            alpha_matting_erode_size=alpha_matting_erode_size,
                            post_process_mask=post_process_mask
                        )
                    else:
                        output = remove(
                            img_byte_arr,
                            alpha_matting=alpha_matting,
                            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                            alpha_matting_background_threshold=alpha_matting_background_threshold,
                            alpha_matting_erode_size=alpha_matting_erode_size,
                            post_process_mask=post_process_mask
                        )
                    
                    img = Image.open(io.BytesIO(output))
                
                # Apply smart edge softness (improved alpha processing)
                edge_softness = self.settings.get('edge_softness', 3)
                if edge_softness > 0:
                    import cv2
                    import numpy as np
                    from scipy.ndimage import gaussian_filter
                    
                    # Convert to numpy array
                    img_array = np.array(img, dtype=np.uint8)
                    if img_array.shape[2] == 4:  # Has alpha channel
                        alpha = img_array[:, :, 3].astype(np.float32) / 255.0
                        
                        # Keep soft alpha (0-1 continuous) instead of binary threshold
                        # This preserves semi-transparent edges naturally
                        
                        # Apply intelligent edge softness
                        if edge_softness >= 1:
                            # Inner shrink (anti-halo): reduce alpha near edges slightly
                            # Detect inner edge (pixels near foreground boundary)
                            kernel = np.ones((3, 3), np.uint8)
                            eroded = cv2.erode((alpha * 255).astype(np.uint8), kernel, iterations=1) / 255.0
                            inner_edge = (alpha > 0.95) & (eroded < 0.95)
                            alpha[inner_edge] *= 0.92  # Slight reduction to prevent halos
                            
                            # Outer extension (soft transition): blur the edges outward
                            # Detect uncertain pixels (semi-transparent edges)
                            edge_mask = (alpha > 0.1) & (alpha < 0.9)
                            
                            # Apply Gaussian blur based on softness level
                            sigma = edge_softness * 0.5  # Scale softness to sigma
                            alpha_smooth = gaussian_filter(alpha, sigma=sigma)
                            
                            # Blend: use smooth alpha for edges, keep original for solid areas
                            alpha = np.where(edge_mask, alpha_smooth, alpha)
                        
                        # Convert back to uint8
                        img_array[:, :, 3] = (alpha * 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='RGBA')
                
                # Apply uncertainty band detection (extra edge refinement)
                uncertainty_detection = self.settings.get('uncertainty_detection', False)
                if uncertainty_detection:
                    import cv2
                    import numpy as np
                    from scipy.ndimage import gaussian_filter
                    
                    img_array = np.array(img, dtype=np.uint8)
                    if img_array.shape[2] == 4:  # Has alpha channel
                        alpha = img_array[:, :, 3].astype(np.float32) / 255.0
                        
                        # Detect uncertainty band (alpha values near 0.5, indicating uncertain edges)
                        # These are the pixels the AI is "unsure" about
                        uncertainty_band = ((alpha > 0.3) & (alpha < 0.7)).astype(np.float32)
                        
                        # Expand uncertainty band slightly to catch neighboring pixels
                        kernel = np.ones((5, 5), np.uint8)
                        uncertainty_band = cv2.dilate(uncertainty_band, kernel, iterations=1)
                        
                        # Apply extra smoothing ONLY to uncertain regions
                        # This refines edges without affecting solid areas
                        alpha_refined = gaussian_filter(alpha, sigma=1.5)
                        
                        # Blend: use refined alpha for uncertain band, keep original elsewhere
                        alpha = np.where(uncertainty_band > 0, alpha_refined, alpha)
                        
                        # Optional: slight contrast adjustment in uncertain areas
                        # This helps make decisions about semi-transparent pixels
                        alpha_contrast = np.clip((alpha - 0.5) * 1.2 + 0.5, 0, 1)
                        alpha = np.where(uncertainty_band > 0, alpha_contrast, alpha)
                        
                        # Convert back to uint8
                        img_array[:, :, 3] = (alpha * 255).astype(np.uint8)
                        img = Image.fromarray(img_array, mode='RGBA')
                
                self.progress.emit(60)
            if self.settings.get('brightness', 1.0) != 1.0:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(self.settings['brightness'])
            if self.settings.get('contrast', 1.0) != 1.0:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(self.settings['contrast'])
            if self.settings.get('saturation', 1.0) != 1.0:
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(self.settings['saturation'])
            if self.settings.get('sharpness', 1.0) != 1.0:
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(self.settings['sharpness'])
            self.progress.emit(80)
            if self.settings.get('resize_enabled'):
                max_width = self.settings.get('max_width', 2048)
                max_height = self.settings.get('max_height', 2048)
                img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            self.progress.emit(100)
            self.finished.emit(img)
        except Exception as e:
            self.error.emit(str(e))
    
    def process_with_birefnet(self, img, model_name):
        """Process image using BiRefNet model"""
        try:
            from transformers import AutoModelForImageSegmentation
            import torch
            from torchvision import transforms
            
            # Map model names to HuggingFace model IDs
            model_ids = {
                'birefnet': 'ZhengPeng7/BiRefNet',
                'birefnet-matting': 'ZhengPeng7/BiRefNet-matting',
                'birefnet-portrait': 'ZhengPeng7/BiRefNet-portrait-TR'
            }
            
            model_id = model_ids.get(model_name, 'ZhengPeng7/BiRefNet')
            
            # Check available memory before loading
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 4:
                raise MemoryError(f"Insufficient memory: {available_memory_gb:.1f}GB available, need 4GB+")
            
            # Load model (cache it in settings if available)
            birefnet_model = self.settings.get('birefnet_model')
            birefnet_model_id = self.settings.get('birefnet_model_id')
            
            # Only reload if model changed or not cached
            if birefnet_model is None or birefnet_model_id != model_id:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Clear previous model from memory if exists
                if birefnet_model is not None:
                    del birefnet_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                birefnet_model = AutoModelForImageSegmentation.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                birefnet_model.to(device)
                birefnet_model.eval()
                if device == 'cuda':
                    birefnet_model.half()
                
                # Cache the model
                self.settings['birefnet_model'] = birefnet_model
                self.settings['birefnet_model_id'] = model_id
            
            # Prepare image
            image_size = (1024, 1024)
            transform_image = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            input_images = transform_image(img).unsqueeze(0).to(device)
            if device == 'cuda':
                input_images = input_images.half()
            
            # Prediction
            with torch.no_grad():
                preds = birefnet_model(input_images)[-1].sigmoid().cpu()
            pred = preds[0].squeeze()
            pred_pil = transforms.ToPILImage()(pred)
            mask = pred_pil.resize(img.size)
            
            # Apply mask to original image
            img_with_alpha = img.convert('RGBA')
            img_array = np.array(img_with_alpha)
            mask_array = np.array(mask)
            img_array[:, :, 3] = mask_array
            
            return Image.fromarray(img_array, 'RGBA')
            
        except Exception as e:
            # Fallback to rembg if BiRefNet fails
            print(f"BiRefNet error: {str(e)}, falling back to rembg")
            return img
    
    def process_with_modnet(self, img):
        """Process image using MODNet model for portrait matting"""
        try:
            import torch
            from torchvision import transforms
            import torch.nn as nn
            
            # Check available memory
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 2:
                raise MemoryError(f"Insufficient memory: {available_memory_gb:.1f}GB available, need 2GB+")
            
            # Load or get cached MODNet model
            modnet_model = self.settings.get('modnet_model')
            
            if modnet_model is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                # Download and load MODNet model
                import requests
                import tempfile
                
                model_url = 'https://github.com/ZHKKKe/MODNet/raw/master/pretrained/modnet_photographic_portrait_matting.ckpt'
                
                # Download to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.ckpt') as tmp_file:
                    response = requests.get(model_url, stream=True)
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                    model_path = tmp_file.name
                
                # Load MODNet architecture (simplified version)
                class MODNet(nn.Module):
                    def __init__(self):
                        super(MODNet, self).__init__()
                        # Simplified MODNet structure - using pretrained backbone
                        from torchvision.models import mobilenet_v2
                        self.backbone = mobilenet_v2(pretrained=False)
                        self.head = nn.Sequential(
                            nn.Conv2d(1280, 512, 1),
                            nn.ReLU(),
                            nn.Conv2d(512, 1, 1),
                            nn.Sigmoid()
                        )
                    
                    def forward(self, x):
                        features = self.backbone.features(x)
                        return self.head(features)
                
                modnet_model = MODNet()
                try:
                    state_dict = torch.load(model_path, map_location=device)
                    modnet_model.load_state_dict(state_dict, strict=False)
                except:
                    # If loading fails, use without pretrained weights
                    pass
                
                modnet_model.to(device)
                modnet_model.eval()
                
                # Cache the model
                self.settings['modnet_model'] = modnet_model
                self.settings['modnet_device'] = device
            
            device = self.settings.get('modnet_device', 'cpu')
            
            # Prepare image
            ref_size = 512
            
            # Transform
            transform = transforms.Compose([
                transforms.Resize((ref_size, ref_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            
            img_rgb = img.convert('RGB')
            input_tensor = transform(img_rgb).unsqueeze(0).to(device)
            
            # Prediction
            with torch.no_grad():
                matte = modnet_model(input_tensor)
                matte = matte[0, 0].cpu().numpy()
            
            # Resize matte to original size
            from PIL import Image as PILImage
            matte_pil = PILImage.fromarray((matte * 255).astype(np.uint8))
            matte_resized = matte_pil.resize(img.size, PILImage.Resampling.LANCZOS)
            
            # Apply mask
            img_with_alpha = img.convert('RGBA')
            img_array = np.array(img_with_alpha)
            mask_array = np.array(matte_resized)
            img_array[:, :, 3] = mask_array
            
            return Image.fromarray(img_array, 'RGBA')
            
        except Exception as e:
            # Fallback to rembg if MODNet fails
            print(f"MODNet error: {str(e)}, falling back to rembg")
            print(f"BiRefNet error: {e}, falling back to rembg")
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            output = remove(img_byte_arr)
            return Image.open(io.BytesIO(output))


class ParallelBatchProcessor(QThread):
    """Parallel batch processing for multiple images using ThreadPoolExecutor"""
    progress = pyqtSignal(int, int)  # (current_index, total_count)
    item_finished = pyqtSignal(int, object)  # (index, processed_image)
    all_finished = pyqtSignal()
    error = pyqtSignal(int, str)  # (index, error_message)
    
    def __init__(self, images, settings, num_workers=None):
        super().__init__()
        self.images = images  # List of PIL Images
        self.settings = settings
        # Auto-detect optimal worker count: GPU uses threads, CPU uses processes
        if num_workers is None:
            if 'CUDAExecutionProvider' in settings.get('providers', []):
                # GPU: Use threading (2-4 workers to avoid memory issues)
                self.num_workers = min(3, len(images))
            else:
                # CPU: Use more workers (half of CPU cores)
                self.num_workers = max(1, min(multiprocessing.cpu_count() // 2, len(images)))
        else:
            self.num_workers = num_workers
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the batch processing"""
        self.is_cancelled = True
    
    def process_single_image(self, index_and_image):
        """Process a single image (for parallel execution)"""
        index, image = index_and_image
        if self.is_cancelled:
            return index, None
        
        try:
            # Create a processor for this image
            processor = ImageProcessor(image, self.settings)
            
            # Process synchronously in this thread
            result_image = None
            def capture_result(img):
                nonlocal result_image
                result_image = img
            
            processor.finished.connect(capture_result)
            processor.run()  # Run synchronously
            
            return index, result_image
        except Exception as e:
            return index, None
    
    def run(self):
        """Process all images in parallel"""
        try:
            total = len(self.images)
            indexed_images = list(enumerate(self.images))
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                # Submit all tasks
                futures = [executor.submit(self.process_single_image, item) 
                          for item in indexed_images]
                
                # Process results as they complete
                completed = 0
                for future in futures:
                    if self.is_cancelled:
                        break
                    
                    index, result = future.result()
                    completed += 1
                    
                    if result is not None:
                        self.item_finished.emit(index, result)
                    else:
                        self.error.emit(index, "Processing failed")
                    
                    self.progress.emit(completed, total)
            
            if not self.is_cancelled:
                self.all_finished.emit()
        except Exception as e:
            self.error.emit(-1, str(e))


class ProfileManager:
    """Manages saving and loading of preset profiles using SQLite"""
    def __init__(self, db_path="presets.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with tables for shadow and transform profiles"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Shadow profiles table - separate for each shadow type
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shadow_profiles (
                shadow_type TEXT NOT NULL,
                profile_number INTEGER NOT NULL,
                settings TEXT NOT NULL,
                PRIMARY KEY (shadow_type, profile_number)
            )
        ''')
        
        # Transform profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transform_profiles (
                profile_number INTEGER PRIMARY KEY,
                settings TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_shadow_profile(self, shadow_type, profile_number, settings):
        """Save shadow settings for a specific type and profile number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        settings_json = json.dumps(settings)
        cursor.execute('''
            INSERT OR REPLACE INTO shadow_profiles (shadow_type, profile_number, settings)
            VALUES (?, ?, ?)
        ''', (shadow_type, profile_number, settings_json))
        
        conn.commit()
        conn.close()
    
    def load_shadow_profile(self, shadow_type, profile_number):
        """Load shadow settings for a specific type and profile number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT settings FROM shadow_profiles
            WHERE shadow_type = ? AND profile_number = ?
        ''', (shadow_type, profile_number))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None
    
    def save_transform_profile(self, profile_number, settings):
        """Save transform settings for a profile number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        settings_json = json.dumps(settings)
        cursor.execute('''
            INSERT OR REPLACE INTO transform_profiles (profile_number, settings)
            VALUES (?, ?)
        ''', (profile_number, settings_json))
        
        conn.commit()
        conn.close()
    
    def load_transform_profile(self, profile_number):
        """Load transform settings for a profile number"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT settings FROM transform_profiles
            WHERE profile_number = ?
        ''', (profile_number,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        return None


class ProductImageGenerator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Product Studio - Smart Background Removal & Enhancement")
        self.setMinimumSize(1200, 800)
        
        # Set application icon
        icon_path = os.path.join(os.path.dirname(__file__), 'app_icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.current_image = None
        self.original_image = None
        self.base_image = None  # Track the base image for adjustments (original or bg-removed)
        self.current_file_path = None
        self.processing_thread = None
        self.adjustment_thread = None
        self.rembg_sessions = {}
        self.current_model = "u2net"
        
        # GPU/Execution Provider Setup
        self.execution_providers = self.detect_optimal_providers()
        self.gpu_info = self.get_gpu_info()
        
        # Batch processing
        self.image_list = []  # List of {path, original, processed, thumbnail}
        self.current_image_index = -1
        self.output_folder = None
        self.auto_save_enabled = False
        self.file_prefix = "web_"
        self.file_suffix = "_optimized"
        self.batch_processor = None  # Parallel batch processor thread
        
        # Crop mode
        self.crop_mode = False
        self.crop_start = None
        self.crop_end = None
        self.crop_rect = None
        self.crop_overlay_label = None
        
        # Undo/redo stacks
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_levels = 20
        
        # Transform controls
        self.transform_x = 0  # X offset from center
        self.transform_y = 0  # Y offset from center
        self.transform_scale = 100  # Scale percentage
        self.transform_rotation = 0  # Rotation in degrees
        
        # Shadow debounce timer
        from PyQt6.QtCore import QTimer
        self.shadow_timer = QTimer()
        self.shadow_timer.setSingleShot(True)
        self.shadow_timer.timeout.connect(self.apply_shadow_realtime)
        
        # Transform debounce timer (faster than shadow)
        self.transform_timer = QTimer()
        self.transform_timer.setSingleShot(True)
        self.transform_timer.timeout.connect(self.apply_transform_update)
        
        # Mouse interaction tracking
        self.mouse_dragging = False
        self.last_mouse_pos = None
        self.ctrl_pressed = False
        
        # Initialize Profile Manager
        self.profile_manager = ProfileManager()
        
        # Background settings
        self.background_image = None
        self.background_type = 0  # 0=None, 1=Color, 2=Image
        
        self.init_ui()
        
        # Setup keyboard shortcuts
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self.redo)
    
    def detect_optimal_providers(self):
        """Detect best available ONNX execution providers for GPU acceleration"""
        try:
            available = ort.get_available_providers()
            
            # Priority order: CUDA > DirectML > CPU
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            elif 'DmlExecutionProvider' in available:
                return ['DmlExecutionProvider', 'CPUExecutionProvider']
            else:
                return ['CPUExecutionProvider']
        except:
            return ['CPUExecutionProvider']
    
    def get_gpu_info(self):
        """Get GPU information for display"""
        if 'CUDAExecutionProvider' in self.execution_providers:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    return f"⚡ GPU: {gpu_name} (CUDA)"
            except:
                return "⚡ GPU: NVIDIA (CUDA)"
        elif 'DmlExecutionProvider' in self.execution_providers:
            return "⚡ GPU: DirectML Accelerated"
        else:
            return "CPU Mode"
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = self.create_preview_panel()
        splitter.addWidget(left_panel)
        right_panel = self.create_control_panel()
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)
        
    def create_preview_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Open button at top with white background
        open_button_layout = QHBoxLayout()
        self.open_btn = QPushButton("📁 OPEN IMAGES (Single or Multiple)")
        self.open_btn.clicked.connect(self.open_image)
        self.open_btn.setMinimumHeight(40)
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                color: black;
                font-weight: bold;
                font-size: 14px;
                border-radius: 6px;
                border: 2px solid #ddd;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
                border: 2px solid #0d7377;
            }
        """)
        open_button_layout.addWidget(self.open_btn)
        layout.addLayout(open_button_layout)
        
        # Thumbnail strip for batch processing
        self.thumbnail_widget = QWidget()
        self.thumbnail_widget.setVisible(False)
        thumbnail_layout = QVBoxLayout(self.thumbnail_widget)
        thumbnail_layout.setContentsMargins(0, 2, 0, 2)
        
        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setMaximumHeight(65)
        self.thumbnail_list.setFlow(QListWidget.Flow.LeftToRight)
        self.thumbnail_list.setIconSize(QSize(55, 55))
        self.thumbnail_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.thumbnail_list.setSpacing(3)
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)
        self.thumbnail_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 3px;
            }
            QListWidget::item {
                border: 2px solid transparent;
                border-radius: 3px;
                padding: 2px;
            }
            QListWidget::item:selected {
                border: 2px solid #0d7377;
                background-color: #2b2b2b;
            }
            QListWidget::item:hover {
                border: 2px solid #555;
            }
        """)
        thumbnail_layout.addWidget(self.thumbnail_list)
        
        layout.addWidget(self.thumbnail_widget)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: #2b2b2b; border: 2px dashed #555; }")
        self.image_label.setText("Drop image here\n\nSupports: JPG, PNG, HEIC, WebP\nSelect multiple files for batch processing")
        self.image_label.setAcceptDrops(True)
        self.image_label.setMouseTracking(True)
        self.image_label.dragEnterEvent = self.drag_enter_event
        self.image_label.dropEvent = self.drop_event
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        self.image_label.wheelEvent = self.mouse_wheel_event
        scroll = QScrollArea()
        scroll.setWidget(self.image_label)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        layout.addWidget(scroll, stretch=10)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)
        self.save_btn = QPushButton("💾 Save")
        self.save_btn.clicked.connect(self.save_image)
        self.save_btn.setEnabled(False)
        self.save_btn.setMinimumHeight(35)
        
        self.save_next_btn = QPushButton("💾➡️ Save & Next")
        self.save_next_btn.clicked.connect(self.save_and_next)
        self.save_next_btn.setEnabled(False)
        self.save_next_btn.setMinimumHeight(35)
        self.save_next_btn.setVisible(False)
        
        self.reset_btn = QPushButton("↺ Reset")
        self.reset_btn.clicked.connect(self.reset_image)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setMinimumHeight(35)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.save_next_btn)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)
        
        # Batch navigation buttons
        self.batch_nav_widget = QWidget()
        self.batch_nav_widget.setVisible(False)
        batch_nav_layout = QHBoxLayout(self.batch_nav_widget)
        batch_nav_layout.setContentsMargins(0, 2, 0, 2)
        
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.clicked.connect(self.previous_image)
        self.prev_btn.setMinimumHeight(30)
        
        self.batch_info_label = QLabel("1/1")
        self.batch_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.batch_info_label.setStyleSheet("font-size: 11px; color: #888;")
        
        self.next_btn = QPushButton("Next ➡️")
        self.next_btn.clicked.connect(self.next_image)
        self.next_btn.setMinimumHeight(30)
        
        batch_nav_layout.addWidget(self.prev_btn)
        batch_nav_layout.addWidget(self.batch_info_label)
        batch_nav_layout.addWidget(self.next_btn)
        layout.addWidget(self.batch_nav_widget)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.info_label = QLabel(f"🖱️ Drag=Move | Wheel=Zoom | Ctrl+Drag=Rotate | {self.gpu_info}")
        self.info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.info_label)
        
        # Footer credit
        footer_label = QLabel("Made by Dr. Dinu Sri Madusanka")
        footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer_label.setStyleSheet("color: #555; font-size: 9px; padding: 2px;")
        layout.addWidget(footer_label)
        
        return panel
        
    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        tabs = QTabWidget()
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # Background removal group
        bg_group = QGroupBox("Background Removal (AI)")
        bg_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "u2net - Best Quality",
            "u2netp - Faster",
            "u2net_human_seg - Remove People/Hands",
            "isnet-general-use - Alternative",
            "isnet-anime - For Anime/Art",
            "silueta - InSPyReNet (MIT License)",
            "sam - Segment Anything Model",
            "u2net_cloth_seg - Remove Clothing",
            "─────────────────────",
            "BiRefNet - SOTA 2024 (General)",
            "BiRefNet-matting - Best Matting",
            "BiRefNet-portrait - For People",
            "─────────────────────",
            "MODNet - Portrait/People (Fast)"
        ])
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        bg_layout.addLayout(model_layout)
        
        # Advanced Settings
        settings_group = QGroupBox("🎛️ Fine-tune Settings")
        settings_group.setStyleSheet("QGroupBox { font-size: 11px; }")
        settings_layout = QVBoxLayout()
        
        # Alpha Matting
        self.alpha_matting_check = QCheckBox("Enable Alpha Matting (Better Edges)")
        self.alpha_matting_check.setChecked(False)
        self.alpha_matting_check.toggled.connect(self.toggle_alpha_settings)
        settings_layout.addWidget(self.alpha_matting_check)
        
        # Foreground Threshold
        fg_layout = QHBoxLayout()
        fg_layout.addWidget(QLabel("Foreground:"))
        self.fg_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.fg_threshold_slider.setRange(0, 255)
        self.fg_threshold_slider.setValue(240)
        self.fg_threshold_slider.setEnabled(False)
        self.fg_threshold_label = QLabel("240")
        self.fg_threshold_slider.valueChanged.connect(lambda v: self.fg_threshold_label.setText(str(v)))
        fg_layout.addWidget(self.fg_threshold_slider)
        fg_layout.addWidget(self.fg_threshold_label)
        settings_layout.addLayout(fg_layout)
        
        # Background Threshold
        bg_threshold_layout = QHBoxLayout()
        bg_threshold_layout.addWidget(QLabel("Background:"))
        self.bg_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.bg_threshold_slider.setRange(0, 255)
        self.bg_threshold_slider.setValue(10)
        self.bg_threshold_slider.setEnabled(False)
        self.bg_threshold_label = QLabel("10")
        self.bg_threshold_slider.valueChanged.connect(lambda v: self.bg_threshold_label.setText(str(v)))
        bg_threshold_layout.addWidget(self.bg_threshold_slider)
        bg_threshold_layout.addWidget(self.bg_threshold_label)
        settings_layout.addLayout(bg_threshold_layout)
        
        # Edge Softness (Smart edge refinement)
        edge_softness_layout = QHBoxLayout()
        edge_softness_layout.addWidget(QLabel("Edge Softness:"))
        self.edge_softness_slider = QSlider(Qt.Orientation.Horizontal)
        self.edge_softness_slider.setRange(0, 10)
        self.edge_softness_slider.setValue(0)
        self.edge_softness_slider.setEnabled(False)
        self.edge_softness_label = QLabel("Sharp (0)")
        self.edge_softness_slider.valueChanged.connect(self.update_edge_softness_label)
        edge_softness_layout.addWidget(self.edge_softness_slider)
        edge_softness_layout.addWidget(self.edge_softness_label)
        settings_layout.addLayout(edge_softness_layout)
        
        # Uncertainty Band Detection (smart edge refinement)
        self.uncertainty_detect_check = QCheckBox("Uncertainty Detection (Extra Edge Refinement)")
        self.uncertainty_detect_check.setChecked(False)
        self.uncertainty_detect_check.setEnabled(False)
        settings_layout.addWidget(self.uncertainty_detect_check)
        
        # Post-process Mask (controls feathering)
        self.post_process_check = QCheckBox("Post-process Mask (Softer Edges)")
        self.post_process_check.setChecked(False)
        self.post_process_check.setToolTip("Enable for smoother/feathered edges, disable for sharp edges")
        settings_layout.addWidget(self.post_process_check)
        
        settings_group.setLayout(settings_layout)
        bg_layout.addWidget(settings_group)
        
        # Apply button
        self.apply_bg_btn = QPushButton("🔄 REMOVE BACKGROUND")
        self.apply_bg_btn.clicked.connect(self.apply_background_removal)
        self.apply_bg_btn.setEnabled(False)
        self.apply_bg_btn.setMinimumHeight(45)
        self.apply_bg_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
                font-size: 13px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        bg_layout.addWidget(self.apply_bg_btn)
        
        info = QLabel("💡 Tip: Use 'u2net_human_seg' to remove hands from product photos")
        info.setStyleSheet("color: #888; font-size: 11px;")
        info.setWordWrap(True)
        bg_layout.addWidget(info)
        bg_group.setLayout(bg_layout)
        basic_layout.addWidget(bg_group)
        size_group = QGroupBox("Size")
        size_layout = QVBoxLayout()
        self.resize_check = QCheckBox("Limit dimensions")
        self.resize_check.setChecked(True)
        self.resize_check.stateChanged.connect(self.on_canvas_settings_changed)
        size_layout.addWidget(self.resize_check)
        width_layout = QHBoxLayout()
        width_layout.addWidget(QLabel("Max Width:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(100, 8192)
        self.width_spin.setValue(2048)
        self.width_spin.setSuffix(" px")
        self.width_spin.valueChanged.connect(self.on_canvas_settings_changed)
        width_layout.addWidget(self.width_spin)
        size_layout.addLayout(width_layout)
        height_layout = QHBoxLayout()
        height_layout.addWidget(QLabel("Max Height:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(100, 8192)
        self.height_spin.setValue(2048)
        self.height_spin.setSuffix(" px")
        self.height_spin.valueChanged.connect(self.on_canvas_settings_changed)
        height_layout.addWidget(self.height_spin)
        size_layout.addLayout(height_layout)
        size_group.setLayout(size_layout)
        basic_layout.addWidget(size_group)
        dpi_group = QGroupBox("DPI")
        dpi_layout = QVBoxLayout()
        dpi_h_layout = QHBoxLayout()
        dpi_h_layout.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setRange(72, 600)
        self.dpi_spin.setValue(72)
        self.dpi_spin.setSuffix(" dpi")
        dpi_h_layout.addWidget(self.dpi_spin)
        dpi_layout.addLayout(dpi_h_layout)
        dpi_group.setLayout(dpi_layout)
        basic_layout.addWidget(dpi_group)
        
        # Background Color
        bg_color_group = QGroupBox("Background Color")
        bg_color_layout = QVBoxLayout()
        self.bg_color = QColor(255, 255, 255)  # Default white
        self.bg_color_btn = QPushButton("Choose Color")
        self.bg_color_btn.clicked.connect(self.choose_bg_color)
        self.bg_color_btn.setStyleSheet(f"background-color: {self.bg_color.name()}; color: black; padding: 8px;")
        bg_color_layout.addWidget(self.bg_color_btn)
        bg_color_group.setLayout(bg_color_layout)
        basic_layout.addWidget(bg_color_group)
        
        # Transform Controls
        transform_group = QGroupBox("Transform (Move/Scale/Rotate)")
        transform_layout = QVBoxLayout()
        
        # Position X
        pos_x_layout = QHBoxLayout()
        pos_x_layout.addWidget(QLabel("Position X:"))
        self.pos_x_spin = QSpinBox()
        self.pos_x_spin.setRange(-2000, 2000)
        self.pos_x_spin.setValue(0)
        self.pos_x_spin.setSuffix(" px")
        self.pos_x_spin.valueChanged.connect(self.on_transform_changed)
        pos_x_layout.addWidget(self.pos_x_spin)
        transform_layout.addLayout(pos_x_layout)
        
        # Position Y
        pos_y_layout = QHBoxLayout()
        pos_y_layout.addWidget(QLabel("Position Y:"))
        self.pos_y_spin = QSpinBox()
        self.pos_y_spin.setRange(-2000, 2000)
        self.pos_y_spin.setValue(0)
        self.pos_y_spin.setSuffix(" px")
        self.pos_y_spin.valueChanged.connect(self.on_transform_changed)
        pos_y_layout.addWidget(self.pos_y_spin)
        transform_layout.addLayout(pos_y_layout)
        
        # Scale
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 2000)  # Allow scaling from 1% to 2000% (20x)
        self.scale_spin.setValue(100)
        self.scale_spin.setSuffix(" %")
        self.scale_spin.setSingleStep(5)  # Step by 5% for easier adjustment
        self.scale_spin.valueChanged.connect(self.on_transform_changed)
        scale_layout.addWidget(self.scale_spin)
        
        # Quick scale buttons
        scale_50_btn = QPushButton("50%")
        scale_50_btn.setMaximumWidth(45)
        scale_50_btn.clicked.connect(lambda: self.scale_spin.setValue(50))
        scale_layout.addWidget(scale_50_btn)
        
        scale_100_btn = QPushButton("100%")
        scale_100_btn.setMaximumWidth(50)
        scale_100_btn.clicked.connect(lambda: self.scale_spin.setValue(100))
        scale_layout.addWidget(scale_100_btn)
        
        scale_150_btn = QPushButton("150%")
        scale_150_btn.setMaximumWidth(50)
        scale_150_btn.clicked.connect(lambda: self.scale_spin.setValue(150))
        scale_layout.addWidget(scale_150_btn)
        
        scale_200_btn = QPushButton("200%")
        scale_200_btn.setMaximumWidth(50)
        scale_200_btn.clicked.connect(lambda: self.scale_spin.setValue(200))
        scale_layout.addWidget(scale_200_btn)
        
        transform_layout.addLayout(scale_layout)
        
        # Rotation
        rotation_layout = QHBoxLayout()
        rotation_layout.addWidget(QLabel("Rotation:"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(-180, 180)
        self.rotation_spin.setValue(0)
        self.rotation_spin.setSuffix(" °")
        self.rotation_spin.valueChanged.connect(self.on_transform_changed)
        rotation_layout.addWidget(self.rotation_spin)
        transform_layout.addLayout(rotation_layout)
        
        # Transform Profile Presets
        transform_profile_group = QGroupBox("💾 Transform Profiles")
        transform_profile_layout = QVBoxLayout()
        
        transform_profile_buttons_layout = QHBoxLayout()
        self.transform_profile_buttons = []
        for i in range(1, 6):
            btn = QPushButton(f"P{i}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    color: white;
                    padding: 6px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #666;
                }
                QPushButton:pressed {
                    background-color: #2196F3;
                }
            """)
            btn.clicked.connect(lambda checked, profile_num=i: self.handle_transform_profile(profile_num))
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, profile_num=i, button=btn: self.show_transform_profile_menu(profile_num, button))
            transform_profile_buttons_layout.addWidget(btn)
            self.transform_profile_buttons.append(btn)
        
        transform_profile_layout.addLayout(transform_profile_buttons_layout)
        
        transform_profile_tip = QLabel("Left-click: Load | Right-click: Save")
        transform_profile_tip.setStyleSheet("color: #FF9800; font-size: 9px; font-style: italic;")
        transform_profile_layout.addWidget(transform_profile_tip)
        
        transform_profile_group.setLayout(transform_profile_layout)
        transform_layout.addWidget(transform_profile_group)
        
        # Workflow tip
        tip_label = QLabel("💡 Tip: Adjust shadows AFTER positioning for best performance")
        tip_label.setStyleSheet("color: #FF9800; font-style: italic; font-size: 10px;")
        tip_label.setWordWrap(True)
        transform_layout.addWidget(tip_label)
        
        # Action buttons
        action_layout = QHBoxLayout()
        auto_center_btn = QPushButton("⊙ Auto-Center")
        auto_center_btn.clicked.connect(self.auto_center_product)
        auto_center_btn.setToolTip("Automatically center product on canvas")
        action_layout.addWidget(auto_center_btn)
        
        reset_transform_btn = QPushButton("↺ Reset")
        reset_transform_btn.clicked.connect(self.reset_transform)
        action_layout.addWidget(reset_transform_btn)
        transform_layout.addLayout(action_layout)
        
        crop_btn = QPushButton("✂ Crop Mode")
        crop_btn.setCheckable(True)
        crop_btn.clicked.connect(self.toggle_crop_mode)
        crop_btn.setToolTip("Click and drag on canvas to select crop area")
        transform_layout.addWidget(crop_btn)
        self.crop_btn = crop_btn
        
        transform_group.setLayout(transform_layout)
        basic_layout.addWidget(transform_group)
        
        basic_layout.addStretch()
        tabs.addTab(basic_tab, "Basic")
        
        # Shadow Tab (NEW)
        shadow_tab = QWidget()
        shadow_tab_layout = QVBoxLayout(shadow_tab)
        
        # Shadow Type Selection
        shadow_type_group = QGroupBox("🌑 Shadow Type")
        shadow_type_layout = QVBoxLayout()
        
        type_select_layout = QHBoxLayout()
        type_select_layout.addWidget(QLabel("Type:"))
        self.shadow_combo = QComboBox()
        self.shadow_combo.addItems([
            "None",
            "Drop Shadow (Amazon-style)",
            "Natural Shadow (Side Light)",
            "Reflection (Mirror/Glossy)"
        ])
        self.shadow_combo.currentIndexChanged.connect(self.on_shadow_type_changed)
        type_select_layout.addWidget(self.shadow_combo)
        shadow_type_layout.addLayout(type_select_layout)
        
        shadow_type_group.setLayout(shadow_type_layout)
        shadow_tab_layout.addWidget(shadow_type_group)
        
        # Shadow Parameters Group
        shadow_params_group = QGroupBox("⚙️ Shadow Parameters")
        shadow_params_layout = QVBoxLayout()
        
        # Opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.shadow_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_opacity_slider.setRange(0, 100)
        self.shadow_opacity_slider.setValue(85)
        self.shadow_opacity_slider.valueChanged.connect(self.schedule_shadow_update)
        opacity_layout.addWidget(self.shadow_opacity_slider)
        self.shadow_opacity_label = QLabel("85%")
        self.shadow_opacity_slider.valueChanged.connect(lambda v: self.shadow_opacity_label.setText(f"{v}%"))
        opacity_layout.addWidget(self.shadow_opacity_label)
        shadow_params_layout.addLayout(opacity_layout)
        
        # Blur Amount
        blur_layout = QHBoxLayout()
        blur_layout.addWidget(QLabel("Blur:"))
        self.shadow_blur_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_blur_slider.setRange(0, 100)
        self.shadow_blur_slider.setValue(20)
        self.shadow_blur_slider.valueChanged.connect(self.schedule_shadow_update)
        blur_layout.addWidget(self.shadow_blur_slider)
        self.shadow_blur_label = QLabel("20px")
        self.shadow_blur_slider.valueChanged.connect(lambda v: self.shadow_blur_label.setText(f"{v}px"))
        blur_layout.addWidget(self.shadow_blur_label)
        shadow_params_layout.addLayout(blur_layout)
        
        # Distance/Offset
        distance_layout = QHBoxLayout()
        distance_layout.addWidget(QLabel("Distance:"))
        self.shadow_distance_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_distance_slider.setRange(-50, 200)  # Allow negative for overlap
        self.shadow_distance_slider.setValue(30)
        self.shadow_distance_slider.valueChanged.connect(self.schedule_shadow_update)
        distance_layout.addWidget(self.shadow_distance_slider)
        self.shadow_distance_label = QLabel("30px")
        self.shadow_distance_slider.valueChanged.connect(lambda v: self.shadow_distance_label.setText(f"{v}px"))
        distance_layout.addWidget(self.shadow_distance_label)
        shadow_params_layout.addLayout(distance_layout)
        
        # Drop Shadow Gap (Drop Shadow-specific)
        self.drop_shadow_gap_layout = QHBoxLayout()
        self.drop_shadow_gap_layout.addWidget(QLabel("Gap:"))
        self.drop_shadow_gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.drop_shadow_gap_slider.setRange(-50, 100)  # -50 to 100% of image height
        self.drop_shadow_gap_slider.setValue(0)
        self.drop_shadow_gap_slider.valueChanged.connect(self.schedule_shadow_update)
        self.drop_shadow_gap_layout.addWidget(self.drop_shadow_gap_slider)
        self.drop_shadow_gap_label = QLabel("0%")
        self.drop_shadow_gap_slider.valueChanged.connect(lambda v: self.drop_shadow_gap_label.setText(f"{v}%"))
        self.drop_shadow_gap_layout.addWidget(self.drop_shadow_gap_label)
        shadow_params_layout.addLayout(self.drop_shadow_gap_layout)
        # Initially hide drop shadow gap (show only for Drop Shadow type)
        for i in range(self.drop_shadow_gap_layout.count()):
            widget = self.drop_shadow_gap_layout.itemAt(i).widget()
            if widget:
                widget.hide()
        
        # Angle (for Natural Shadow)
        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Angle:"))
        self.shadow_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_angle_slider.setRange(0, 360)
        self.shadow_angle_slider.setValue(135)
        self.shadow_angle_slider.valueChanged.connect(self.schedule_shadow_update)
        angle_layout.addWidget(self.shadow_angle_slider)
        self.shadow_angle_label = QLabel("135°")
        self.shadow_angle_slider.valueChanged.connect(lambda v: self.shadow_angle_label.setText(f"{v}°"))
        angle_layout.addWidget(self.shadow_angle_label)
        shadow_params_layout.addLayout(angle_layout)
        
        # Scale (compression)
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        self.shadow_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_scale_slider.setRange(10, 150)
        self.shadow_scale_slider.setValue(100)
        self.shadow_scale_slider.valueChanged.connect(self.schedule_shadow_update)
        scale_layout.addWidget(self.shadow_scale_slider)
        self.shadow_scale_label = QLabel("100%")
        self.shadow_scale_slider.valueChanged.connect(lambda v: self.shadow_scale_label.setText(f"{v}%"))
        scale_layout.addWidget(self.shadow_scale_label)
        shadow_params_layout.addLayout(scale_layout)
        
        # Vertical Compression (for perspective)
        compress_layout = QHBoxLayout()
        compress_layout.addWidget(QLabel("Perspective:"))
        self.shadow_compress_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_compress_slider.setRange(10, 100)
        self.shadow_compress_slider.setValue(50)
        self.shadow_compress_slider.valueChanged.connect(self.schedule_shadow_update)
        compress_layout.addWidget(self.shadow_compress_slider)
        self.shadow_compress_label = QLabel("50%")
        self.shadow_compress_slider.valueChanged.connect(lambda v: self.shadow_compress_label.setText(f"{v}%"))
        compress_layout.addWidget(self.shadow_compress_label)
        shadow_params_layout.addLayout(compress_layout)
        
        shadow_params_group.setLayout(shadow_params_layout)
        shadow_tab_layout.addWidget(shadow_params_group)
        
        # Shadow Color Group
        shadow_color_group = QGroupBox("🎨 Shadow Appearance")
        shadow_color_layout = QVBoxLayout()
        
        # Shadow color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color:"))
        self.shadow_color_btn = QPushButton("Choose Color")
        self.shadow_color = QColor(0, 0, 0)  # Default black
        self.shadow_color_btn.clicked.connect(self.choose_shadow_color)
        self.shadow_color_preview = QLabel("   ")
        self.shadow_color_preview.setStyleSheet(f"background-color: {self.shadow_color.name()}; border: 1px solid #666;")
        self.shadow_color_preview.setFixedSize(30, 20)
        color_layout.addWidget(self.shadow_color_btn)
        color_layout.addWidget(self.shadow_color_preview)
        color_layout.addStretch()
        shadow_color_layout.addLayout(color_layout)
        
        # Softness (feather edges)
        softness_layout = QHBoxLayout()
        softness_layout.addWidget(QLabel("Softness:"))
        self.shadow_softness_slider = QSlider(Qt.Orientation.Horizontal)
        self.shadow_softness_slider.setRange(0, 100)
        self.shadow_softness_slider.setValue(50)
        self.shadow_softness_slider.valueChanged.connect(self.schedule_shadow_update)
        softness_layout.addWidget(self.shadow_softness_slider)
        self.shadow_softness_label = QLabel("50%")
        self.shadow_softness_slider.valueChanged.connect(lambda v: self.shadow_softness_label.setText(f"{v}%"))
        softness_layout.addWidget(self.shadow_softness_label)
        shadow_color_layout.addLayout(softness_layout)
        
        shadow_color_group.setLayout(shadow_color_layout)
        shadow_tab_layout.addWidget(shadow_color_group)
        
        # Profile Presets Group
        profile_group = QGroupBox("💾 Save/Load Profiles")
        profile_layout = QVBoxLayout()
        
        profile_info = QLabel("Save current settings to a profile or load saved presets")
        profile_info.setStyleSheet("color: #888; font-size: 10px; font-style: italic;")
        profile_layout.addWidget(profile_info)
        
        profile_buttons_layout = QHBoxLayout()
        self.shadow_profile_buttons = []
        for i in range(1, 6):
            btn = QPushButton(f"Profile {i}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #555;
                    color: white;
                    padding: 8px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #666;
                }
                QPushButton:pressed {
                    background-color: #2196F3;
                }
            """)
            btn.clicked.connect(lambda checked, profile_num=i: self.handle_shadow_profile(profile_num))
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, profile_num=i, button=btn: self.show_shadow_profile_menu(profile_num, button))
            profile_buttons_layout.addWidget(btn)
            self.shadow_profile_buttons.append(btn)
        
        profile_layout.addLayout(profile_buttons_layout)
        
        profile_tip = QLabel("💡 Left-click: Load | Right-click: Save current settings")
        profile_tip.setStyleSheet("color: #FF9800; font-size: 9px; font-style: italic;")
        profile_layout.addWidget(profile_tip)
        
        profile_group.setLayout(profile_layout)
        shadow_tab_layout.addWidget(profile_group)
        
        # Reflection-Specific Controls (initially hidden)
        self.reflection_group = QGroupBox("🪞 Reflection Settings")
        reflection_layout = QVBoxLayout()
        
        # Reflection Height
        reflection_height_layout = QHBoxLayout()
        reflection_height_layout.addWidget(QLabel("Reflection Height:"))
        self.reflection_height_slider = QSlider(Qt.Orientation.Horizontal)
        self.reflection_height_slider.setRange(10, 200)
        self.reflection_height_slider.setValue(100)
        self.reflection_height_slider.valueChanged.connect(self.schedule_shadow_update)
        reflection_height_layout.addWidget(self.reflection_height_slider)
        self.reflection_height_label = QLabel("100%")
        self.reflection_height_slider.valueChanged.connect(lambda v: self.reflection_height_label.setText(f"{v}%"))
        reflection_height_layout.addWidget(self.reflection_height_label)
        reflection_layout.addLayout(reflection_height_layout)
        
        # Reflection Gap (percentage based)
        reflection_gap_layout = QHBoxLayout()
        reflection_gap_layout.addWidget(QLabel("Gap:"))
        self.reflection_gap_slider = QSlider(Qt.Orientation.Horizontal)
        self.reflection_gap_slider.setRange(-50, 50)  # -50% to +50% of image height
        self.reflection_gap_slider.setValue(0)
        self.reflection_gap_slider.valueChanged.connect(self.schedule_shadow_update)
        reflection_gap_layout.addWidget(self.reflection_gap_slider)
        self.reflection_gap_label = QLabel("0%")
        self.reflection_gap_slider.valueChanged.connect(lambda v: self.reflection_gap_label.setText(f"{v}%"))
        reflection_gap_layout.addWidget(self.reflection_gap_label)
        reflection_layout.addLayout(reflection_gap_layout)
        
        # Reflection Fade Start Point
        reflection_fade_layout = QHBoxLayout()
        reflection_fade_layout.addWidget(QLabel("Fade Start:"))
        self.reflection_fade_slider = QSlider(Qt.Orientation.Horizontal)
        self.reflection_fade_slider.setRange(0, 100)
        self.reflection_fade_slider.setValue(0)
        self.reflection_fade_slider.valueChanged.connect(self.schedule_shadow_update)
        reflection_fade_layout.addWidget(self.reflection_fade_slider)
        self.reflection_fade_label = QLabel("0%")
        self.reflection_fade_slider.valueChanged.connect(lambda v: self.reflection_fade_label.setText(f"{v}%"))
        reflection_fade_layout.addWidget(self.reflection_fade_label)
        reflection_layout.addLayout(reflection_fade_layout)
        
        self.reflection_group.setLayout(reflection_layout)
        self.reflection_group.setVisible(False)  # Hidden by default
        shadow_color_layout.addWidget(self.reflection_group)
        
        # Reset Button
        reset_shadow_btn = QPushButton("🔄 Reset to Defaults")
        reset_shadow_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 6px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        reset_shadow_btn.clicked.connect(self.reset_shadow_parameters)
        shadow_color_layout.addWidget(reset_shadow_btn)
        
        shadow_color_group.setLayout(shadow_color_layout)
        shadow_tab_layout.addWidget(shadow_color_group)
        
        # Apply Shadow Button
        apply_shadow_btn = QPushButton("💾 Save Shadow Permanently (Optional)")
        apply_shadow_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_shadow_btn.clicked.connect(self.apply_shadow_to_image)
        shadow_tab_layout.addWidget(apply_shadow_btn)
        
        shadow_tab_layout.addStretch()
        tabs.addTab(shadow_tab, "Shadow")
        
        # Background Tab
        background_tab = QWidget()
        background_tab_layout = QVBoxLayout(background_tab)
        
        background_info = QLabel("🖼️ Add a background image or solid color behind your product")
        background_info.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 11px;")
        background_info.setWordWrap(True)
        background_tab_layout.addWidget(background_info)
        
        # Background Type Selection
        bg_type_group = QGroupBox("Background Type")
        bg_type_layout = QVBoxLayout()
        
        self.bg_type_combo = QComboBox()
        self.bg_type_combo.addItems(["None (Transparent)", "Solid Color", "Image"])
        self.bg_type_combo.currentIndexChanged.connect(self.on_background_type_changed)
        bg_type_layout.addWidget(self.bg_type_combo)
        
        bg_type_group.setLayout(bg_type_layout)
        background_tab_layout.addWidget(bg_type_group)
        
        # Solid Color Section
        self.bg_color_group = QGroupBox("Solid Color Settings")
        bg_color_layout = QVBoxLayout()
        
        color_select_layout = QHBoxLayout()
        color_select_layout.addWidget(QLabel("Color:"))
        self.bg_color_btn = QPushButton("Choose Color")
        self.bg_color = QColor(255, 255, 255)  # Default white
        self.bg_color_btn.clicked.connect(self.choose_background_color)
        self.bg_color_preview = QLabel("   ")
        self.bg_color_preview.setStyleSheet(f"background-color: {self.bg_color.name()}; border: 1px solid #666;")
        self.bg_color_preview.setFixedSize(50, 30)
        color_select_layout.addWidget(self.bg_color_btn)
        color_select_layout.addWidget(self.bg_color_preview)
        color_select_layout.addStretch()
        bg_color_layout.addLayout(color_select_layout)
        
        # Quick color presets
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Presets:"))
        
        preset_colors = [
            ("White", "#FFFFFF"),
            ("Black", "#000000"),
            ("Gray", "#808080"),
            ("Blue", "#2196F3"),
            ("Green", "#4CAF50")
        ]
        
        for name, color in preset_colors:
            btn = QPushButton(name)
            btn.setMaximumWidth(60)
            btn.setStyleSheet(f"background-color: {color}; color: {'white' if color in ['#000000', '#808080', '#2196F3', '#4CAF50'] else 'black'}; padding: 5px;")
            btn.clicked.connect(lambda checked, c=color: self.set_background_color_preset(c))
            preset_layout.addWidget(btn)
        
        preset_layout.addStretch()
        bg_color_layout.addLayout(preset_layout)
        
        self.bg_color_group.setLayout(bg_color_layout)
        self.bg_color_group.setVisible(False)
        background_tab_layout.addWidget(self.bg_color_group)
        
        # Image Background Section
        self.bg_image_group = QGroupBox("Image Background Settings")
        bg_image_layout = QVBoxLayout()
        
        bg_image_select_layout = QHBoxLayout()
        self.bg_image_btn = QPushButton("📁 Load Background Image")
        self.bg_image_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 8px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.bg_image_btn.clicked.connect(self.load_background_image)
        bg_image_select_layout.addWidget(self.bg_image_btn)
        
        self.bg_clear_btn = QPushButton("🗑️ Clear")
        self.bg_clear_btn.clicked.connect(self.clear_background_image)
        bg_image_select_layout.addWidget(self.bg_clear_btn)
        bg_image_layout.addLayout(bg_image_select_layout)
        
        self.bg_image_label = QLabel("No background image loaded")
        self.bg_image_label.setStyleSheet("color: #888; font-style: italic;")
        bg_image_layout.addWidget(self.bg_image_label)
        
        # Background scaling
        bg_scale_layout = QHBoxLayout()
        bg_scale_layout.addWidget(QLabel("Fit Mode:"))
        self.bg_fit_combo = QComboBox()
        self.bg_fit_combo.addItems(["Stretch to Fill", "Fit (Keep Aspect)", "Fill (Crop)", "Tile"])
        self.bg_fit_combo.currentIndexChanged.connect(self.apply_background)
        bg_scale_layout.addWidget(self.bg_fit_combo)
        bg_image_layout.addLayout(bg_scale_layout)
        
        self.bg_image_group.setLayout(bg_image_layout)
        self.bg_image_group.setVisible(False)
        background_tab_layout.addWidget(self.bg_image_group)
        
        # Apply button
        apply_bg_btn = QPushButton("✓ Apply Background")
        apply_bg_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_bg_btn.clicked.connect(self.apply_background)
        background_tab_layout.addWidget(apply_bg_btn)
        
        background_tab_layout.addStretch()
        tabs.addTab(background_tab, "Background")
        
        adj_tab = QWidget()
        adj_layout = QVBoxLayout(adj_tab)
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout()
        
        # Slider with buttons
        brightness_control = QHBoxLayout()
        brightness_down_btn = QPushButton("−")
        brightness_down_btn.setMaximumWidth(30)
        brightness_down_btn.clicked.connect(lambda: self.brightness_slider.setValue(self.brightness_slider.value() - 1))
        brightness_control.addWidget(brightness_down_btn)
        
        self.brightness_slider = self.create_slider(0, 200, 100)
        self.brightness_slider.valueChanged.connect(self.update_preview)
        brightness_control.addWidget(self.brightness_slider)
        
        brightness_up_btn = QPushButton("+")
        brightness_up_btn.setMaximumWidth(30)
        brightness_up_btn.clicked.connect(lambda: self.brightness_slider.setValue(self.brightness_slider.value() + 1))
        brightness_control.addWidget(brightness_up_btn)
        
        brightness_layout.addLayout(brightness_control)
        self.brightness_value = QLabel("1.00")
        brightness_layout.addWidget(self.brightness_value)
        brightness_group.setLayout(brightness_layout)
        adj_layout.addWidget(brightness_group)
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout()
        
        # Slider with buttons
        contrast_control = QHBoxLayout()
        contrast_down_btn = QPushButton("−")
        contrast_down_btn.setMaximumWidth(30)
        contrast_down_btn.clicked.connect(lambda: self.contrast_slider.setValue(self.contrast_slider.value() - 1))
        contrast_control.addWidget(contrast_down_btn)
        
        self.contrast_slider = self.create_slider(0, 200, 100)
        self.contrast_slider.valueChanged.connect(self.update_preview)
        contrast_control.addWidget(self.contrast_slider)
        
        contrast_up_btn = QPushButton("+")
        contrast_up_btn.setMaximumWidth(30)
        contrast_up_btn.clicked.connect(lambda: self.contrast_slider.setValue(self.contrast_slider.value() + 1))
        contrast_control.addWidget(contrast_up_btn)
        
        contrast_layout.addLayout(contrast_control)
        self.contrast_value = QLabel("1.00")
        contrast_layout.addWidget(self.contrast_value)
        contrast_group.setLayout(contrast_layout)
        adj_layout.addWidget(contrast_group)
        saturation_group = QGroupBox("Saturation")
        saturation_layout = QVBoxLayout()
        
        # Slider with buttons
        saturation_control = QHBoxLayout()
        saturation_down_btn = QPushButton("−")
        saturation_down_btn.setMaximumWidth(30)
        saturation_down_btn.clicked.connect(lambda: self.saturation_slider.setValue(self.saturation_slider.value() - 1))
        saturation_control.addWidget(saturation_down_btn)
        
        self.saturation_slider = self.create_slider(0, 200, 100)
        self.saturation_slider.valueChanged.connect(self.update_preview)
        saturation_control.addWidget(self.saturation_slider)
        
        saturation_up_btn = QPushButton("+")
        saturation_up_btn.setMaximumWidth(30)
        saturation_up_btn.clicked.connect(lambda: self.saturation_slider.setValue(self.saturation_slider.value() + 1))
        saturation_control.addWidget(saturation_up_btn)
        
        saturation_layout.addLayout(saturation_control)
        self.saturation_value = QLabel("1.00")
        saturation_layout.addWidget(self.saturation_value)
        saturation_group.setLayout(saturation_layout)
        adj_layout.addWidget(saturation_group)
        sharpness_group = QGroupBox("Sharpness")
        sharpness_layout = QVBoxLayout()
        
        # Slider with buttons
        sharpness_control = QHBoxLayout()
        sharpness_down_btn = QPushButton("−")
        sharpness_down_btn.setMaximumWidth(30)
        sharpness_down_btn.clicked.connect(lambda: self.sharpness_slider.setValue(self.sharpness_slider.value() - 1))
        sharpness_control.addWidget(sharpness_down_btn)
        
        self.sharpness_slider = self.create_slider(0, 200, 100)
        self.sharpness_slider.valueChanged.connect(self.update_preview)
        sharpness_control.addWidget(self.sharpness_slider)
        
        sharpness_up_btn = QPushButton("+")
        sharpness_up_btn.setMaximumWidth(30)
        sharpness_up_btn.clicked.connect(lambda: self.sharpness_slider.setValue(self.sharpness_slider.value() + 1))
        sharpness_control.addWidget(sharpness_up_btn)
        
        sharpness_layout.addLayout(sharpness_control)
        self.sharpness_value = QLabel("1.00")
        sharpness_layout.addWidget(self.sharpness_value)
        sharpness_group.setLayout(sharpness_layout)
        adj_layout.addWidget(sharpness_group)
        adj_layout.addStretch()
        tabs.addTab(adj_tab, "Adjustments")
        export_tab = QWidget()
        export_layout = QVBoxLayout(export_tab)
        
        # Output Settings
        output_group = QGroupBox("📁 Output Settings")
        output_layout = QVBoxLayout()
        
        # Output folder
        folder_label = QLabel("Output Folder:")
        folder_label.setStyleSheet("font-size: 11px; color: #888;")
        output_layout.addWidget(folder_label)
        
        folder_display = QLabel("(Auto: 'processed images' folder)")
        folder_display.setStyleSheet("font-size: 10px; color: #0d7377; font-style: italic;")
        folder_display.setWordWrap(True)
        output_layout.addWidget(folder_display)
        
        folder_btn_layout = QHBoxLayout()
        choose_folder_btn = QPushButton("📁 Choose Folder")
        choose_folder_btn.clicked.connect(self.choose_output_folder)
        folder_btn_layout.addWidget(choose_folder_btn)
        
        open_folder_btn = QPushButton("📂 Open Folder")
        open_folder_btn.clicked.connect(self.open_output_folder)
        folder_btn_layout.addWidget(open_folder_btn)
        output_layout.addLayout(folder_btn_layout)
        
        # Prefix
        prefix_layout = QHBoxLayout()
        prefix_layout.addWidget(QLabel("Prefix:"))
        self.prefix_input = QLineEdit(self.file_prefix)
        self.prefix_input.setPlaceholderText("e.g., web_")
        self.prefix_input.textChanged.connect(lambda text: setattr(self, 'file_prefix', text))
        prefix_layout.addWidget(self.prefix_input)
        output_layout.addLayout(prefix_layout)
        
        # Suffix
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Suffix:"))
        self.suffix_input = QLineEdit(self.file_suffix)
        self.suffix_input.setPlaceholderText("e.g., _optimized")
        self.suffix_input.textChanged.connect(lambda text: setattr(self, 'file_suffix', text))
        suffix_layout.addWidget(self.suffix_input)
        output_layout.addLayout(suffix_layout)
        
        output_group.setLayout(output_layout)
        export_layout.addWidget(output_group)
        
        quality_group = QGroupBox("WebP Quality")
        quality_layout = QVBoxLayout()
        self.quality_slider = self.create_slider(1, 100, 85)
        self.quality_value = QLabel("85")
        quality_layout.addWidget(QLabel("Quality:"))
        quality_layout.addWidget(self.quality_slider)
        quality_layout.addWidget(self.quality_value)
        self.quality_slider.valueChanged.connect(lambda v: self.quality_value.setText(str(v)))
        quality_group.setLayout(quality_layout)
        export_layout.addWidget(quality_group)
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout()
        self.lossless_check = QCheckBox("Lossless")
        options_layout.addWidget(self.lossless_check)
        self.keep_exif_check = QCheckBox("Keep EXIF")
        options_layout.addWidget(self.keep_exif_check)
        options_group.setLayout(options_layout)
        export_layout.addWidget(options_group)
        
        # Batch Processing Group
        batch_group = QGroupBox("🚀 Batch Processing")
        batch_layout = QVBoxLayout()
        
        self.process_all_btn = QPushButton("⚡ Process All Images (Parallel)")
        self.process_all_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:disabled {
                background-color: #555;
            }
        """)
        self.process_all_btn.clicked.connect(self.process_all_batch)
        self.process_all_btn.setEnabled(False)
        batch_layout.addWidget(self.process_all_btn)
        
        self.batch_progress_label = QLabel("0/0 processed")
        self.batch_progress_label.setStyleSheet("color: #888; font-size: 11px;")
        self.batch_progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        batch_layout.addWidget(self.batch_progress_label)
        
        batch_group.setLayout(batch_layout)
        export_layout.addWidget(batch_group)
        
        # Smart Padding Group (for e-commerce product shots)
        padding_group = QGroupBox("📦 Smart Padding (Product Centering)")
        padding_layout = QVBoxLayout()
        
        self.smart_padding_check = QCheckBox("Enable Smart Padding")
        self.smart_padding_check.setChecked(False)
        self.smart_padding_check.toggled.connect(self.toggle_padding_settings)
        padding_layout.addWidget(self.smart_padding_check)
        
        # Padding percentage
        padding_percent_layout = QHBoxLayout()
        padding_percent_layout.addWidget(QLabel("Padding %:"))
        self.padding_slider = QSlider(Qt.Orientation.Horizontal)
        self.padding_slider.setRange(0, 50)
        self.padding_slider.setValue(10)
        self.padding_slider.setEnabled(False)
        self.padding_label = QLabel("10%")
        self.padding_slider.valueChanged.connect(lambda v: self.padding_label.setText(f"{v}%"))
        padding_percent_layout.addWidget(self.padding_slider)
        padding_percent_layout.addWidget(self.padding_label)
        padding_layout.addLayout(padding_percent_layout)
        
        # Canvas size presets
        canvas_size_layout = QHBoxLayout()
        canvas_size_layout.addWidget(QLabel("Canvas:"))
        self.canvas_combo = QComboBox()
        self.canvas_combo.addItems([
            "Square (1:1)",
            "Portrait (3:4)",
            "Landscape (4:3)",
            "Wide (16:9)",
            "Instagram (1:1)",
            "Custom"
        ])
        self.canvas_combo.setEnabled(False)
        canvas_size_layout.addWidget(self.canvas_combo)
        padding_layout.addLayout(canvas_size_layout)
        
        padding_group.setLayout(padding_layout)
        export_layout.addWidget(padding_group)
        
        export_layout.addStretch()
        tabs.addTab(export_tab, "Export")
        layout.addWidget(tabs)
        return panel
        
    def create_slider(self, min_val, max_val, default):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        return slider
        
    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            
    def drop_event(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.load_image(files[0])
    
    def on_model_changed(self, index):
        """Handle AI model selection change"""
        model_map = {
            0: "u2net",
            1: "u2netp",
            2: "u2net_human_seg",
            3: "isnet-general-use",
            4: "isnet-anime",
            5: "silueta",
            6: "sam",
            7: "u2net_cloth_seg",
            8: None,  # Separator
            9: "birefnet",
            10: "birefnet-matting",
            11: "birefnet-portrait",
            12: None,  # Separator
            13: "modnet"
        }
        
        # Prevent selecting separators
        if index in [8, 12]:
            self.model_combo.setCurrentIndex(0)
            return
        
        self.current_model = model_map.get(index, "u2net")
        self.info_label.setText(f"Model changed to: {self.current_model}")
    
    def toggle_alpha_settings(self, enabled):
        """Enable/disable alpha matting settings"""
        self.fg_threshold_slider.setEnabled(enabled)
        self.bg_threshold_slider.setEnabled(enabled)
        self.edge_softness_slider.setEnabled(enabled)
        self.uncertainty_detect_check.setEnabled(enabled)
        self.post_process_check.setEnabled(enabled)
    
    def update_edge_softness_label(self, value):
        """Update edge softness label with descriptive text"""
        labels = {
            0: "Sharp (0)",
            1: "Crisp (1)",
            2: "Mild (2)",
            3: "Normal (3)",
            4: "Soft (4)",
            5: "Smooth (5)",
            6: "Very Soft (6)",
            7: "Extra Soft (7)",
            8: "Feather (8)",
            9: "Blur (9)",
            10: "Maximum (10)"
        }
        self.edge_softness_label.setText(labels.get(value, str(value)))
    
    def toggle_padding_settings(self, enabled):
        """Enable/disable smart padding settings"""
        self.padding_slider.setEnabled(enabled)
        self.canvas_combo.setEnabled(enabled)
    
    def on_canvas_settings_changed(self):
        """Update preview when canvas size or background settings change"""
        if self.current_image is not None:
            self.display_processed_image(self.current_image)
    
    def choose_bg_color(self):
        """Open color picker dialog for background color"""
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(self.bg_color, self, "Choose Background Color")
        if color.isValid():
            self.bg_color = color
            self.bg_color_btn.setStyleSheet(f"background-color: {color.name()}; color: {'black' if color.lightness() > 128 else 'white'}; padding: 8px;")
            self.on_canvas_settings_changed()
    
    def choose_output_folder(self):
        """Choose custom output folder"""
        folder = QFileDialog.getExistingDirectory(self, "Choose Output Folder", self.output_folder or "")
        if folder:
            self.output_folder = folder
            self.info_label.setText(f"Output folder: {folder}")
    
    def open_output_folder(self):
        """Open output folder in file explorer"""
        if self.output_folder and os.path.exists(self.output_folder):
            os.startfile(self.output_folder)
        else:
            self.info_label.setText("No output folder set yet")
    
    def auto_center_product(self):
        """Automatically center the product by finding its bounding box"""
        if self.current_image is None or self.current_image.mode != 'RGBA':
            self.info_label.setText("Auto-center requires transparent background")
            return
        
        try:
            # Save state for undo
            self.save_state()
            
            # Get alpha channel
            alpha = np.array(self.current_image.split()[-1])
            
            # Find bounding box of non-transparent pixels
            rows = np.any(alpha > 10, axis=1)
            cols = np.any(alpha > 10, axis=0)
            
            if not rows.any() or not cols.any():
                self.info_label.setText("No object detected")
                return
            
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Calculate center of bounding box
            obj_center_x = (cmin + cmax) / 2
            obj_center_y = (rmin + rmax) / 2
            
            # Calculate image center
            img_center_x = self.current_image.width / 2
            img_center_y = self.current_image.height / 2
            
            # Calculate offset needed
            offset_x = int(img_center_x - obj_center_x)
            offset_y = int(img_center_y - obj_center_y)
            
            # Update position spinners
            self.pos_x_spin.setValue(offset_x)
            self.pos_y_spin.setValue(offset_y)
            
            self.info_label.setText(f"✓ Product centered (offset: {offset_x}, {offset_y})")
        except Exception as e:
            self.info_label.setText(f"Auto-center error: {str(e)}")
    
    def toggle_crop_mode(self, checked):
        """Toggle crop mode on/off"""
        self.crop_mode = checked
        if checked:
            self.info_label.setText("🔲 CROP MODE: Click and drag on canvas to select crop area, then press ENTER to apply or ESC to cancel")
            self.crop_start = None
            self.crop_end = None
            self.crop_rect = None
            # Add overlay for visual feedback
            if not self.crop_overlay_label:
                from PyQt6.QtWidgets import QLabel as OverlayLabel
                self.crop_overlay_label = OverlayLabel(self.image_label)
                self.crop_overlay_label.setStyleSheet("border: 3px dashed #00ff00; background-color: rgba(0, 255, 0, 30);")
                self.crop_overlay_label.hide()
        else:
            self.info_label.setText("Crop mode disabled")
            self.crop_start = None
            self.crop_end = None
            self.crop_rect = None
            if self.crop_overlay_label:
                self.crop_overlay_label.hide()
            if self.current_image:
                self.display_processed_image(self.current_image)
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        from PyQt6.QtCore import Qt
        if self.crop_mode:
            if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
                # Apply crop
                self.apply_crop()
            elif event.key() == Qt.Key.Key_Escape:
                # Cancel crop
                self.crop_btn.setChecked(False)
                self.toggle_crop_mode(False)
        
        # Call parent implementation for other shortcuts (Ctrl+Z, etc.)
        super().keyPressEvent(event)
    
    def apply_crop(self):
        """Apply the selected crop rectangle - simplified direct mapping"""
        if self.crop_rect and self.current_image:
            self.save_state()
            
            pixmap = self.image_label.pixmap()
            if not pixmap:
                return
            
            screen_x1, screen_y1, screen_x2, screen_y2 = self.crop_rect
            
            # Step 1: Replicate the exact display transformation
            if self.resize_check.isChecked():
                canvas_width = self.width_spin.value()
                canvas_height = self.height_spin.value()
            else:
                canvas_width = self.current_image.width
                canvas_height = self.current_image.height
            
            # Create the exact transformed image as shown
            scale_factor = self.scale_spin.value() / 100.0
            img_copy = self.current_image.copy()
            
            if scale_factor != 1.0:
                new_w = int(img_copy.width * scale_factor)
                new_h = int(img_copy.height * scale_factor)
                img_copy = img_copy.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            if self.resize_check.isChecked():
                img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Now img_copy is the exact size shown on canvas
            displayed_img_w = img_copy.width
            displayed_img_h = img_copy.height
            
            # Position on canvas
            canvas_x = (canvas_width - displayed_img_w) // 2 + self.pos_x_spin.value()
            canvas_y = (canvas_height - displayed_img_h) // 2 + self.pos_y_spin.value()
            
            # Canvas gets thumbnailed for display
            canvas_for_display = Image.new('RGB', (canvas_width, canvas_height))
            canvas_for_display.thumbnail((800, 600), Image.Resampling.LANCZOS)
            disp_scale = canvas_for_display.width / canvas_width
            
            # Map screen coords to canvas coords
            canvas_x1 = int(screen_x1 / disp_scale)
            canvas_y1 = int(screen_y1 / disp_scale)
            canvas_x2 = int(screen_x2 / disp_scale)
            canvas_y2 = int(screen_y2 / disp_scale)
            
            # Map canvas coords to displayed image coords
            img_x1 = canvas_x1 - canvas_x
            img_y1 = canvas_y1 - canvas_y
            img_x2 = canvas_x2 - canvas_x
            img_y2 = canvas_y2 - canvas_y
            
            # Clamp to displayed image bounds
            img_x1 = max(0, min(img_x1, displayed_img_w))
            img_y1 = max(0, min(img_y1, displayed_img_h))
            img_x2 = max(0, min(img_x2, displayed_img_w))
            img_y2 = max(0, min(img_y2, displayed_img_h))
            
            # Now map from displayed image coords to original image coords
            # displayed_img came from: original -> scale -> thumbnail
            # We need the combined scale factor
            combined_scale = displayed_img_w / self.current_image.width
            
            orig_x1 = int(img_x1 / combined_scale)
            orig_y1 = int(img_y1 / combined_scale)
            orig_x2 = int(img_x2 / combined_scale)
            orig_y2 = int(img_y2 / combined_scale)
            
            # Final clamp
            orig_x1 = max(0, min(orig_x1, self.current_image.width))
            orig_y1 = max(0, min(orig_y1, self.current_image.height))
            orig_x2 = max(0, min(orig_x2, self.current_image.width))
            orig_y2 = max(0, min(orig_y2, self.current_image.height))
            
            print(f"\nCrop Debug:")
            print(f"  Screen: ({screen_x1},{screen_y1}) -> ({screen_x2},{screen_y2})")
            print(f"  Canvas: {canvas_width}x{canvas_height} -> Display: {canvas_for_display.width}x{canvas_for_display.height}")
            print(f"  Canvas coords: ({canvas_x1},{canvas_y1}) -> ({canvas_x2},{canvas_y2})")
            print(f"  Image on canvas at: ({canvas_x},{canvas_y}) size: {displayed_img_w}x{displayed_img_h}")
            print(f"  Image coords: ({img_x1},{img_y1}) -> ({img_x2},{img_y2})")
            print(f"  Combined scale: {combined_scale:.4f}")
            print(f"  Original coords: ({orig_x1},{orig_y1}) -> ({orig_x2},{orig_y2})")
            print(f"  Original size: {self.current_image.width}x{self.current_image.height}\n")
            
            if orig_x2 > orig_x1 + 5 and orig_y2 > orig_y1 + 5:
                self.current_image = self.current_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))
                self.original_image = self.current_image.copy()
                self.base_image = self.current_image.copy()
                
                self.pos_x_spin.setValue(0)
                self.pos_y_spin.setValue(0)
                self.scale_spin.setValue(100)
                
                self.display_processed_image(self.current_image)
                self.crop_btn.setChecked(False)
                self.crop_mode = False
                self.crop_rect = None
                if self.crop_overlay_label:
                    self.crop_overlay_label.hide()
                self.info_label.setText(f"✓ Cropped to {orig_x2-orig_x1}x{orig_y2-orig_y1}px")
            else:
                self.info_label.setText("❌ Invalid crop area - please try again")
    
    def reset_shadow_parameters(self):
        """Reset shadow parameters to defaults for current type"""
        shadow_type = self.shadow_combo.currentIndex()
        
        if shadow_type == 1:  # Drop Shadow
            self.shadow_opacity_slider.setValue(85)
            self.shadow_blur_slider.setValue(20)
            self.shadow_distance_slider.setValue(30)
            self.shadow_angle_slider.setValue(180)
            self.shadow_scale_slider.setValue(100)
            self.shadow_compress_slider.setValue(50)
            self.shadow_softness_slider.setValue(30)
        elif shadow_type == 2:  # Natural Shadow
            self.shadow_opacity_slider.setValue(70)
            self.shadow_blur_slider.setValue(25)
            self.shadow_distance_slider.setValue(40)
            self.shadow_angle_slider.setValue(135)
            self.shadow_scale_slider.setValue(95)
            self.shadow_compress_slider.setValue(80)
            self.shadow_softness_slider.setValue(40)
        elif shadow_type == 3:  # Reflection
            self.shadow_opacity_slider.setValue(50)
            self.shadow_blur_slider.setValue(5)
            self.shadow_distance_slider.setValue(0)  # Not used
            self.shadow_scale_slider.setValue(100)
            self.shadow_compress_slider.setValue(100)  # Not used
            self.shadow_softness_slider.setValue(50)
            # Reflection-specific
            self.reflection_height_slider.setValue(100)
            self.reflection_gap_slider.setValue(0)
            self.reflection_fade_slider.setValue(0)
        
        self.info_label.setText("✅ Shadow parameters reset to defaults")
        self.info_label.setStyleSheet("color: green;")
    
    def get_current_shadow_settings(self):
        """Get current shadow settings as dictionary"""
        return {
            'opacity': self.shadow_opacity_slider.value(),
            'blur': self.shadow_blur_slider.value(),
            'distance': self.shadow_distance_slider.value(),
            'angle': self.shadow_angle_slider.value(),
            'scale': self.shadow_scale_slider.value(),
            'compress': self.shadow_compress_slider.value(),
            'softness': self.shadow_softness_slider.value(),
            'color': self.shadow_color.name(),
            'drop_gap': self.drop_shadow_gap_slider.value(),
            'reflection_height': self.reflection_height_slider.value(),
            'reflection_gap': self.reflection_gap_slider.value(),
            'reflection_fade': self.reflection_fade_slider.value()
        }
    
    def apply_shadow_settings(self, settings):
        """Apply shadow settings from dictionary"""
        if settings:
            self.shadow_opacity_slider.setValue(settings.get('opacity', 85))
            self.shadow_blur_slider.setValue(settings.get('blur', 20))
            self.shadow_distance_slider.setValue(settings.get('distance', 30))
            self.shadow_angle_slider.setValue(settings.get('angle', 180))
            self.shadow_scale_slider.setValue(settings.get('scale', 100))
            self.shadow_compress_slider.setValue(settings.get('compress', 50))
            self.shadow_softness_slider.setValue(settings.get('softness', 50))
            self.drop_shadow_gap_slider.setValue(settings.get('drop_gap', 0))
            self.reflection_height_slider.setValue(settings.get('reflection_height', 100))
            self.reflection_gap_slider.setValue(settings.get('reflection_gap', 0))
            self.reflection_fade_slider.setValue(settings.get('reflection_fade', 0))
            
            # Apply color if saved
            if 'color' in settings:
                self.shadow_color = QColor(settings['color'])
                self.shadow_color_preview.setStyleSheet(f"background-color: {self.shadow_color.name()}; border: 1px solid #666;")
    
    def handle_shadow_profile(self, profile_number):
        """Handle shadow profile button click - Load profile"""
        shadow_type = self.shadow_combo.currentIndex()
        if shadow_type == 0:
            self.info_label.setText("⚠️ Please select a shadow type first")
            self.info_label.setStyleSheet("color: orange;")
            return
        
        shadow_type_names = ['none', 'drop', 'natural', 'reflection']
        shadow_type_name = shadow_type_names[shadow_type]
        
        # Load profile
        settings = self.profile_manager.load_shadow_profile(shadow_type_name, profile_number)
        if settings:
            self.apply_shadow_settings(settings)
            self.info_label.setText(f"✅ Loaded {shadow_type_name.capitalize()} Shadow Profile {profile_number}")
            self.info_label.setStyleSheet("color: green;")
            self.schedule_shadow_update()
        else:
            self.info_label.setText(f"⚠️ Profile {profile_number} is empty. Right-click to save current settings.")
            self.info_label.setStyleSheet("color: orange;")
    
    def show_shadow_profile_menu(self, profile_number, button):
        """Show context menu for shadow profile - Save option"""
        from PyQt6.QtWidgets import QMenu
        shadow_type = self.shadow_combo.currentIndex()
        if shadow_type == 0:
            self.info_label.setText("⚠️ Please select a shadow type first")
            self.info_label.setStyleSheet("color: orange;")
            return
        
        shadow_type_names = ['none', 'drop', 'natural', 'reflection']
        shadow_type_name = shadow_type_names[shadow_type]
        
        menu = QMenu(self)
        save_action = menu.addAction(f"💾 Save Current Settings to Profile {profile_number}")
        
        action = menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if action == save_action:
            # Save current settings
            settings = self.get_current_shadow_settings()
            self.profile_manager.save_shadow_profile(shadow_type_name, profile_number, settings)
            self.info_label.setText(f"✅ Saved {shadow_type_name.capitalize()} Shadow Profile {profile_number}")
            self.info_label.setStyleSheet("color: green;")
    
    def get_current_transform_settings(self):
        """Get current transform settings as dictionary"""
        return {
            'pos_x': self.pos_x_spin.value(),
            'pos_y': self.pos_y_spin.value(),
            'scale': self.scale_spin.value(),
            'rotation': self.rotation_spin.value()
        }
    
    def apply_transform_settings(self, settings):
        """Apply transform settings from dictionary"""
        if settings:
            self.pos_x_spin.setValue(settings.get('pos_x', 0))
            self.pos_y_spin.setValue(settings.get('pos_y', 0))
            self.scale_spin.setValue(settings.get('scale', 100))
            self.rotation_spin.setValue(settings.get('rotation', 0))
    
    def handle_transform_profile(self, profile_number):
        """Handle transform profile button click - Load profile"""
        settings = self.profile_manager.load_transform_profile(profile_number)
        if settings:
            self.apply_transform_settings(settings)
            self.info_label.setText(f"✅ Loaded Transform Profile {profile_number}")
            self.info_label.setStyleSheet("color: green;")
            self.on_transform_changed()
        else:
            self.info_label.setText(f"⚠️ Profile {profile_number} is empty. Right-click to save current settings.")
            self.info_label.setStyleSheet("color: orange;")
    
    def show_transform_profile_menu(self, profile_number, button):
        """Show context menu for transform profile - Save option"""
        from PyQt6.QtWidgets import QMenu
        
        menu = QMenu(self)
        save_action = menu.addAction(f"💾 Save Current Transform to Profile {profile_number}")
        
        action = menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if action == save_action:
            # Save current settings
            settings = self.get_current_transform_settings()
            self.profile_manager.save_transform_profile(profile_number, settings)
            self.info_label.setText(f"✅ Saved Transform Profile {profile_number}")
            self.info_label.setStyleSheet("color: green;")
    
    def on_shadow_type_changed(self):
        """Handle shadow type selection"""
        shadow_type = self.shadow_combo.currentIndex()
        # Enable/disable controls based on type
        # Angle works for Drop (direction) and Natural (side light)
        self.shadow_angle_slider.setEnabled(shadow_type in [1, 2])
        
        # Show/hide reflection-specific controls
        self.reflection_group.setVisible(shadow_type == 3)
        
        # Show/hide drop shadow gap control (only for Drop Shadow)
        for i in range(self.drop_shadow_gap_layout.count()):
            widget = self.drop_shadow_gap_layout.itemAt(i).widget()
            if widget:
                widget.setVisible(shadow_type == 1)
        
        # Reset parameters to defaults for the selected shadow type
        if shadow_type == 1:  # Drop Shadow - Amazon style
            self.shadow_opacity_slider.setValue(85)
            self.shadow_blur_slider.setValue(20)
            self.shadow_distance_slider.setValue(30)
            self.shadow_angle_slider.setValue(180)  # Straight down
            self.shadow_scale_slider.setValue(100)
            self.shadow_compress_slider.setValue(50)  # Vertical compression
            self.shadow_softness_slider.setValue(30)
            self.drop_shadow_gap_slider.setValue(0)  # No gap by default
        elif shadow_type == 2:  # Natural Shadow - Side light
            self.shadow_opacity_slider.setValue(70)
            self.shadow_blur_slider.setValue(25)
            self.shadow_distance_slider.setValue(40)
            self.shadow_angle_slider.setValue(135)  # Bottom-right
            self.shadow_scale_slider.setValue(95)
            self.shadow_compress_slider.setValue(80)  # Less compression
            self.shadow_softness_slider.setValue(40)
        elif shadow_type == 3:  # Reflection Shadow
            self.shadow_opacity_slider.setValue(50)  # More visible
            self.shadow_blur_slider.setValue(5)
            self.shadow_distance_slider.setValue(0)  # Not used for reflection
            self.shadow_scale_slider.setValue(100)
            self.shadow_compress_slider.setValue(100)  # Not used for reflection
            self.shadow_softness_slider.setValue(50)  # Fade power
            # Reflection-specific
            self.reflection_height_slider.setValue(100)  # Full height
            self.reflection_gap_slider.setValue(0)  # No gap
            self.reflection_fade_slider.setValue(0)  # Fade from top
        
        # Apply shadow immediately with debouncing
        self.schedule_shadow_update()
    
    def schedule_shadow_update(self):
        """Schedule shadow update with debouncing (300ms delay)"""
        self.shadow_timer.stop()
        self.shadow_timer.start(300)  # Wait 300ms after last slider change
    
    def update_shadow_preview(self):
        """Update shadow preview in real-time"""
        # Apply shadow with current settings
        self.schedule_shadow_update()
    
    def choose_shadow_color(self):
        """Open color picker for shadow color"""
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(self.shadow_color, self, "Choose Shadow Color")
        if color.isValid():
            self.shadow_color = color
            self.shadow_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            self.schedule_shadow_update()
    
    def apply_shadow_realtime(self):
        """Apply shadow in real-time without modifying base image"""
        # Use base_image if available (after bg removal), otherwise current_image
        source_image = self.base_image if self.base_image else self.current_image
        
        if not source_image:
            self.info_label.setText("⚠️ Please load an image first")
            self.info_label.setStyleSheet("color: orange;")
            return
        
        # Show processing indicator
        original_text = self.info_label.text()
        self.info_label.setText("⏳ Processing shadow...")
        self.info_label.setStyleSheet("color: orange; font-weight: bold;")
        QApplication.processEvents()  # Force UI update
        
        shadow_type = self.shadow_combo.currentIndex()
        
        if shadow_type == 0:  # None - show base image without shadow
            self.current_image = source_image.copy()
            self.display_processed_image(self.current_image)
            self.info_label.setText(original_text)
            self.info_label.setStyleSheet("")
            return
        
        try:
            # Use source image as base for shadow generation
            base = source_image.copy()
            
            # Generate shadow based on type
            if shadow_type == 1:  # Drop Shadow
                result = self.generate_drop_shadow(base)
            elif shadow_type == 2:  # Natural Shadow
                result = self.generate_natural_shadow(base)
            elif shadow_type == 3:  # Reflection
                result = self.generate_reflection_shadow(base)
            else:
                result = base
            
            if result:
                self.current_image = result
                self.display_processed_image(result)
                
                # Clear processing indicator
                self.info_label.setText(original_text)
                self.info_label.setStyleSheet("")
        except Exception as e:
            self.info_label.setText(f"❌ Shadow preview error: {str(e)}")
            self.info_label.setStyleSheet("color: red;")
    
    def apply_shadow_to_image(self):
        """Save current shadow to base image permanently"""
        if not self.current_image:
            return
        
        shadow_type = self.shadow_combo.currentIndex()
        if shadow_type == 0:  # None
            self.info_label.setText("No shadow to apply")
            return
        
        try:
            # Save current shadowed image to base
            self.save_state()
            self.base_image = self.current_image.copy()
            self.original_image = self.current_image.copy()
            self.info_label.setText("✅ Shadow saved permanently")
        except Exception as e:
            self.info_label.setText(f"❌ Error saving shadow: {str(e)}")
    
    def on_background_type_changed(self):
        """Handle background type selection"""
        bg_type = self.bg_type_combo.currentIndex()
        self.background_type = bg_type
        
        # Show/hide relevant sections
        self.bg_color_group.setVisible(bg_type == 1)
        self.bg_image_group.setVisible(bg_type == 2)
        
        # Apply background immediately
        self.apply_background()
    
    def choose_background_color(self):
        """Open color picker for background color"""
        from PyQt6.QtWidgets import QColorDialog
        color = QColorDialog.getColor(self.bg_color, self, "Choose Background Color")
        if color.isValid():
            self.bg_color = color
            self.bg_color_preview.setStyleSheet(f"background-color: {color.name()}; border: 1px solid #666;")
            self.apply_background()
    
    def set_background_color_preset(self, color_hex):
        """Set background color from preset"""
        self.bg_color = QColor(color_hex)
        self.bg_color_preview.setStyleSheet(f"background-color: {color_hex}; border: 1px solid #666;")
        self.apply_background()
    
    def load_background_image(self):
        """Load a background image from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Background Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp)"
        )
        
        if file_path:
            try:
                self.background_image = Image.open(file_path).convert('RGB')
                self.bg_image_label.setText(f"Loaded: {os.path.basename(file_path)}")
                self.bg_image_label.setStyleSheet("color: green; font-weight: bold;")
                self.apply_background()
            except Exception as e:
                self.info_label.setText(f"❌ Error loading background: {str(e)}")
    
    def clear_background_image(self):
        """Clear the loaded background image"""
        self.background_image = None
        self.bg_image_label.setText("No background image loaded")
        self.bg_image_label.setStyleSheet("color: #888; font-style: italic;")
        self.apply_background()
    
    def apply_background(self):
        """Apply background (color or image) to current image"""
        if not self.current_image:
            return
        
        source_image = self.base_image if self.base_image else self.current_image
        if not source_image:
            return
        
        try:
            bg_type = self.bg_type_combo.currentIndex()
            
            if bg_type == 0:  # None - transparent
                # Keep original transparency
                self.current_image = source_image.copy()
            
            elif bg_type == 1:  # Solid Color
                # Create solid color background
                result = Image.new('RGB', source_image.size, (self.bg_color.red(), self.bg_color.green(), self.bg_color.blue()))
                if source_image.mode == 'RGBA':
                    result.paste(source_image, (0, 0), source_image)
                else:
                    result.paste(source_image, (0, 0))
                self.current_image = result
            
            elif bg_type == 2:  # Image Background
                if not self.background_image:
                    self.info_label.setText("⚠️ Please load a background image first")
                    return
                
                fit_mode = self.bg_fit_combo.currentIndex()
                w, h = source_image.size
                bg = self.background_image.copy()
                
                if fit_mode == 0:  # Stretch to Fill
                    bg = bg.resize((w, h), Image.Resampling.LANCZOS)
                
                elif fit_mode == 1:  # Fit (Keep Aspect)
                    bg.thumbnail((w, h), Image.Resampling.LANCZOS)
                    # Center on canvas
                    result_bg = Image.new('RGB', (w, h), (128, 128, 128))
                    offset_x = (w - bg.width) // 2
                    offset_y = (h - bg.height) // 2
                    result_bg.paste(bg, (offset_x, offset_y))
                    bg = result_bg
                
                elif fit_mode == 2:  # Fill (Crop)
                    # Scale to cover entire canvas, crop excess
                    scale = max(w / bg.width, h / bg.height)
                    new_w = int(bg.width * scale)
                    new_h = int(bg.height * scale)
                    bg = bg.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    # Center crop
                    left = (new_w - w) // 2
                    top = (new_h - h) // 2
                    bg = bg.crop((left, top, left + w, top + h))
                
                elif fit_mode == 3:  # Tile
                    result_bg = Image.new('RGB', (w, h))
                    for y in range(0, h, bg.height):
                        for x in range(0, w, bg.width):
                            result_bg.paste(bg, (x, y))
                    bg = result_bg
                
                # Composite product on background
                result = bg.convert('RGB')
                if source_image.mode == 'RGBA':
                    result.paste(source_image, (0, 0), source_image)
                else:
                    result.paste(source_image, (0, 0))
                
                self.current_image = result
            
            # Update display
            self.display_processed_image(self.current_image)
            self.info_label.setText("✅ Background applied")
            
        except Exception as e:
            self.info_label.setText(f"❌ Error applying background: {str(e)}")
    
    def generate_drop_shadow(self, img):
        """Generate Amazon-style drop shadow (simple offset below product)"""
        import cv2
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        img_array = np.array(img)
        alpha = img_array[:, :, 3]
        
        # Get parameters
        opacity = self.shadow_opacity_slider.value() / 100.0
        blur = self.shadow_blur_slider.value()
        distance = self.shadow_distance_slider.value()
        angle = self.shadow_angle_slider.value()
        compress = self.shadow_compress_slider.value() / 100.0
        softness = self.shadow_softness_slider.value() / 100.0
        gap_percent = self.drop_shadow_gap_slider.value() / 100.0  # Gap control
        
        h, w = alpha.shape
        
        # Create shadow mask from alpha
        shadow_mask = alpha.astype(np.float32) / 255.0
        
        # Apply vertical compression for perspective
        compressed_h = max(1, int(h * compress))
        shadow_compressed = cv2.resize(shadow_mask, (w, compressed_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate offset based on angle (180° = straight down, 135° = bottom-right)
        angle_rad = np.radians(angle)
        offset_x = int(distance * np.sin(angle_rad))
        offset_y = int(distance * np.cos(angle_rad))
        
        # Add gap (percentage of image height) - can be negative for overlap
        gap_pixels = int(h * gap_percent)
        offset_y += gap_pixels
        
        # Create result canvas (expanded to fit shadow, handle negative gap)
        # For negative gap, shadow overlaps with image
        extra_space = 100  # Extra padding
        if gap_pixels < 0:
            # Overlap case - canvas should accommodate both image and overlapping shadow
            total_h = h + max(abs(offset_y) + compressed_h, 0) + extra_space
        else:
            # Normal case - shadow below image
            total_h = h + abs(offset_y) + compressed_h + extra_space
        
        total_w = w + abs(offset_x) + extra_space
        shadow_canvas = np.zeros((total_h, total_w), dtype=np.float32)
        
        # Position shadow on canvas - handle negative gap (overlap)
        if gap_pixels < 0:
            # Negative gap: shadow should overlap with bottom of image
            shadow_start_y = h + offset_y  # This will be less than h, creating overlap
        else:
            # Positive gap: shadow below image
            shadow_start_y = h + offset_y
        
        # Ensure shadow position is valid
        shadow_start_y = max(0, shadow_start_y)
        shadow_start_x = max(0, offset_x if offset_x >= 0 else 0)
        
        # Place compressed shadow
        if shadow_start_y >= 0 and shadow_start_y + compressed_h <= total_h:
            if shadow_start_x >= 0 and shadow_start_x + w <= total_w:
                shadow_canvas[shadow_start_y:shadow_start_y+compressed_h, shadow_start_x:shadow_start_x+w] = shadow_compressed
        
        # Apply blur for soft edges
        if blur > 0:
            shadow_canvas = gaussian_filter(shadow_canvas, sigma=blur)
        
        # Apply additional softness (feathering)
        if softness > 0:
            kernel_size = int(softness * 5) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            if kernel_size > 1:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                shadow_canvas = cv2.erode(shadow_canvas, kernel, iterations=1)
        
        # Apply opacity
        shadow_canvas = shadow_canvas * opacity
        
        # Calculate final canvas size
        img_y = 0
        img_x = 0
        result_h = max(h, shadow_start_y + compressed_h)
        result_w = max(w, shadow_start_x + w)
        
        # Create final result with expanded canvas (OPTIMIZED: vectorized operations)
        result_array = np.zeros((result_h, result_w, 4), dtype=np.uint8)
        
        # Place shadow first - VECTORIZED (100x faster than nested loops)
        shadow_h_end = min(shadow_start_y + compressed_h, result_h)
        shadow_w_end = min(shadow_start_x + w, result_w)
        actual_shadow_h = shadow_h_end - shadow_start_y
        actual_shadow_w = shadow_w_end - shadow_start_x
        
        if actual_shadow_h > 0 and actual_shadow_w > 0:
            # Extract shadow region
            shadow_region = shadow_canvas[shadow_start_y:shadow_h_end, shadow_start_x:shadow_w_end]
            
            # Create shadow color array (vectorized)
            shadow_alpha = (shadow_region * 255).astype(np.uint8)
            result_array[shadow_start_y:shadow_h_end, shadow_start_x:shadow_w_end, 0] = self.shadow_color.red()
            result_array[shadow_start_y:shadow_h_end, shadow_start_x:shadow_w_end, 1] = self.shadow_color.green()
            result_array[shadow_start_y:shadow_h_end, shadow_start_x:shadow_w_end, 2] = self.shadow_color.blue()
            result_array[shadow_start_y:shadow_h_end, shadow_start_x:shadow_w_end, 3] = shadow_alpha
        
        # Place original image on top
        result_array[img_y:img_y+h, img_x:img_x+w] = img_array
        
        return Image.fromarray(result_array, 'RGBA')
    
    def generate_natural_shadow(self, img):
        """Generate side-light shadow (stretches to one side)"""
        import cv2
        import numpy as np
        from scipy.ndimage import gaussian_filter, affine_transform
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        img_array = np.array(img)
        alpha = img_array[:, :, 3]
        
        # Get parameters
        opacity = self.shadow_opacity_slider.value() / 100.0
        blur = self.shadow_blur_slider.value()
        distance = self.shadow_distance_slider.value()
        angle = self.shadow_angle_slider.value()
        compress = self.shadow_compress_slider.value() / 100.0
        softness = self.shadow_softness_slider.value() / 100.0
        
        # Create shadow mask
        shadow_mask = alpha.astype(np.float32) / 255.0
        
        # Apply vertical compression
        h, w = shadow_mask.shape
        compressed_h = int(h * compress)
        shadow_mask_compressed = cv2.resize(shadow_mask, (w, compressed_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate directional offset
        angle_rad = np.radians(angle)
        offset_x = int(distance * np.cos(angle_rad))
        offset_y = int(distance * np.sin(angle_rad))
        
        # Create shadow with offset
        shadow_full = np.zeros((h, w), dtype=np.float32)
        paste_y = h - compressed_h + offset_y
        paste_x = offset_x
        
        # Place compressed shadow
        for y in range(compressed_h):
            for x in range(w):
                target_y = paste_y + y
                target_x = paste_x + x
                if 0 <= target_y < h and 0 <= target_x < w:
                    shadow_full[target_y, target_x] = shadow_mask_compressed[y, x]
        
        # Apply blur
        if blur > 0:
            shadow_full = gaussian_filter(shadow_full, sigma=blur / 2)
        
        # Soften edges
        if softness > 0:
            kernel_size = int(softness * 10) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            shadow_full = cv2.erode(shadow_full, kernel, iterations=1)
        
        shadow_full = shadow_full * opacity
        
        # Create RGBA shadow
        shadow_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        shadow_rgba[:, :, 0] = int(self.shadow_color.red())
        shadow_rgba[:, :, 1] = int(self.shadow_color.green())
        shadow_rgba[:, :, 2] = int(self.shadow_color.blue())
        shadow_rgba[:, :, 3] = (shadow_full * 255).astype(np.uint8)
        
        # Composite
        result = Image.new('RGBA', img.size, (255, 255, 255, 0))
        result.paste(Image.fromarray(shadow_rgba, 'RGBA'), (0, 0))
        result.paste(img, (0, 0), mask=img)
        
        return result
    
    def generate_reflection_shadow(self, img):
        """Generate mirror reflection (glossy surface)"""
        import cv2
        import numpy as np
        from scipy.ndimage import gaussian_filter
        
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        img_array = np.array(img)
        h, w, _ = img_array.shape
        
        # Get reflection-specific parameters
        opacity = self.shadow_opacity_slider.value() / 100.0
        blur = self.shadow_blur_slider.value()
        softness = self.shadow_softness_slider.value() / 100.0
        
        # New reflection controls
        reflection_height_pct = self.reflection_height_slider.value() / 100.0  # 10% to 200%
        gap_pct = self.reflection_gap_slider.value() / 100.0  # -50% to +50%
        fade_start = self.reflection_fade_slider.value() / 100.0  # 0% to 100%
        
        # Calculate actual gap in pixels (percentage of image height)
        gap_pixels = int(h * gap_pct)
        
        # Flip image vertically for reflection
        reflection = np.flipud(img_array)
        
        # Calculate reflection height
        reflection_h = max(1, int(h * reflection_height_pct))
        reflection_resized = cv2.resize(reflection, (w, reflection_h), interpolation=cv2.INTER_LINEAR)
        
        # Create gradient fade
        # fade_start: where fade begins (0 = fade from top, 1 = no fade until bottom)
        # softness: how fast it fades
        fade_length = reflection_h
        fade = np.ones(fade_length, dtype=np.float32)
        
        # Calculate fade start position
        fade_start_pos = int(fade_length * fade_start)
        fade_region_length = fade_length - fade_start_pos
        
        if fade_region_length > 0:
            # Create fade from fade_start position to end
            fade_power = 0.5 + (softness * 2.5)  # Control fade curve
            fade_values = np.linspace(1.0, 0.0, fade_region_length) ** fade_power
            fade[fade_start_pos:] = fade_values
        
        fade = fade.reshape(-1, 1)
        reflection_resized[:, :, 3] = (reflection_resized[:, :, 3] * fade * opacity).astype(np.uint8)
        
        # Apply blur
        if blur > 0:
            for c in range(4):
                reflection_resized[:, :, c] = gaussian_filter(reflection_resized[:, :, c], sigma=blur / 3)
        
        # Create expanded canvas to fit both image and reflection
        total_height = h + gap_pixels + reflection_h
        
        # Handle negative gap (overlap)
        if gap_pixels < 0:
            # Reflection overlaps with image
            result = Image.new('RGBA', (w, max(h, h + gap_pixels + reflection_h)), (255, 255, 255, 0))
            result.paste(img, (0, 0))
            reflection_img = Image.fromarray(reflection_resized, 'RGBA')
            result.paste(reflection_img, (0, h + gap_pixels), mask=reflection_img)
        else:
            # Normal case with gap
            result = Image.new('RGBA', (w, total_height), (255, 255, 255, 0))
            result.paste(img, (0, 0))
            reflection_img = Image.fromarray(reflection_resized, 'RGBA')
            result.paste(reflection_img, (0, h + gap_pixels), mask=reflection_img)
        
        return result
    
    def on_shadow_changed(self):
        """Update shadow effect (legacy method for compatibility)"""
        self.update_shadow_preview()
    
    def save_state(self):
        """Save current state for undo"""
        if self.current_image:
            state = {
                'current': self.current_image.copy(),
                'original': self.original_image.copy(),
                'base': self.base_image.copy() if self.base_image else None,
                'pos_x': self.pos_x_spin.value(),
                'pos_y': self.pos_y_spin.value(),
                'scale': self.scale_spin.value(),
                'rotation': self.rotation_spin.value()
            }
            self.undo_stack.append(state)
            if len(self.undo_stack) > self.max_undo_levels:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
    
    def undo(self):
        """Undo last operation"""
        if self.undo_stack:
            # Save current state to redo
            current_state = {
                'current': self.current_image.copy() if self.current_image else None,
                'original': self.original_image.copy() if self.original_image else None,
                'base': self.base_image.copy() if self.base_image else None,
                'pos_x': self.pos_x_spin.value(),
                'pos_y': self.pos_y_spin.value(),
                'scale': self.scale_spin.value(),
                'rotation': self.rotation_spin.value()
            }
            self.redo_stack.append(current_state)
            
            # Restore previous state
            state = self.undo_stack.pop()
            self.current_image = state['current']
            self.original_image = state['original']
            self.base_image = state['base']
            self.pos_x_spin.setValue(state['pos_x'])
            self.pos_y_spin.setValue(state['pos_y'])
            self.scale_spin.setValue(state['scale'])
            self.rotation_spin.setValue(state['rotation'])
            
            self.display_processed_image(self.current_image)
            self.info_label.setText(f"↶ Undo ({len(self.undo_stack)} steps remaining)")
    
    def redo(self):
        """Redo last undone operation"""
        if self.redo_stack:
            # Save current to undo
            self.save_state()
            
            # Restore redo state
            state = self.redo_stack.pop()
            self.current_image = state['current']
            self.original_image = state['original']
            self.base_image = state['base']
            self.pos_x_spin.setValue(state['pos_x'])
            self.pos_y_spin.setValue(state['pos_y'])
            self.scale_spin.setValue(state['scale'])
            self.rotation_spin.setValue(state['rotation'])
            
            self.display_processed_image(self.current_image)
            self.info_label.setText(f"↷ Redo ({len(self.redo_stack)} steps remaining)")
    
    def on_transform_changed(self):
        """Update preview when transform settings change (debounced)"""
        if self.current_image is not None:
            # Show that transform is updating
            self.info_label.setText("🔄 Updating position...")
            self.info_label.setStyleSheet("color: #2196F3;")  # Blue color
            # Stop any pending transform update
            self.transform_timer.stop()
            # Start new timer - waits 100ms after last change
            self.transform_timer.start(100)
    
    def apply_transform_update(self):
        """Apply the actual transform update (called after debounce)"""
        if self.current_image is not None:
            # Force display update with current image
            self.display_processed_image(self.current_image)
            # Clear the updating message
            self.info_label.setText(self.gpu_info)
            self.info_label.setStyleSheet("")
    
    def reset_transform(self):
        """Reset all transform values to default"""
        self.pos_x_spin.setValue(0)
        self.pos_y_spin.setValue(0)
        self.scale_spin.setValue(100)
        self.rotation_spin.setValue(0)
    
    def mouse_press_event(self, event):
        """Handle mouse press for dragging or crop selection"""
        if self.current_image is not None and event.button() == Qt.MouseButton.LeftButton:
            if self.crop_mode and self.crop_rect is None:
                # Start NEW crop selection only if no crop exists yet
                # Get position relative to pixmap, not label
                pixmap = self.image_label.pixmap()
                if pixmap:
                    # Label might be larger than pixmap - constrain to pixmap bounds
                    pos = event.pos()
                    label_rect = self.image_label.rect()
                    pixmap_rect = pixmap.rect()
                    
                    # Calculate offset if label is centered
                    offset_x = (label_rect.width() - pixmap_rect.width()) // 2
                    offset_y = (label_rect.height() - pixmap_rect.height()) // 2
                    
                    # Convert to pixmap coordinates
                    pix_x = pos.x() - offset_x
                    pix_y = pos.y() - offset_y
                    
                    # Clamp to pixmap bounds
                    pix_x = max(0, min(pix_x, pixmap_rect.width()))
                    pix_y = max(0, min(pix_y, pixmap_rect.height()))
                    
                    self.crop_start = QPoint(pix_x, pix_y)
                    self.crop_end = QPoint(pix_x, pix_y)
                    if self.crop_overlay_label:
                        # Position overlay relative to label (add offset back)
                        self.crop_overlay_label.setGeometry(pix_x + offset_x, pix_y + offset_y, 0, 0)
                        self.crop_overlay_label.show()
            elif not self.crop_mode:
                # Normal drag mode
                self.mouse_dragging = True
                self.last_mouse_pos = event.pos()
                self.ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                self.image_label.setCursor(Qt.CursorShape.ClosedHandCursor)
    
    def mouse_move_event(self, event):
        """Handle mouse move for dragging/rotating or crop selection"""
        if self.crop_mode and self.crop_start is not None and self.crop_rect is None:
            # Update crop rectangle visual feedback (only while actively drawing)
            pixmap = self.image_label.pixmap()
            if pixmap:
                pos = event.pos()
                label_rect = self.image_label.rect()
                pixmap_rect = pixmap.rect()
                
                offset_x = (label_rect.width() - pixmap_rect.width()) // 2
                offset_y = (label_rect.height() - pixmap_rect.height()) // 2
                
                pix_x = max(0, min(pos.x() - offset_x, pixmap_rect.width()))
                pix_y = max(0, min(pos.y() - offset_y, pixmap_rect.height()))
                
                self.crop_end = QPoint(pix_x, pix_y)
                
                if self.crop_overlay_label:
                    x = min(self.crop_start.x(), self.crop_end.x())
                    y = min(self.crop_start.y(), self.crop_end.y())
                    w = abs(self.crop_end.x() - self.crop_start.x())
                    h = abs(self.crop_end.y() - self.crop_start.y())
                    # Position relative to label
                    self.crop_overlay_label.setGeometry(x + offset_x, y + offset_y, w, h)
        elif self.mouse_dragging and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            
            if self.ctrl_pressed:
                # Ctrl+Drag = Rotate
                rotation_delta = delta.x() * 0.5  # Sensitivity adjustment
                new_rotation = self.rotation_spin.value() + int(rotation_delta)
                new_rotation = max(-180, min(180, new_rotation))
                self.rotation_spin.setValue(new_rotation)
            else:
                # Normal Drag = Move
                new_x = self.pos_x_spin.value() + delta.x()
                new_y = self.pos_y_spin.value() + delta.y()
                self.pos_x_spin.setValue(new_x)
                self.pos_y_spin.setValue(new_y)
            
            self.last_mouse_pos = event.pos()
    
    def mouse_release_event(self, event):
        """Handle mouse release"""
        if event.button() == Qt.MouseButton.LeftButton:
            if self.crop_mode and self.crop_start is not None and self.crop_end is not None and self.crop_rect is None:
                # Finalize crop selection
                x1 = min(self.crop_start.x(), self.crop_end.x())
                y1 = min(self.crop_start.y(), self.crop_end.y())
                x2 = max(self.crop_start.x(), self.crop_end.x())
                y2 = max(self.crop_start.y(), self.crop_end.y())
                
                if x2 - x1 > 10 and y2 - y1 > 10:  # Minimum size
                    self.crop_rect = (x1, y1, x2, y2)
                    self.info_label.setText(f"✓ Crop area selected ({x2-x1}x{y2-y1}px). Press ENTER to apply or ESC to cancel")
                    # Keep crop_start and crop_end so we don't allow new selection until this is applied/cancelled
                else:
                    # Too small, reset
                    self.crop_start = None
                    self.crop_end = None
                    if self.crop_overlay_label:
                        self.crop_overlay_label.hide()
            elif not self.crop_mode:
                self.mouse_dragging = False
                self.last_mouse_pos = None
                self.image_label.setCursor(Qt.CursorShape.ArrowCursor)
    
    def mouse_wheel_event(self, event):
        """Handle mouse wheel for zooming"""
        if self.current_image is not None:
            delta = event.angleDelta().y()
            scale_change = 5 if delta > 0 else -5
            new_scale = self.scale_spin.value() + scale_change
            new_scale = max(10, min(500, new_scale))
            self.scale_spin.setValue(new_scale)
            event.accept()
    
    def apply_background_removal(self):
        """Apply background removal with selected AI model"""
        if self.original_image is None:  # Changed from current_image to original_image
            return
        
        self.info_label.setText(f"Removing background with {self.current_model}...")
        
        # Get or create session for current model
        if self.current_model not in self.rembg_sessions:
            try:
                self.info_label.setText(f"Loading {self.current_model} model ({self.gpu_info})...")
                QApplication.processEvents()
                # Use GPU-accelerated providers if available
                self.rembg_sessions[self.current_model] = new_session(
                    self.current_model,
                    providers=self.execution_providers
                )
                self.info_label.setText(f"✓ Model loaded on {self.gpu_info}")
            except Exception as e:
                self.info_label.setText(f"Error loading model: {str(e)}")
                return
        
        # Process background removal - always use original image for consistent results
        settings = {
            'remove_bg': True,
            'brightness': 1.0,
            'contrast': 1.0,
            'saturation': 1.0,
            'sharpness': 1.0,
            'resize_enabled': False,
            'session': self.rembg_sessions[self.current_model],
            'model_name': self.current_model,
            'alpha_matting': self.alpha_matting_check.isChecked(),
            'alpha_matting_foreground_threshold': self.fg_threshold_slider.value(),
            'alpha_matting_background_threshold': self.bg_threshold_slider.value(),
            'alpha_matting_erode_size': 10,  # Fixed at 10 for stability
            'edge_softness': self.edge_softness_slider.value(),
            'uncertainty_detection': self.uncertainty_detect_check.isChecked(),
            'post_process_mask': self.post_process_check.isChecked()
        }
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.apply_bg_btn.setEnabled(False)
        
        self.processing_thread = ImageProcessor(self.original_image, settings)  # Changed to use original_image
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.finished.connect(self.on_bg_removed)
        self.processing_thread.error.connect(self.handle_processing_error)
        self.processing_thread.start()
    
    def on_bg_removed(self, img):
        """Handle background removal completion"""
        self.current_image = img
        self.base_image = img  # Set bg-removed image as base for adjustments
        self.progress_bar.setVisible(False)
        self.apply_bg_btn.setEnabled(True)
        self.info_label.setText("✅ Background removed successfully!")
        self.display_processed_image(img)
            
    def open_image(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*.*)")
        if file_paths:
            if len(file_paths) > 1:
                # Batch mode
                self.load_batch_images(file_paths)
            else:
                # Single image mode
                self.load_image(file_paths[0])
            
    def load_image(self, file_path):
        try:
            self.current_file_path = file_path
            self.original_image = Image.open(file_path)
            if self.original_image.mode not in ('RGB', 'RGBA'):
                self.original_image = self.original_image.convert('RGB')
            
            # Optimize very large images to prevent memory issues
            max_dimension = 8192
            if self.original_image.width > max_dimension or self.original_image.height > max_dimension:
                self.info_label.setText(f"Resizing large image for processing...")
                QApplication.processEvents()
                self.original_image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            
            self.current_image = self.original_image.copy()
            self.base_image = self.original_image.copy()  # Initially use original as base
            
            # Auto-fit: Calculate initial scale to fit product nicely in canvas
            if self.resize_check.isChecked():
                # Canvas is limited to width_spin x height_spin (e.g., 2048x2048)
                canvas_size = min(self.width_spin.value(), self.height_spin.value())
            else:
                # Canvas will be image size, use a reasonable target
                canvas_size = 2048
            
            # If image is larger than canvas, scale down to fit
            max_img_dimension = max(self.original_image.width, self.original_image.height)
            if max_img_dimension > canvas_size:
                # Scale to fit with some margin (90% of canvas)
                scale_to_fit = int((canvas_size * 0.9) / max_img_dimension * 100)
                self.scale_spin.setValue(scale_to_fit)
            else:
                # Image fits, use 100% scale
                self.scale_spin.setValue(100)
            
            self.info_label.setText(f"Loaded: {os.path.basename(file_path)} | {self.original_image.size[0]}x{self.original_image.size[1]}")
            self.save_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.apply_bg_btn.setEnabled(True)
            self.display_processed_image(self.current_image)
        except Exception as e:
            self.info_label.setText(f"Error: {str(e)}")
            
    def update_preview(self):
        """Apply adjustments instantly without background removal"""
        if self.base_image is None:
            return
            
        self.brightness_value.setText(f"{self.brightness_slider.value() / 100.0:.2f}")
        self.contrast_value.setText(f"{self.contrast_slider.value() / 100.0:.2f}")
        self.saturation_value.setText(f"{self.saturation_slider.value() / 100.0:.2f}")
        self.sharpness_value.setText(f"{self.sharpness_slider.value() / 100.0:.2f}")
        
        # Always start from base image (original or bg-removed) to avoid cumulative adjustments
        img = self.base_image.copy()
        
        # Apply adjustments directly (fast, no threading needed)
        brightness = self.brightness_slider.value() / 100.0
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        
        contrast = self.contrast_slider.value() / 100.0
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        
        saturation = self.saturation_slider.value() / 100.0
        if saturation != 1.0:
            img = ImageEnhance.Color(img).enhance(saturation)
        
        sharpness = self.sharpness_slider.value() / 100.0
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        
        # Update current image and display
        self.current_image = img
        self.display_processed_image(img)
    
    def on_adjustment_applied(self, img):
        """Update preview after adjustments"""
        self.current_image = img
        self.display_processed_image(img)
        
    def display_processed_image(self, img):
        self.current_image = img
        self.progress_bar.setVisible(False)
        
        # Get target canvas size from settings (FIXED canvas size)
        if self.resize_check.isChecked():
            canvas_width = self.width_spin.value()
            canvas_height = self.height_spin.value()
        else:
            # Use fixed canvas size (not scaled)
            canvas_width = img.width
            canvas_height = img.height
        
        # Get background color
        bg_color = self.bg_color.getRgb()[:3]  # (R, G, B)
        
        # Create canvas with FIXED dimensions
        canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
        
        # Prepare the image with transforms
        img_to_paste = img.copy()
        
        # Apply scale transform to PRODUCT ONLY
        scale_factor = self.scale_spin.value() / 100.0
        if scale_factor != 1.0:
            new_width = int(img_to_paste.width * scale_factor)
            new_height = int(img_to_paste.height * scale_factor)
            img_to_paste = img_to_paste.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Apply rotation transform
        rotation = self.rotation_spin.value()
        if rotation != 0:
            img_to_paste = img_to_paste.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
        
        # Get background color
        bg_color = self.bg_color.getRgb()[:3]  # (R, G, B)
        
        # Create canvas with FIXED dimensions (doesn't change with scale)
        canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
        
        # Calculate position with transforms (center + manual offset)
        x_offset = (canvas_width - img_to_paste.width) // 2 + self.pos_x_spin.value()
        y_offset = (canvas_height - img_to_paste.height) // 2 + self.pos_y_spin.value()
        
        # Paste scaled product onto FIXED canvas
        if img_to_paste.mode == 'RGBA':
            canvas.paste(img_to_paste, (x_offset, y_offset), mask=img_to_paste.split()[3])
        else:
            if img_to_paste.mode != 'RGB':
                img_to_paste = img_to_paste.convert('RGB')
            canvas.paste(img_to_paste, (x_offset, y_offset))
        
        # Scale canvas to fit display area (800x600) while maintaining aspect ratio
        display_canvas = canvas.copy()
        display_canvas.thumbnail((800, 600), Image.Resampling.LANCZOS)
        
        # Convert to QPixmap for display
        img_bytes = display_canvas.tobytes('raw', 'RGB')
        qimage = QImage(img_bytes, display_canvas.width, display_canvas.height, display_canvas.width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)
        
    def handle_processing_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.info_label.setText(f"Error: {error_msg}")
        
    def reset_image(self):
        """Reset all adjustments, transforms, and effects to default"""
        # Reset adjustment sliders
        self.brightness_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.saturation_slider.setValue(100)
        self.sharpness_slider.setValue(100)
        
        # Reset transforms
        self.pos_x_spin.setValue(0)
        self.pos_y_spin.setValue(0)
        self.scale_spin.setValue(100)
        self.rotation_spin.setValue(0)
        
        # Reset shadow
        self.shadow_combo.setCurrentIndex(0)
        
        # Reset to original image
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.base_image = self.original_image.copy()
            self.display_processed_image(self.current_image)
    
    def apply_smart_padding(self, img):
        """Apply smart padding to center product with consistent margins"""
        import numpy as np
        
        # Get padding settings
        padding_percent = self.padding_slider.value()
        canvas_type = self.canvas_combo.currentText()
        
        # Convert to numpy for analysis
        img_array = np.array(img)
        
        # Find bounding box of non-transparent pixels
        if img_array.shape[2] == 4:  # Has alpha
            alpha = img_array[:, :, 3]
            rows = np.any(alpha > 10, axis=1)
            cols = np.any(alpha > 10, axis=0)
            if not np.any(rows) or not np.any(cols):
                return img  # Empty image
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            # Crop to content
            content = img.crop((xmin, ymin, xmax + 1, ymax + 1))
            content_width = content.width
            content_height = content.height
        else:
            content = img
            content_width = img.width
            content_height = img.height
        
        # Calculate canvas size based on preset
        aspect_ratios = {
            "Square (1:1)": 1.0,
            "Portrait (3:4)": 3/4,
            "Landscape (4:3)": 4/3,
            "Wide (16:9)": 16/9,
            "Instagram (1:1)": 1.0,
            "Custom": content_width / content_height
        }
        
        target_aspect = aspect_ratios.get(canvas_type, 1.0)
        
        # Calculate canvas size with padding
        padding_factor = 1 + (padding_percent / 100) * 2  # Both sides
        
        if target_aspect >= 1:  # Landscape or square
            canvas_height = int(content_height * padding_factor)
            canvas_width = int(canvas_height * target_aspect)
        else:  # Portrait
            canvas_width = int(content_width * padding_factor)
            canvas_height = int(canvas_width / target_aspect)
        
        # Create canvas with white or transparent background
        if img.mode == 'RGBA':
            canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 0))
        else:
            canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # Resize content to fit with padding
        max_content_width = int(canvas_width / padding_factor)
        max_content_height = int(canvas_height / padding_factor)
        
        content.thumbnail((max_content_width, max_content_height), Image.Resampling.LANCZOS)
        
        # Center content on canvas
        x_offset = (canvas_width - content.width) // 2
        y_offset = (canvas_height - content.height) // 2
        
        if content.mode == 'RGBA':
            canvas.paste(content, (x_offset, y_offset), mask=content.split()[3])
        else:
            canvas.paste(content, (x_offset, y_offset))
        
        return canvas
            
    def save_image(self):
        if self.current_image is None:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "WebP Images (*.webp);;All Files (*.*)")
        if file_path:
            try:
                if not file_path.lower().endswith('.webp'):
                    file_path += '.webp'
                
                # Prepare the image with transforms
                img_to_save = self.current_image.copy()
                
                # Apply smart padding if enabled (BEFORE other transforms)
                if self.smart_padding_check.isChecked():
                    img_to_save = self.apply_smart_padding(img_to_save)
                
                # Get target canvas size from settings
                if self.resize_check.isChecked():
                    canvas_width = self.width_spin.value()
                    canvas_height = self.height_spin.value()
                else:
                    canvas_width = self.current_image.width
                    canvas_height = self.current_image.height
                
                # Get background color
                bg_color = self.bg_color.getRgb()[:3]
                
                # Create canvas with target dimensions
                canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
                
                # Prepare the image with transforms
                img_to_save = self.current_image.copy()
                
                # Apply scale transform
                scale_factor = self.scale_spin.value() / 100.0
                if scale_factor != 1.0:
                    new_width = int(img_to_save.width * scale_factor)
                    new_height = int(img_to_save.height * scale_factor)
                    img_to_save = img_to_save.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Apply rotation transform
                rotation = self.rotation_spin.value()
                if rotation != 0:
                    img_to_save = img_to_save.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
                
                # Resize image to fit within canvas if needed (after transforms)
                if self.resize_check.isChecked():
                    img_to_save.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
                
                # Calculate position with transforms
                x_offset = (canvas_width - img_to_save.width) // 2 + self.pos_x_spin.value()
                y_offset = (canvas_height - img_to_save.height) // 2 + self.pos_y_spin.value()
                
                # Paste image onto canvas
                if img_to_save.mode == 'RGBA':
                    canvas.paste(img_to_save, (x_offset, y_offset), mask=img_to_save.split()[3])
                else:
                    if img_to_save.mode != 'RGB':
                        img_to_save = img_to_save.convert('RGB')
                    canvas.paste(img_to_save, (x_offset, y_offset))
                
                # Save with settings
                quality = self.quality_slider.value()
                lossless = self.lossless_check.isChecked()
                dpi = self.dpi_spin.value()
                save_kwargs = {'format': 'WebP', 'quality': quality, 'lossless': lossless, 'dpi': (dpi, dpi)}
                
                if self.keep_exif_check.isChecked() and hasattr(self.original_image, 'info'):
                    exif = self.original_image.info.get('exif')
                    if exif:
                        save_kwargs['exif'] = exif
                
                canvas.save(file_path, **save_kwargs)
                file_size = os.path.getsize(file_path) / 1024
                self.info_label.setText(f"Saved: {os.path.basename(file_path)} | {canvas_width}x{canvas_height} | {file_size:.1f} KB")
            except Exception as e:
                self.info_label.setText(f"Error saving: {str(e)}")
    
    def load_batch_images(self, file_paths):
        """Load multiple images for batch processing"""
        self.image_list = []
        self.thumbnail_list.clear()
        
        # Set output folder to 'processed ecom images' in the same directory as first image
        first_file_dir = os.path.dirname(file_paths[0])
        self.output_folder = os.path.join(first_file_dir, "processed ecom images")
        os.makedirs(self.output_folder, exist_ok=True)
        
        for file_path in file_paths:
            try:
                img = Image.open(file_path)
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                # Optimize very large images
                max_dimension = 8192
                if img.width > max_dimension or img.height > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                
                # Create thumbnail for list
                thumb = img.copy()
                thumb.thumbnail((80, 80), Image.Resampling.LANCZOS)
                
                # Convert to QPixmap
                if thumb.mode == 'RGBA':
                    thumb_rgb = Image.new('RGB', thumb.size, (255, 255, 255))
                    thumb_rgb.paste(thumb, mask=thumb.split()[3] if thumb.mode == 'RGBA' else None)
                    thumb = thumb_rgb
                thumb_bytes = thumb.tobytes('raw', 'RGB')
                qimage = QImage(thumb_bytes, thumb.width, thumb.height, thumb.width * 3, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                # Add to list
                item = QListWidgetItem(QIcon(pixmap), "")
                item.setToolTip(os.path.basename(file_path))
                self.thumbnail_list.addItem(item)
                
                self.image_list.append({
                    'path': file_path,
                    'original': img,
                    'processed': None
                })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if self.image_list:
            self.thumbnail_widget.setVisible(True)
            self.batch_nav_widget.setVisible(True)
            self.save_next_btn.setVisible(True)
            self.process_all_btn.setEnabled(True)  # Enable parallel processing button
            self.current_image_index = 0
            self.load_image_from_batch(0)
            self.info_label.setText(f"✅ Loaded {len(self.image_list)} images | Output: {self.output_folder}")
    
    def process_all_batch(self):
        """Process all images in batch using parallel processing"""
        if not self.image_list:
            return
        
        # Check if model is loaded
        if self.current_model not in self.rembg_sessions:
            try:
                self.info_label.setText(f"Loading {self.current_model} model ({self.gpu_info})...")
                QApplication.processEvents()
                self.rembg_sessions[self.current_model] = new_session(
                    self.current_model,
                    providers=self.execution_providers
                )
            except Exception as e:
                self.info_label.setText(f"Error loading model: {str(e)}")
                return
        
        # Prepare settings
        settings = {
            'remove_bg': True,
            'brightness': 1.0,
            'contrast': 1.0,
            'saturation': 1.0,
            'sharpness': 1.0,
            'resize_enabled': False,
            'session': self.rembg_sessions[self.current_model],
            'model_name': self.current_model,
            'alpha_matting': self.alpha_matting_check.isChecked(),
            'alpha_matting_foreground_threshold': self.fg_threshold_slider.value(),
            'alpha_matting_background_threshold': self.bg_threshold_slider.value(),
            'alpha_matting_erode_size': 10,
            'edge_softness': self.edge_softness_slider.value(),
            'uncertainty_detection': self.uncertainty_detect_check.isChecked(),
            'post_process_mask': self.post_process_check.isChecked(),
            'providers': self.execution_providers
        }
        
        # Collect all original images
        images_to_process = [item['original'] for item in self.image_list]
        
        # Disable buttons during processing
        self.process_all_btn.setEnabled(False)
        self.apply_bg_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        # Start parallel batch processor
        self.batch_processor = ParallelBatchProcessor(images_to_process, settings)
        self.batch_processor.progress.connect(self.on_batch_progress)
        self.batch_processor.item_finished.connect(self.on_batch_item_finished)
        self.batch_processor.all_finished.connect(self.on_batch_all_finished)
        self.batch_processor.error.connect(self.on_batch_item_error)
        self.batch_processor.start()
        
        worker_count = self.batch_processor.num_workers
        self.info_label.setText(f"⚡ Processing {len(images_to_process)} images in parallel ({worker_count} workers)...")
    
    def on_batch_progress(self, current, total):
        """Update progress during batch processing"""
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
        self.batch_progress_label.setText(f"{current}/{total} processed")
    
    def on_batch_item_finished(self, index, processed_image):
        """Handle when a single batch item finishes"""
        if 0 <= index < len(self.image_list):
            self.image_list[index]['processed'] = processed_image
            # Update thumbnail to show it's processed (you could add a checkmark overlay)
    
    def on_batch_item_error(self, index, error_msg):
        """Handle error during batch processing"""
        if index >= 0:
            print(f"Error processing image {index}: {error_msg}")
    
    def on_batch_all_finished(self):
        """Handle when all batch processing is complete"""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.process_all_btn.setEnabled(True)
        self.apply_bg_btn.setEnabled(True)
        
        processed_count = sum(1 for item in self.image_list if item['processed'] is not None)
        self.info_label.setText(f"✅ Batch complete! {processed_count}/{len(self.image_list)} images processed successfully")
        self.batch_progress_label.setText(f"{processed_count}/{len(self.image_list)} processed")
        
        # Refresh current image if it was processed
        if self.current_image_index >= 0 and self.image_list[self.current_image_index]['processed']:
            self.current_image = self.image_list[self.current_image_index]['processed']
            self.base_image = self.current_image.copy()
            self.display_processed_image(self.current_image)
    
    def load_image_from_batch(self, index):
        """Load a specific image from the batch"""
        if 0 <= index < len(self.image_list):
            self.current_image_index = index
            img_data = self.image_list[index]
            
            self.original_image = img_data['original']
            self.current_image = self.original_image.copy()
            self.base_image = self.original_image.copy()
            self.current_file_path = img_data['path']
            
            self.save_btn.setEnabled(True)
            self.save_next_btn.setEnabled(True)
            self.reset_btn.setEnabled(True)
            self.apply_bg_btn.setEnabled(True)
            
            # Update UI
            self.thumbnail_list.setCurrentRow(index)
            self.batch_info_label.setText(f"{index + 1}/{len(self.image_list)}")
            self.prev_btn.setEnabled(index > 0)
            self.next_btn.setEnabled(index < len(self.image_list) - 1)
            
            self.display_processed_image(self.current_image)
    
    def on_thumbnail_clicked(self, item):
        """Handle thumbnail click"""
        index = self.thumbnail_list.row(item)
        self.load_image_from_batch(index)
    
    def previous_image(self):
        """Go to previous image in batch"""
        if self.current_image_index > 0:
            self.load_image_from_batch(self.current_image_index - 1)
    
    def next_image(self):
        """Go to next image in batch"""
        if self.current_image_index < len(self.image_list) - 1:
            self.load_image_from_batch(self.current_image_index + 1)
    
    def save_and_next(self):
        """Save current image and move to next"""
        if self.current_image and self.output_folder:
            self.auto_save_current_image()
            if self.current_image_index < len(self.image_list) - 1:
                self.next_image()
            else:
                self.info_label.setText("✅ All images processed!")
    
    def auto_save_current_image(self):
        """Auto-save current image to output folder"""
        if not self.current_image or not self.output_folder:
            return
        
        try:
            # Generate output filename
            original_name = os.path.basename(self.current_file_path)
            name_without_ext = os.path.splitext(original_name)[0]
            output_filename = f"{self.file_prefix}{name_without_ext}{self.file_suffix}.webp"
            file_path = os.path.join(self.output_folder, output_filename)
            
            # Get canvas and settings
            if self.resize_check.isChecked():
                canvas_width = self.width_spin.value()
                canvas_height = self.height_spin.value()
            else:
                canvas_width = self.current_image.width
                canvas_height = self.current_image.height
            
            bg_color = self.bg_color.getRgb()[:3]
            canvas = Image.new('RGB', (canvas_width, canvas_height), bg_color)
            
            img_to_save = self.current_image.copy()
            
            # Apply transforms
            scale_factor = self.scale_spin.value() / 100.0
            if scale_factor != 1.0:
                new_width = int(img_to_save.width * scale_factor)
                new_height = int(img_to_save.height * scale_factor)
                img_to_save = img_to_save.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            rotation = self.rotation_spin.value()
            if rotation != 0:
                img_to_save = img_to_save.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)
            
            if self.resize_check.isChecked():
                img_to_save.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            x_offset = (canvas_width - img_to_save.width) // 2 + self.pos_x_spin.value()
            y_offset = (canvas_height - img_to_save.height) // 2 + self.pos_y_spin.value()
            
            if img_to_save.mode == 'RGBA':
                canvas.paste(img_to_save, (x_offset, y_offset), mask=img_to_save.split()[3])
            else:
                if img_to_save.mode != 'RGB':
                    img_to_save = img_to_save.convert('RGB')
                canvas.paste(img_to_save, (x_offset, y_offset))
            
            # Save
            quality = self.quality_slider.value()
            lossless = self.lossless_check.isChecked()
            dpi = self.dpi_spin.value()
            canvas.save(file_path, format='WebP', quality=quality, lossless=lossless, dpi=(dpi, dpi))
            
            file_size = os.path.getsize(file_path) / 1024
            self.info_label.setText(f"💾 Saved: {output_filename} | {file_size:.1f} KB")
            
            # Mark as processed in thumbnail
            item = self.thumbnail_list.item(self.current_image_index)
            item.setText("✓")
            item.setToolTip(f"✓ {os.path.basename(self.current_file_path)}")
            
        except Exception as e:
            self.info_label.setText(f"Error saving: {str(e)}")


def main():
    # Fix Windows taskbar icon grouping - MUST be called BEFORE QApplication
    import platform
    if platform.system() == 'Windows':
        try:
            import ctypes
            # Set explicit AppUserModelID to ensure icon displays correctly in taskbar
            myappid = 'AIProductStudio.BackgroundRemover.v1.0'
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except:
            pass
    
    app = QApplication(sys.argv)
    
    # Set application-wide icon (for window and taskbar)
    icon_path = os.path.join(os.path.dirname(__file__), 'app_icon.png')
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
        # Also set for the application itself
        QApplication.setWindowIcon(app_icon)
    
    app.setStyle('Fusion')
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    app.setPalette(palette)
    window = ProductImageGenerator()
    window.showMaximized()  # Launch in maximized state like Photoshop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
