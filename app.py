# ================================================================
# Ophthalmology Segmentation API - ULTRA FREE TIER OPTIMIZED
# ================================================================

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
import cv2
import base64
import os
import gc

# Force CPU and disable GPU libraries
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)  # Single thread to save memory

# ================================================================
# DEVICE SETUP
# ================================================================
DEVICE = torch.device("cpu")
print(f"🔹 Running on: {DEVICE}")

# ================================================================
# SEGMENTATION MODEL DEFINITION
# ================================================================

class UpBlock(nn.Module):
    """Upsampling block for U-Net decoder"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip_feature):
        x = self.conv_transpose(x)
        x = torch.cat([x, skip_feature], dim=1)
        return self.conv_block(x)


class SegmentationModel(nn.Module):
    """ResNet50-based U-Net - Memory optimized"""
    def __init__(self, n_classes=3):
        super().__init__()
        
        from torchvision.models import resnet50
        resnet = resnet50(weights=None)
        
        # Build encoder
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Freeze encoder to save memory
        for param in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in param.parameters():
                p.requires_grad = False
        
        # Decoder
        enc_ch = [256, 512, 1024, 2048]
        self.up1 = UpBlock(enc_ch[3], enc_ch[2], 512)
        self.up2 = UpBlock(512, enc_ch[1], 256)
        self.up3 = UpBlock(256, enc_ch[0], 128)
        self.up4 = nn.Sequential(nn.ConvTranspose2d(128, 64, 2, 2), nn.ReLU(inplace=True))
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.stem(x)
        skip1 = self.layer1(x)
        skip2 = self.layer2(skip1)
        skip3 = self.layer3(skip2)
        x = self.layer4(skip3)
        
        # Decoder
        x = self.up1(x, skip3)
        x = self.up2(x, skip2)
        x = self.up3(x, skip1)
        x = self.up4(x)
        x = self.final_conv(x)
        
        return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def calculate_cdr(mask):
    """Calculate CDR with error handling"""
    try:
        disc = (mask == 1).astype(np.uint8)
        cup = (mask == 2).astype(np.uint8)
        
        d_contours, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_contours, _ = cv2.findContours(cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(d_contours) == 0:
            return 0.0
        
        v_disc = max([cv2.boundingRect(c)[3] for c in d_contours], default=0)
        v_cup = max([cv2.boundingRect(c)[3] for c in c_contours], default=0) if len(c_contours) > 0 else 0
        
        return float(v_cup) / float(v_disc) if v_disc > 0 else 0.0
    except Exception as e:
        print(f"⚠️ CDR calculation error: {e}")
        return 0.0


def create_overlay(image, mask):
    """Create visualization overlay"""
    try:
        image = cv2.resize(image, (224, 224))
        overlay = image.copy()
        
        disc_bin = (mask == 1).astype(np.uint8)
        cup_bin = (mask == 2).astype(np.uint8)
        
        d_contours, _ = cv2.findContours(disc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_contours, _ = cv2.findContours(cup_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv2.drawContours(overlay, d_contours, -1, (255, 255, 0), 2)  # Yellow
        cv2.drawContours(overlay, c_contours, -1, (0, 255, 0), 2)    # Green
        
        return overlay
    except Exception as e:
        print(f"⚠️ Overlay error: {e}")
        return image


def encode_image_base64(image):
    """Encode image to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"⚠️ Encoding error: {e}")
        return ""


def preprocess_image(image):
    """Preprocess with proper float32 handling"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    image = cv2.resize(image, (224, 224))
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Normalize
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # To tensor
    tensor = torch.from_numpy(image).permute(2, 0, 1).float()
    return tensor


# ================================================================
# LAZY MODEL LOADING
# ================================================================

segmentation_model = None
MODEL_PATH = "best_seg_model.pth"


def load_segmentation_model():
    """Lazy load model with aggressive memory management"""
    global segmentation_model
    
    if segmentation_model is not None:
        return segmentation_model
    
    print("🔹 Loading segmentation model...")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    # Clear memory before loading
    gc.collect()
    
    # Initialize model
    segmentation_model = SegmentationModel(n_classes=3).to(DEVICE)
    
    # Load weights
    try:
        print(f"📥 Loading weights from {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Convert float64 to float32
        state_dict = {k: v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v 
                     for k, v in state_dict.items()}
        
        segmentation_model.load_state_dict(state_dict, strict=False)
        segmentation_model.eval()
        segmentation_model = segmentation_model.float()
        
        # Free checkpoint memory
        del checkpoint, state_dict
        gc.collect()
        
        print(f"✅ Model loaded! Size: {os.path.getsize(MODEL_PATH)/(1024*1024):.1f}MB")
        return segmentation_model
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


# ================================================================
# FLASK APP
# ================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max


@app.route("/", methods=["GET", "HEAD"])
def home():
    return jsonify({
        "status": "active",
        "message": "Ophthalmology Segmentation API",
        "version": "1.0.0-free-tier",
        "endpoints": {
            "/health": "GET - Health check",
            "/segment": "POST - Segment image",
            "/info": "GET - Model info"
        },
        "device": str(DEVICE),
        "model_loaded": segmentation_model is not None,
        "note": "First request takes 60-90s (model loading)"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": segmentation_model is not None,
        "memory_optimized": True
    })


@app.route("/segment", methods=["POST"])
def segment():
    """Segmentation endpoint with aggressive memory management"""
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        print("📸 Processing image...")
        
        # Load model
        model = load_segmentation_model()
        
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        
        # Preprocess
        tensor = preprocess_image(img).unsqueeze(0).to(DEVICE)
        
        # Inference
        print("🔮 Running inference...")
        with torch.no_grad():
            logits = model(tensor)
            mask = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()
        
        # Free memory immediately
        del tensor, logits
        gc.collect()
        
        # Calculate CDR
        cdr = calculate_cdr(mask)
        diagnosis = "Glaucomatous" if cdr >= 0.65 else "Healthy"
        
        if cdr < 0.3:
            interpretation = "Very low CDR - Normal"
        elif cdr < 0.5:
            interpretation = "Low CDR - Normal"
        elif cdr < 0.65:
            interpretation = "Moderate CDR - Borderline"
        elif cdr < 0.8:
            interpretation = "High CDR - Likely glaucomatous"
        else:
            interpretation = "Very high CDR - Severe"
        
        # Create overlay
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        overlay = create_overlay(img_bgr, mask)
        overlay_b64 = encode_image_base64(overlay)
        
        # Clean up
        del img, img_np, img_bgr, mask, overlay
        gc.collect()
        
        print("✅ Processing complete!")
        
        return jsonify({
            "success": True,
            "task": "segmentation",
            "cdr": round(float(cdr), 3),
            "diagnosis": diagnosis,
            "interpretation": interpretation,
            "segmentation_overlay": overlay_b64,
            "details": {
                "optic_disc": "Yellow contour",
                "optic_cup": "Green contour",
                "threshold": 0.65
            }
        })

    except FileNotFoundError as e:
        return jsonify({
            "error": "Model not found",
            "message": str(e)
        }), 500
    
    except MemoryError:
        gc.collect()
        return jsonify({
            "error": "Out of memory",
            "message": "Server memory limit reached. Try again or use smaller image.",
            "tip": "Free tier has 512MB RAM limit"
        }), 503
    
    except Exception as e:
        print(f"❌ Error: {e}")
        gc.collect()
        return jsonify({
            "error": "Processing failed",
            "message": str(e),
            "type": type(e).__name__
        }), 500


@app.route("/info", methods=["GET"])
def model_info():
    return jsonify({
        "model": "ResNet50-UNet",
        "file": MODEL_PATH,
        "loaded": segmentation_model is not None,
        "classes": ["Background", "Optic Disc", "Optic Cup"],
        "input_size": "224x224",
        "cdr_threshold": 0.65,
        "tier": "free (512MB RAM)",
        "file_exists": os.path.exists(MODEL_PATH),
        "file_size_mb": round(os.path.getsize(MODEL_PATH)/(1024*1024), 2) if os.path.exists(MODEL_PATH) else 0
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "endpoints": ["/", "/health", "/segment", "/info"]}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal error", "message": str(e)}), 500


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large", "limit": "10MB"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("🚀 Segmentation API (FREE TIER)")
    print(f"   Port: {port}")
    print(f"   Device: {DEVICE}")
    print(f"   Model: {MODEL_PATH}")
    print("   Optimized: Aggressive memory management")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=False)