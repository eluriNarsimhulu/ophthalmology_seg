# ================================================================
# Ophthalmology Segmentation API - Optimized for Render
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

# ================================================================
# DEVICE SETUP
# ================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """ResNet50-based U-Net for optic disc and cup segmentation"""
    def __init__(self, n_classes=3):
        super().__init__()
        
        # Try to import timm, fallback to manual ResNet if not available
        try:
            import timm
            self.encoder = timm.create_model(
                'resnet50', 
                pretrained=False,  # Don't download pretrained weights
                features_only=True, 
                out_indices=[1, 2, 3, 4]
            )
            # Freeze encoder
            for p in self.encoder.parameters():
                p.requires_grad = False
            
            # Get encoder channel dimensions
            enc_ch = self.encoder.feature_info.channels()
        except ImportError:
            # Fallback: Use torchvision ResNet50
            from torchvision.models import resnet50
            resnet = resnet50(pretrained=False)
            
            # Extract encoder layers
            self.encoder = nn.ModuleList([
                nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            ])
            
            # Freeze encoder
            for layer in self.encoder:
                for p in layer.parameters():
                    p.requires_grad = False
            
            enc_ch = [256, 512, 1024, 2048]
        
        # Decoder
        self.up1 = UpBlock(enc_ch[3], enc_ch[2], 512)
        self.up2 = UpBlock(512, enc_ch[1], 256)
        self.up3 = UpBlock(256, enc_ch[0], 128)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        if isinstance(self.encoder, nn.ModuleList):
            # Fallback encoder
            skips = []
            for i, layer in enumerate(self.encoder):
                x = layer(x)
                if i > 0:  # Skip first layer (stem)
                    skips.append(x)
        else:
            # timm encoder
            skips = self.encoder(x)
        
        # Decoder
        x = skips.pop()
        x = self.up1(x, skips.pop())
        x = self.up2(x, skips.pop())
        x = self.up3(x, skips.pop())
        x = self.up4(x)
        x = self.final_conv(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def calculate_cdr(mask):
    """
    Calculate vertical Cup-to-Disc Ratio (CDR)
    
    Args:
        mask: Segmentation mask where 0=background, 1=disc, 2=cup
    
    Returns:
        float: Vertical CDR value
    """
    disc = (mask == 1).astype(np.uint8)
    cup = (mask == 2).astype(np.uint8)
    
    # Find contours
    d_contours, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_contours, _ = cv2.findContours(cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate vertical diameters
    v_disc = max([cv2.boundingRect(c)[3] for c in d_contours], default=0)
    v_cup = max([cv2.boundingRect(c)[3] for c in c_contours], default=0)
    
    # Return CDR
    return float(v_cup) / float(v_disc) if v_disc > 0 else 0.0


def create_overlay(image, mask):
    """
    Create visualization overlay with contours
    
    Args:
        image: Original RGB image (numpy array)
        mask: Segmentation mask
    
    Returns:
        numpy array: Image with overlaid contours
    """
    # Resize image to match mask
    image = cv2.resize(image, (224, 224))
    overlay = image.copy()
    
    # Colors: Yellow for disc, Green for cup
    disc_color = (0, 255, 255)  # Yellow in RGB
    cup_color = (0, 255, 0)     # Green in RGB
    
    # Extract binary masks
    disc_bin = (mask == 1).astype(np.uint8)
    cup_bin = (mask == 2).astype(np.uint8)
    
    # Find contours
    d_contours, _ = cv2.findContours(disc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c_contours, _ = cv2.findContours(cup_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    cv2.drawContours(overlay, d_contours, -1, disc_color, 2)
    cv2.drawContours(overlay, c_contours, -1, cup_color, 2)
    
    return overlay


def encode_image_base64(image):
    """
    Encode numpy image to base64 string for JSON response
    
    Args:
        image: RGB numpy array
    
    Returns:
        str: Base64-encoded JPEG image
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def simple_normalize(image):
    """
    Simple normalization for images
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        torch.Tensor: Normalized tensor
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize
    image = cv2.resize(image, (224, 224))
    
    # Convert to float32 and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    # Ensure float32 dtype
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image


# ================================================================
# LAZY MODEL LOADING
# ================================================================

segmentation_model = None
MODEL_PATH = "best_seg_model.pth"


def load_segmentation_model():
    """Lazy load segmentation model on first request"""
    global segmentation_model
    
    if segmentation_model is not None:
        return segmentation_model
    
    print("🔹 Loading segmentation model...")
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    # Initialize model
    segmentation_model = SegmentationModel(n_classes=3).to(DEVICE)
    
    # Load weights with float32 conversion
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    
    # Convert all tensors to float32 to avoid dtype mismatches
    state_dict = {k: v.float() if v.dtype == torch.float64 else v 
                  for k, v in state_dict.items()}
    
    segmentation_model.load_state_dict(state_dict)
    segmentation_model.eval()
    
    # Ensure model is in float32
    segmentation_model = segmentation_model.float()
    
    print("✅ Segmentation model loaded successfully!")
    print(f"   Model size: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} MB")
    return segmentation_model


# ================================================================
# FLASK APP
# ================================================================

app = Flask(__name__)


@app.route("/", methods=["GET", "HEAD"])
def home():
    """API information and health check"""
    return jsonify({
        "message": "Ophthalmology Segmentation API",
        "version": "1.0.0",
        "status": "active",
        "description": "Optic disc and cup segmentation with CDR calculation",
        "endpoints": {
            "/": "GET - API information",
            "/health": "GET - Health check",
            "/segment": "POST - Segment fundus image and calculate CDR"
        },
        "device": str(DEVICE),
        "model_loaded": segmentation_model is not None,
        "usage": {
            "endpoint": "/segment",
            "method": "POST",
            "body": "multipart/form-data with 'image' file",
            "returns": {
                "cdr": "Cup-to-Disc Ratio (float)",
                "diagnosis": "Glaucomatous or Healthy",
                "segmentation_overlay": "Base64-encoded image with contours"
            }
        }
    })


@app.route("/health", methods=["GET"])
def health():
    """Simple health check"""
    return jsonify({
        "status": "healthy",
        "device": str(DEVICE),
        "model_loaded": segmentation_model is not None
    })


@app.route("/segment", methods=["POST"])
def segment():
    """
    Main segmentation endpoint
    
    Expects:
        - 'image': Fundus image file (JPEG/PNG)
    
    Returns:
        - cdr: Vertical Cup-to-Disc Ratio
        - diagnosis: Glaucomatous (CDR >= 0.65) or Healthy
        - segmentation_overlay: Base64-encoded image with contours
        - interpretation: Detailed interpretation of CDR
    """
    try:
        # Validate request
        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded",
                "message": "Please upload an image file with key 'image'"
            }), 400

        file = request.files["image"]
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Load model (lazy loading)
        model = load_segmentation_model()
        
        # Read and preprocess image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(img)
        
        # Normalize and convert to tensor
        tensor = simple_normalize(img).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            logits = model(tensor)
            mask = torch.argmax(logits.squeeze(0), dim=0).cpu().numpy()
        
        # Calculate CDR
        cdr = calculate_cdr(mask)
        
        # Determine diagnosis
        diagnosis = "Glaucomatous" if cdr >= 0.65 else "Healthy"
        
        # Interpret CDR
        if cdr < 0.3:
            interpretation = "Very low CDR - Normal"
        elif cdr < 0.5:
            interpretation = "Low CDR - Normal"
        elif cdr < 0.65:
            interpretation = "Moderate CDR - Borderline, monitor closely"
        elif cdr < 0.8:
            interpretation = "High CDR - Likely glaucomatous"
        else:
            interpretation = "Very high CDR - Severe glaucoma suspected"
        
        # Create visualization
        overlay = create_overlay(img_np, mask)
        overlay_b64 = encode_image_base64(overlay)
        
        # Prepare response
        response = {
            "success": True,
            "task": "segmentation",
            "cdr": round(cdr, 3),
            "diagnosis": diagnosis,
            "interpretation": interpretation,
            "segmentation_overlay": overlay_b64,
            "details": {
                "optic_disc": "Segmented (yellow contour)",
                "optic_cup": "Segmented (green contour)",
                "threshold": 0.65,
                "note": "CDR >= 0.65 indicates potential glaucoma"
            }
        }
        
        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({
            "error": "Model file not found",
            "message": str(e),
            "solution": "Ensure 'best_seg_model.pth' is in the same directory"
        }), 500
    
    except Exception as e:
        return jsonify({
            "error": "Processing failed",
            "message": str(e),
            "type": type(e).__name__
        }), 500


@app.route("/info", methods=["GET"])
def model_info():
    """Get detailed model information"""
    model_loaded = segmentation_model is not None
    
    info = {
        "model_name": "ResNet50-UNet Segmentation",
        "model_file": MODEL_PATH,
        "model_loaded": model_loaded,
        "classes": {
            "0": "Background",
            "1": "Optic Disc",
            "2": "Optic Cup"
        },
        "input_size": "224x224 RGB",
        "output_size": "224x224 segmentation mask",
        "metrics": {
            "cdr_threshold": 0.65,
            "normal_range": "< 0.65",
            "glaucomatous_range": ">= 0.65"
        }
    }
    
    if os.path.exists(MODEL_PATH):
        info["model_size_mb"] = round(os.path.getsize(MODEL_PATH) / (1024*1024), 2)
    
    return jsonify(info)


# ================================================================
# ERROR HANDLERS
# ================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/segment", "/info"]
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": str(e)
    }), 500


@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({
        "error": "File too large",
        "message": "Maximum file size is 16MB"
    }), 413


# ================================================================
# RUN SERVER
# ================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("🚀 Ophthalmology Segmentation API Starting...")
    print(f"   Port: {port}")
    print(f"   Device: {DEVICE}")
    print(f"   Model: {MODEL_PATH}")
    print("   Mode: Lazy loading (model loads on first request)")
    print("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=False)