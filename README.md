# 🍇 GRAPES
### **High-Fidelity ASCII Art Generator**
*Transform images and video into stunning terminal art with AI-powered precision*

---

## ✨ Features

- **🧠 Neural AI** - MediaPipe-powered subject isolation
- **🔬 Hyper-Fidelity Mode** - Sub-pixel rendering for 2x vertical resolution
- **✂️ Smart Zoom** - Auto-crop to subject with bounding box detection
- **🎨 6 Preset Modes** - Portrait, Neon, Matrix, Thermal, Sketch, Pixel
- **🎬 Real-Time Video** - Webcam ASCII streaming
- **🌈 TrueColor Support** - 16.7 million colors in your terminal
- **🎯 Advanced Rendering** - Edge detection, color isolation, focus modes

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/J0J0M0J0/grapes.git
cd grapes

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download AI model (required for Neural Brain)
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
```

### Basic Usage

```bash
# Simple ASCII conversion
python grapes.py photo.jpg

# High-fidelity with AI isolation
python grapes.py photo.jpg --mode portrait

# Matrix-style rendering
python grapes.py photo.jpg --mode matrix

# Real-time webcam
python grapes.py --video
```

---

## 🎨 Rendering Modes

### **Portrait Mode** (`--mode portrait`)
Optimized for human subjects with AI-powered cropping and high detail.
```bash
python grapes.py selfie.jpg --mode portrait
```
**Features**: Auto-crop, Neural AI, Complex charset

### **Neon Mode** (`--mode neon`)
Cyberpunk aesthetic with vibrant colors and HD rendering.
```bash
python grapes.py cityscape.jpg --mode neon
```
**Features**: Color boost (1.8x saturation), HD, Hybrid charset

### **Matrix Mode** (`--mode matrix`)
Classic green-on-black hacker aesthetic.
```bash
python grapes.py code.jpg --mode matrix
```
**Features**: Green gradient, Complex charset

### **Thermal Mode** (`--mode thermal`)
Heat map visualization (Blue → Yellow → Red).
```bash
python grapes.py landscape.jpg --mode thermal
```
**Features**: Temperature color mapping, Block charset

### **Sketch Mode** (`--mode sketch`)
Pencil drawing effect with edge detection.
```bash
python grapes.py portrait.jpg --mode sketch
```
**Features**: Directional hatching, Edge detection

### **Pixel Mode** (`--mode pixel`)
Retro 8-bit pixel art style.
```bash
python grapes.py game_screenshot.jpg --mode pixel
```
**Features**: Block charset, Color quantization, Low resolution

---

## 🛠️ Advanced Usage

### Manual Configuration

```bash
# HD Mode + Neural AI + Smart Crop
python grapes.py photo.jpg --hd --brain --crop --width 120

# Color isolation (extract specific color)
python grapes.py flower.jpg --target-color purple

# Edge detection blueprint
python grapes.py building.jpg --edge

# Custom charset
python grapes.py photo.jpg --charset hybrid --color
```

### Video Streaming

```bash
# Basic webcam stream
python grapes.py --video

# HD webcam with AI focus
python grapes.py --video --hd --brain --crop

# Webcam ID selection
python grapes.py --video --webcam 1
```

### Save Output

```bash
# Save to file
python grapes.py photo.jpg --mode neon -o output.txt

# Redirect to file (preserves ANSI colors)
python grapes.py photo.jpg --mode matrix > matrix_art.ans
```

---

## 🧙 Interactive Wizard

Run without arguments to launch the **Cognitive Interface**:

```bash
python grapes.py
```

The wizard will guide you through:
1. **Source**: Image or Video?
2. **Subject**: Human, Structure, Detail, Color?
3. **Aesthetic**: Standard, Cyberpunk, High-Fidelity, Blueprint, Retro?
4. **Zoom**: Full Scene or Smart-Crop?

The AI will auto-configure optimal settings based on your choices.

---

## 📋 Command Reference

### Core Arguments
| Flag | Description |
|------|-------------|
| `image` | Path to input image |
| `--width` | Output width in characters (default: 100) |
| `--mode` | Preset mode: `portrait`, `neon`, `matrix`, `thermal`, `sketch`, `pixel` |

### Rendering Modes
| Flag | Description |
|------|-------------|
| `--color` | Enable TrueColor (16.7M colors) |
| `--hd` | Hyper-Fidelity Mode (sub-pixel rendering) |
| `--edge` | Edge detection (blueprint style) |
| `--focus` | Background removal (mathematical) |
| `--brain` | Neural AI segmentation (MediaPipe) |
| `--crop` | Smart Zoom (auto-crop to subject) |

### Charsets
| Flag | Description |
|------|-------------|
| `--charset standard` | Default ASCII ramp (70+ chars) |
| `--charset block` | Block characters (`█▓▒░`) |
| `--charset complex` | High-detail ASCII |
| `--charset hybrid` | Blocks + ASCII fusion |

### Video
| Flag | Description |
|------|-------------|
| `--video` | Enable webcam mode |
| `--webcam N` | Select webcam ID (default: 0) |

### Color Isolation
| Flag | Description |
|------|-------------|
| `--target-color COLOR` | Isolate specific color: `red`, `green`, `blue`, `purple`, `yellow`, `cyan`, `magenta` |

---

## 🎯 Examples

### Example 1: Professional Headshot
```bash
python grapes.py headshot.jpg --mode portrait --width 100
```
**Output**: AI-cropped, high-detail portrait with clean background isolation.

### Example 2: Cyberpunk Cityscape
```bash
python grapes.py city.jpg --mode neon --hd --width 150
```
**Output**: Vibrant neon colors with sub-pixel HD rendering.

### Example 3: Code Matrix Effect
```bash
python grapes.py code_editor.png --mode matrix --width 80
```
**Output**: Green-on-black Matrix-style rendering.

### Example 4: Thermal Camera Simulation
```bash
python grapes.py person.jpg --mode thermal --width 100
```
**Output**: Blue-to-red heat map visualization.

### Example 5: Live Webcam Portrait
```bash
python grapes.py --video --mode portrait
```
**Output**: Real-time AI-powered portrait stream.

---

## 🧬 Technical Details

### Neural AI (MediaPipe)
- **Model**: Selfie Segmenter (Float16)
- **Purpose**: Human subject isolation
- **Accuracy**: 95%+ on frontal portraits
- **Speed**: ~50ms per frame (CPU)

### Hyper-Fidelity Mode
- **Technique**: Sub-pixel rendering using `▀` (upper half block)
- **Resolution**: 2x vertical density
- **Aspect Ratio**: Square pixels (1:1)
- **Enhancement**: CLAHE auto-contrast

### Smart Zoom
- **Algorithm**: Bounding box detection from AI mask
- **Padding**: 10% border
- **Scaling**: Proportional to mask dimensions

---

## 📦 Requirements

```
Pillow>=10.0.0
numpy>=1.24.0
rich>=13.0.0
opencv-python>=4.8.0
mediapipe>=0.10.0
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👤 Creator

**J0J0M0J0**

Cybersecurity researcher and terminal artist.

---

## 🌟 Showcase

*Add your ASCII art creations here!*

```
Submit your renders via PR to be featured in the showcase.
```

---

**⭐ Star this repo if you found it useful!**
# grapes
# grapes
