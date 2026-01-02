# 🍇 Grapes - Complete Walkthrough

## Table of Contents
1. [Installation](#installation)
2. [Basic Usage](#basic-usage)
3. [Rendering Modes](#rendering-modes)
4. [Advanced Features](#advanced-features)
5. [Interactive Wizard](#interactive-wizard)
6. [Video Streaming](#video-streaming)
7. [Troubleshooting](#troubleshooting)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/J0J0M0J0/grapes.git
cd grapes
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download AI Model (Required for Neural Brain)
```bash
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
```

### Step 5: Verify Installation
```bash
python grapes.py --help
```

---

## Basic Usage

### Convert an Image to ASCII
```bash
python grapes.py photo.jpg
```
**Output**: Standard ASCII art in your terminal

### Adjust Width
```bash
python grapes.py photo.jpg --width 120
```
**Output**: Wider ASCII art (more detail)

### Enable Color
```bash
python grapes.py photo.jpg --color
```
**Output**: Full 16.7M color ASCII art

### Save to File
```bash
python grapes.py photo.jpg -o output.txt
```
**Output**: ASCII art saved to `output.txt`

---

## Rendering Modes

### Portrait Mode (Recommended for People)
```bash
python grapes.py selfie.jpg --mode portrait
```
**What it does**:
- Activates Neural AI for subject detection
- Auto-crops to face/person
- Uses high-detail charset
- Removes background distractions

**Best for**: Headshots, selfies, portraits

### Neon Mode (Cyberpunk Aesthetic)
```bash
python grapes.py cityscape.jpg --mode neon
```
**What it does**:
- Boosts color saturation by 1.8x
- Enables HD sub-pixel rendering
- Uses hybrid block+ASCII charset
- Creates vibrant, glowing effect

**Best for**: Night scenes, neon signs, cyberpunk themes

### Matrix Mode (Green Hacker Style)
```bash
python grapes.py code.jpg --mode matrix
```
**What it does**:
- Renders everything in green gradient
- Uses complex charset for detail
- Classic Matrix aesthetic

**Best for**: Code screenshots, terminal art, hacker themes

### Thermal Mode (Heat Map)
```bash
python grapes.py person.jpg --mode thermal
```
**What it does**:
- Maps brightness to temperature colors
- Blue (cold) → Yellow (warm) → Red (hot)
- Uses block charset for smooth gradients

**Best for**: Artistic effects, data visualization

### Sketch Mode (Pencil Drawing)
```bash
python grapes.py portrait.jpg --mode sketch
```
**What it does**:
- Detects edges in the image
- Uses directional line characters (`/`, `\`, `-`, `|`)
- Creates pencil sketch effect

**Best for**: Portraits, architectural drawings

### Pixel Mode (Retro 8-Bit)
```bash
python grapes.py game.jpg --mode pixel
```
**What it does**:
- Forces low resolution (max 50 chars wide)
- Uses only block characters
- Quantizes colors for retro look

**Best for**: Game screenshots, retro art

---

## Advanced Features

### Hyper-Fidelity Mode (HD)
Doubles vertical resolution using sub-pixel rendering.

```bash
python grapes.py photo.jpg --hd --width 100
```

**Technical Details**:
- Uses `▀` (upper half block) character
- Renders 2 pixels per character cell
- Applies CLAHE auto-contrast enhancement
- Creates square pixel aspect ratio

**Example Output**: Near-photographic quality in terminal

### Neural AI Brain
AI-powered subject isolation using MediaPipe.

```bash
python grapes.py person.jpg --brain --focus
```

**What it does**:
1. Detects human subjects in the image
2. Creates segmentation mask
3. Isolates subject (mask > 0.5)
4. Renders background as void (empty space)

**Requirements**: `selfie_segmenter.tflite` model file

### Smart Zoom (Auto-Crop)
Automatically crops to the subject.

```bash
python grapes.py group_photo.jpg --crop
```

**Algorithm**:
1. Runs Neural AI to detect subject
2. Calculates bounding box of mask
3. Adds 10% padding
4. Crops original image
5. Renders cropped region

**Result**: Subject fills entire terminal width (maximum detail)

### Color Isolation
Extract and highlight specific colors.

```bash
python grapes.py flower.jpg --target-color purple
```

**Available Colors**:
- `red`, `green`, `blue`
- `yellow`, `cyan`, `magenta`, `purple`

**Use Case**: Highlight specific objects by color

### Edge Detection
Blueprint-style line art.

```bash
python grapes.py building.jpg --edge
```

**Algorithm**:
- Calculates image gradients
- Maps gradient angles to directional characters
- Creates architectural blueprint effect

---

## Interactive Wizard

### Launch the Wizard
```bash
python grapes.py
```

### Wizard Flow

**Step 1: Source Selection**
```
Analysis Source [image/video] (image): image
Enter image path: photo.jpg
```

**Step 2: Subject Analysis**
```
What is the primary focus? [human/structure/detail/color] (human): human
```
- `human` → Activates Neural AI
- `structure` → Optimizes for buildings/objects
- `detail` → Enables HD mode
- `color` → Enables color isolation

**Step 3: Aesthetic Profiling**
```
Select Render Protocol [standard/cyberpunk/high-fidelity/blueprint/retro] (standard): cyberpunk
```
- `standard` → Basic ASCII
- `cyberpunk` → Neon + Hybrid charset
- `high-fidelity` → HD + Complex charset
- `blueprint` → Edge detection
- `retro` → Block charset + color

**Step 4: Zoom Level**
```
Zoom Level [full/smart-crop] (full): smart-crop
```
- `full` → Render entire image
- `smart-crop` → Auto-crop to subject

**Step 5: Neural Calibration**
The wizard simulates "deep thinking" and auto-configures optimal settings.

---

## Video Streaming

### Basic Webcam Stream
```bash
python grapes.py --video
```
**Output**: Real-time ASCII video from webcam 0

### HD Webcam with AI
```bash
python grapes.py --video --hd --brain --crop
```
**Output**: High-fidelity, AI-cropped live stream

### Select Webcam
```bash
python grapes.py --video --webcam 1
```
**Use Case**: Multiple cameras (0, 1, 2, etc.)

### Stop Stream
Press `Ctrl+C` to terminate

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'mediapipe'"
**Solution**:
```bash
pip install mediapipe
```

### "FileNotFoundError: selfie_segmenter.tflite"
**Solution**:
```bash
wget https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite
```

### "Could not open video source 0"
**Solution**:
- Check webcam permissions
- Try different webcam ID: `--webcam 1`
- Verify camera is not in use by another app

### Background Still Showing (Commas/Dots)
**Solution**:
- Ensure `--brain` or `--crop` is enabled
- Use `--mode portrait` for automatic configuration
- Check that `selfie_segmenter.tflite` is in the current directory

### Colors Not Showing
**Solution**:
- Add `--color` flag
- Verify terminal supports TrueColor (most modern terminals do)
- Try a different terminal emulator

### Image Too Small/Large
**Solution**:
- Adjust `--width` parameter
- For HD mode: `--width 120` (recommended)
- For standard: `--width 80-100`

---

## Command Cheat Sheet

### Quick Reference
```bash
# Portrait with AI
python grapes.py photo.jpg --mode portrait

# Cyberpunk neon
python grapes.py city.jpg --mode neon

# Matrix green
python grapes.py code.jpg --mode matrix

# Thermal heat map
python grapes.py person.jpg --mode thermal

# Pencil sketch
python grapes.py face.jpg --mode sketch

# Retro pixel art
python grapes.py game.jpg --mode pixel

# Manual HD + AI + Crop
python grapes.py photo.jpg --hd --brain --crop --width 120

# Live webcam portrait
python grapes.py --video --mode portrait

# Save colored output
python grapes.py photo.jpg --mode neon -o neon_art.txt
```

---

## Tips & Best Practices

1. **For Portraits**: Always use `--mode portrait` or `--brain --crop`
2. **For HD Quality**: Combine `--hd` with `--width 120+`
3. **For Speed**: Use standard mode without `--brain` or `--hd`
4. **For Artistic Effects**: Experiment with `--mode thermal` or `--mode matrix`
5. **For Webcam**: Start with `--video --mode portrait` for best results

---

**Created by J0J0M0J0** 🍇
