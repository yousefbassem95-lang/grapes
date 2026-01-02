# PROJECT GRAPES 🍇
**High-Fidelity ASCII Art Generator**

![Grapes Banner](https://img.shields.io/badge/Project-GRAPES-purple) ![Made By](https://img.shields.io/badge/Made_By-J0J0M0J0-blue)

>**"Engineering Art from Chaos."**

## 🖼️ Overview
**Grapes** is a powerful CLI engine capable of converting images into Detailed ASCII Art. It features an advanced "Retina" engine that uses `numpy` for pixel-perfect mapping and aspect ratio correction.

## ✨ Features
-   **Smart Resizing**: Automatically corrects for terminal character aspect ratio (0.55).
-   **TrueColor Mode**: Renders 24-bit ANSI colors for 1:1 image replication.
-   **Edge Detection**: Uses **Sobel Filters** to draw "Blueprint" style ASCII using directional characters (`| / - \`).
-   **Export**: Save outputs to text files effortlessly.

## 🛠️ Installation
```bash
cd Grapes_tool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 🍇 Usage

### Basic Grayscale
```bash
python grapes.py image.jpg
```

### TrueColor Mode (Vibrant)
```bash
python grapes.py image.jpg --color --width 120
```

### Edge Detection (Blueprint Style)
```bash
python grapes.py blueprint.jpg --edge --width 150
```

### Smart Focus (Remove Background)
Automatically dimmed background to highlight the subject.
```bash
python grapes.py portrait.jpg --focus --width 100
```

### Color Targeting (Intelligence Mode)
Isolate specific objects by color. Supported: red, green, blue, yellow, cyan, magenta, purple.
```bash
python grapes.py grapes.jpg --target-color purple --width 80
```

### Interactive Mode (The Sentinel)
Just run the tool without arguments for a guided targeting system.
```bash
python grapes.py
```

### Options
| Flag | Description |
|------|-------------|
| `--width` | Set output width (height calculated automatically) |
| `--color` | Enable ANSI TrueColor mode |
| `--edge` | Enable Edge Detection mode |
| `--focus` | Enable Smart Focus (Background Removal) |
| `--target-color` | Isolate specific color region |
| `--charset` | `standard`, `block`, or `complex` |
| `--output` | Save result to a file |

---
*Made with precision by **J0J0M0J0**.*
