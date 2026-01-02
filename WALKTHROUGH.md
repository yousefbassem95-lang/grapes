# PROJECT GRAPES 🍇 - The Walkthrough
**Operator:** J0J0M0J0

## 1. Setup
Ensure dependencies are installed:
```bash
cd Grapes_tool
pip install -r requirements.txt
```

## 2. Interactive Sentinel (Recommended)
This mode guides you through all options, including video source selection.
```bash
python grapes.py
```

## 3. Operations Manual

### Static Analysis (Images)
**Basic Scan:**
```bash
python grapes.py target.jpg
```

**Retina & Focus (Smart Focus):**
Removes background noise to focus on the subject.
```bash
python grapes.py target.jpg --focus
```

**Color Targeting:**
Isolate specific colors (e.g., Red, Purple, Green).
```bash
python grapes.py target.jpg --target-color purple
```

### The Matrix (Live Video) 🕶️
**Standard Feed:**
```bash
python grapes.py --video
```

**Hacker Vision (Edge Detection):**
See the world in blueprint code.
```bash
python grapes.py --video --edge
```

**Targeting Mode (Live):**
Hold up a red object and watch the machine ignore everything else.
```bash
python grapes.py --video --target-color red
```

## 4. Troubleshooting
-   **Webcam Error?**: Ensure no other apps are using the camera. Try `--webcam 1` if you have multiple.
-   **Slow Video?**: Reduce width with `--width 60`.

---
*Engineering Art from Chaos.*
