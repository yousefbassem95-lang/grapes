#!/usr/bin/env python3
import argparse
import sys
import time
import numpy as np
from PIL import Image
from rich.console import Console
from rich.prompt import Prompt

console = Console()

# --- Branding ---
BANNER = r"""
              #####                    ****                     #####   
              #######                 ******                   ######             
          ####   ##########           ******             ##########  ###          
         #####           #######       ****        ########         #####        
         #####                #####             #####               #####        
           #####################  #   #####     # ######################         
                              ###   #########   ###                            
                       #######    ############      ######                       
                     ##########   #############    #########                     
                   ###########    #############    ###########                   
                 ######       ##  ############# ##         ######                 
               #####        #####  ###########  ####         #####               
             ****#        *******#  #########   #******         #****             
           ****         **+++***     ######      ***+++***        ****           
         ***          *++++*+  ***  *+++++++*  *** ++++++**          ***         
       **           *+++++    *+++  ++++++++   +++*    ++++++*          **       
                  +++++       +=+    +=====+    +=+      +++++*                  
                ++++         +=+     +=====+     +=+        ++++*                
              ++==          +=+      +=====+      +=+          ==++              
            ++=            ===       =----=        ===            ==+            
                           ==        =----=         ==              =++         
                          ==         =----=          ==                          
                         ==          =-::-=           ==                         
                         =            =::=             =                         
                        =             =::=              =                        
                       -              =..=               -                       
                      :               :.:                 :                       
                                      :::                                       
                                      :::                                        
                                      ---                                        
                                      |||
    
  ██████╗ ██████╗  █████╗ ██████╗ ███████╗███████╗
 ██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝
 ██║  ███╗██████╔╝███████║██████╔╝█████╗  ███████╗
 ██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══╝  ╚════██║
 ╚██████╔╝██║  ██║██║  ██║██║     ███████╗███████║
  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝
            High-Fidelity ASCII Engine
"""

# --- Configuration & Charsets ---
# Standard ASCII ramp (High density -> Low density)
CHARSET_STANDARD = r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
# Simple block set
CHARSET_BLOCKS = "█▓▒░ "
# Detailed specialized set
CHARSET_COMPLEX = r"$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
# Hybrid set (Blocks for weight, ASCII for detail)
CHARSET_HYBRID = r"█▓▒░$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# Terminal character aspect ratio (Height is usually ~2x Width)
# Adjusting output height by this factor prevents squashed images.
FONT_ASPECT_RATIO = 0.55

class ImageProcessor:
    @staticmethod
    def load_image(path: str) -> Image.Image:
        try:
            return Image.open(path)
        except Exception as e:
            console.print(f"[bold red]Error loading image: {e}[/bold red]")
            sys.exit(1)

    @staticmethod
    def resize_image(image: Image.Image, new_width: int) -> Image.Image:
        width, height = image.size
        # Calculate new height based on aspect ratio correction (Standard ASCII)
        new_height = int(height * (new_width / width) * FONT_ASPECT_RATIO)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def resize_hd(image: Image.Image, new_width: int) -> Image.Image:
        """Resize for square-pixel aspect ratio (HD Mode)."""
        width, height = image.size
        # No aspect ratio correction needed for subpixel rendering (it creates square cells)
        new_height = int(height * (new_width / width))
        # Ensure height is even for row pairing
        if new_height % 2 != 0:
            new_height -= 1
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    @staticmethod
    def apply_clahe(image: Image.Image) -> Image.Image:
        """Applies Contrast Limited Adaptive Histogram Equalization."""
        try:
            import cv2
            img_np = np.array(image.convert("RGB"))
            
            # Convert to LAB to process Lightness channel
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            limg = cv2.merge((cl, a, b))
            final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(final)
            return image
        except Exception:
            return image

    @staticmethod
    def smart_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
        """Crops the image to the bounding box of the non-zero mask entries with padding."""
        if mask is None: return image
        
        # Find indices of the subject
        rows = np.any(mask > 0, axis=1)
        cols = np.any(mask > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return image # Mask empty, return full image
            
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # Padding (10%)
        h, w = mask.shape
        pad_y = int(h * 0.1)
        pad_x = int(w * 0.1)
        
        ymin = max(0, ymin - pad_y)
        ymax = min(h, ymax + pad_y)
        xmin = max(0, xmin - pad_x)
        xmax = min(w, xmax + pad_x)
        
        # Crop
        return image.crop((xmin, ymin, xmax, ymax))

class ASCIIEngine:
    @staticmethod
    def map_pixels_to_ascii(image: Image.Image, charset: str) -> str:
        """Converts resized image to Grayscale ASCII string."""
        grayscale_image = image.convert("L")
        pixels = np.array(grayscale_image)
        # Invert logic for Dark Terminal: 255 (White) -> Dense (Index 0), 0 (Black) -> Space (Index -1)
        indices = ((255 - pixels) / 255 * (len(charset) - 1)).astype(int)
        
        lines = []
        chars = np.array(list(charset))
        ascii_grid = chars[indices]
        
        for row in ascii_grid:
            lines.append("".join(row))
            
        return "\n".join(lines)

    @staticmethod
    def render_truecolor(image: Image.Image, charset: str, mask: np.ndarray = None) -> str:
        """Converts resized image to ANSI TrueColor ASCII string. Optional mask for transparency."""
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        grayscale_image = image.convert("L")
        pixels_gray = np.array(grayscale_image)
        pixels_rgb = np.array(image)
        
        # Invert logic for Dark Terminal: 255 (White) -> Dense (Index 0), 0 (Black) -> Space (Index -1)
        indices = ((255 - pixels_gray) / 255 * (len(charset) - 1)).astype(int)
        chars = np.array(list(charset))
        ascii_chars = chars[indices]
        
        output_lines = []
        height, width, _ = pixels_rgb.shape
        
        for y in range(height):
            line_buffer = []
            for x in range(width):
                # Apply mask if exists
                if mask is not None:
                    if mask[y, x] < 0.5: # Absolute Isolation Threshold
                        # Force Empty Space (No Commas, No Dimming)
                        line_buffer.append(" ") 
                        continue
                    else:
                        r, g, b = pixels_rgb[y, x]
                else:
                    r, g, b = pixels_rgb[y, x]
                    
                char = ascii_chars[y, x]
                line_buffer.append(f"\x1b[38;2;{r};{g};{b}m{char}")
            line_buffer.append("\x1b[0m")
            output_lines.append("".join(line_buffer))
            
        return "\n".join(output_lines)

    @staticmethod
    def render_subpixel(image: Image.Image) -> str:
        """Renders image using subpixel block characters for 2x vertical resolution. Assumes square pixel sizing."""
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixels = np.array(image)
        height, width, _ = pixels.shape
        
        output_lines = []
        
        # Iterate y in steps of 2
        for y in range(0, height - 1, 2):
            line_buffer = []
            for x in range(width):
                # Top Pixel (Foreground)
                r1, g1, b1 = pixels[y, x]
                # Bottom Pixel (Background)
                r2, g2, b2 = pixels[y+1, x]
                
                # Upper Half Block ▀: FG color is top half, BG color is bottom half.
                # ANSI: \033[38;2;R;G;Bm (FG) \033[48;2;R;G;Bm (BG) ▀
                line_buffer.append(f"\x1b[38;2;{r1};{g1};{b1};48;2;{r2};{g2};{b2}m▀")
            
            line_buffer.append("\x1b[0m")
            output_lines.append("".join(line_buffer))
            
        return "\n".join(output_lines)
        
    @staticmethod
    def render_focus_mode(image: Image.Image, charset: str) -> str:
        """Removes background by focusing on high-detail areas (edges + density)."""
        from PIL import ImageFilter
        
        # 1. Calculate Edge Density (Energy Map)
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        mask_img = edges.filter(ImageFilter.GaussianBlur(radius=3))
        
        mask = np.array(mask_img, dtype=float) / 255.0
        mask = (mask - 0.15) * 3 
        mask = np.clip(mask, 0, 1)
        
        return ASCIIEngine.render_truecolor(image, charset, mask=mask)

    @staticmethod
    def render_color_segmentation(image: Image.Image, charset: str, target_color: str) -> str:
        """Focuses on a specific color (Red, Green, Blue, Yellow, Cyan, Magenta)."""
        import colorsys
        
        # Define HSV targets (Hue 0-1)
        # Red: 0.0 or 1.0, Green: 0.33, Blue: 0.66, Yellow: 0.16, Cyan: 0.5, Magenta: 0.83
        targets = {
            "red": 0.0,
            "green": 0.33,
            "blue": 0.66,
            "yellow": 0.16,
            "cyan": 0.5,
            "magenta": 0.83,
            "purple": 0.75 # Bonus
        }
        
        if target_color.lower() not in targets:
            console.print(f"[yellow]Warning: Color '{target_color}' not found. Defaulting to Red.[/yellow]")
            target_h = 0.0
        else:
            target_h = targets[target_color.lower()]

        if image.mode != "RGB":
            image = image.convert("RGB")
            
        pixels = np.array(image) / 255.0
        height, width, _ = pixels.shape
        
        # Vectorized RGB -> HSV is complex in pure numpy without heavy libs like scikit-image.
        # Stick to a fast iteration or simple numpy math for Hue.
        # Simple Hue calc:
        r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
        mx = np.max(pixels, axis=2)
        mn = np.min(pixels, axis=2)
        df = mx - mn
        
        # Avoid division by zero
        df[df == 0] = 0.00001
        
        # Calculate Hue
        h = np.zeros_like(mx)
        
        # If max is Red
        mask_r = (mx == r)
        h[mask_r] = (g[mask_r] - b[mask_r]) / df[mask_r]
        
        # If max is Green
        mask_g = (mx == g)
        h[mask_g] = (b[mask_g] - r[mask_g]) / df[mask_g] + 2.0
        
        # If max is Blue
        mask_b = (mx == b)
        h[mask_b] = (r[mask_b] - g[mask_b]) / df[mask_b] + 4.0
        
        h = (h / 6.0) % 1.0
        
        # S & V
        # s = np.zeros_like(mx)
        # mask_nz = (mx != 0)
        # s[mask_nz] = df[mask_nz] / mx[mask_nz]
        v = mx
        
        # Calculate Distance to Target Hue
        # Circular distance: min(|a-b|, 1-|a-b|)
        diff = np.abs(h - target_h)
        dist = np.minimum(diff, 1.0 - diff)
        
        # Create Mask
        # If distance < threshold (e.g. 0.1) AND Value > 0.2 (not black) AND Saturation (approx by df) > 0.1 (not grey)
        # Using df (delta) as proxy for saturation/chroma
        
        mask = np.zeros_like(mx)
        
        # Thresholds
        HUE_THRESH = 0.10  # +/- 10% of color wheel
        SAT_THRESH = 0.15  # Must have some color
        VAL_THRESH = 0.20  # Must be visible
        
        selected = (dist < HUE_THRESH) & (df > SAT_THRESH) & (v > VAL_THRESH)
        mask[selected] = 1.0
        
        # Soften edges? simple clip
        
        return ASCIIEngine.render_truecolor(image, charset, mask=mask)

    @staticmethod
    def render_edge_detection(image: Image.Image) -> str:
        """Uses Numpy gradients to map high-gradient edges to directional characters."""
        
        # 1. Convert to grayscale numpy array
        if image.mode != "L":
            image = image.convert("L")
        pixels = np.array(image, dtype=float)

        # 2. Apply Gradients using Numpy
        grad_y, grad_x = np.gradient(pixels)
        
        magnitude = np.hypot(grad_x, grad_y)
        angle = np.arctan2(grad_y, grad_x)
        
        # Normalize magnitude
        max_mag = np.max(magnitude)
        if max_mag == 0: max_mag = 1
        magnitude = magnitude / max_mag
        
        output_lines = []
        height, width = pixels.shape
        
        # Threshold for considering something an edge
        EDGE_THRESHOLD = 0.15
        
        for y in range(height):
            line_buffer = []
            for x in range(width):
                mag = magnitude[y, x]
                if mag > EDGE_THRESHOLD:
                    # Map angle to char
                    # Angles are -pi to pi
                    ang = angle[y, x]
                    # -pi .. pi mapped to directions
                    if -np.pi/8 <= ang < np.pi/8:
                        char = "-" # Horizontal
                    elif np.pi/8 <= ang < 3*np.pi/8:
                        char = "\\" # Diagonal Down-Right
                    elif 3*np.pi/8 <= ang < 5*np.pi/8:
                        char = "|" # Vertical
                    elif 5*np.pi/8 <= ang < 7*np.pi/8:
                        char = "/" # Diagonal Down-Left
                    elif -3*np.pi/8 <= ang < -np.pi/8:
                        char = "/" # Diagonal Up-Right
                    elif -5*np.pi/8 <= ang < -3*np.pi/8:
                        char = "|" # Vertical
                    elif -7*np.pi/8 <= ang < -5*np.pi/8:
                        char = "\\" # Diagonal Up-Left
                    else:
                        char = "-" # Horizontal
                else:
                    # Non-edge: Contextual
                    lum = pixels[y, x]
                    if lum < 50: char = "@"
                    elif lum < 100: char = "%"
                    elif lum < 150: char = ":"
                    else: char = " "
                
                line_buffer.append(char)
            output_lines.append("".join(line_buffer))
            
        return "\n".join(output_lines)

    @staticmethod
    def render_matrix(image: Image.Image, charset: str) -> str:
        """Matrix-style green-on-black rendering."""
        grayscale_image = image.convert("L")
        pixels = np.array(grayscale_image)
        indices = ((255 - pixels) / 255 * (len(charset) - 1)).astype(int)
        
        output_lines = []
        chars = np.array(list(charset))
        ascii_grid = chars[indices]
        
        for row in ascii_grid:
            line_buffer = []
            for char in row:
                if char == ' ':
                    line_buffer.append(' ')
                else:
                    # Green gradient
                    intensity = min(255, max(50, int(ord(char) % 150 + 100)))
                    line_buffer.append(f"\x1b[38;2;0;{intensity};0m{char}")
            line_buffer.append("\x1b[0m")
            output_lines.append("".join(line_buffer))
        
        return "\n".join(output_lines)
    
    @staticmethod
    def render_thermal(image: Image.Image, charset: str) -> str:
        """Thermal heat map rendering (Blue -> Red)."""
        grayscale_image = image.convert("L")
        pixels = np.array(grayscale_image)
        indices = ((255 - pixels) / 255 * (len(charset) - 1)).astype(int)
        
        output_lines = []
        chars = np.array(list(charset))
        ascii_grid = chars[indices]
        
        for y, row in enumerate(ascii_grid):
            line_buffer = []
            for x, char in enumerate(row):
                if char == ' ':
                    line_buffer.append(' ')
                else:
                    # Map brightness to temperature color
                    temp = pixels[y, x]
                    if temp < 85:
                        # Cold (Blue)
                        r, g, b = 0, int(temp * 1.5), 255
                    elif temp < 170:
                        # Warm (Yellow)
                        r, g, b = int((temp - 85) * 3), 255, int(255 - (temp - 85) * 3)
                    else:
                        # Hot (Red)
                        r, g, b = 255, int(255 - (temp - 170) * 3), 0
                    
                    line_buffer.append(f"\x1b[38;2;{r};{g};{b}m{char}")
            line_buffer.append("\x1b[0m")
            output_lines.append("".join(line_buffer))
        
        return "\n".join(output_lines)
    
    @staticmethod
    def render_neon(image: Image.Image, charset: str) -> str:
        """Neon cyberpunk rendering with boosted saturation."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Boost saturation
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.8)
        
        # Use standard truecolor rendering
        return ASCIIEngine.render_truecolor(image, charset)

class NeuralEngine:
    segmenter = None
    
    @staticmethod
    def segment_body(image: Image.Image) -> np.ndarray:
        """Uses MediaPipe Tasks API for AI Selfie Segmentation."""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            console.print("[bold red]Error: mediapipe not installed.[/bold red]")
            return None

        if NeuralEngine.segmenter is None:
            try:
                base_options = python.BaseOptions(model_asset_path='selfie_segmenter.tflite')
                options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
                NeuralEngine.segmenter = vision.ImageSegmenter.create_from_options(options)
            except Exception as e:
                console.print(f"[bold red]Error loading AI model: {e}[/bold red]")
                return None
        
        # Convert PIL to MP Image
        # Ensure RGB
        if image.mode != "RGB":
             image = image.convert("RGB")
             
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(image))
        
        segmentation_result = NeuralEngine.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask
        
        # Convert to numpy
        mask_np = category_mask.numpy_view()
        
        # Mask: 0=Background, >0=Person
        final_mask = (mask_np > 0).astype(float)
        
        return final_mask

class VideoEngine:
    @staticmethod
    def stream_video(source_id: int, width: int, mode: str, charset: list, target_color: str = None, use_brain: bool = False, hd: bool = False, crop: bool = False):
        """Streams video from webcam and renders ASCII in real-time."""
        try:
            import cv2
        except ImportError:
            console.print("[bold red]Error: opencv-python not installed. Run 'pip install opencv-python'[/bold red]")
            return

        cap = cv2.VideoCapture(source_id)
        if not cap.isOpened():
            console.print(f"[bold red]Error: Could not open video source {source_id}[/bold red]")
            return

        console.print("[bold green]Stream Active. Press Ctrl+C to stop.[/bold green]")
        
        try:
            # Hide cursor
            print("\033[?25l", end="")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert BGR (OpenCV) to RGB (PIL)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize
                # If CROP is active, we need to mask FIRST (at low res for speed) then Crop High Res
                
                if crop and use_brain:
                     # 1. Low res mask for speed/bbox
                     temp_small = ImageProcessor.resize_image(img, 200)
                     mask_small = NeuralEngine.segment_body(temp_small)
                     
                     if mask_small is not None:
                         # 2. Crop original image
                         # We need to scale bbox from 200 to original
                         # Actually, NeuralEngine.segment_body works fast on reasonable sizes. 
                         # Let's try segmentation on a mid-sized copy (e.g. 300px) then crop original.
                         scale_w = 300
                         scale_factor = img.width / scale_w
                         
                         temp_scale = ImageProcessor.resize_image(img, scale_w)
                         mask_scale = NeuralEngine.segment_body(temp_scale)
                         
                         if mask_scale is not None:
                             # Calc BBox on mask_scale
                             rows = np.any(mask_scale > 0, axis=1)
                             cols = np.any(mask_scale > 0, axis=0)
                             if np.any(rows) and np.any(cols):
                                ymin, ymax = np.where(rows)[0][[0, -1]]
                                xmin, xmax = np.where(cols)[0][[0, -1]]
                                
                                # Scale up to original
                                ymin = int(ymin * (img.height / temp_scale.height))
                                ymax = int(ymax * (img.height / temp_scale.height))
                                xmin = int(xmin * scale_factor)
                                xmax = int(xmax * scale_factor)
                                
                                # Padding
                                pad_y = int(img.height * 0.1)
                                pad_x = int(img.width * 0.1)
                                ymin = max(0, ymin - pad_y)
                                ymax = min(img.height, ymax + pad_y)
                                xmin = max(0, xmin - pad_x)
                                xmax = min(img.width, xmax + pad_x)
                                
                                img = img.crop((xmin, ymin, xmax, ymax))

                if hd:
                    img_resized = ImageProcessor.resize_hd(img, width)
                    # Auto-Contrast for HD
                    img_resized = ImageProcessor.apply_clahe(img_resized)
                else:
                    img_resized = ImageProcessor.resize_image(img, width)
                
                # Pre-calculate Mask if needed for HD or if Brain is active
                mask = None
                if use_brain:
                     mask = NeuralEngine.segment_body(img_resized)
                elif mode == "focus":
                     # Mathematical Focus Mask Calc (Need to extract mask logic if we want to separate it)
                     # For now, if HD + Focus (Math), we might rely on the old method or implement a mask extractor.
                     # Let's support Brain+HD fully. Math+HD might be tricky without refactor.
                     # We'll skip Math mask for HD for now and stick to Brain for HD focus.
                     pass 
                
                if hd:
                     # HD Mode Rendering
                     # Apply Mask if exists (Brain)
                     if mask is not None:
                         # Apply mask to image (Dim background)
                         img_np = np.array(img_resized)
                         if mask.ndim == 2:
                             mask = mask[..., np.newaxis]
                             
                         bg_dimmed = (img_np * 0.3).astype(np.uint8) # Dimmed BG
                         
                         # Combine
                         final_np = (img_np * mask + bg_dimmed * (1 - mask)).astype(np.uint8)
                         img_resized = Image.fromarray(final_np)
                         
                     result = ASCIIEngine.render_subpixel(img_resized)
                else:
                    # Standard Rendering Pipeline
                    # (Keep existing logic...)
                    if mode == "edge":
                        result = ASCIIEngine.render_edge_detection(img_resized)
                    elif mode == "focus":
                        if use_brain:
                            if mask is not None:
                                result = ASCIIEngine.render_truecolor(img_resized, "".join(charset), mask=mask)
                            else:
                                result = ASCIIEngine.render_focus_mode(img_resized, "".join(charset))
                        else:
                            result = ASCIIEngine.render_focus_mode(img_resized, "".join(charset))
                    elif mode == "color_target":
                         result = ASCIIEngine.render_color_segmentation(img_resized, "".join(charset), target_color)
                    elif mode == "truecolor":
                        result = ASCIIEngine.render_truecolor(img_resized, "".join(charset))
                    else:
                        if use_brain:
                             if mask is not None:
                                 result = ASCIIEngine.render_truecolor(img_resized, "".join(charset), mask=mask)
                             else:
                                 result = ASCIIEngine.map_pixels_to_ascii(img_resized, charset)
                        else:
                            result = ASCIIEngine.map_pixels_to_ascii(img_resized, charset)
                
                # Move cursor to top-left to overwrite
                print("\033[H" + result)
                
        except KeyboardInterrupt:
            pass
        finally:
            # Show cursor
            print("\033[?25h", end="")
            cap.release()
            console.print("\n[bold yellow]Stream Terminated.[/bold yellow]")

def run_wizard():
    """Runs the Deep Thinking interactive configuration wizard."""
    console.print("\n[bold cyan]Initiating Cognitive Interface...[/bold cyan]")
    time.sleep(0.5)

    # 1. Source Analysis
    source_type = Prompt.ask("Analysis Source", choices=["image", "video"], default="image")
    image_path = None
    if source_type == "image":
        image_path = Prompt.ask("Enter image path")
    
    # 2. Subject Focus (The "Deep Thinking" part)
    console.print("\n[bold]Semantic Analysis[/bold]")
    console.print("[dim]I need to understand the subject matter to optimize the neural weights.[/dim]")
    subject = Prompt.ask("What is the primary focus?", choices=["human", "structure", "detail", "color"], default="human")
    
    # 3. Aesthetic Profile
    console.print("\n[bold]Aesthetic Profiling[/bold]")
    style = Prompt.ask("Select Render Protocol", choices=["standard", "cyberpunk", "high-fidelity", "blueprint", "retro"], default="standard")
    
    # 4. Zoom Control
    zoom = Prompt.ask("Zoom Level", choices=["full", "smart-crop"], default="full")

    # Thinking Simulation
    console.print("")
    with console.status("[bold green]Calibrating Neural Link...[/bold green]"):
        time.sleep(0.8)
    with console.status("[bold green]Scanning Subject Boundaries...[/bold green]"): 
        if zoom == "smart-crop": time.sleep(0.8)
        else: time.sleep(0.2)
    with console.status("[bold green]Optimizing Charset Density...[/bold green]"):
        time.sleep(0.8)
    with console.status("[bold green]Configuring Render Engine...[/bold green]"):
        time.sleep(0.6)
        
    # Configuration Logic
    config = {
        "image_path": image_path,
        "video": (source_type == "video"),
        "width": 100,
        "mode": "standard",
        "brain": False,
        "hd": False,
        "case_crop": False,
        "edge": False,
        "formatted_charset": "standard", # 'standard', 'block', 'complex', 'hybrid'
        "target_color": None,
        "color": False,
        "focus": False
    }

    # Subject Logic
    if subject == "human":
        config["brain"] = True # Always use brain for humans
        config["focus"] = True # Default to focus for humans
    elif subject == "structure":
        config["edge"] = False 
    elif subject == "detail":
        config["hd"] = True
        config["width"] = 120
    elif subject == "color":
        config["color"] = True
        target = Prompt.ask("Specific color to isolate? (Leave empty for full color)", default="none")
        if target != "none":
            config["target_color"] = target
            config["mode"] = "color_target"
            
    if zoom == "smart-crop":
        config["case_crop"] = True
        config["brain"] = True # Need brain for crop
    
    # Style Logic (Overrides)
    if style == "cyberpunk":
        config["formatted_charset"] = "hybrid"
        config["color"] = True
        if not config["mode"] == "color_target":
             config["mode"] = "truecolor" # Cyberpunk needs color
    elif style == "high-fidelity":
        config["hd"] = True
        config["formatted_charset"] = "complex"
        config["width"] = max(config["width"], 120)
    elif style == "blueprint":
        config["mode"] = "edge"
        config["edge"] = True
    elif style == "retro":
        config["formatted_charset"] = "block"
        config["color"] = True
        
    console.print("[bold green]System Configured.[/bold green]\n")
    return config

def main():
    parser = argparse.ArgumentParser(description="Grapes: High-Fidelity ASCII Generator")
    parser.add_argument("image", nargs="?", help="Path to input image")
    parser.add_argument("--width", type=int, default=100, help="Output width in characters")
    parser.add_argument("--color", action="store_true", help="Enable TrueColor mode")
    parser.add_argument("--edge", action="store_true", help="Enable Edge Detection mode (Blueprint style)")
    parser.add_argument("--focus", action="store_true", help="Enable Focus Mode (Remove Background)")
    parser.add_argument("--video", action="store_true", help="Enable Real-Time Video Mode (Webcam)")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam ID (Default: 0)")
    parser.add_argument("--brain", action="store_true", help="Enable Neural Engine (AI Segmentation) [Requires mediapipe]")
    parser.add_argument("--target-color", help="Target color for segmentation (e.g., 'purple', 'red')")
    parser.add_argument("--charset", choices=["standard", "block", "complex", "hybrid"], default="standard", help="Character set style")
    parser.add_argument("--output", "-o", help="Save output to file")
    parser.add_argument("--hd", action="store_true", help="Enable Hyper-Fidelity Mode (Subpixel + Auto-Contrast)")
    parser.add_argument("--crop", action="store_true", help="Enable Smart Zoom (Auto-Crop to Subject)")
    parser.add_argument("--mode", choices=["portrait", "neon", "matrix", "thermal", "sketch", "pixel"], help="Preset rendering mode")
    
    args = parser.parse_args()
    
    # Branding
    if not args.hd:
        console.print(f"[bold magenta]{BANNER}[/bold magenta]")
    else:
        # Minimal header for HD to save space? Or just standard.
        console.print(f"[bold magenta]{BANNER}[/bold magenta]")
        
    console.print("[bold white]Created by J0J0M0J0[/bold white]")
    
    # Configuration State
    image_path = args.image
    width = args.width
    mode = "standard" # Internal mode tracker
    target_color = args.target_color
    is_video = args.video
    use_brain = args.brain
    hd_mode = args.hd
    use_crop = args.crop
    use_color = args.color
    charset_name = args.charset
    
    # Mode resolution from args
    if args.focus: mode = "focus"
    elif args.edge: mode = "edge"
    elif args.target_color: mode = "color_target"
    elif args.color: mode = "truecolor"
    
    # Auto-enable focus mode if crop is used (crop needs brain+mask)
    if args.crop:
        use_brain = True
        use_crop = True
        if mode == "standard":
            mode = "focus"
    
    # Apply Mode Presets
    if args.mode:
        if args.mode == "portrait":
            use_brain = True
            use_crop = True
            mode = "focus"
            charset_name = "complex"
        elif args.mode == "neon":
            use_color = True
            hd_mode = True
            charset_name = "hybrid"
            mode = "neon"
        elif args.mode == "matrix":
            charset_name = "complex"
            mode = "matrix"
        elif args.mode == "thermal":
            charset_name = "block"
            mode = "thermal"
        elif args.mode == "sketch":
            mode = "edge"
        elif args.mode == "pixel":
            charset_name = "block"
            use_color = True
            width = min(width, 50)

    # Wizard Trigger
    if not image_path and not args.video:
        # Run Wizard
        wiz_config = run_wizard()
        
        # Apply Wizard Config
        image_path = wiz_config["image_path"]
        is_video = wiz_config["video"]
        width = wiz_config["width"]
        use_brain = wiz_config["brain"]
        hd_mode = wiz_config["hd"]
        charset_name = wiz_config["formatted_charset"]
        
        # Mode Logic Merging
        if wiz_config["mode"] != "standard":
             mode = wiz_config["mode"]
        elif wiz_config["focus"]:
             mode = "focus"
        elif wiz_config["color"]:
             mode = "truecolor"
             
        if wiz_config["target_color"]:
            target_color = wiz_config["target_color"]
            mode = "color_target"
            
        if wiz_config["edge"]:
            mode = "edge"
            
        if wiz_config["case_crop"]:
            use_crop = True

    # Status Display
    tech_str = "Neural Network" if use_brain else "Standard Algorithm"
    if hd_mode: tech_str += " + Hyper-Fidelity"
    console.print(f"[dim]Mode: {mode.upper()} | Source: {'VIDEO' if is_video else 'IMAGE'} | Tech: {tech_str} | Width: {width}[/dim]\n")
    
    # Select Charset
    if charset_name == "block":
        selected_charset = CHARSET_BLOCKS
    elif charset_name == "complex":
        selected_charset = CHARSET_COMPLEX
    elif charset_name == "hybrid":
        selected_charset = CHARSET_HYBRID
    else:
        selected_charset = CHARSET_STANDARD

    # Process
    try:
        import cv2 # Ensure cv2 is available for NeuralEngine logic even in image mode
        
        if is_video:
            VideoEngine.stream_video(args.webcam, width, mode, selected_charset, target_color, use_brain, args.hd, use_crop)
        else:
            if not image_path:
                console.print("[red]Error: No image path provided.[/red]")
                return

            img = ImageProcessor.load_image(image_path)
            
            # Smart Zoom (Crop)
            if use_crop:
                 if not use_brain:
                     # Force activate brain for crop if not active
                     # (Assuming simple logic: needs mask)
                     use_brain = True
                     
                 # We need to render the mask first to find the crop
                 # NOTE: segment_body might be slow on full image 4K, so maybe resize first?
                 # But we want high res crop.
                 # Let's run segment on a scaled version to find bbox, then crop original.
                 w, h = img.size
                 scale_w = 400
                 if w > scale_w:
                     thumb = img.copy()
                     thumb.thumbnail((scale_w, scale_w))
                     mask_thumb = NeuralEngine.segment_body(thumb)
                     
                     if mask_thumb is not None:
                         # Calc bbox on thumb
                         rows = np.any(mask_thumb > 0, axis=1)
                         cols = np.any(mask_thumb > 0, axis=0)
                         
                         if np.any(rows) and np.any(cols):
                             ymin, ymax = np.where(rows)[0][[0, -1]]
                             xmin, xmax = np.where(cols)[0][[0, -1]]
                             
                             # Restore scale
                             scale = w / thumb.size[0]
                             ymin = int(ymin * scale)
                             ymax = int(ymax * scale)
                             xmin = int(xmin * scale)
                             xmax = int(xmax * scale)
                             
                             # Pad
                             pad_y = int(h * 0.1)
                             pad_x = int(w * 0.1)
                             ymin = max(0, ymin - pad_y)
                             ymax = min(h, ymax + pad_y)
                             xmin = max(0, xmin - pad_x)
                             xmax = min(w, xmax + pad_x)
                             
                             img = img.crop((xmin, ymin, xmax, ymax))
                             console.print(f"[dim]Auto-Cropped to Subject: {xmax-xmin}x{ymax-ymin}[/dim]")
                 else:
                     # small enough
                     mask = NeuralEngine.segment_body(img)
                     img = ImageProcessor.smart_crop(img, mask)

            if args.hd:
                img_resized = ImageProcessor.resize_hd(img, width)
                img_resized = ImageProcessor.apply_clahe(img_resized)
                
                if use_brain:
                     mask = NeuralEngine.segment_body(img_resized)
                     if mask is not None:
                         # Apply Void Logic for HD
                         img_np = np.array(img_resized)
                         if mask.ndim == 2:
                             mask = mask[..., np.newaxis]
                         
                         # Void Cutoff (0.5) -> Black
                         # Note: In HD, Black renders as Black Block. 
                         # Ideally it should be Space, but render_subpixel logic needs update for that.
                         # For now, Black is good enough for Dark Terminal.
                         cutoff = 0.5
                         void_idx = (mask < cutoff).squeeze()
                         img_np[void_idx] = [0, 0, 0]
                         
                         img_resized = Image.fromarray(img_np)
                
                result = ASCIIEngine.render_subpixel(img_resized)
            else:
                img_resized = ImageProcessor.resize_image(img, width)
                
                mask = None
                if use_brain:
                    mask = NeuralEngine.segment_body(img_resized)
                
                if mode == "edge":
                    result = ASCIIEngine.render_edge_detection(img_resized)
                elif mode == "matrix":
                    result = ASCIIEngine.render_matrix(img_resized, selected_charset)
                elif mode == "thermal":
                    result = ASCIIEngine.render_thermal(img_resized, selected_charset)
                elif mode == "neon":
                    result = ASCIIEngine.render_neon(img_resized, selected_charset)
                elif mode == "focus":
                    if use_brain:
                        mask = NeuralEngine.segment_body(img_resized)
                        if mask is not None:
                            result = ASCIIEngine.render_truecolor(img_resized, selected_charset, mask=mask)
                        else:
                            result = ASCIIEngine.render_focus_mode(img_resized, selected_charset)
                    else:
                        result = ASCIIEngine.render_focus_mode(img_resized, selected_charset)
                elif mode == "color_target":
                    result = ASCIIEngine.render_color_segmentation(img_resized, selected_charset, target_color)
                elif mode == "truecolor":
                    result = ASCIIEngine.render_truecolor(img_resized, selected_charset)
                else:
                    if use_brain:
                         # Standard mode + Brain essentially acts like Focus mode but with AI
                         mask = NeuralEngine.segment_body(img_resized)
                         if mask is not None:
                             result = ASCIIEngine.render_truecolor(img_resized, selected_charset, mask=mask)
                         else:
                             result = ASCIIEngine.map_pixels_to_ascii(img_resized, selected_charset)
                    else:
                        result = ASCIIEngine.map_pixels_to_ascii(img_resized, selected_charset)
            
            print(result)
            
            if args.output:
                with open(args.output, "w") as f:
                    f.write(result)
                console.print(f"[bold green]Saved to {args.output}[/bold green]")
            
    except KeyboardInterrupt:
        console.print("\n[red]Aborted.[/red]")
    except Exception as e:
        console.print(f"[bold red]Unexpected Error: {e}[/bold red]")

if __name__ == "__main__":
    main()
