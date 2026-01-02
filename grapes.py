#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from PIL import Image
from rich.console import Console

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
        # Calculate new height based on aspect ratio correction
        new_height = int(height * (new_width / width) * FONT_ASPECT_RATIO)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

class ASCIIEngine:
    @staticmethod
    def map_pixels_to_ascii(image: Image.Image, charset: str) -> str:
        """Converts resized image to Grayscale ASCII string."""
        grayscale_image = image.convert("L")
        pixels = np.array(grayscale_image)
        indices = (pixels / 255 * (len(charset) - 1)).astype(int)
        
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
        
        indices = (pixels_gray / 255 * (len(charset) - 1)).astype(int)
        chars = np.array(list(charset))
        ascii_chars = chars[indices]
        
        output_lines = []
        height, width, _ = pixels_rgb.shape
        
        for y in range(height):
            line_buffer = []
            for x in range(width):
                # Apply mask if exists
                if mask is not None:
                    if mask[y, x] < 0.2: # Threshold for "Background"
                        line_buffer.append(" ") # Transparent
                        continue
                    elif mask[y, x] < 0.4:
                        # Faint / Transition
                        r, g, b = (pixels_rgb[y, x] * 0.5).astype(int) # Dimmed
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

class VideoEngine:
    @staticmethod
    def stream_video(source_id: int, width: int, mode: str, charset: list, target_color: str = None):
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
                img_resized = ImageProcessor.resize_image(img, width)
                
                # Render
                if mode == "edge":
                    result = ASCIIEngine.render_edge_detection(img_resized)
                elif mode == "focus":
                    result = ASCIIEngine.render_focus_mode(img_resized, "".join(charset))
                elif mode == "color_target":
                    result = ASCIIEngine.render_color_segmentation(img_resized, "".join(charset), target_color)
                elif mode == "truecolor":
                    result = ASCIIEngine.render_truecolor(img_resized, "".join(charset))
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

def main():
    from rich.prompt import Prompt

    parser = argparse.ArgumentParser(description="Grapes: High-Fidelity ASCII Generator")
    parser.add_argument("image", nargs="?", help="Path to input image")
    parser.add_argument("--width", type=int, default=100, help="Output width in characters")
    parser.add_argument("--color", action="store_true", help="Enable TrueColor mode")
    parser.add_argument("--edge", action="store_true", help="Enable Edge Detection mode (Blueprint style)")
    parser.add_argument("--focus", action="store_true", help="Enable Focus Mode (Remove Background)")
    parser.add_argument("--video", action="store_true", help="Enable Real-Time Video Mode (Webcam)")
    parser.add_argument("--webcam", type=int, default=0, help="Webcam ID (Default: 0)")
    parser.add_argument("--charset", choices=["standard", "block", "complex"], default="standard", help="Character set style")
    parser.add_argument("--output", "-o", help="Save output to file")
    
    args = parser.parse_args()
    
    # Branding
    console.print(f"[bold magenta]{BANNER}[/bold magenta]")
    console.print("[bold white]Made by J0J0M0J0[/bold white]")
    
    # Interactive Logic
    image_path = args.image
    width = args.width
    mode = "standard"
    target_color = None
    is_video = args.video

    if not image_path and not args.video:
        console.print("[bold cyan]Interactive Mode Initiated[/bold cyan]")
        choice_src = Prompt.ask("Select Source", choices=["image", "video"], default="image")
        
        if choice_src == "video":
            is_video = True
            
        if not is_video:
            image_path = Prompt.ask("Enter the path to your image")
        
        console.print("\n[bold]Select Analysis Mode:[/bold]")
        console.print("1. [green]Standard[/green] (Full Image)")
        console.print("2. [cyan]Smart Focus[/cyan] (Auto-Remove Background)")
        console.print("3. [magenta]Color Target[/magenta] (Isolate specific color)")
        console.print("4. [blue]Edge Detection[/blue] (Blueprint)")
        
        choice = Prompt.ask("Choose", choices=["1", "2", "3", "4"], default="1")
        
        if choice == "2":
            mode = "focus"
        elif choice == "3":
            mode = "color_target"
            target_color = Prompt.ask("Which color should I isolate?", choices=["red", "green", "blue", "yellow", "cyan", "magenta", "purple"])
        elif choice == "4":
            mode = "edge"
            
        width = int(Prompt.ask("Output Width (chars)", default="100"))

    # Argument Overrides
    if args.focus: mode = "focus"
    elif args.edge: mode = "edge"
    elif args.target_color: 
        mode = "color_target"
        target_color = args.target_color
    elif args.color:
        mode = "truecolor"
    
    console.print(f"[dim]Mode: {mode.upper()} | source: {'VIDEO' if is_video else 'IMAGE'} | Width: {width}[/dim]\n")
    
    # Select Charset
    if args.charset == "block":
        selected_charset = CHARSET_BLOCKS
    elif args.charset == "complex":
        selected_charset = CHARSET_COMPLEX
    else:
        selected_charset = CHARSET_STANDARD

    # Process
    try:
        if is_video:
            VideoEngine.stream_video(args.webcam, width, mode, selected_charset, target_color)
        else:
            if not image_path:
                console.print("[red]Error: No image path provided.[/red]")
                return

            img = ImageProcessor.load_image(image_path)
            img_resized = ImageProcessor.resize_image(img, width)
            
            if mode == "edge":
                result = ASCIIEngine.render_edge_detection(img_resized)
            elif mode == "focus":
                result = ASCIIEngine.render_focus_mode(img_resized, selected_charset)
            elif mode == "color_target":
                result = ASCIIEngine.render_color_segmentation(img_resized, selected_charset, target_color)
            elif mode == "truecolor":
                result = ASCIIEngine.render_truecolor(img_resized, selected_charset)
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
