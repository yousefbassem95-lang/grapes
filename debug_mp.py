import sys
try:
    import mediapipe as mp
    print(f"MediaPipe Version: {mp.__version__}")
    print("Dir(mp):", dir(mp))
    
    try:
        import mediapipe.python.solutions
        print("Explicit import of solutions successful.")
        print("mp.solutions available?", hasattr(mp, 'solutions'))
    except ImportError as e:
        print(f"Explicit import failed: {e}")

except Exception as e:
    print(f"FAILED: {e}")
