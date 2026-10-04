# Obscura Studio 📸

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Ultralytics YOLO](https://img.shields.io/badge/YOLO-v11-yellow)

**Obscura Studio** is a privacy-first, web-based image redaction and editing suite. Designed to strip away the complexity of traditional photo editors, it provides an intuitive interface to anonymize faces, crop out sensitive information, and apply spatial transformations directly in the browser. 

Powered by **OpenCV** for lightning-fast matrix transformations and **YOLOv11** for state-of-the-art face detection, Obscura Studio ensures robust privacy protection without sacrificing image quality.

## ✨ Features

### 🛡️ Intelligent Redaction
*   **YOLOv11 Face Detection:** Leverages Ultralytics' latest object detection models to accurately locate faces, even at extreme angles or in complex lighting conditions.
*   **Multiple Censor Styles:**
    *   **Gaussian Blur:** Smooth, professional anonymization.
    *   **Pixelate (Mosaic):** Classic 8-bit block destruction via interpolation downscaling.
    *   **Censor Bar:** Cinematic black bars precisely mapped to the eye-line based on facial geometry.
    *   **Emoji Overlays:** Replace faces with custom graphics utilizing mathematically perfect Alpha Blending to preserve transparency.
    *   **Solid Color Box:** Complete blackout redaction.

### 📐 Spatial & Color Operations
*   **Interactive Cropping:** Draw bounding boxes directly on the image using `streamlit-cropper` instead of manually typing coordinates.
*   **Transformations:** Arbitrary rotation angles and directional flipping (horizontal, vertical, both).
*   **Global Filters:** Bilateral blurring (preserves sharp edges while smoothing textures) and destructive color space visualizations (HSV, RGB, Grayscale).

### ⚙️ Professional UX
*   **State Management (Undo/Redo):** Complete non-destructive editing history. Roll back or step forward through your pipeline at any time.
*   **Custom UI:** Sleek, custom-injected Light Mode CSS tailored for a premium workspace feel.
*   **One-Click Download:** Instantly encode and download the final edited BGR array as a high-quality JPEG.

---
