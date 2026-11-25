# Fall Armyworm (FAW) Leaf Damage Detection Demo

A concise, modular CLI application for detecting *Fall Armyworm* leaf damage on corn leaves using YOLO.

## Directory Structure

```text
Demo/
├── main.py          # Main application (interactive CLI, image & webcam detection)
├── config.json      # Configuration settings (model, confidence, resolution)
├── requirements.txt # Python package dependencies
├── model/           # YOLO checkpoints (.pt) & OpenVINO INT8 model folder
├── upload/          # Input directory for leaf images
└── output/          # Annotated output images (images/) and CSV report (laporan_deteksi.csv)
```

## Quick Start

1. Install dependencies (if not already installed):
   ```bash
   pip install -r Demo/requirements.txt
   ```

2. Run the interactive CLI:
   ```bash
   .venv/bin/python Demo/main.py
   ```

3. Direct CLI Inference (Optional):
   ```bash
   .venv/bin/python Demo/main.py /path/to/image.jpg
   ```

## Application Menus

- **Detect Now**: Runs inference on all images in `upload/`, displays performance metrics, saves annotated images to `output/images/`, and logs results into `output/laporan_deteksi.csv`.
- **Upload**: Uploads new image(s) to the `upload/` directory without running immediate inference.
  - Options: `Upload Image`, `Back`.
- **Camera**: Lightweight real-time webcam scanner (press `q` on the video window to return to menu).
- **Settings**: Configure model checkpoint, confidence threshold, and input resolution.
- **Exit**: Exit the program.

## Navigation Controls

- Use **↑ / ↓** arrow keys and press **Enter**, or directly press **number keys** (`1`, `2`, `3`...).
- Press **q** to go back or exit.
