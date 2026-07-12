# Edge Detection Project (OpenCV)

Real-time and batch edge detection using six methods — **Canny**, **Sobel**,
**Laplacian**, **Scharr**, **Laplacian of Gaussian (LoG)**, and **Prewitt** —
built with OpenCV and Python.

## Setup

```bash
pip install -r requirements.txt
```

## 1. Interactive mode — `edge_detection.py`

Runs live on your webcam, a video file, or a still image, with on-screen
trackbars to tune parameters in real time.

```bash
# Webcam (default camera)
python edge_detection.py

# Still image
python edge_detection.py --image path/to/photo.jpg

# Video file
python edge_detection.py --video path/to/clip.mp4
```

**Keyboard controls** (click the video window first so it has focus):

| Key | Action |
|-----|--------|
| `c` | Canny edge detection |
| `s` | Sobel edge detection |
| `l` | Laplacian edge detection |
| `x` | Scharr edge detection |
| `g` | Laplacian of Gaussian (LoG) |
| `p` | Prewitt edge detection |
| `o` | Show original (no processing) |
| `b` | Toggle Gaussian blur pre-processing |
| `q` | Quit |

**Trackbars** (in the "Controls" window):
- **Canny Low / Canny High** — hysteresis thresholds for Canny
- **Sobel Ksize** — Sobel kernel size (maps to 1, 3, 5, 7)
- **Blur Ksize** — Gaussian blur strength before detection (0 = off)

The window shows the original frame side-by-side with the processed result.

## 2. Batch / headless mode — `batch_compare.py`

No GUI required — good for servers, scripts, or quick reports. Runs all
six methods on one image and saves a 3x3 comparison grid.

```bash
python batch_compare.py --image path/to/photo.jpg --out comparison.png
```

Optional flags:
```bash
python batch_compare.py --image photo.jpg \
    --canny-low 50 --canny-high 150 \
    --blur-ksize 5 \
    --out comparison.png
```

## How it works

1. **Grayscale conversion** — edge detectors operate on intensity, not color.
2. **Gaussian blur** (optional) — reduces noise so it isn't mistaken for edges.
3. **Edge detection**:
   - **Canny**: multi-stage algorithm (gradient computation, non-max
     suppression, double-threshold hysteresis). Generally the cleanest,
     thinnest edges. The default go-to for most use cases.
   - **Sobel**: computes intensity gradients in X and Y directions
     separately using a weighted 3x3+ kernel, then combines them into a
     gradient magnitude image.
   - **Laplacian**: uses the second derivative to find edges — sensitive to
     noise but detects edges regardless of direction in one pass.
   - **Scharr**: like Sobel, but with a kernel specifically tuned for
     rotational symmetry — more accurate gradient magnitude and direction,
     especially on diagonal edges.
   - **Laplacian of Gaussian (LoG)**: deliberately blurs with a larger
     Gaussian kernel before applying the Laplacian, taming its noise
     sensitivity while keeping the "all directions at once" property.
   - **Prewitt**: same idea as Sobel (separate X/Y kernels combined into a
     magnitude) but with a simpler, unweighted 3x3 kernel — a classic,
     lightweight alternative.

## Files

- `edge_detection.py` — interactive webcam/image/video tool with trackbars
- `batch_compare.py` — headless batch comparison tool
- `requirements.txt` — Python dependencies

---

## 👨‍💻 Author

**Ashutosh Tare** — Aspiring ML Engineer | Data Science Enthusiast
