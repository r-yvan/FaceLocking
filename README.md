# FaceLocking

A real-time face recognition and tracking system built with Python, OpenCV, MediaPipe, and ArcFace embeddings. This project provides a complete pipeline for face detection, enrollment, and recognition with advanced tracking capabilities.

## Features

- **Multi-Face Detection**: Detect and track multiple faces simultaneously using Haar Cascade and MediaPipe FaceMesh
- **Face Enrollment**: Enroll new identities with automatic or manual capture modes
- **Real-Time Recognition**: Recognize enrolled faces in real-time with cosine similarity matching
- **Face Locking**: Lock onto a specific face and track movements, blinks, and expressions
- **Action Detection**: Detect head movements (left/right), eye blinks, and smiles
- **Action History Logging**: Save detailed logs of all detected actions for each tracked face
- **Live Threshold Adjustment**: Dynamically adjust recognition thresholds during runtime
- **Database Management**: Persistent storage of face embeddings with easy reload capabilities

## Architecture

The system uses a robust pipeline:

1. **Detection**: Haar Cascade for initial face detection
2. **Landmarks**: MediaPipe FaceMesh for 5-point facial landmarks (eyes, nose, mouth)
3. **Alignment**: 5-point alignment to normalize face orientation (112x112)
4. **Embedding**: ArcFace ONNX model to generate 512-dimensional face embeddings
5. **Matching**: Cosine similarity comparison against enrolled database

## Project Structure

```
FaceLocking-main/
├── src/
│   ├── camera.py          # Camera utilities
│   ├── detect.py          # Face detection
│   ├── landmarks.py       # Facial landmark detection
│   ├── align.py           # Face alignment
│   ├── embed.py           # ArcFace embedding generation
│   ├── enroll.py          # Face enrollment tool
│   ├── recognize.py       # Face recognition and tracking
│   ├── evaluate.py        # Evaluation utilities
│   └── haar_5pt.py        # Haar + 5-point landmark detector
├── data/
│   ├── enroll/            # Enrolled face crops (per identity)
│   └── db/                # Face database (embeddings)
│       ├── face_db.npz    # Numpy archive of embeddings
│       └── face_db.json   # Metadata
├── models/
│   ├── embedder_arcface.onnx      # ArcFace embedding model
│   └── face_landmarker.task       # MediaPipe FaceMesh model
├── logs/                  # Action history logs
├── requirements.txt       # Python dependencies
├── init_project.py        # Project structure initialization
└── .gitignore
```

## Installation

### Prerequisites

- Python 3.7 or higher
- Webcam or camera device

### Setup

1. **Clone the repository**:
   ```bash
   
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .env
   source .env/bin/activate  # On Windows: .env\Scripts\activate
   ```

3. **Install dependencies**:
   
   pip install -r requirements.txt
   ```

4. **Initialize project structure**:
   ```bash
   python init_project.py
   ```

5. **Download required models**:
   - Place `embedder_arcface.onnx` in the `models/` directory
   - Place `face_landmarker.task` (MediaPipe model) in the `models/` directory

## Usage

### 1. Enroll Faces

Enroll new identities into the database:

```bash
python -m src.enroll
```

**Controls during enrollment**:
- `SPACE`: Capture one sample (manual mode)
- `a`: Toggle auto-capture mode (captures every 0.25s)
- `s`: Save enrollment (requires minimum samples)
- `r`: Reset new samples (keeps existing crops)
- `q`: Quit

**Tips for best results**:
- Ensure stable lighting
- Move slightly left/right during capture
- Try different expressions
- Capture at least 15 samples per identity

### 2. Recognize Faces

Run real-time face recognition:

```bash
python -m src.recognize
```

**Controls during recognition**:
- `q`: Quit
- `r`: Reload database from disk
- `+`/`=`: Increase recognition threshold
- `-`: Decrease recognition threshold
- `d`: Toggle debug overlay
- `l`: Lock/unlock face tracking

**Face Locking Feature**:
When a face is recognized, press `l` to lock onto that person. The system will then:
- Track their movements in real-time
- Detect head movements (left/right)
- Detect eye blinks
- Detect smiles
- Log all actions with timestamps
- Save action history when unlocked or lost

### 3. Re-enrollment

To add more samples to an existing identity:

```bash
python -m src.enroll
# Enter the same name as before
```

The system will:
- Load existing samples from disk
- Allow you to capture additional samples
- Combine old and new samples for an updated template

## Configuration

### Enrollment Settings

Edit `src/enroll.py` to adjust:
- `samples_needed`: Minimum samples required (default: 15)
- `auto_capture_every_s`: Auto-capture interval (default: 0.25s)
- `max_existing_crops`: Maximum existing samples to load (default: 300)

### Recognition Settings

Edit `src/recognize.py` to adjust:
- `dist_thresh`: Recognition threshold (default: 0.34)
- `min_size`: Minimum face size for detection (default: 70x70)
- `max_faces`: Maximum faces to detect (default: 5)
- `max_timeout`: Face lock timeout in seconds (default: 2.0)

## Technical Details

### Face Embeddings

- **Model**: ArcFace (ONNX format)
- **Input**: 112x112 RGB aligned face
- **Output**: 512-dimensional L2-normalized embedding
- **Distance Metric**: Cosine distance (1 - cosine similarity)

### Recognition Threshold

- **Default**: 0.34 (cosine distance)
- **Lower values**: Stricter matching (fewer false positives)
- **Higher values**: Looser matching (more false positives)
- Typical range: 0.25 - 0.45

### Action Detection

The face locking system detects:
- **Head Movement**: Tracks horizontal displacement (>10px threshold)
- **Eye Blinks**: Monitors vertical eye-nose distance (70% reduction)
- **Smiles**: Analyzes mouth width/height ratio (1.5x increase)

## Output Files

### Database Files

- `data/db/face_db.npz`: Binary numpy archive containing embeddings
- `data/db/face_db.json`: Metadata including names, timestamps, and sample counts

### Enrolled Crops

- `data/enroll/<name>/*.jpg`: Aligned 112x112 face crops for each identity

### Action Logs

- `logs/<name>_history_<timestamp>.txt`: Detailed action history for tracked faces

## Dependencies

- **numpy**: Numerical operations
- **opencv-python**: Computer vision and image processing
- **mediapipe**: Facial landmark detection
- **onnxruntime**: ONNX model inference
- **scipy**: Scientific computing utilities
- **tqdm**: Progress bars

## Troubleshooting

### Camera not opening
- Check camera permissions
- Ensure no other application is using the camera
- Try changing camera index in `cv2.VideoCapture(0)` to `1` or `2`

### Model not found errors
- Verify models are in the `models/` directory
- Check file names match exactly: `embedder_arcface.onnx` and `face_landmarker.task`

### Poor recognition accuracy
- Enroll with more samples (20-30 recommended)
- Ensure good lighting during enrollment and recognition
- Adjust recognition threshold using `+`/`-` keys
- Re-enroll with varied poses and expressions

### Face not detected
- Ensure adequate lighting
- Move closer to the camera
- Face the camera directly
- Adjust `min_size` parameter if needed

## Performance

- **FPS**: 15-30 FPS on modern CPUs (varies by hardware)
- **Detection**: Multi-face capable (up to 5 faces by default)
- **Latency**: Real-time (<100ms per frame)
- **CPU-Friendly**: Optimized for CPU inference (no GPU required)

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- **ArcFace**: For the face recognition model
- **MediaPipe**: For facial landmark detection
- **OpenCV**: For computer vision utilities

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Future Enhancements

- [ ] GPU acceleration support
- [ ] Multi-camera support
- [ ] Web interface for enrollment and monitoring
- [ ] Advanced emotion detection
- [ ] Face anti-spoofing
- [ ] Export action history to CSV/JSON
- [ ] Real-time alerts and notifications
