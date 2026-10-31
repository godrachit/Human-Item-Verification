# Human & Item Verification System

A local laptop-webcam project using YOLO object detection, OpenCV, Flask, SQLite, and Pandas.

## What it does

- Detects people and objects supported by the pretrained COCO model.
- Uses OR logic: the overall result is VERIFIED if at least one person OR one non-person object is detected.
- Displays a live annotated camera feed and a web dashboard.
- Stores timestamped detection summaries in SQLite.
- Exports CSV and Excel reports.

Important: this verifies visible presence only. It does not establish a person's identity, prove ownership of an item, or confirm that an item belongs to the detected person.

## Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10–3.12 recommended
- Built-in or USB webcam
- Internet for the first YOLO model download

## Windows installation

1. Install Python 3.11 from https://www.python.org/downloads/ . During setup, tick "Add Python to PATH".
2. Extract this ZIP file.
3. Open Command Prompt in the extracted `human_item_verification` folder.
4. Create a virtual environment:

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\activate
   ```

5. Upgrade pip and install packages:

   ```bat
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the project:

   ```bat
   python app.py
   ```

7. On first launch, Ultralytics downloads `yolov8n.pt`. Wait until the model loads.
8. Open this address in Chrome or Edge:

   http://127.0.0.1:5000

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## Camera selection

The default camera is index 0. If the wrong camera opens or the camera cannot be opened:

Windows Command Prompt:
```bat
set CAMERA_INDEX=1
python app.py
```

macOS/Linux:
```bash
CAMERA_INDEX=1 python app.py
```

Close other applications using the camera before starting.

## Settings

Optional environment variables:

- `CAMERA_INDEX=0` — camera number
- `CONFIDENCE=0.35` — detection confidence threshold (0 to 1)
- `YOLO_MODEL=yolov8n.pt` — model filename/name
- `LOG_INTERVAL_SECONDS=5` — repeat log interval while a state remains unchanged

Example (Windows):
```bat
set CONFIDENCE=0.45
python app.py
```

## Supported item classes

The pretrained YOLOv8 nano model is trained on COCO classes, including common objects such as:
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, bench, bird, cat, dog, backpack, umbrella, handbag, tie, suitcase, frisbee, sports ball, skateboard, bottle, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, TV, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, and toothbrush.

"Any item" means any item in the model's learned class list that is visible enough for the model to recognize. It cannot detect literally every possible object. Add a custom-trained model if you need specific local objects or ID cards.

## Verification logic

- Human verified = at least one `person` detection.
- Item verified = at least one non-person detection.
- Overall verified = human detected OR item detected.

A detected object may be unrelated to the person. This is not a secure access-control or ownership-verification system.

## Testing

1. Start with an empty camera view. Expected: NOT VERIFIED.
2. Stand in front of the camera. Expected: HUMAN DETECTED and overall VERIFIED.
3. Leave the camera view and place a supported object (for example a bottle, book, or phone) in view. Expected: ITEM DETECTED and overall VERIFIED.
4. Show both a person and an object. Expected: both detections and overall VERIFIED.
5. Click Download CSV and Download Excel. Open the files and check timestamps, labels, and status.
6. Check recent verification history on the dashboard.

## Troubleshooting

- `Cannot open camera`: close Zoom/Teams/browser camera tabs; check Windows Settings > Privacy & security > Camera; try `CAMERA_INDEX=1`.
- Model download fails: check internet, then retry `python app.py`.
- `ModuleNotFoundError`: activate `.venv` and run `pip install -r requirements.txt`.
- Excel export error: run `pip install --upgrade openpyxl pandas`.
- Slow detection: close other applications; keep the laptop plugged in; try a smaller input resolution or a machine with better CPU/GPU.
- Camera permission issue on macOS: allow Terminal/Python camera access in Privacy & Security settings.

## Privacy and safety

The app does not record or save camera video. It stores detection summaries (time, counts, labels, status) in `verification_logs.db`. Keep the dashboard bound to localhost; do not expose it publicly without authentication, HTTPS, and a privacy review.