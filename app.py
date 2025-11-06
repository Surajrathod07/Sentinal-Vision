# app.py
# Sentinal-vision — Full final robust Streamlit app with ONNX support, safe camera worker, and polished UI
# Run: streamlit run app.py
#
# Recommended packages:
# pip install streamlit ultralytics torch torchvision opencv-python pillow onnxruntime streamlit-autorefresh

import streamlit as st
import os, sys, time, glob, json, hashlib, tempfile, traceback, importlib, threading, queue
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

# Optional heavy imports wrapped safely
try:
    import torch
except Exception:
    torch = None

# ultralytics (YOLOv8) optional
try:
    ultralytics = importlib.import_module("ultralytics")
    YOLO_IMPL_AVAILABLE = True
except Exception:
    ultralytics = None
    YOLO_IMPL_AVAILABLE = False

# OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False

# onnxruntime fallback (optional)
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except Exception:
    ort = None
    ONNXRUNTIME_AVAILABLE = False

# autorefresh optional
try:
    from streamlit_autorefresh import st_autorefresh
    HAS_AUTORELOAD = True
except Exception:
    HAS_AUTORELOAD = False

# ---------------- Config & folders ----------------
APP_TITLE = "Sentinal-vision"
PROJECT_ROOT = os.path.abspath(os.getcwd())
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
EXAMPLE_IMG_DIR = os.path.join(DATA_DIR, "example_images")
EXAMPLE_VID_DIR = os.path.join(DATA_DIR, "example_videos")
for d in (MODELS_DIR, DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, EXAMPLE_IMG_DIR, EXAMPLE_VID_DIR):
    os.makedirs(d, exist_ok=True)

MODELS_JSON = os.path.join(PROJECT_ROOT, "models.json")
USERS_JSON = os.path.join(PROJECT_ROOT, "users.json")

# Persisted/Default mapping of display-name -> file path (custom friendly names)
DEFAULT_MODELS = {
    # Display name : file path (you should have these files present in models/)
    "Drone": os.path.join(MODELS_DIR, "Drone.pt"),
    "Detect-Mosuito": os.path.join(MODELS_DIR, "Mosquito-High.pt"),
    "Detect-mosquito-video": os.path.join(MODELS_DIR, "Mosquito-low.pt"),
    "Custom1": os.path.join(MODELS_DIR, "custom1.pt"),
    "Custom2": os.path.join(MODELS_DIR, "custom2.pt"),
    # If you added yolov8n.onnx / yolov8s.onnx in models/ these can be referenced here as well:
    "yolov8n (onnx)": os.path.join(MODELS_DIR, "yolov8n.onnx"),
    "yolov8s (onnx)": os.path.join(MODELS_DIR, "yolov8s.onnx"),
}

STREAM_LATEST_PATH = os.path.join(OUTPUT_DIR, "stream_latest.jpg")  # persistent path for worker frames
THREAD_LOG_QUEUE = queue.Queue()  # thread-safe queue for worker -> main logs

# ---------------- Utilities ----------------
def now_ts(): return int(time.time() * 1000)
def fmt_time(ts=None):
    if ts is None: ts = time.time()
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def safe_json_load(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
    except Exception:
        pass
    return default.copy()

def safe_json_save(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------- Streamlit session initialization ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# initialize session_state defaults
DEFAULT_SESSION = {
    "users": safe_json_load(USERS_JSON, {}),
    "models": safe_json_load(MODELS_JSON, DEFAULT_MODELS),
    "logged_in": False,
    "username": "",
    "page": "start",
    "model_obj": None,           # ultralytics model object OR torch hub model OR None
    "model_type": None,          # 'yolov8', 'yolov5', 'onnxrt'
    "loaded_model_path": None,
    "model_device": "cpu",
    "log_lines": [],             # main UI logs
    "camera_thread": None,       # info dict with thread and stop_event
    "streaming": False,
    "ui_theme_dark": True,
}
for k, v in DEFAULT_SESSION.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Drain THREAD_LOG_QUEUE into st.session_state.log_lines (call only from main thread)
def drain_thread_logs():
    pushed = 0
    while True:
        try:
            msg = THREAD_LOG_QUEUE.get_nowait()
        except queue.Empty:
            break
        logs = st.session_state.get("log_lines", [])
        logs.insert(0, f"[{fmt_time()}] {msg}")
        st.session_state["log_lines"] = logs[:1000]
        pushed += 1
    return pushed

# main-thread logger that updates session_state
def ui_log(msg: str):
    logs = st.session_state.get("log_lines", [])
    logs.insert(0, f"[{fmt_time()}] {msg}")
    st.session_state["log_lines"] = logs[:1000]
    # also print to stdout for local logs
    print(f"[{fmt_time()}] {msg}")

# ---------------- Styling / CSS ----------------
st.markdown("""
<style>
:root { --bg:#0b0f1a; --card:#0f1724; --muted:#9fb0d6; --accent:#0b73d6; --accent2:#0687ff; --input:#e6eef8; }
.stApp { background: linear-gradient(180deg, #060712 0%, #0b0f1a 100%); color: #e6eef8; }
.card { background: var(--card); border-radius:12px; padding:18px; box-shadow: 0 6px 20px rgba(0,0,0,0.6); }
.small-muted { color: var(--muted); font-size:0.95rem; }
.app-header { display:flex; align-items:center; gap:14px; }
.logo { font-weight:800; font-size:20px; color: var(--accent); }
.btn-prim { background: linear-gradient(90deg,var(--accent),var(--accent2)); color:white; padding:10px 14px; border-radius:10px; border:none; font-weight:700; }
.center-card { display:flex; justify-content:center; align-items:center; min-height:70vh; }
.login-card { max-width:540px; width:100%; padding:28px; border-radius:12px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow: 0 20px 50px rgba(3,19,54,0.6); }
input[type="text"], input[type="password"], textarea, .stTextInput>div>div>input { background: rgba(255,255,255,0.04) !important; color: #e6eef8 !important; border-radius:8px !important; padding:10px !important; border: 1px solid rgba(255,255,255,0.03) !important; }
.kpi { font-size:22px; font-weight:700; color: #ffffff; }
.sidebar-title { color:#9fb0d6; font-weight:700; padding:8px 10px; }
.footer-small { color: #97a0b3; font-size:0.85rem; padding-top:10px; }
a { color: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ---------------- Model loaders (ultralytics + onnxruntime fallback) ----------------
def try_load_yolov8_ultralytics(path: str, device="cpu"):
    """Use ultralytics.YOLO to load `.pt` or `.onnx` if possible."""
    if not YOLO_IMPL_AVAILABLE:
        raise RuntimeError("ultralytics package not available")
    YOLO = getattr(ultralytics, "YOLO")
    try:
        model = YOLO(path)
        # try moving to device if requested
        if device == "cuda" and torch and torch.cuda.is_available():
            try: model.to("cuda")
            except Exception: pass
        else:
            try: model.to("cpu")
            except Exception: pass
        return model
    except Exception as e:
        raise RuntimeError(f"ultralytics failed to load {path}: {e}")

def init_onnx_session(path: str):
    """Create onnxruntime session (cached)"""
    if not ONNXRUNTIME_AVAILABLE:
        raise RuntimeError("onnxruntime not installed")
    providers = ['CPUExecutionProvider']
    # prefer CUDA if available
    try:
        avail = ort.get_available_providers()
        if 'CUDAExecutionProvider' in avail:
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
    except Exception:
        providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(path, providers=providers)
    return sess

@st.cache_resource
def load_model(path: str, device="cpu"):
    """
    Try ultralytics loader (v8) first.
    If ultralytics fails on an ONNX file, fallback to onnxruntime session (type 'onnxrt').
    Returns (model_obj_or_session, model_type_str)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    try:
        # 1) try ultralytics (works for .pt and many .onnx exports)
        m = try_load_yolov8_ultralytics(path, device=device)
        return m, "yolov8"
    except Exception as e_ul:
        # If path is .onnx we might try onnxruntime fallback
        if path.lower().endswith(".onnx"):
            try:
                sess = init_onnx_session(path)
                return sess, "onnxrt"
            except Exception as e_onnx:
                raise RuntimeError(f"ultralytics load failed: {e_ul}\nonnxruntime fallback failed: {e_onnx}")
        else:
            # attempt YOLOv5 hub fallback for .pt or if ultralytics failed
            if torch is None:
                raise RuntimeError(f"ultralytics load failed and torch not available: {e_ul}")
            try:
                hub_load = getattr(torch.hub, "load")
                model = hub_load('ultralytics/yolov5', 'custom', path=path, force_reload=False)
                return model, "yolov5"
            except Exception as e_v5:
                raise RuntimeError(f"ultralytics load failed: {e_ul}\nYOLOv5 hub fallback failed: {e_v5}")

# ---------------- ONNX runtime helper (basic) ----------------
# NOTE: ONNX fallback is basic and may need adjustments depending on export specifics.
# The app will prefer ultralytics for ONNX. Keep this as a defensive fallback.
import numpy as np

def letterbox(im, new_shape=(640,640), color=(114,114,114), auto=False):
    # Resize and pad image to new_shape (similar to YOLO letterbox)
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, r, (left, top)

def onnx_infer_and_draw(session, image_path, conf=0.25, iou_thres=0.45, input_size=(640,640)):
    """
    Very generic ONNX inference + plotting - attempts to decode typical YOLOv8-style outputs.
    It may not work for every ONNX variant; recommended to use ultralytics first.
    Returns path to saved annotated image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image for ONNX inference")
    img0 = img.copy()
    img_proc, scale, (pad_x, pad_y) = letterbox(img0, new_shape=input_size)
    img_rgb = cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    # HWC -> NCHW
    input_tensor = np.transpose(img_norm, (2,0,1))
    input_tensor = np.expand_dims(input_tensor, 0).astype(np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    # try to interpret outputs[0] as predictions Nx85 or (1,N,85)
    preds = outputs[0]
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]
    # preds assumed to be (N, 85) where 0:4 = xywh, 4 = objectness, 5: = class scores
    boxes = []
    scores = []
    classes = []
    for row in preds:
        if row.shape[0] < 5:
            continue
        conf_obj = float(row[4])
        class_conf = 0.0
        class_idx = -1
        if row.shape[0] > 5:
            class_scores = row[5:]
            class_idx = int(np.argmax(class_scores))
            class_conf = float(np.max(class_scores))
        final_conf = conf_obj * class_conf if class_conf>0 else conf_obj
        if final_conf < conf:
            continue
        # xywh to xyxy in original image coordinates
        x_center, y_center, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        # coordinates are relative to padded image size if model was exported that way; here we assume pixel coords relative to input_size
        # convert to pixel coords in padded image:
        x1 = x_center - w/2
        y1 = y_center - h/2
        x2 = x_center + w/2
        y2 = y_center + h/2
        # scale back to original image coordinates:
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale
        boxes.append([x1, y1, x2, y2])
        scores.append(final_conf)
        classes.append(class_idx if class_idx>=0 else 0)
    # apply NMS (simple)
    keep = nms_fast(np.array(boxes), np.array(scores), iou_thres)
    # draw boxes
    out_im = Image.fromarray(cv2.cvtColor(img0, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(out_im)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for idx in keep:
        if idx >= len(boxes): continue
        x1,y1,x2,y2 = boxes[int(idx)]
        sc = scores[int(idx)]
        cls = classes[int(idx)]
        draw.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
        draw.text((x1+4, y1+2), f"{cls}:{sc:.2f}", fill=(255,255,255), font=font)
    out_path = os.path.join(OUTPUT_DIR, f"onnx_pred_{now_ts()}_{os.path.basename(image_path)}")
    out_im.save(out_path)
    return out_path

def nms_fast(boxes, scores, iou_threshold=0.45):
    """
    boxes: Nx4 (x1,y1,x2,y2)
    scores: N
    returns indices kept
    """
    if len(boxes)==0:
        return []
    boxes = boxes.astype(float)
    x1 = boxes[:,0]; y1 = boxes[:,1]; x2 = boxes[:,2]; y2 = boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# ---------------- Inference unified function ----------------
def infer_image_and_save(model_obj, model_type, image_path, conf=0.25):
    """
    model_type: 'yolov8' -> ultralytics YOLO object
                'yolov5' -> torch hub object
                'onnxrt' -> onnxruntime session (fallback)
    """
    if model_obj is None:
        raise RuntimeError("No model loaded")
    if model_type == "yolov8":
        # ultralytics API
        results = model_obj.predict(source=image_path, conf=conf)
        r = results[0]
        np_img = r.plot()
        if isinstance(np_img, (list, tuple)):
            np_img = np_img[0]
        out_path = os.path.join(OUTPUT_DIR, f"pred_{now_ts()}_{os.path.basename(image_path)}")
        Image.fromarray(np_img).save(out_path)
        return out_path
    elif model_type == "yolov5":
        # torch.hub API
        results = model_obj(image_path)
        try:
            results.render()
            np_img = results.ims[0]
            out_path = os.path.join(OUTPUT_DIR, f"pred_{now_ts()}_{os.path.basename(image_path)}")
            Image.fromarray(np_img).save(out_path)
            return out_path
        except Exception as e:
            raise RuntimeError(f"YOLOv5 render failed: {e}")
    elif model_type == "onnxrt":
        # attempt defensive ONNX fallback
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime not installed")
        return onnx_infer_and_draw(model_obj, image_path, conf=conf)
    else:
        raise RuntimeError(f"Unknown model_type: {model_type}")

# ---------------- Camera worker - SAFE (background thread NOT calling Streamlit) ----------------
def _open_camera_try(indices=(0,1,2,3)):
    """Try to open camera with multiple backends & indices. Returns (cap, desc) or (None, reason)."""
    if not OPENCV_AVAILABLE:
        return None, "OpenCV not installed"
    backends = []
    if sys.platform.startswith("win"):
        for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_VFW", "CAP_ANY"):
            val = getattr(cv2, name, None)
            if val is not None and val not in backends:
                backends.append(val)
    else:
        for name in ("CAP_V4L2", "CAP_ANY"):
            val = getattr(cv2, name, None)
            if val is not None and val not in backends:
                backends.append(val)
    if not backends:
        backends = [cv2.CAP_ANY]
    for backend in backends:
        for idx in indices:
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                cap = None
            if not cap:
                continue
            try:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        return cap, f"idx={idx},backend={backend}"
                    else:
                        try: cap.release()
                        except Exception: pass
                else:
                    try: cap.release()
                    except Exception: pass
            except Exception:
                try: cap.release()
                except Exception: pass
    return None, "Unable to open camera"

def camera_worker_main(model_path, model_type, conf, max_fps, stop_event):
    """
    Background worker that:
      - loads its own model instance / session (so it never touches Streamlit objects),
      - opens camera,
      - writes annotated frames to STREAM_LATEST_PATH continuously,
      - pushes plain-text logs to THREAD_LOG_QUEUE (main thread reads them).
    """
    # load model in worker
    try:
        if model_type == "yolov8":
            if not YOLO_IMPL_AVAILABLE:
                THREAD_LOG_QUEUE.put("[worker] ultralytics not available for loading model")
                return
            YOLO = getattr(importlib.import_module("ultralytics"), "YOLO")
            local_model = YOLO(model_path)
        elif model_type == "yolov5":
            if torch is None:
                THREAD_LOG_QUEUE.put("[worker] torch not available for yolov5")
                return
            hub_load = getattr(torch.hub, "load")
            local_model = hub_load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        elif model_type == "onnxrt":
            if not ONNXRUNTIME_AVAILABLE:
                THREAD_LOG_QUEUE.put("[worker] onnxruntime not available")
                return
            local_model = init_onnx_session(model_path)
        else:
            THREAD_LOG_QUEUE.put(f"[worker] unknown model type: {model_type}")
            return
    except Exception as e:
        THREAD_LOG_QUEUE.put(f"[worker] failed to load model: {e}")
        return

    THREAD_LOG_QUEUE.put(f"[worker] model loaded: {os.path.basename(model_path)} ({model_type})")
    # open camera
    cap, msg = _open_camera_try((0,1,2,3))
    if not cap:
        THREAD_LOG_QUEUE.put(f"[worker] camera open failed: {msg}")
        return
    THREAD_LOG_QUEUE.put(f"[worker] camera opened: {msg}")

    last_time = 0.0
    min_interval = 1.0 / max(1, int(max_fps))
    try:
        while not stop_event.is_set():
            tnow = time.time()
            if (tnow - last_time) < min_interval:
                time.sleep(0.005)
                continue
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            last_time = time.time()
            # write tmp input
            tmp_in = os.path.join(tempfile.gettempdir(), f"sv_stream_in_{now_ts()}.jpg")
            try:
                cv2.imwrite(tmp_in, frame)
            except Exception as e:
                THREAD_LOG_QUEUE.put(f"[worker] failed writing tmp frame: {e}")
                continue
            # run inference & write annotated output to STREAM_LATEST_PATH
            try:
                if model_type in ("yolov8", "yolov5"):
                    # use local_model as ultralytics or yolov5 hub
                    if model_type == "yolov8":
                        results = local_model.predict(source=tmp_in, conf=conf)
                        r = results[0]
                        np_img = r.plot()
                        if isinstance(np_img, (list, tuple)):
                            np_img = np_img[0]
                        Image.fromarray(np_img).save(STREAM_LATEST_PATH)
                    else:
                        results = local_model(tmp_in)
                        try:
                            results.render()
                            np_img = results.ims[0]
                            Image.fromarray(np_img).save(STREAM_LATEST_PATH)
                        except Exception:
                            # fallback: save raw frame
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(STREAM_LATEST_PATH)
                elif model_type == "onnxrt":
                    try:
                        out = onnx_infer_and_draw(local_model, tmp_in, conf=conf)
                        # copy generated out into STREAM_LATEST_PATH
                        try:
                            Image.open(out).save(STREAM_LATEST_PATH)
                        except Exception:
                            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(STREAM_LATEST_PATH)
                    except Exception as e:
                        THREAD_LOG_QUEUE.put(f"[worker] onnx inference failed: {e}")
                        Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(STREAM_LATEST_PATH)
                else:
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(STREAM_LATEST_PATH)
            except Exception as e:
                THREAD_LOG_QUEUE.put(f"[worker] inference writing failed: {e}")
                try:
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(STREAM_LATEST_PATH)
                except Exception:
                    pass
            finally:
                try:
                    if os.path.exists(tmp_in):
                        os.remove(tmp_in)
                except Exception:
                    pass
    except Exception as e:
        THREAD_LOG_QUEUE.put(f"[worker] unexpected error: {e}")
    finally:
        try:
            if cap and cap.isOpened():
                cap.release()
        except Exception:
            pass
        THREAD_LOG_QUEUE.put("[worker] exiting")

# ---------------- UI helpers ----------------
def top_header():
    st.markdown(f"""
    <div class="app-header">
      <div class="logo">{APP_TITLE}</div>
      <div style="flex:1"></div>
      <div class="small-muted">Local inference • {fmt_time()}</div>
    </div>
    <hr style="border:1px solid rgba(255,255,255,0.03); margin-top:10px; margin-bottom:18px;" />
    """, unsafe_allow_html=True)

def sidebar_nav():
    with st.sidebar:
        st.markdown("<div class='sidebar-title'>Navigation</div>", unsafe_allow_html=True)
        pages = [
            ("Dashboard", "dashboard"),
            ("Detection", "detection"),
            ("Live Camera", "live_camera"),
            ("Models", "models"),
            ("Logs", "logs"),
            ("About", "about"),
            ("Logout", "logout")
        ]
        for label, key in pages:
            if st.button(label, key=f"nav_{key}"):
                st.session_state.page = key
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='footer-small'>Signed in: <strong>{st.session_state.username or 'Guest'}</strong></div>", unsafe_allow_html=True)

# ---------------- Pages ----------------
def page_login():
    drain_thread_logs()
    st.markdown("<div class='center-card'>", unsafe_allow_html=True)
    st.markdown("<div class='login-card card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; color:var(--accent); margin-bottom:6px'>Welcome to Sentinal-vision</h2>", unsafe_allow_html=True)
    st.markdown("<p class='small-muted' style='text-align:center; margin-top:0;'>Sign in or create a local account</p>", unsafe_allow_html=True)
    # Make input text color highly visible on dark background using Streamlit's input API plus CSS override above
    with st.form("login_form"):
        email = st.text_input("Email", key="login_email", placeholder="you@example.com")
        pwd = st.text_input("Password", type="password", key="login_pwd")
        col1, col2 = st.columns([1,1])
        with col1:
            signin = st.form_submit_button("Sign In")
        with col2:
            signup = st.form_submit_button("Sign Up")
        if signin:
            if not email or not pwd:
                st.error("Enter email and password")
            else:
                users = st.session_state.users
                if email in users and users[email] == hash_pw(pwd):
                    st.session_state.logged_in = True
                    st.session_state.username = email
                    st.session_state.page = "dashboard"
                    st.success("Signed in")
                else:
                    st.error("Invalid credentials")
        if signup:
            if not email or not pwd:
                st.error("Enter email and password to sign up")
            else:
                users = st.session_state.users
                if email in users:
                    st.warning("User exists")
                else:
                    users[email] = hash_pw(pwd)
                    st.session_state.users = users
                    safe_json_save(USERS_JSON, users)
                    st.session_state.logged_in = True
                    st.session_state.username = email
                    st.session_state.page = "dashboard"
                    st.success("Account created")
    st.markdown("</div></div>", unsafe_allow_html=True)

def page_dashboard():
    drain_thread_logs()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div style='display:flex;align-items:center;justify-content:space-between'><div><h2 style='margin:0;color:var(--accent)'>Dashboard</h2><div class='small-muted'>Overview</div></div></div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.03)' />", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    uploaded_count = len(glob.glob(os.path.join(UPLOAD_DIR,"*")))
    outputs_count = sum([len(files) for r,d,files in os.walk(OUTPUT_DIR)])
    active_model = os.path.basename(st.session_state.loaded_model_path) if st.session_state.loaded_model_path else "None"
    c1.markdown(f"<div class='card'><div class='small-muted'>Uploads</div><div class='kpi'>{uploaded_count}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><div class='small-muted'>Outputs</div><div class='kpi'>{outputs_count}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><div class='small-muted'>Active model</div><div class='kpi'>{active_model}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><div class='small-muted'>Signed in</div><div class='kpi'>{st.session_state.username or 'Guest'}</div></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:12px;margin-bottom:12px;border:1px solid rgba(255,255,255,0.03)' />", unsafe_allow_html=True)
    st.markdown("<h4>Recent outputs</h4>", unsafe_allow_html=True)
    outs = sorted(glob.glob(os.path.join(OUTPUT_DIR,"**","*.*"), recursive=True), key=os.path.getmtime, reverse=True)[:8]
    if outs:
        cols = st.columns(4)
        for i,f in enumerate(outs):
            col = cols[i%4]
            with col:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".png",".jpg",".jpeg"):
                    try:
                        col.image(f, caption=os.path.basename(f), width="stretch")
                    except Exception:
                        col.write(os.path.basename(f))
                else:
                    col.write("Video: "+os.path.basename(f))
                    try: col.video(f)
                    except Exception: pass
                if col.button("Download", key=f"dl_{i}"):
                    with open(f,"rb") as fh:
                        st.download_button("Download file", fh, file_name=os.path.basename(f))
    else:
        st.info("No outputs yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_models():
    drain_thread_logs()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:var(--accent)'>Models</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Friendly names are shown here (file paths stored behind the scenes)</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.03)' />", unsafe_allow_html=True)
    models = st.session_state.models
    if models:
        idx = 0
        for display_name, path in list(models.items()):
            st.markdown(f"<div style='display:flex; align-items:center; gap:8px;'><div style='flex:1;'><strong>{display_name}</strong><div class='small-muted' style='margin-top:4px'>{path}</div></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1,1,1])
            if col1.button("Load", key=f"load_{display_name}"):
                try:
                    mobj, mtype = load_model(path, device=st.session_state.model_device)
                    st.session_state.model_obj = mobj
                    st.session_state.model_type = mtype
                    st.session_state.loaded_model_path = path
                    st.success(f"Loaded {display_name} as {mtype}")
                except Exception as e:
                    st.error(f"Load failed: {e}")
                    st.text(traceback.format_exc())
            if col2.button("Test (single image)", key=f"test_{display_name}"):
                # quick test run on example image if exists
                ex = None
                for ext in (".jpg",".png",".jpeg"):
                    exlist = glob.glob(os.path.join(EXAMPLE_IMG_DIR, f"*{ext}"))
                    if exlist:
                        ex = exlist[0]; break
                if not ex:
                    st.warning("No example image found in data/example_images. Upload one to test.")
                else:
                    try:
                        mobj, mtype = load_model(path, device=st.session_state.model_device)
                        out = infer_image_and_save(mobj, mtype, ex, conf=0.25)
                        st.image(out, caption="Test result", width="stretch")
                    except Exception as e:
                        st.error(f"Test failed: {e}")
                        st.text(traceback.format_exc())
            if col3.button("Remove", key=f"rm_{display_name}"):
                models.pop(display_name, None)
                st.session_state.models = models
                safe_json_save(MODELS_JSON, models)
                st.success("Removed")
                st.experimental_rerun()
            st.markdown("</div>", unsafe_allow_html=True)
            idx += 1
    else:
        st.info("No models configured.")
    st.markdown("---")
    with st.form("add_model_form"):
        new_name = st.text_input("Friendly name", key="new_model_name")
        new_path = st.text_input("Full path to model file (.pt / .onnx)", key="new_model_path")
        if st.form_submit_button("Add / Update"):
            if not new_name or not new_path:
                st.error("Provide both name and model path")
            else:
                st.session_state.models[new_name] = new_path
                safe_json_save(MODELS_JSON, st.session_state.models)
                st.success("Saved model mapping")
    if torch and torch.cuda.is_available():
        dev = st.selectbox("Device for loading model", ["cpu", "cuda"], index=0, key="model_device_sel")
        st.session_state.model_device = dev
    else:
        st.markdown("<div class='small-muted'>Using CPU</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def page_detection():
    drain_thread_logs()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:var(--accent)'>Detection</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Image & Video inference</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.03)' />", unsafe_allow_html=True)
    models = st.session_state.models
    sel = st.selectbox("Model (friendly name)", list(models.keys()) if models else [], key="det_model_sel")
    conf = st.slider("Confidence", 0.05, 0.99, 0.25, 0.01, key="det_conf")
    input_type = st.radio("Input", ["Image", "Video"], horizontal=True, key="det_input_type")
    if input_type == "Image":
        uploaded = st.file_uploader("Upload image (.jpg/.png)", type=["png","jpg","jpeg"], key="det_upload_img")
        if uploaded:
            saved = os.path.join(UPLOAD_DIR, f"{now_ts()}_{uploaded.name}")
            with open(saved,"wb") as f:
                f.write(uploaded.getbuffer())
            st.image(saved, caption="Uploaded", width="stretch")
            if st.button("Run detection", key="run_img_det"):
                if not sel:
                    st.error("Select a model")
                else:
                    try:
                        model_path = st.session_state.models[sel]
                        if st.session_state.loaded_model_path != model_path or st.session_state.model_obj is None:
                            mobj, mtype = load_model(model_path, device=st.session_state.model_device)
                            st.session_state.model_obj = mobj
                            st.session_state.model_type = mtype
                            st.session_state.loaded_model_path = model_path
                        out = infer_image_and_save(st.session_state.model_obj, st.session_state.model_type, saved, conf=conf)
                        st.success("Inference complete")
                        st.image(out, caption="Result", width="stretch")
                        with open(out,"rb") as fh:
                            st.download_button("Download result", fh, file_name=os.path.basename(out), key=f"dl_result")
                    except Exception as e:
                        st.error(f"Inference error: {e}")
                        st.text(traceback.format_exc())
    else:
        uploaded = st.file_uploader("Upload video (.mp4/.mov/.avi/.mkv)", type=["mp4","mov","avi","mkv"], key="det_upload_vid")
        if uploaded:
            saved = os.path.join(UPLOAD_DIR, f"{now_ts()}_{uploaded.name}")
            with open(saved,"wb") as f:
                f.write(uploaded.getbuffer())
            try:
                st.video(saved)
            except Exception:
                st.write(os.path.basename(saved))
            if st.button("Run video detection", key="run_vid_det"):
                if not sel:
                    st.error("Select a model")
                else:
                    try:
                        model_path = st.session_state.models[sel]
                        if st.session_state.loaded_model_path != model_path or st.session_state.model_obj is None:
                            mobj, mtype = load_model(model_path, device=st.session_state.model_device)
                            st.session_state.model_obj = mobj
                            st.session_state.model_type = mtype
                            st.session_state.loaded_model_path = model_path
                        with st.spinner("Processing video — this may take time..."):
                            out_vid = infer_video_and_get_output(st.session_state.model_obj, st.session_state.model_type, saved, conf=conf)
                        st.success("Video processed")
                        try:
                            st.video(out_vid)
                        except Exception:
                            st.write(f"Output: {out_vid}")
                        with open(out_vid,"rb") as fh:
                            st.download_button("Download video", fh, file_name=os.path.basename(out_vid), key="dl_vid")
                    except Exception as e:
                        st.error(f"Video inference failed: {e}")
                        st.text(traceback.format_exc())
    st.markdown("</div>", unsafe_allow_html=True)

# small helper for video inference (moved here)
def infer_video_and_get_output(model_obj, model_type, video_path, conf=0.25):
    name = f"sv_video_{now_ts()}"
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    try:
        if model_type == "yolov8":
            # FIX: 'overwrite' -> 'exist_ok' to match Ultralytics API
            model_obj.predict(source=video_path, conf=conf, save=True, project=OUTPUT_DIR, name=name, exist_ok=True)
        else:
            try:
                model_obj(video_path)
            except Exception:
                pass
        # search for common video outputs
        patterns = ["*.mp4","*.mov","*.avi","*.mkv"]
        candidates = []
        for p in patterns:
            candidates.extend(glob.glob(os.path.join(out_dir,"**",p), recursive=True))
        runs = os.path.join(PROJECT_ROOT, "runs", "detect")
        if os.path.exists(runs):
            for p in patterns:
                candidates.extend(glob.glob(os.path.join(runs,"**",p), recursive=True))
        for p in patterns:
            candidates.extend(glob.glob(os.path.join(OUTPUT_DIR,"**",p), recursive=True))
        candidates = list(set(candidates))
        if not candidates:
            raise RuntimeError("No output video found; check runs/detect or outputs folder for frames.")
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]
    except Exception as e:
        raise RuntimeError(f"Video inference failed: {e}")

def page_live_camera():
    drain_thread_logs()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:var(--accent)'>Live Camera</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Default system webcam (autodetect). Start/Stop stream & snapshot.</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.03)' />", unsafe_allow_html=True)

    cam_col, settings_col = st.columns([3,1])
    with settings_col:
        live_model = st.selectbox("Live model (friendly)", list(st.session_state.models.keys()) if st.session_state.models else [], key="live_model")
        conf = st.slider("Confidence", 0.05, 0.99, 0.25, 0.01, key="live_conf")
        max_fps = st.slider("Max FPS", 1, 30, 8, key="live_maxfps")
        if st.button("Load model for live", key="live_load"):
            if not live_model:
                st.error("Select a model")
            else:
                try:
                    path = st.session_state.models[live_model]
                    mobj, mtype = load_model(path, device=st.session_state.model_device)
                    st.session_state.model_obj = mobj
                    st.session_state.model_type = mtype
                    st.session_state.loaded_model_path = path
                    st.success(f"Loaded {live_model}")
                except Exception as e:
                    st.error(f"Load failed: {e}")
                    st.text(traceback.format_exc())

    with cam_col:
        placeholder = st.empty()
        if os.path.exists(STREAM_LATEST_PATH):
            try:
                placeholder.image(STREAM_LATEST_PATH, caption="Live stream", width="stretch")
            except Exception:
                placeholder.markdown("<div style='text-align:center;color:#9fb0d6;'>Waiting for frames...</div>", unsafe_allow_html=True)
        else:
            placeholder.markdown("<div style='text-align:center;color:#9fb0d6;'>Click <strong>Start Stream</strong> to begin<br/><small class='small-muted'>If no frames appear, check Logs for worker messages</small></div>", unsafe_allow_html=True)

    start_btn = st.button("Start Stream", key="start_stream")
    stop_btn = st.button("Stop Stream", key="stop_stream")
    snap_btn = st.button("Save Snapshot", key="snapshot")

    # handle start
    if start_btn:
        if st.session_state.model_obj is None or st.session_state.model_type is None or st.session_state.loaded_model_path is None:
            st.error("Load a model first (Models -> Load or Load model for live).")
        else:
            info = st.session_state.get("camera_thread")
            if info and info.get("thread") and info["thread"].is_alive():
                st.warning("Stream already running")
            else:
                stop_event = threading.Event()
                model_path = st.session_state.loaded_model_path
                model_type = st.session_state.model_type
                # start background worker thread
                t = threading.Thread(target=camera_worker_main, args=(model_path, model_type, conf, max_fps, stop_event), daemon=True)
                t.start()
                st.session_state.camera_thread = {
                    "thread": t,
                    "stop_event": stop_event,
                    "model_path": model_path,
                    "model_type": model_type,
                    "conf": conf,
                    "max_fps": max_fps,
                    "started_at": now_ts()
                }
                st.session_state.streaming = True
                ui_log("Started background camera worker. Check Logs for details.")

    # handle stop
    if stop_btn:
        info = st.session_state.get("camera_thread")
        if info and info.get("stop_event"):
            try:
                info["stop_event"].set()
                tr = info.get("thread")
                if tr and tr.is_alive():
                    tr.join(timeout=3.0)
            except Exception as e:
                ui_log(f"Error while stopping thread: {e}")
        st.session_state.camera_thread = None
        st.session_state.streaming = False
        ui_log("Stopped streaming")
        placeholder.markdown("<div style='text-align:center;color:#9fb0d6;'>Stream stopped</div>", unsafe_allow_html=True)

    # handle snapshot
    if snap_btn:
        if not OPENCV_AVAILABLE:
            st.error("OpenCV is required for snapshot capture")
        else:
            cap, msg = _open_camera_try((0,1,2,3))
            if not cap:
                st.error(f"Snapshot failed: {msg}")
            else:
                try:
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        path = os.path.join(OUTPUT_DIR, f"snapshot_{now_ts()}.jpg")
                        cv2.imwrite(path, frame)
                        st.image(path, width="stretch", caption="Snapshot saved")
                        st.success(f"Snapshot saved: {path}")
                    else:
                        st.error("Failed to capture frame for snapshot")
                finally:
                    try: cap.release()
                    except Exception: pass

    # auto-refresh strategy
    info = st.session_state.get("camera_thread")
    if info and info.get("thread") and info["thread"].is_alive():
        if HAS_AUTORELOAD:
            # refresh every 600 ms to fetch updated STREAM_LATEST_PATH
            st_autorefresh(interval=600, limit=None, key="auto_live_refresh")
        else:
            st.info("Install streamlit-autorefresh for smoother live preview: pip install streamlit-autorefresh")
            if st.button("Refresh Frame", key="manual_refresh_frame"):
                pass

    st.markdown("</div>", unsafe_allow_html=True)

def page_logs():
    drain_thread_logs()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:var(--accent)'>Logs</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Worker & UI logs (most recent first)</div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:1px solid rgba(255,255,255,0.03)' />", unsafe_allow_html=True)
    lines = st.session_state.get("log_lines", [])
    if lines:
        for ln in lines[:500]:
            st.markdown(f"<div style='font-family:monospace;color:#cfe6ff'>{ln}</div>", unsafe_allow_html=True)
    else:
        st.info("No logs yet.")
    st.markdown("</div>", unsafe_allow_html=True)

def page_about():
    drain_thread_logs()
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:var(--accent)'>About Sentinal-vision</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class='small-muted'>
      Sentinal-vision — local YOLO inference dashboard supporting:
      <ul>
        <li>YOLOv8 (.pt & many .onnx) via ultralytics</li>
        <li>YOLOv5 fallback via torch.hub</li>
        <li>ONNXRuntime fallback for ONNX files when ultralytics cannot load</li>
        <li>Robust background camera worker that writes frames to a persistent file (no Streamlit calls from threads)</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def page_logout():
    # shutdown camera worker if running
    info = st.session_state.get("camera_thread")
    if info and info.get("stop_event"):
        try:
            info["stop_event"].set()
            tr = info.get("thread")
            if tr and tr.is_alive():
                tr.join(timeout=2.0)
        except Exception:
            pass
    st.session_state.camera_thread = None
    st.session_state.streaming = False
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "start"
    ui_log("Logged out")

# ---------------- Router ----------------
def route():
    # always drain queue at start of route
    drain_thread_logs()
    top_header()
    sidebar_nav()
    if not st.session_state.logged_in:
        page_login()
        return
    p = st.session_state.get("page", "dashboard")
    if p == "dashboard":
        page_dashboard()
    elif p == "detection":
        page_detection()
    elif p == "live_camera":
        page_live_camera()
    elif p == "models":
        page_models()
    elif p == "logs":
        page_logs()
    elif p == "about":
        page_about()
    elif p == "logout":
        page_logout()
    else:
        page_dashboard()

# ---------------- Start app ----------------
if __name__ == "__main__":
    try:
        ui_log("Starting Sentinal-vision (final edition) — safe streaming & ONNX support")
        route()
    except Exception as e:
        # final fallback - show traceback in UI and logs
        try:
            st.error(f"Fatal error: {e}")
            st.text(traceback.format_exc())
        except Exception:
            print(f"Fatal: {e}\n{traceback.format_exc()}")
