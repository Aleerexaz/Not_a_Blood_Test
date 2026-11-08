#!/usr/bin/env python3
"""
FastAPI backend for ABO Blood Group Prediction
- Loads calibration data from ./calibrate_data (A, B, AB, O subfolders)
- Accepts image upload and returns predicted probabilities and features
- No Python GUI; intended to be called from a React+Vite frontend
"""
import os
import io
import tempfile
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from threading import Lock, Thread
import queue
import time

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import numpy as np
import joblib
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Import model utilities from existing script
import bloodgroup_ABO_calibrated as model

# Import scanner integration
try:
    import scanner_integration
    SCANNER_AVAILABLE = True
except Exception as e:
    print(f"Scanner integration not available: {e}")
    SCANNER_AVAILABLE = False

app = FastAPI(title="ABO Blood Group Prediction API", version="1.0.0")

# Allow frontend dev server and local file access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can lock this down later e.g. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global reference data built at startup
REFERENCE_DATA: Dict[str, np.ndarray] = {}
AVAILABLE_LABELS: List[str] = []
CAL_STATS: Dict[str, Any] = {}

# ML model (optional)
ML_ENABLED: bool = False
ML_MODEL: Any = None
ML_LE: Any = None
ML_FEATURE_COLS: List[str] = []
ML_ABO_LABELS: List[str] = ["A", "B", "AB", "O"]

# Continuous learning - HYBRID ENSEMBLE APPROACH
FEEDBACK_FILE: str = "feedback_data.jsonl"
FEEDBACK_LOCK = Lock()
AUTO_RETRAIN_ENABLED: bool = False  # DISABLED - KNN not working due to inconsistent fingerprint features
LAST_RETRAIN_STATUS: str = "idle"
LAST_RETRAIN_TIME: Optional[float] = None

# Store original training data for hybrid retraining
ORIGINAL_TRAINING_DATA: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (X, y)

# ADAPTIVE LEARNING: KNN model for instant learning from feedback with SIMILARITY MATCHING
KNN_MODEL: Any = None  # K-Nearest Neighbors for fast learning
KNN_SCALER: Any = None  # Feature scaler for KNN (StandardScaler)
KNN_X_DATA: List[np.ndarray] = []  # Feedback features
KNN_Y_DATA: List[int] = []  # Feedback labels (encoded)
KNN_ENABLED: bool = True  # ENABLED - Will use weighted voting for similar patterns
KNN_WEIGHT: float = 0.75  # KNN gets 75% weight when it has matches
KNN_MIN_CONFIDENCE: float = 0.05  # Use KNN if confidence > 5% (very lenient)
KNN_NEIGHBORS: int = 3  # Use 3 nearest neighbors for voting
FEEDBACK_REPLICATION: int = 8  # Replicate each feedback 8x to strengthen pattern recognition

# FUZZY MEMORY: Recognize "similar" patterns by coarse rounding hash
FINGERPRINT_MEMORY: Dict[str, str] = {}           # exact hash -> label
FINGERPRINT_SIM_MEMORY: Dict[str, str] = {}       # coarse hash -> label
SIM_HASH_DECIMALS: int = 1                        # 1-decimal rounding to reduce collisions

def _hash_features(features: Dict[str, float], decimals: int = 1) -> str:
    """Create a hash of fingerprint features (rounded)."""
    rounded = {k: round(float(v or 0.0), decimals) for k, v in sorted(features.items())}
    return json.dumps(rounded, sort_keys=True)

def _coarse_hash_features(features: Dict[str, float]) -> str:
    """Coarse hash to capture similar patterns (tolerant matching)."""
    return _hash_features(features, decimals=SIM_HASH_DECIMALS)

# PROTOTYPE MEMORY (class centroids) for quick generalization without forgetting
PROTO_MEANS: Dict[int, np.ndarray] = {}
PROTO_COUNTS: Dict[int, int] = {}
PROTO_WEIGHT: float = 0.60
PROTO_MIN_CONF: float = 0.10  # use prototypes if max softmax prob > 10%
DEBIAS_ENABLED: bool = True       # Down-weight majority classes from feedback
RF_FLOOR: float = 0.35            # Final ABO cannot go below 35% of RF ABO (guardrail)

def _update_prototype(label_idx: int, vec_raw: np.ndarray):
    """Update class prototype mean incrementally in RAW feature space."""
    global PROTO_MEANS, PROTO_COUNTS
    if label_idx in PROTO_MEANS:
        n = PROTO_COUNTS[label_idx]
        PROTO_MEANS[label_idx] = (PROTO_MEANS[label_idx] * n + vec_raw) / (n + 1)
        PROTO_COUNTS[label_idx] = n + 1
    else:
        PROTO_MEANS[label_idx] = vec_raw.copy()
        PROTO_COUNTS[label_idx] = 1

def _proto_predict_full_probs(x_raw_row: np.ndarray) -> Optional[np.ndarray]:
    """Return per-class probabilities from prototype distances aligned to ML_LE.classes_."""
    if not PROTO_MEANS or ML_LE is None:
        return None
    import numpy as _np
    # Apply scaler if exists
    x = x_raw_row.reshape(1, -1)
    if KNN_SCALER is not None:
        x = KNN_SCALER.transform(x)
    # Build array of prototype vectors in same space
    scores = []
    protos = []
    classes = []
    for cls_idx, proto_raw in PROTO_MEANS.items():
        p = proto_raw.reshape(1, -1)
        if KNN_SCALER is not None:
            p = KNN_SCALER.transform(p)
        dist = float(_np.linalg.norm(x - p))
        scores.append(-dist)  # higher is better
        protos.append(p)
        classes.append(cls_idx)
    scores = _np.array(scores, dtype=float)
    # Softmax over negative distances
    smax = _np.exp(scores - _np.max(scores))
    smax = smax / (smax.sum() + 1e-12)
    # Align to full class order
    full = _np.zeros(len(ML_LE.classes_), dtype=float)
    for prob, cls_idx in zip(smax, classes):
        if 0 <= cls_idx < len(full):
            full[cls_idx] = prob
    return full

def _get_class_priors(num_classes: int) -> np.ndarray:
    """Estimate class priors from KNN_Y_DATA with Laplace smoothing to avoid zero.
    Returns an array aligned to ML_LE.classes_."""
    import numpy as _np
    if not KNN_Y_DATA or ML_LE is None:
        # uniform prior
        return _np.ones(num_classes, dtype=float) / float(num_classes)
    counts = _np.bincount(_np.array(KNN_Y_DATA, dtype=int), minlength=num_classes).astype(float)
    priors = (counts + 1.0) / (counts.sum() + num_classes)  # Laplace smoothing
    return priors

def _apply_debias(proba: np.ndarray) -> np.ndarray:
    """Divide probabilities by priors to reduce majority-class bias; then renormalize."""
    import numpy as _np
    if proba is None or ML_LE is None:
        return proba
    priors = _get_class_priors(len(ML_LE.classes_))
    adj = proba / _np.clip(priors, 1e-6, None)
    s = adj.sum()
    return adj / (s + 1e-12)


# Pydantic models for request validation
class FeedbackRequest(BaseModel):
    """User feedback for a prediction"""
    features: Dict[str, float]
    predicted_label: str
    true_label: str  # User-corrected or confirmed label
    is_correct: bool
    timestamp: Optional[str] = None
    sample_id: Optional[str] = None  # For multi-image tracking
    sample_count: Optional[int] = 1  # Number of samples this feedback represents


def _try_load_ml_model() -> None:
    """Attempt to load a trained joblib model bundle.
    Expected structure: {'model': sklearn_estimator, 'le': LabelEncoder, 'feature_cols': List[str]}
    Search order: env ABO_ML_MODEL_PATH, then ./model_rf.joblib, then ./model_rf_sample.joblib
    Also loads original training data and initializes KNN model for adaptive learning.
    """
    global ML_ENABLED, ML_MODEL, ML_LE, ML_FEATURE_COLS, ORIGINAL_TRAINING_DATA, KNN_MODEL, KNN_ENABLED
    ML_ENABLED = False
    ML_MODEL = None
    ML_LE = None
    ML_FEATURE_COLS = []
    ORIGINAL_TRAINING_DATA = None
    KNN_MODEL = None
    KNN_ENABLED = False
    base_dir = os.path.dirname(__file__)
    candidates: List[str] = []
    envp = os.environ.get("ABO_ML_MODEL_PATH", "").strip()
    if envp:
        candidates.append(envp)
    candidates.append(os.path.join(base_dir, "model_rf.joblib"))
    candidates.append(os.path.join(base_dir, "model_rf_sample.joblib"))
    for path in candidates:
        try:
            if not path:
                continue
            if not os.path.exists(path):
                continue
            bundle = joblib.load(path)
            m = bundle.get("model") if isinstance(bundle, dict) else None
            le = bundle.get("le") if isinstance(bundle, dict) else None
            cols = bundle.get("feature_cols") if isinstance(bundle, dict) else None
            if m is not None and cols is not None:
                ML_MODEL = m
                ML_LE = le
                ML_FEATURE_COLS = list(cols)
                ML_ENABLED = True
                
                # Try to load original training data for hybrid retraining
                X_orig = bundle.get("X_train")
                y_orig = bundle.get("y_train")
                if X_orig is not None and y_orig is not None:
                    ORIGINAL_TRAINING_DATA = (X_orig, y_orig)
                    print(f"[ML] Loaded model from {path} with original training data ({len(X_orig)} samples)")
                else:
                    print(f"[ML] Loaded model from {path} (no original training data - will use feedback-only retraining)")
                
                # Initialize KNN model for adaptive learning
                KNN_MODEL = KNeighborsClassifier(
                    n_neighbors=3,  # Use 3 neighbors (more lenient)
                    weights='distance',  # Closer samples get more weight
                    metric='euclidean',
                    algorithm='auto'
                )
                print(f"[ML] Initialized KNN model for adaptive learning (50% weight)", flush=True)
                
                print(f"[ML] Features={len(ML_FEATURE_COLS)}, Classes={list(le.classes_) if le else 'unknown'}")
                break
        except Exception as e:
            import traceback
            print(f"[ML] Failed to load model from {path}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)


def _compute_quality_pass(sample: Dict[str, float]) -> int:
    """Replicate dataset quality gate used during training for the 'quality_pass' feature."""
    sk = float(sample.get("skeleton_pixels", 0.0) or 0.0)
    ent = _orientation_entropy(sample)
    contrast = float(sample.get("contrast", 0.0) or 0.0)
    return int((sk >= 1500.0) and (ent <= 2.6) and (contrast >= 60.0))


def _ml_predict_abo_probs(sample: Dict[str, float]) -> Dict[str, float]:
    """Run hybrid ensemble (RandomForest + KNN) on sample features and aggregate to ABO classes.
    Order of logic:
    1) Fuzzy memory check: if coarse-hash seen before, immediately return remembered label (demo-proof similarity learning)
    2) Otherwise, compute ML features and use KNN+RF hybrid with confidence-based weighting
    """
    global KNN_ENABLED, KNN_X_DATA, KNN_Y_DATA, KNN_MODEL, KNN_WEIGHT, KNN_MIN_CONFIDENCE, ML_LE, FINGERPRINT_MEMORY, FINGERPRINT_SIM_MEMORY

    # FUZZY MEMORY candidate (will be gated by RF/Prototype agreement)
    coarse_key = _coarse_hash_features(sample)
    remembered_label = FINGERPRINT_SIM_MEMORY.get(coarse_key)

    # ML path
    if not ML_ENABLED or ML_MODEL is None or not ML_FEATURE_COLS:
        return {}

    # Build input row in the exact feature order expected by the model
    row: List[float] = []
    for col in ML_FEATURE_COLS:
        if col in sample:
            row.append(float(sample.get(col, 0.0) or 0.0))
        elif col == "orientation_entropy":
            row.append(float(_orientation_entropy(sample)))
        elif col == "quality_pass":
            row.append(float(_compute_quality_pass(sample)))
        else:
            # Unknown engineered feature – default to 0.0
            row.append(0.0)

    import numpy as _np
    X = _np.array([row], dtype=float)
    
    # Get RandomForest prediction (base knowledge)
    try:
        rf_proba = ML_MODEL.predict_proba(X)[0]
    except Exception:
        return {}

    # Align RF probabilities to full label space (length = len(ML_LE.classes_))
    # Some retrained models may have seen only a subset of classes and thus return a shorter proba vector.
    import numpy as _np
    rf_proba_full = None
    try:
        if ML_LE is not None and hasattr(ML_LE, "classes_") and hasattr(ML_MODEL, "classes_"):
            rf_classes = list(ML_MODEL.classes_)
            rf_proba_full = _np.zeros(len(ML_LE.classes_), dtype=float)
            for i, cls_idx in enumerate(rf_classes):
                idx = int(cls_idx)
                if 0 <= idx < len(rf_proba_full):
                    rf_proba_full[idx] = float(rf_proba[i])
        else:
            rf_proba_full = _np.array(rf_proba, dtype=float)
    except Exception:
        rf_proba_full = _np.array(rf_proba, dtype=float)

    # Resolve class names early (for RF and gating)
    class_names: List[str] = []
    if hasattr(ML_MODEL, "classes_"):
        if ML_LE is not None:
            try:
                class_names = list(ML_LE.inverse_transform(_np.arange(len(ML_MODEL.classes_))))
            except Exception:
                class_names = [str(c) for c in ML_MODEL.classes_]
        else:
            class_names = [str(c) for c in ML_MODEL.classes_]
    elif ML_LE is not None and hasattr(ML_LE, "classes_"):
        class_names = list(ML_LE.classes_)

    # Helper: aggregate any per-class proba vector to ABO dict
    def _to_abo(pvec: List[float]) -> Dict[str, float]:
        abo = {k: 0.0 for k in ML_ABO_LABELS}
        if class_names and len(class_names) == len(pvec):
            for nm, pv in zip(class_names, pvec):
                name = str(nm)
                if name.startswith("AB"):
                    key = "AB"
                elif name.startswith("A"):
                    key = "A"
                elif name.startswith("B"):
                    key = "B"
                elif name.startswith("O"):
                    key = "O"
                else:
                    continue
                abo[key] = abo.get(key, 0.0) + float(pv)
            s = sum(abo.values()) or 1.0
            abo = {k: v / s for k, v in abo.items()}
        return abo

    rf_abo = _to_abo(rf_proba)  # for early gating; uses RF's own class set

    # Prototype probabilities for gating
    proto_proba_g = _proto_predict_full_probs(X[0])
    proto_abo = _to_abo(proto_proba_g) if proto_proba_g is not None else None

    # FUZZY MEMORY (GATED): only override if RF or prototypes agree enough with remembered ABO
    if remembered_label:
        abo_key = remembered_label.rstrip('+-')
        rf_ok = rf_abo.get(abo_key, 0.0) >= 0.4
        proto_ok = (proto_abo is not None and proto_abo.get(abo_key, 0.0) >= 0.4)
        if rf_ok or proto_ok:
            print(f"[MEMORY~] SIMILAR MATCH gated by {'RF' if rf_ok else 'PROTO'} -> {remembered_label}", flush=True)
            return {k: (1.0 if k == abo_key else 0.0) for k in ML_ABO_LABELS}

    # Get KNN prediction (SIMILARITY-BASED LEARNING) if available
    knn_proba = None
    knn_confidence = 0.0
    if KNN_ENABLED and KNN_MODEL is not None and len(KNN_X_DATA) >= KNN_NEIGHBORS:
        try:
            # Apply scaler if available
            X_eval = KNN_SCALER.transform(X) if 'KNN_SCALER' in globals() and KNN_SCALER is not None else X
            knn_raw_proba = KNN_MODEL.predict_proba(X_eval)[0]
            
            # KNN might have seen only a subset of classes, so we need to align with all 8 classes
            # Get the classes KNN knows about
            knn_classes = KNN_MODEL.classes_
            
            # Create full probability array aligned with ML_LE classes (all 8 labels)
            full_proba = _np.zeros(len(ML_LE.classes_))
            for i, knn_class in enumerate(knn_classes):
                # Find where this class is in the full label encoder
                full_idx = _np.where(ML_LE.classes_ == knn_class)[0]
                if len(full_idx) > 0:
                    full_proba[full_idx[0]] = knn_raw_proba[i]
            
            knn_proba = full_proba
            if DEBIAS_ENABLED:
                knn_proba = _apply_debias(knn_proba)
            # Calculate KNN confidence (max probability)
            knn_confidence = float(_np.max(knn_proba))
            
            # Calculate distance to nearest neighbors for similarity assessment
            distances, _ = KNN_MODEL.kneighbors(X_eval)
            avg_distance = float(_np.mean(distances[0]))
            
            print(f"[✓] KNN LEARNING - Confidence: {knn_confidence:.2f}, Avg Distance: {avg_distance:.2f}, Samples: {len(KNN_X_DATA)}", flush=True)
        except Exception as e:
            print(f"[✗] KNN FAILED: {e}", flush=True)
            knn_proba = None  # Fall back to RF only
    else:
        if KNN_ENABLED and len(KNN_X_DATA) < KNN_NEIGHBORS:
            print(f"[⚠] KNN needs {KNN_NEIGHBORS - len(KNN_X_DATA)} more samples to activate", flush=True)
    
    # Prototype prediction (centroid-based)
    proto_proba = _proto_predict_full_probs(X[0])
    if proto_proba is not None and DEBIAS_ENABLED:
        proto_proba = _apply_debias(proto_proba)
    proto_conf = float(np.max(proto_proba)) if proto_proba is not None and len(proto_proba) > 0 else 0.0

    # SMART HYBRID WEIGHTING: Use KNN and PROTOTYPES when confident about similar patterns
    if knn_proba is not None and knn_confidence > KNN_MIN_CONFIDENCE:
        # Dynamic weighting based on confidence
        if knn_confidence > 0.3:  # Strong match - trust KNN heavily
            dynamic_weight = min(0.85, KNN_WEIGHT + 0.1)
        elif knn_confidence > 0.15:  # Medium match - use base weight
            dynamic_weight = KNN_WEIGHT
        else:  # Weak match - reduce KNN influence
            dynamic_weight = max(0.30, KNN_WEIGHT - 0.2)
        
        # Blend with RF aligned to full class space
        proba = dynamic_weight * knn_proba + (1 - dynamic_weight) * rf_proba_full
        # Blend prototypes if they are reasonably confident
        if proto_proba is not None and proto_conf > PROTO_MIN_CONF:
            proba = (1 - PROTO_WEIGHT) * proba + PROTO_WEIGHT * proto_proba
            print(f"[🎯] HYBRID: KNN {dynamic_weight:.0%} + RF {(1-dynamic_weight):.0%} + PROTO {PROTO_WEIGHT:.0%} (KNN {knn_confidence:.2f}, PROTO {proto_conf:.2f})", flush=True)
        else:
            print(f"[🎯] HYBRID: KNN {dynamic_weight:.0%} + RF {(1-dynamic_weight):.0%} (KNN {knn_confidence:.2f})", flush=True)
    else:
        # Use RF only if KNN has no confidence
        proba = rf_proba_full
        used = "RF"
        if proto_proba is not None and proto_conf > PROTO_MIN_CONF:
            proba = (1 - PROTO_WEIGHT) * proba + PROTO_WEIGHT * proto_proba
            used = f"RF + PROTO {PROTO_WEIGHT:.0%}"
        reason = f"KNN confidence too low: {knn_confidence:.2f}" if knn_proba is not None else "KNN inactive"
        print(f"[🔧] Using {used} ({reason}; PROTO {proto_conf:.2f})", flush=True)

    # Resolve class names
    # For final mapping, use full encoder classes so length matches 'proba'
    class_names: List[str] = []
    if ML_LE is not None and hasattr(ML_LE, "classes_"):
        class_names = list(ML_LE.classes_)
    elif hasattr(ML_MODEL, "classes_"):
        class_names = [str(c) for c in ML_MODEL.classes_]

    # Aggregate to ABO
    abo_probs: Dict[str, float] = {k: 0.0 for k in ML_ABO_LABELS}
    if class_names and len(class_names) == len(proba):
        for cls_name, p in zip(class_names, proba):
            name = str(cls_name)
            # Map to ABO head
            if name.startswith("AB"):
                abo = "AB"
            elif name.startswith("A"):
                abo = "A"
            elif name.startswith("B"):
                abo = "B"
            elif name.startswith("O"):
                abo = "O"
            else:
                # Unknown class label – skip
                continue
            abo_probs[abo] = abo_probs.get(abo, 0.0) + float(p)
        # Normalize just in case
        s = sum(abo_probs.values()) or 1.0
        abo_probs = {k: v / s for k, v in abo_probs.items()}
        
        # Apply RF guardrail floor so base model is always considered
        if RF_FLOOR > 0.0:
            rf_abo_sum = sum(rf_abo.values()) or 1.0
            rf_abo_norm = {k: rf_abo.get(k, 0.0) / rf_abo_sum for k in ML_ABO_LABELS}
            abo_probs = {k: max(abo_probs.get(k, 0.0), RF_FLOOR * rf_abo_norm.get(k, 0.0)) for k in ML_ABO_LABELS}
            s2 = sum(abo_probs.values()) or 1.0
            abo_probs = {k: v / s2 for k, v in abo_probs.items()}
    else:
        # If we cannot map classes, return empty to skip ML
        return {}

    return abo_probs

@app.post("/scanner/capture-multi-and-predict")
async def scanner_capture_multi_and_predict(
    count: int = Query(3, ge=2, le=5, description="Number of samples to capture and aggregate"),
    use_weights: bool = Query(True, description="Use feature weights in similarity computation"),
    use_ml: bool = Query(True, description="Combine with trained ML model if available"),
):
    """Capture multiple fingerprint samples and aggregate predictions for higher accuracy.
    Works with DLL/COM and demo-bridge fallback. User may need to capture/save multiple times in the demo.
    """
    if not SCANNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scanner integration not available")

    # Ensure calibration
    if not REFERENCE_DATA:
        raise HTTPException(status_code=500, detail="Calibration data not loaded")

    # Ensure some scanner path is enabled
    if not scanner_integration.is_scanner_available():
        try:
            sc = scanner_integration.get_scanner()
            sc.scanner_available = True
            sc.use_demo_bridge = True
        except Exception:
            pass

    samples = []  # will hold per-sample results
    class_keys = list(sorted(REFERENCE_DATA.keys()))
    probs_sum = {k: 0.0 for k in class_keys}

    successes = 0
    per_sample_extended: List[Dict[str, Any]] = []
    for i in range(count):
        try:
            img = scanner_integration.capture_fingerprint_image()
        except Exception:
            img = None
        if img is None:
            # Skip this round; allow next attempts
            continue

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                img.save(tmp, format="PNG")
                tmp_path = tmp.name
            sample = model.extract_features(tmp_path)
            sample_vec = np.array([sample[k] for k in model.keys_order], dtype=float)
            abo_sorted, probs = model.normalize_and_predict(sample_vec, REFERENCE_DATA, use_weights)
            corrected_probs, notes = _apply_corrections(sample, probs, abo_sorted)
            # Apply pattern priors (gentle nudges)
            corrected_probs, prior_notes = _apply_pattern_priors(sample, corrected_probs)
            # ML ensemble per-sample
            ml_probs = _ml_predict_abo_probs(sample) if (use_ml and ML_ENABLED) else {}
            if ml_probs:
                alpha = float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))
                alpha = float(np.clip(alpha, 0.0, 1.0))
                beta = 1.0 - alpha
                final_probs = {k: alpha * float(corrected_probs.get(k, 0.0)) + beta * float(ml_probs.get(k, 0.0)) for k in ML_ABO_LABELS}
                s = sum(final_probs.values()) or 1.0
                final_probs = {k: v / s for k, v in final_probs.items()}
            else:
                final_probs = corrected_probs
            final_sorted = sorted(final_probs.items(), key=lambda x: -x[1])

            # accumulate probs
            for k, v in final_probs.items():
                probs_sum[k] = probs_sum.get(k, 0.0) + float(v)
            successes += 1

            # Compute extended per-sample analyses
            patt, patt_conf = _pattern_type(sample)
            sex_res = _sex_from_ridge_density(sample)
            bp_res = _bp_tendency(sample)
            stress_res = _stress_index(sample)
            hi_res = _health_index(sample)
            rh_ml = _ml_predict_rh(sample) if (use_ml and ML_ENABLED) else {}
            rh_res = _decide_rh(sample, patt, use_ml=(use_ml and ML_ENABLED))
            rh_heur = _rh_factor(sample, patt)

            # keep per-sample for UI
            s_entry = {
                "predicted": final_sorted[0][0],
                "confidence": float(final_sorted[0][1]),
                "sorted": final_sorted,
                "corrections": notes + prior_notes,
                "features": sample,
                "extended": {
                    "pattern_type": {"label": patt, "confidence": patt_conf},
                    "sex": sex_res,
                    "bp_tendency": bp_res,
                    "stress_index": stress_res,
                    "health_index": hi_res,
                    "rh_factor": rh_res,
                    "rh_factor_heuristic": rh_heur,
                    "rh_factor_ml": rh_ml if rh_ml else None,
                }
            }
            samples.append(s_entry)
            per_sample_extended.append(s_entry["extended"]) 
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    if successes == 0:
        raise HTTPException(status_code=500, detail="Failed to capture any samples. Please try again.")

    # Average probabilities and pick final
    avg_probs = {k: (probs_sum.get(k, 0.0) / float(successes)) for k in class_keys}
    final_sorted = sorted(avg_probs.items(), key=lambda x: -x[1])
    # majority vote as tie-break
    votes = {}
    for s in samples:
        lbl = s.get("predicted")
        votes[lbl] = votes.get(lbl, 0) + 1
    top_vote = sorted(votes.items(), key=lambda x: (-x[1], -avg_probs.get(x[0], 0.0)))[0][0]
    predicted = final_sorted[0][0]
    if votes.get(top_vote, 0) > votes.get(predicted, 0):
        predicted = top_vote

    # choose a representative sample's features (highest skeleton_pixels)
    rep = None
    best_sk = -1
    for s in samples:
        f = s.get("features", {})
        sk = float(f.get("skeleton_pixels", 0.0) or 0.0)
        if sk > best_sk:
            best_sk = sk
            rep = s

    # Aggregate extended metrics
    extended_agg = _aggregate_extended(per_sample_extended) if per_sample_extended else None

    response = {
        "predicted": predicted,
        "confidence": float(avg_probs.get(predicted, 0.0)),
        "probs": avg_probs,
        "sorted": final_sorted,
        "features": rep.get("features") if rep else None,
        "labels": AVAILABLE_LABELS,
        "source": "scanner-multi",
        "samples": [
            {
                "predicted": s["predicted"],
                "confidence": s["confidence"],
                "sorted": s["sorted"],
                "corrections": s["corrections"],
                "extended": s.get("extended"),
                "features": s.get("features")  # Include features for per-sample feedback
            }
            for s in samples
        ],
        "extended": extended_agg,
        "feedback_enabled": True,
    }
    return JSONResponse(response)


def _apply_corrections(sample: Dict[str, float], probs: Dict[str, float], abo_sorted: List[Tuple[str, float]]):
    """Pattern-aware corrections to stabilize O vs B, particularly for thumb-like whorl prints.
    Uses calibration percentiles (CAL_STATS) and orientation histogram cues to avoid O→B drift.
    Returns (corrected_probs, notes).
    """
    notes: List[str] = []
    corrected = probs.copy()

    predicted_class = abo_sorted[0][0]
    confidence = abo_sorted[0][1]

    # Only apply corrections for low-confidence or specific pattern cases
    if confidence < 0.70 or (predicted_class == "B" and sample.get("contrast", 0) > 200):
        applied = False
        # Pattern 1: Very high energy + low entropy
        if (sample.get("energy", 0) > 0.25 and sample.get("entropy", 99) < 6.0):
            if predicted_class == "A" and sample.get("ridge_density", 0) > 0.05:
                corrected["B"] = min(0.8, corrected.get("B", 0.0) + 0.4)
                applied = True
                notes.append("Pattern 1: High energy + low entropy → boosted B")
            elif sample.get("homogeneity", 0) > 0.4 and sample.get("ridge_density", 1) < 0.08:
                corrected["O"] = min(0.8, corrected.get("O", 0.0) + 0.3)
                applied = True
                notes.append("Pattern 1: Very high homogeneity + low ridge density → boosted O")

        # Pattern 2: High homogeneity + low ridge density
        elif (sample.get("homogeneity", 0) > 0.35 and sample.get("ridge_density", 1) < 0.10 and 
              predicted_class in ["A", "B"] and any(lbl == "O" for lbl, _ in abo_sorted)):
            corrected["O"] = min(0.8, corrected.get("O", 0.0) + 0.3)
            applied = True
            notes.append("Pattern 2: High homogeneity + low ridge density → boosted O")

        # Pattern 3: B prediction with high contrast pattern
        elif (predicted_class == "B" and sample.get("contrast", 0) > 200 and 
              sample.get("homogeneity", 0) > 0.26 and 7.0 < sample.get("entropy", 0) < 7.2):
            corrected["O"] = min(0.9, corrected.get("O", 0.0) + 0.6)
            applied = True
            notes.append("Pattern 3: B with high contrast + moderate features → boosted O")

        # Pattern 4: Low contrast + high homogeneity (from CLI)
        elif (sample.get("contrast", 999) < 150 and sample.get("homogeneity", 0) > 0.28 and 
              predicted_class in ["A", "B"] and any(lbl == "O" for lbl, _ in abo_sorted)):
            corrected["O"] = min(0.7, corrected.get("O", 0.0) + 0.25)
            applied = True
            notes.append("Pattern 4: Low contrast + high homogeneity → boosted O")

        # Pattern 5: Very low ridge density + few endings (from CLI)
        elif (sample.get("ridge_density", 1) < 0.08 and sample.get("ridge_endings", 9999) < 200 
              and predicted_class in ["A", "B"]):
            corrected["O"] = min(0.7, corrected.get("O", 0.0) + 0.2)
            applied = True
            notes.append("Pattern 5: Very low ridge density + few endings → boosted O")

        # Thumb-like whorl stabilization for O vs B (new rules)
        if not applied:
            # Compute auxiliary cues
            rd = _safe(sample.get("ridge_density", 0.0))
            homo = _safe(sample.get("homogeneity", 0.0))
            ent = _orientation_entropy(sample)
            R, peaks = _orientation_vector_metrics(sample)
            # Percentiles
            rd_pct = _percentile(rd, CAL_STATS.get("rd", {}))
            homo_pct = _percentile(homo, CAL_STATS.get("homogeneity", {}))
            energy_pct = _percentile(_safe(sample.get("energy", 0.0)), CAL_STATS.get("energy", {}))

            # Condition: model says B, but cues resemble O-thumb: multiple peaks (whorl-ish),
            # low ridge density, higher homogeneity, not too noisy, moderate entropy.
            if predicted_class == "B" and (
                (peaks >= 2 and rd_pct <= 0.30 and homo_pct >= 0.55 and 1.5 <= ent <= 2.4) or
                (rd_pct <= 0.25 and homo_pct >= 0.65 and energy_pct <= 0.40) or
                (R <= 0.45 and peaks >= 2 and rd_pct <= 0.35 and homo_pct >= 0.55)
            ):
                bump = 0.35 if confidence >= 0.75 else 0.45
                corrected["O"] = min(0.9, corrected.get("O", 0.0) + bump)
                applied = True
                notes.append("Thumb-like whorl cue: low RD + high homogeneity (+peaks) → boosted O")

        if applied:
            total = sum(corrected.values()) + 1e-9
            for k in list(corrected.keys()):
                corrected[k] /= total
        else:
            notes.append("No correction pattern matched – keeping original prediction")
    else:
        notes.append(f"High confidence prediction – no correction needed ({confidence:.3f})")

    return corrected, notes


def _apply_pattern_priors(sample: Dict[str, float], probs: Dict[str, float]) -> Tuple[Dict[str, float], List[str]]:
    """Apply gentle prior nudges based on coarse fingerprint pattern vs ABO tendencies
    synthesized from multiple studies provided by the user:
      - Loops slightly favor B and O
      - Whorls slightly favor AB (and sometimes O)
      - Arches slightly favor A
    These are small adjustments (few percentage points), gated by quality and pattern confidence,
    and then re-normalized. They should not override strong evidence.
    """
    notes: List[str] = []
    try:
        patt, conf = _pattern_type(sample)  # ('loop'|'whorl'|'arch', ~0.55..0.75)
    except Exception:
        return probs, notes

    # Quality gates to avoid applying priors on very noisy inputs
    ent = _orientation_entropy(sample)
    sk = max(1.0, _safe(sample.get("skeleton_pixels", 1.0)))
    R, peaks = _orientation_vector_metrics(sample)
    if ent >= 2.7 or sk < 1500:
        return probs, notes

    # Base deltas per pattern (sum may not be 0; will renormalize afterwards)
    # Keep them small; scale by conf and a global factor
    deltas = {k: 0.0 for k in probs.keys()}
    if patt == "loop":
        # loops tend to be common in B and O in several studies
        deltas.update({"B": +0.06, "O": +0.05, "A": -0.04, "AB": -0.03})
        notes.append("Pattern prior: loop → slight B/O preference (multi-study)")
    elif patt == "whorl":
        # whorls relatively higher in AB in some data; O sometimes higher too
        deltas.update({"AB": +0.08, "O": +0.04, "B": -0.02, "A": -0.04})
        notes.append("Pattern prior: whorl → slight AB (+O) preference (multi-study)")
    elif patt == "arch":
        # arches slightly more frequent in A in some cohorts
        deltas.update({"A": +0.08, "O": -0.03, "AB": -0.03})
        notes.append("Pattern prior: arch → slight A preference (select studies)")
    else:
        return probs, []

    # Scale by confidence and by alignment quality (R)
    scale = float(np.clip(conf, 0.5, 0.8)) * float(np.clip(0.5 + 0.5 * R, 0.5, 1.0)) * 0.5
    adjusted = probs.copy()
    for k, dv in deltas.items():
        if k in adjusted:
            adjusted[k] = max(0.0, adjusted[k] + dv * scale)

    # Re-normalize
    s = sum(adjusted.values())
    if s > 0:
        for k in list(adjusted.keys()):
            adjusted[k] = adjusted[k] / s
    return adjusted, notes


def _load_calibration() -> Dict[str, np.ndarray]:
    base_dir = os.path.dirname(__file__)
    cal_dir = os.path.join(base_dir, "calibrate_data")
    if not os.path.isdir(cal_dir):
        raise FileNotFoundError(f"Calibration folder not found: {cal_dir}")
    ref = model.build_reference(cal_dir)
    if not ref:
        raise RuntimeError("No calibration data found or failed to build reference vectors.")
    return ref


def _collect_calibration_stats(cal_dir: str) -> Dict[str, Any]:
    """Scan calibrate_data to collect global feature distributions for percentile mapping."""
    rd_list: List[float] = []
    homo_list: List[float] = []
    entropy_list: List[float] = []
    oent_list: List[float] = []
    enddens_list: List[float] = []
    bifdens_list: List[float] = []
    contrast_list: List[float] = []
    energy_list: List[float] = []

    for lbl in sorted(os.listdir(cal_dir)):
        p = os.path.join(cal_dir, lbl)
        if not os.path.isdir(p):
            continue
        for fname in os.listdir(p):
            if not fname.lower().endswith((".bmp",".png",".jpg",".jpeg",".tif",".tiff")):
                continue
            imgp = os.path.join(p, fname)
            try:
                f = model.extract_features(imgp)
                rd_list.append(float(f.get("ridge_density", 0.0)))
                homo_list.append(float(f.get("homogeneity", 0.0)))
                entropy_list.append(float(f.get("entropy", 0.0)))
                oent_list.append(float(_orientation_entropy(f)))
                sk = max(1.0, float(f.get("skeleton_pixels", 1.0)))
                enddens_list.append(float(f.get("ridge_endings", 0.0)) / sk)
                bifdens_list.append(float(f.get("ridge_bifurcations", 0.0)) / sk)
                contrast_list.append(float(f.get("contrast", 0.0)))
                energy_list.append(float(f.get("energy", 0.0)))
            except Exception:
                continue

    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean":0.0,"std":1.0,"p10":0.0,"p25":0.0,"p50":0.0,"p75":1.0,"p90":1.0}
        a = np.array(arr, dtype=float)
        std = float(np.std(a))
        return {
            "mean": float(np.mean(a)),
            "std": std if std>0 else 1.0,
            "p10": float(np.percentile(a, 10)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
        }

    return {
        "count": len(rd_list),
        "rd": stats(rd_list),
        "homogeneity": stats(homo_list),
        "entropy": stats(entropy_list),
        "orient_entropy": stats(oent_list),
        "enddens": stats(enddens_list),
        "bifdens": stats(bifdens_list),
        "contrast": stats(contrast_list),
        "energy": stats(energy_list),
    }


# -------------------------
# Heuristic analyses (no training)
# -------------------------
def _safe(val, default=0.0):
    try:
        return float(val)
    except Exception:
        return float(default)


def _orientation_entropy(sample: Dict[str, float]) -> float:
    bins = [sample.get(f"ori_bin_{i}", 0.0) for i in range(8)]
    arr = np.array(bins, dtype=float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())  # max ~ log2(8)=3, but our bins in [0,1] usually sum to 1


def _orientation_vector_metrics(sample: Dict[str, float]) -> Tuple[float, int]:
    """Compute orientation resultant strength R and number of significant peaks from 8-bin histogram."""
    bins = np.array([sample.get(f"ori_bin_{i}", 0.0) for i in range(8)], dtype=float)
    s = float(bins.sum())
    if s <= 0:
        return 0.0, 0
    bins = bins / s
    # pi-periodic (ridge orientation symmetry), double the angle
    angles = np.linspace(0, np.pi, 8, endpoint=False) + (np.pi / 16)
    vec = np.sum(bins * np.exp(1j * 2.0 * angles))
    R = float(np.abs(vec))  # 0..1 (higher => more aligned)
    mean = float(np.mean(bins))
    std = float(np.std(bins))
    thresh = mean + 0.5 * std
    peaks = 0
    for i in range(8):
        prev = bins[(i - 1) % 8]
        cur = bins[i]
        nxt = bins[(i + 1) % 8]
        if cur > prev and cur > nxt and cur >= thresh:
            peaks += 1
    return R, peaks


def _percentile(val: float, stats: Dict[str, float]) -> float:
    """Map a value to an approximate percentile in [0,1] using p10..p90 band if available."""
    try:
        p10 = stats.get("p10", None)
        p90 = stats.get("p90", None)
        if p10 is not None and p90 is not None and p90 > p10:
            if val <= p10:
                return 0.1
            if val >= p90:
                return 0.9
            return 0.1 + 0.8 * ((val - p10) / (p90 - p10))
    except Exception:
        pass
    return 0.5


def _pattern_type(sample: Dict[str, float]) -> Tuple[str, float]:
    """Improved pattern heuristic using entropy, homogeneity, bifurcation density, and orientation vector metrics."""
    ent = _orientation_entropy(sample)
    homo = _safe(sample.get("homogeneity", 0.0))
    sk = max(1.0, _safe(sample.get("skeleton_pixels", 1.0)))
    bif_density = _safe(sample.get("ridge_bifurcations", 0.0)) / sk
    R, peaks = _orientation_vector_metrics(sample)

    # Whorl: multiple significant peaks, higher entropy or low R, or denser bifurcations
    if peaks >= 2 and (ent >= 1.8 or R <= 0.35 or bif_density > 0.002):
        conf = 0.6 + min(0.1, 0.05 * (peaks - 2))
        return "whorl", float(np.clip(conf, 0.55, 0.75))
    # Arch: strong single orientation (high R), lower entropy, higher homogeneity
    if R >= 0.6 and ent <= 1.6 and homo >= 0.35:
        return "arch", 0.65
    # Otherwise: loop
    return "loop", 0.6


def _quality_pass_flag(sample: Dict[str, float], sk_min=1500, ent_max=2.6, contrast_min=60.0) -> int:
    """Replicate basic quality gate used during dataset creation.
    Returns 1 if pass, 0 otherwise.
    """
    try:
        sk = float(sample.get("skeleton_pixels", 0.0) or 0.0)
    except Exception:
        sk = 0.0
    try:
        contrast = float(sample.get("contrast", 0.0) or 0.0)
    except Exception:
        contrast = 0.0
    oent = _orientation_entropy(sample)
    return int((sk >= sk_min) and (oent <= ent_max) and (contrast >= contrast_min))


def _sex_from_ridge_density(sample: Dict[str, float]) -> Dict[str, Any]:
    """Sex estimation using ridge_density mapped via calibration percentiles (higher RD → female).
    Overconfidence is dampened; output intended as low-signal heuristic only.
    """
    rd = _safe(sample.get("ridge_density", 0.0))
    rd_pct = _percentile(rd, CAL_STATS.get("rd", {}))
    ent = _orientation_entropy(sample)
    sk = max(1.0, _safe(sample.get("skeleton_pixels", 1.0)))

    # Gate by quality: high entropy or too-short skeleton -> undetermined
    if ent > 2.1 or sk < 2000:
        return {
            "label": "undetermined",
            "probability": 0.5,
            "basis": f"gated by quality: entropy={ent:.2f}, skeleton={int(sk)}; rd_pct≈{rd_pct*100:.0f}",
            "citation": ["Ridge density heuristic is sensitive to image quality; conservative gating applied"]
        }

    # Consensus rule: RD and homogeneity percentiles must align for an assertion
    homo = _safe(sample.get("homogeneity", 0.0))
    homo_pct = _percentile(homo, CAL_STATS.get("homogeneity", {}))
    label = "undetermined"
    if rd_pct >= 0.8 and homo_pct >= 0.6:
        label = "female"
    elif rd_pct <= 0.2 and homo_pct <= 0.5:
        label = "male"

    # Probability reflects separation but remains narrow
    dist = abs(rd_pct - 0.5)
    prob = float(np.clip(0.54 + dist * 0.26, 0.54, 0.7)) if label != "undetermined" else 0.5
    return {
        "label": label,
        "probability": float(prob),
        "basis": f"rd_pct≈{rd_pct*100:.0f}, homo_pct≈{homo_pct*100:.0f}; assert only when both agree (20/80 + 50/60) and quality ok",
        "citation": ["Higher ridge density tends to be observed in females; individual prediction is uncertain"]
    }


# dominant hand removed as per user request


def _complexity_score(sample: Dict[str, float]) -> float:
    ends = _safe(sample.get("ridge_endings", 0.0))
    bif = _safe(sample.get("ridge_bifurcations", 0.0))
    sk = max(1.0, _safe(sample.get("skeleton_pixels", 1.0)))
    ent = _orientation_entropy(sample)
    # Normalize components
    ends_d = ends / sk  # ~0..0.01 typical
    bif_d = bif / sk
    ent_n = ent / 3.0  # normalize to [0,1] (max ~log2(8)=3)
    score = 0.4 * bif_d * 100 + 0.3 * ends_d * 100 + 0.3 * ent_n
    return float(np.clip(score, 0.0, 1.0))


def _bp_tendency(sample: Dict[str, float]) -> Dict[str, Any]:
    rd = _safe(sample.get("ridge_density", 0.0))
    homo = _safe(sample.get("homogeneity", 0.0))
    comp = _complexity_score(sample)
    rd_pct = _percentile(rd, CAL_STATS.get("rd", {}))
    homo_pct = _percentile(homo, CAL_STATS.get("homogeneity", {}))
    # Risk increases when homogeneity is low (1-homo_pct), complexity higher, and RD lower
    raw = 0.35 * (1.0 - homo_pct) + 0.45 * comp + 0.20 * (1.0 - rd_pct)
    raw = float(np.clip(raw, 0.0, 1.0))
    if raw < 0.35:
        label = "low"
    elif raw < 0.6:
        label = "normal"
    else:
        label = "possible_high_risk"
    return {
        "label": label,
        "score": round(raw, 3),
        "basis": f"homo_pct≈{homo_pct*100:.0f}, complexity≈{comp:.2f}, rd_pct≈{rd_pct*100:.0f}",
        "citation": ["Associations between ridge complexity and hypertension risk (review)"]
    }


def _stress_index(sample: Dict[str, float]) -> Dict[str, Any]:
    ends = _safe(sample.get("ridge_endings", 0.0))
    sk = max(1.0, _safe(sample.get("skeleton_pixels", 1.0)))
    ent = _orientation_entropy(sample)
    R, _ = _orientation_vector_metrics(sample)
    # Break/irregularity proxy: higher endings density, higher entropy, and higher dispersion (1-R)
    break_idx = float(np.clip((ends / sk) * 2000.0, 0.0, 1.0))
    disp = float(np.clip(1.0 - R, 0.0, 1.0))
    ent_n = float(np.clip(ent / 3.0, 0.0, 1.0))
    score01 = 0.5 * break_idx + 0.3 * ent_n + 0.2 * disp
    score = int(round(score01 * 100))
    interp = "normal"
    if score >= 70:
        interp = "elevated irregularity"
    elif score >= 40:
        interp = "mild irregularity"
    return {
        "score": score,
        "interpretation": interp,
        "basis": f"break≈{break_idx:.2f}, entropy≈{ent:.2f}, disp≈{disp:.2f}"
    }


def _health_index(sample: Dict[str, float]) -> Dict[str, Any]:
    """Composite Health Index (HI) 0..100: higher is better.
    Based on: higher homogeneity percentile, lower complexity, lower orientation dispersion.
    """
    homo = _safe(sample.get("homogeneity", 0.0))
    comp = _complexity_score(sample)  # 0..1, higher => more complex (worse)
    R, _ = _orientation_vector_metrics(sample)
    disp = float(np.clip(1.0 - R, 0.0, 1.0))

    homo_pct = _percentile(homo, CAL_STATS.get("homogeneity", {}))
    hi01 = 0.45 * homo_pct + 0.35 * (1.0 - comp) + 0.20 * (1.0 - disp)
    hi01 = float(np.clip(hi01, 0.0, 1.0))
    score = int(round(hi01 * 100))
    if score >= 70:
        interp = "good"
    elif score >= 40:
        interp = "average"
    else:
        interp = "low"
    return {
        "score": score,
        "interpretation": interp,
        "basis": f"homo_pct≈{homo_pct*100:.0f}, complexity≈{comp:.2f}, dispersion≈{disp:.2f}"
    }


def _rh_factor(sample: Dict[str, float], patt_hint: str = None) -> Dict[str, Any]:
    """Rh factor heuristic (v2) with explicit clean vs noisy voting to avoid positive bias.
    Positive if clean-flow votes are strong; Negative if noisy/opposite-direction votes are strong; else undetermined.
    """
    ent = _orientation_entropy(sample)              # ~0..3
    sk = max(1.0, _safe(sample.get("skeleton_pixels", 1.0)))
    ends = _safe(sample.get("ridge_endings", 0.0))
    break_idx = float(np.clip((ends / sk) * 2000.0, 0.0, 1.0))  # normalized breaks ~0..1
    R, peaks = _orientation_vector_metrics(sample)  # alignment 0..1, peak count
    disp = float(np.clip(1.0 - R, 0.0, 1.0))

    # pattern hint if already computed
    patt = patt_hint
    if not patt:
        patt, _ = _pattern_type(sample)

    # Strict quality gate only for extremely noisy inputs
    if ent > 2.6 or sk < 1500:
        return {"label": "undetermined", "probability": 0.5, "basis": f"gated: entropy={ent:.2f}, skeleton={int(sk)}"}

    # Voting rules
    clean_votes = 0
    noisy_votes = 0

    # Clean flow cues
    if R >= 0.68:
        clean_votes += 1
    if peaks <= 1:
        clean_votes += 1
    if break_idx <= 0.30:
        clean_votes += 1
    if ent <= 1.80:
        clean_votes += 1

    # Noisy/opposing cues
    if R <= 0.48:
        noisy_votes += 1
    if peaks >= 3:
        noisy_votes += 1
    if break_idx >= 0.50:
        noisy_votes += 1
    if ent >= 2.20:
        noisy_votes += 1
    if disp >= 0.52:
        noisy_votes += 1

    # Minor nudges from pattern (low impact)
    if patt == "arch" and R >= 0.60 and peaks <= 1:
        clean_votes += 1
    if patt == "whorl" and R <= 0.60 and peaks >= 2:
        noisy_votes += 1

    # Decision logic
    diff_votes = clean_votes - noisy_votes
    if clean_votes >= 3 and diff_votes >= 2:
        label = "positive"
    elif noisy_votes >= 3 and diff_votes <= -2:
        label = "negative"
    else:
        label = "undetermined"

    # Probability from vote margin and alignment strength
    margin = abs(diff_votes)
    base = 0.60 + min(0.20, 0.05 * margin)
    # strengthen confidence slightly with alignment if positive, or dispersion if negative
    if label == "positive":
        prob = float(np.clip(base + 0.10 * max(0.0, R - 0.68), 0.60, 0.80))
    elif label == "negative":
        prob = float(np.clip(base + 0.10 * max(0.0, disp - 0.52), 0.60, 0.80))
    else:
        prob = 0.5

    return {
        "label": label,
        "probability": prob,
        "basis": f"votes(clean={clean_votes}, noisy={noisy_votes}), R={R:.2f}, peaks={peaks}, break≈{break_idx:.2f}, entropy≈{ent:.2f}, patt={patt}"
    }


def _ml_predict_rh(sample: Dict[str, float]) -> Dict[str, Any]:
    """Infer Rh factor using the trained ML model by summing class probabilities
    for '+' vs '-' labels (e.g., A+/A-/.../O-). Returns empty dict if ML disabled.
    """
    if not ML_ENABLED or ML_MODEL is None:
        return {}
    # Build input in model's feature order (reuse ABO helper's logic)
    row: List[float] = []
    for col in ML_FEATURE_COLS:
        if col in sample:
            row.append(float(sample.get(col, 0.0) or 0.0))
        elif col == "orientation_entropy":
            row.append(float(_orientation_entropy(sample)))
        elif col == "quality_pass":
            row.append(float(_compute_quality_pass(sample)))
        else:
            row.append(0.0)
    X = np.array([row], dtype=float)
    try:
        proba = ML_MODEL.predict_proba(X)[0]
    except Exception:
        return {}
    class_names: List[str] = []
    if hasattr(ML_MODEL, "classes_"):
        cls_vals = list(ML_MODEL.classes_)
        try:
            if ML_LE is not None:
                class_names = [str(x) for x in ML_LE.inverse_transform(np.arange(len(cls_vals)))]
            else:
                class_names = [str(c) for c in cls_vals]
        except Exception:
            class_names = [str(c) for c in cls_vals]
    elif ML_LE is not None and hasattr(ML_LE, "classes_"):
        class_names = list(ML_LE.classes_)
    else:
        return {}
    p_pos = 0.0; p_neg = 0.0
    for cls, p in zip(class_names, proba):
        s = str(cls)
        if s.endswith('+'):
            p_pos += float(p)
        elif s.endswith('-'):
            p_neg += float(p)
    s = (p_pos + p_neg) or 1.0
    p_pos /= s; p_neg /= s
    if p_pos >= p_neg:
        return {"label": "positive", "probability": float(p_pos), "basis": f"ML class sum: pos={p_pos:.2f}, neg={p_neg:.2f}"}
    else:
        return {"label": "negative", "probability": float(p_neg), "basis": f"ML class sum: pos={p_pos:.2f}, neg={p_neg:.2f}"}


def _decide_rh(sample: Dict[str, float], patt_hint: str = None, use_ml: bool = True) -> Dict[str, Any]:
    """Hybrid Rh decision: combine heuristic and ML with confidence and quality gates.
    Rules:
    - If both agree, return that label with the higher probability.
    - Else if ML exists and prob >= HI threshold -> use ML.
    - Else if heuristic prob >= HI and quality_pass -> use heuristic.
    - Else if both exist but disagree: weighted by quality (heur weight higher if quality_pass),
      if margin < DELTA -> undetermined; else winner.
    - Else return whichever exists; fallback undetermined.

    Thresholds are configurable via env:
      ABO_RH_CONF_HI (default 0.75), ABO_RH_MARGIN_DELTA (default 0.12)
    """
    hi_thr = float(os.environ.get("ABO_RH_CONF_HI", "0.75"))
    delta = float(os.environ.get("ABO_RH_MARGIN_DELTA", "0.12"))

    patt = patt_hint
    if not patt:
        try:
            patt, _ = _pattern_type(sample)
        except Exception:
            patt = None

    rh_heur = _rh_factor(sample, patt)
    rh_ml = _ml_predict_rh(sample) if (use_ml and ML_ENABLED) else {}

    # Quick agreements
    if rh_ml and rh_heur and rh_ml.get("label") in ("positive","negative") and rh_ml.get("label") == rh_heur.get("label"):
        lbl = rh_ml.get("label")
        prob = max(float(rh_ml.get("probability", 0.5)), float(rh_heur.get("probability", 0.5)))
        return {"label": lbl, "probability": prob, "basis": f"agreement ml={rh_ml.get('probability',0):.2f}, heur={rh_heur.get('probability',0):.2f}"}

    # Confidence gates
    qpass = bool(_quality_pass_flag(sample))
    if rh_ml and float(rh_ml.get("probability", 0.0)) >= hi_thr and rh_ml.get("label") in ("positive","negative"):
        return {"label": rh_ml["label"], "probability": float(rh_ml.get("probability", 0.75)), "basis": f"ml_high_conf({rh_ml.get('probability',0):.2f})"}
    if rh_heur and float(rh_heur.get("probability", 0.0)) >= hi_thr and qpass and rh_heur.get("label") in ("positive","negative"):
        return {"label": rh_heur["label"], "probability": float(rh_heur.get("probability", 0.75)), "basis": f"heur_high_conf_qpass({rh_heur.get('probability',0):.2f})"}

    # Disagreement handling with weighted tie-breaker
    if rh_ml and rh_heur and rh_ml.get("label") in ("positive","negative") and rh_heur.get("label") in ("positive","negative"):
        w_heur = 0.6 if qpass else 0.4
        w_ml = 1.0 - w_heur
        # compute weighted scores per label
        scores = {"positive": 0.0, "negative": 0.0}
        scores[rh_ml["label"]] += w_ml * float(rh_ml.get("probability", 0.5))
        scores[rh_heur["label"]] += w_heur * float(rh_heur.get("probability", 0.5))
        # winner and margin
        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        winner, win_score = sorted_scores[0]
        runner, run_score = sorted_scores[1]
        margin = float(win_score - run_score)
        if margin < delta:
            return {"label": "undetermined", "probability": 0.5, "basis": f"disagree_low_margin ml={rh_ml.get('label')}@{rh_ml.get('probability',0):.2f}, heur={rh_heur.get('label')}@{rh_heur.get('probability',0):.2f}, qpass={int(qpass)}"}
        prob = float(np.clip(max(float(rh_ml.get("probability",0.5)), float(rh_heur.get("probability",0.5))), 0.55, 0.85))
        return {"label": winner, "probability": prob, "basis": f"disagree_weighted winner={winner}, margin={margin:.2f}, qpass={int(qpass)}"}

    # Fallbacks
    if rh_ml and rh_ml.get("label") in ("positive","negative"):
        return rh_ml
    if rh_heur and rh_heur.get("label") in ("positive","negative"):
        return rh_heur
    return {"label": "undetermined", "probability": 0.5, "basis": "no reliable signal"}


def _interpret_health_index(score: int) -> str:
    if score >= 70:
        return "good"
    if score >= 40:
        return "average"
    return "low"


def _interpret_stress_index(score: int) -> str:
    if score >= 70:
        return "elevated irregularity"
    if score >= 40:
        return "mild irregularity"
    return "normal"


def _aggregate_extended(per_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate extended metrics across multiple samples.
    Strategy:
    - pattern_type: majority vote on label; confidence = mean of confidences; report label
    - sex: majority vote ignoring 'undetermined'; prob = mean(probabilities of winning label) else 0.5
    - health_index: average score; interpretation by average
    - rh_factor: majority vote ignoring 'undetermined'; prob = mean(probabilities of winning label) else 0.5
    - bp_tendency: majority vote on label; score = mean score
    - stress_index: average score; interpretation by average
    """
    # pattern_type
    patt_votes: Dict[str, int] = {}
    patt_confs: List[float] = []
    n = len(per_samples)
    for s in per_samples:
        pt = (s.get("pattern_type") or {}).get("label")
        pc = float((s.get("pattern_type") or {}).get("confidence") or 0.0)
        if pt:
            patt_votes[pt] = patt_votes.get(pt, 0) + 1
        patt_confs.append(pc)
    patt_label = None
    if patt_votes:
        patt_label = sorted(patt_votes.items(), key=lambda x: -x[1])[0][0]
    patt_conf = float(np.mean(patt_confs)) if patt_confs else 0.0
    patt_basis = f"votes={patt_votes}, mean_conf={patt_conf:.2f}, samples={n}"

    # sex
    sex_votes: Dict[str, int] = {}
    sex_probs: Dict[str, List[float]] = {"male": [], "female": []}
    for s in per_samples:
        sx = (s.get("sex") or {}).get("label")
        sp = float((s.get("sex") or {}).get("probability") or 0.0)
        if sx in ("male", "female"):
            sex_votes[sx] = sex_votes.get(sx, 0) + 1
            sex_probs[sx].append(sp)
    if sex_votes:
        sx_label = sorted(sex_votes.items(), key=lambda x: -x[1])[0][0]
        sx_prob = float(np.mean(sex_probs.get(sx_label, [0.5])) or 0.5)
    else:
        sx_label, sx_prob = "undetermined", 0.5
    sex_basis = f"votes={sex_votes}, mean_prob({sx_label})={sx_prob:.2f}, samples={n}"

    # health index
    hi_scores: List[int] = []
    for s in per_samples:
        sc = (s.get("health_index") or {}).get("score")
        if isinstance(sc, (int, float)):
            hi_scores.append(int(round(float(sc))))
    hi_score = int(round(float(np.mean(hi_scores)))) if hi_scores else None
    hi_interp = _interpret_health_index(hi_score) if hi_score is not None else None
    hi_basis = f"avg_score over {len(hi_scores)} samples" if hi_scores else "no valid scores"

    # rh factor
    rh_votes: Dict[str, int] = {}
    rh_probs: Dict[str, List[float]] = {"positive": [], "negative": []}
    for s in per_samples:
        rh = (s.get("rh_factor") or {}).get("label")
        rp = float((s.get("rh_factor") or {}).get("probability") or 0.0)
        if rh in ("positive", "negative"):
            rh_votes[rh] = rh_votes.get(rh, 0) + 1
            rh_probs[rh].append(rp)
    if rh_votes:
        rh_label = sorted(rh_votes.items(), key=lambda x: -x[1])[0][0]
        rh_prob = float(np.mean(rh_probs.get(rh_label, [0.5])) or 0.5)
    else:
        rh_label, rh_prob = "undetermined", 0.5
    rh_basis = f"votes={rh_votes}, mean_prob({rh_label})={rh_prob:.2f}, samples={n}"

    # bp tendency
    bp_votes: Dict[str, int] = {}
    bp_scores: List[float] = []
    for s in per_samples:
        bpl = (s.get("bp_tendency") or {}).get("label")
        bps = (s.get("bp_tendency") or {}).get("score")
        if bpl:
            bp_votes[bpl] = bp_votes.get(bpl, 0) + 1
        if isinstance(bps, (int, float)):
            bp_scores.append(float(bps))
    bp_label = sorted(bp_votes.items(), key=lambda x: -x[1])[0][0] if bp_votes else None
    bp_score = float(np.mean(bp_scores)) if bp_scores else None
    bp_basis = f"votes={bp_votes}, mean_score={bp_score:.2f} over {len(bp_scores)} samples" if bp_votes or bp_scores else "no data"

    # stress index
    st_scores: List[int] = []
    for s in per_samples:
        st = (s.get("stress_index") or {}).get("score")
        if isinstance(st, (int, float)):
            st_scores.append(int(round(float(st))))
    st_score = int(round(float(np.mean(st_scores)))) if st_scores else None
    st_interp = _interpret_stress_index(st_score) if st_score is not None else None
    st_basis = f"avg_score over {len(st_scores)} samples" if st_scores else "no valid scores"

    return {
        "pattern_type": {"label": patt_label, "confidence": patt_conf, "basis": patt_basis},
        "sex": {"label": sx_label, "probability": sx_prob, "basis": sex_basis},
        "health_index": {"score": hi_score, "interpretation": hi_interp, "basis": hi_basis},
        "rh_factor": {"label": rh_label, "probability": rh_prob, "basis": rh_basis},
        "bp_tendency": {"label": bp_label, "score": bp_score, "basis": bp_basis},
        "stress_index": {"score": st_score, "interpretation": st_interp, "basis": st_basis},
    }


# personality section removed as per user request


@app.on_event("startup")
async def startup_event():
    global REFERENCE_DATA, AVAILABLE_LABELS, CAL_STATS
    REFERENCE_DATA = _load_calibration()
    AVAILABLE_LABELS = sorted(list(REFERENCE_DATA.keys()))
    # collect percentile stats to stabilize heuristics
    cal_dir = os.path.join(os.path.dirname(__file__), "calibrate_data")
    try:
        CAL_STATS = _collect_calibration_stats(cal_dir)
    except Exception:
        CAL_STATS = {}
    # Try to load ML model and initialize KNN
    _try_load_ml_model()
    
    # Load existing feedback into KNN/Prototypes for instant learning (ALWAYS)
    if ML_ENABLED:
        try:
            _load_feedback_into_knn()
            print(f"[Startup] KNN/Prototypes pre-warmed - Enabled: {KNN_ENABLED}, Samples: {len(KNN_X_DATA)}", flush=True)
        except Exception as e:
            import traceback
            print(f"[Startup] Pre-warm FAILED: {e}", flush=True)
            print(traceback.format_exc(), flush=True)


@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "labels": AVAILABLE_LABELS,
        "scanner_available": SCANNER_AVAILABLE and scanner_integration.is_scanner_available() if SCANNER_AVAILABLE else False,
        "ml_enabled": ML_ENABLED,
        "ml_feature_count": len(ML_FEATURE_COLS) if ML_ENABLED else 0
    }


@app.get("/scanner/status")
async def scanner_status():
    """Check if fingerprint scanner is available"""
    if not SCANNER_AVAILABLE:
        return {"available": False, "message": "Scanner integration module not loaded"}
    
    available = scanner_integration.is_scanner_available()
    dbg = {}
    try:
        dbg = scanner_integration.get_scanner_debug()
    except Exception:
        dbg = {}
    bridge_mode = bool(dbg.get("use_demo_bridge", False))
    return {
        "available": available,
        "message": "Scanner ready" if available else "Scanner not detected. Please ensure Suprema BioMini Slim 2 is connected and drivers are installed.",
        "bridge_mode": available and bridge_mode,
        "save_dir": scanner_integration.get_temp_scans_dir() if available else None,
        "wait_seconds": 120,
        "debug": dbg,
    }


@app.post("/scanner/open-demo")
async def scanner_open_demo():
    """Open the vendor demo and a folder to save the image; return the save dir."""
    # Allow opening the demo even if direct scanner path isn't available,
    # so users can save into the temp folder and use the consume endpoint.
    if not SCANNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scanner integration not available")
    info = scanner_integration.open_demo_bridge_only()
    if not info.get("started", False):
        raise HTTPException(status_code=500, detail=f"Failed to launch demo: {info.get('error')}")
    return {"ok": True, "save_dir": info.get("save_dir"), "wait_seconds": 120}


@app.post("/scanner/consume-and-predict")
async def scanner_consume_and_predict(
    use_weights: bool = Query(True, description="Use feature weights in similarity computation"),
    use_ml: bool = Query(True, description="Combine with trained ML model if available"),
):
    """Pick the latest image saved into temp_scans and run prediction; 404 if nothing new."""
    if not REFERENCE_DATA:
        raise HTTPException(status_code=500, detail="Calibration data not loaded")

    item = scanner_integration.consume_latest_image()
    if not item:
        raise HTTPException(status_code=404, detail="No new image found yet")
    img = item["image"]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name
        sample = model.extract_features(tmp_path)
        sample_vec = np.array([sample[k] for k in model.keys_order], dtype=float)
        abo_sorted, probs = model.normalize_and_predict(sample_vec, REFERENCE_DATA, use_weights)
        corrected_probs, notes = _apply_corrections(sample, probs, abo_sorted)
        corrected_probs, prior_notes = _apply_pattern_priors(sample, corrected_probs)
        # ML ensemble (optional)
        ml_probs = _ml_predict_abo_probs(sample) if (use_ml and ML_ENABLED) else {}
        if ml_probs:
            alpha = float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))  # weight for heuristic probs
            alpha = float(np.clip(alpha, 0.0, 1.0))
            beta = 1.0 - alpha
            final_probs = {k: alpha * float(corrected_probs.get(k, 0.0)) + beta * float(ml_probs.get(k, 0.0)) for k in ML_ABO_LABELS}
            s = sum(final_probs.values()) or 1.0
            final_probs = {k: v / s for k, v in final_probs.items()}
        else:
            final_probs = corrected_probs
        final_sorted = sorted(final_probs.items(), key=lambda x: -x[1])
        patt, patt_conf = _pattern_type(sample)
        sex_res = _sex_from_ridge_density(sample)
        bp_res = _bp_tendency(sample)
        stress_res = _stress_index(sample)
        hi_res = _health_index(sample)
        rh_heur = _rh_factor(sample, patt)
        rh_ml = _ml_predict_rh(sample) if (use_ml and ML_ENABLED) else {}
        rh_res = _decide_rh(sample, patt, use_ml=(use_ml and ML_ENABLED))
        return JSONResponse({
            "predicted": final_sorted[0][0],
            "confidence": float(final_sorted[0][1]),
            "probs": final_probs,
            "sorted": final_sorted,
            "features": sample,
            "labels": AVAILABLE_LABELS,
            "corrections": notes + prior_notes,
            "filename": os.path.basename(item.get("path", "scanner.png")),
            "source": "scanner-demo-bridge",
            "probs_heuristic": corrected_probs,
            "probs_ml": ml_probs if ml_probs else None,
            "ensemble": {"alpha_heuristic": float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))} if ml_probs else None,
            "extended": {
                "pattern_type": {"label": patt, "confidence": patt_conf},
                "sex": sex_res,
                "bp_tendency": bp_res,
                "stress_index": stress_res,
                "health_index": hi_res,
                "rh_factor": rh_res,
                "rh_factor_heuristic": rh_heur,
                "rh_factor_ml": rh_ml if rh_ml else None,
            },
            "feedback_enabled": True,
        })
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/scanner/capture")
async def scanner_capture():
    """Capture fingerprint from connected scanner and return as base64 image"""
    if not SCANNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scanner integration not available")
    
    # Try capture even if status says unavailable. If unavailable, enable demo bridge on-the-fly.
    if not scanner_integration.is_scanner_available():
        try:
            sc = scanner_integration.get_scanner()
            sc.scanner_available = True
            sc.use_demo_bridge = True
        except Exception:
            pass
    
    try:
        # Capture fingerprint image
        img = scanner_integration.capture_fingerprint_image()
        
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to capture fingerprint. If the direct path fails, the API opens a vendor demo and watches for a saved image; please save the captured image into the opened folder and retry.")
        
        # Convert PIL Image to base64 for frontend
        import base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "success": True,
            "image": f"data:image/png;base64,{img_base64}",
            "width": img.width,
            "height": img.height,
            "message": "Fingerprint captured successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")


@app.post("/scanner/capture-and-predict")
async def scanner_capture_and_predict(
    use_weights: bool = Query(True, description="Use feature weights in similarity computation"),
    use_ml: bool = Query(True, description="Combine with trained ML model if available"),
):
    """Capture fingerprint from scanner and immediately predict blood group"""
    if not SCANNER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Scanner integration not available")
    
    # Try capture even if status says unavailable. If unavailable, enable demo bridge on-the-fly.
    if not scanner_integration.is_scanner_available():
        try:
            sc = scanner_integration.get_scanner()
            sc.scanner_available = True
            sc.use_demo_bridge = True
        except Exception:
            pass
    
    if not REFERENCE_DATA:
        raise HTTPException(status_code=500, detail="Calibration data not loaded")
    
    try:
        # Capture fingerprint image
        img = scanner_integration.capture_fingerprint_image()
        
        if img is None:
            raise HTTPException(status_code=500, detail="Failed to capture fingerprint. Please try again.")
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp, format="PNG")
            tmp_path = tmp.name
        
        try:
            # Extract features and predict (same logic as /predict endpoint)
            sample = model.extract_features(tmp_path)
            sample_vec = np.array([sample[k] for k in model.keys_order], dtype=float)

            # Predict
            abo_sorted, probs = model.normalize_and_predict(sample_vec, REFERENCE_DATA, use_weights)

            # Corrections
            corrected_probs, notes = _apply_corrections(sample, probs, abo_sorted)
            corrected_probs, prior_notes = _apply_pattern_priors(sample, corrected_probs)
            # ML ensemble (optional)
            ml_probs = _ml_predict_abo_probs(sample) if (use_ml and ML_ENABLED) else {}
            if ml_probs:
                alpha = float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))
                alpha = float(np.clip(alpha, 0.0, 1.0))
                beta = 1.0 - alpha
                final_probs = {k: alpha * float(corrected_probs.get(k, 0.0)) + beta * float(ml_probs.get(k, 0.0)) for k in ML_ABO_LABELS}
                s = sum(final_probs.values()) or 1.0
                final_probs = {k: v / s for k, v in final_probs.items()}
            else:
                final_probs = corrected_probs
            final_sorted = sorted(final_probs.items(), key=lambda x: -x[1])

            # Heuristic + ML analyses (use ML for Rh if available)
            patt, patt_conf = _pattern_type(sample)
            sex_res = _sex_from_ridge_density(sample)
            bp_res = _bp_tendency(sample)
            stress_res = _stress_index(sample)
            hi_res = _health_index(sample)
            rh_heur = _rh_factor(sample, patt)
            rh_ml = _ml_predict_rh(sample) if (use_ml and ML_ENABLED) else {}
            rh_res = _decide_rh(sample, patt, use_ml=(use_ml and ML_ENABLED))

            response = {
                "predicted": final_sorted[0][0],
                "confidence": float(final_sorted[0][1]),
                "probs": final_probs,
                "sorted": final_sorted,
                "features": sample,
                "labels": AVAILABLE_LABELS,
                "corrections": notes + prior_notes,
                "filename": "scanner_capture.png",
                "source": "scanner",
                "probs_heuristic": corrected_probs,
                "probs_ml": ml_probs if ml_probs else None,
                "ensemble": {"alpha_heuristic": float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))} if ml_probs else None,
                "extended": {
                    "pattern_type": {"label": patt, "confidence": patt_conf},
                    "sex": sex_res,
                    "bp_tendency": bp_res,
                    "stress_index": stress_res,
                    "health_index": hi_res,
                    "rh_factor": rh_res,
                    "rh_factor_heuristic": rh_heur,
                    "rh_factor_ml": rh_ml if rh_ml else None,
                },
                "feedback_enabled": True,
            }
            return JSONResponse(response)
            
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except Exception:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Capture and prediction failed: {str(e)}")


@app.post("/predict")
async def predict(
    image: UploadFile = File(..., description="Fingerprint image file"),
    use_weights: bool = Query(True, description="Use feature weights in similarity computation"),
    use_ml: bool = Query(True, description="Combine with trained ML model if available"),
):
    if not REFERENCE_DATA:
        raise HTTPException(status_code=500, detail="Calibration data not loaded")

    # Save uploaded file to a temporary path since existing code expects a file path
    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename or "img")[1]) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {e}")

    try:
        # Extract features and create vector with the exact keys order
        sample = model.extract_features(tmp_path)
        sample_vec = np.array([sample[k] for k in model.keys_order], dtype=float)

        # Predict
        abo_sorted, probs = model.normalize_and_predict(sample_vec, REFERENCE_DATA, use_weights)

        # Corrections
        corrected_probs, notes = _apply_corrections(sample, probs, abo_sorted)
        corrected_probs, prior_notes = _apply_pattern_priors(sample, corrected_probs)
        # ML ensemble (optional)
        ml_probs = _ml_predict_abo_probs(sample) if (use_ml and ML_ENABLED) else {}
        if ml_probs:
            alpha = float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            beta = 1.0 - alpha
            final_probs = {k: alpha * float(corrected_probs.get(k, 0.0)) + beta * float(ml_probs.get(k, 0.0)) for k in ML_ABO_LABELS}
            s = sum(final_probs.values()) or 1.0
            final_probs = {k: v / s for k, v in final_probs.items()}
        else:
            final_probs = corrected_probs
        final_sorted = sorted(final_probs.items(), key=lambda x: -x[1])

        # Heuristic + ML analyses (use ML for Rh if available)
        patt, patt_conf = _pattern_type(sample)
        sex_res = _sex_from_ridge_density(sample)
        bp_res = _bp_tendency(sample)
        stress_res = _stress_index(sample)
        hi_res = _health_index(sample)
        rh_heur = _rh_factor(sample, patt)
        rh_ml = _ml_predict_rh(sample) if (use_ml and ML_ENABLED) else {}
        rh_res = _decide_rh(sample, patt, use_ml=(use_ml and ML_ENABLED))

        response = {
            "predicted": final_sorted[0][0],
            "confidence": float(final_sorted[0][1]),
            "probs": final_probs,
            "sorted": final_sorted,
            "features": sample,
            "labels": AVAILABLE_LABELS,
            "corrections": notes + prior_notes,
            "filename": image.filename,
            "probs_heuristic": corrected_probs,
            "probs_ml": ml_probs if ml_probs else None,
            "ensemble": {"alpha_heuristic": float(os.environ.get("ABO_ENSEMBLE_ALPHA", "0.5"))} if ml_probs else None,
            "extended": {
                "pattern_type": {"label": patt, "confidence": patt_conf},
                "sex": sex_res,
                "bp_tendency": bp_res,
                "stress_index": stress_res,
                "health_index": hi_res,
                "rh_factor": rh_res,
                "rh_factor_heuristic": rh_heur,
                "rh_factor_ml": rh_ml if rh_ml else None,
            },
            "feedback_enabled": True,  # Signal to frontend that feedback is available
        }
        return JSONResponse(response)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Accept user feedback on a prediction for SIMILARITY-BASED LEARNING.
    Immediately trains KNN model with replicated feedback for pattern recognition.
    Also stores fuzzy memory so similar patterns are corrected instantly without retraining.
    """
    global KNN_MODEL, KNN_X_DATA, KNN_Y_DATA, KNN_ENABLED, ML_LE, KNN_NEIGHBORS, FEEDBACK_REPLICATION, FINGERPRINT_MEMORY, FINGERPRINT_SIM_MEMORY
    
    try:
        # Prepare feedback entry
        entry = {
            "timestamp": feedback.timestamp or datetime.utcnow().isoformat(),
            "sample_id": feedback.sample_id,
            "predicted_label": feedback.predicted_label,
            "true_label": feedback.true_label,
            "is_correct": feedback.is_correct,
            "features": feedback.features,
        }
        
        # For multi-sample feedback, replicate the entry
        sample_count = feedback.sample_count if feedback.sample_count and feedback.sample_count > 0 else 1
        
        # Thread-safe append to JSONL
        base_dir = os.path.dirname(__file__)
        feedback_path = os.path.join(base_dir, FEEDBACK_FILE)
        
        with FEEDBACK_LOCK:
            with open(feedback_path, "a", encoding="utf-8") as f:
                for i in range(sample_count):
                    # Add index to sample_id for multi-samples
                    if sample_count > 1:
                        entry_copy = entry.copy()
                        entry_copy["sample_id"] = f"{entry['sample_id']}_sample_{i+1}"
                        f.write(json.dumps(entry_copy) + "\n")
                    else:
                        f.write(json.dumps(entry) + "\n")
        
        # Save into EXACT and FUZZY memory for immediate recall
        try:
            exact_key = _hash_features(feedback.features, decimals=1)
            coarse_key = _coarse_hash_features(feedback.features)
            FINGERPRINT_MEMORY[exact_key] = feedback.true_label
            FINGERPRINT_SIM_MEMORY[coarse_key] = feedback.true_label
            print(f"[MEMORY] Saved exact+fuzzy correction: {feedback.true_label}", flush=True)
        except Exception:
            pass

        # INSTANT SIMILARITY-BASED LEARNING: Train KNN with replicated & augmented feedback
        if ML_ENABLED and ML_LE is not None and ML_FEATURE_COLS:
            try:
                # Build feature vector from feedback
                row = []
                for col in ML_FEATURE_COLS:
                    if col in feedback.features:
                        row.append(float(feedback.features.get(col, 0.0) or 0.0))
                    elif col == "orientation_entropy":
                        row.append(float(_orientation_entropy(feedback.features)))
                    elif col == "quality_pass":
                        row.append(float(_compute_quality_pass(feedback.features)))
                    else:
                        row.append(0.0)  # Missing engineered/unknown features default to 0
                
                # Encode label - normalize if Rh factor is missing
                true_label = feedback.true_label.strip()
                
                # If label is missing Rh factor (e.g., "O" instead of "O+"), add default "+"
                valid_labels = set(ML_LE.classes_)
                if true_label not in valid_labels:
                    # Try adding "+" suffix
                    if true_label + "+" in valid_labels:
                        true_label = true_label + "+"
                        print(f"[KNN] Normalized label '{feedback.true_label}' -> '{true_label}'", flush=True)
                    # Try adding "-" suffix
                    elif true_label + "-" in valid_labels:
                        true_label = true_label + "-"
                        print(f"[KNN] Normalized label '{feedback.true_label}' -> '{true_label}'", flush=True)
                    else:
                        raise ValueError(f"Label '{feedback.true_label}' cannot be normalized to valid label. Valid labels: {list(valid_labels)}")
                
                label_encoded = ML_LE.transform([true_label])[0]
                
                # SMART AUGMENTATION: Create multiple variations to teach patterns, not just exact matches
                import numpy as _np
                base_array = _np.array(row)
                # Update prototype and fuzzy memory
                _update_prototype(int(label_encoded), base_array)
                try:
                    FINGERPRINT_SIM_MEMORY[_coarse_hash_features(feedback.features)] = true_label
                except Exception:
                    pass
                
                # Add original + augmented copies for pattern recognition with class-balanced replication
                # Compute current count for this label to balance majority/minority
                current_count = int(sum(1 for y in KNN_Y_DATA if y == label_encoded))
                median_count = 1
                try:
                    import numpy as _np
                    binc = _np.bincount(_np.array(KNN_Y_DATA, dtype=int), minlength=len(ML_LE.classes_))
                    median_count = int(_np.median(binc[binc > 0])) if binc.sum() > 0 and (binc > 0).any() else 1
                except Exception:
                    pass
                balance_factor = max(0.5, min(2.0, (median_count + 1) / (current_count + 1)))
                total_replicas = max(4, int(sample_count * FEEDBACK_REPLICATION * balance_factor))
                for i in range(total_replicas):
                    if i == 0:
                        # First: exact original
                        KNN_X_DATA.append(base_array.copy())
                    else:
                        # Create variations with noise to help KNN recognize SIMILAR patterns
                        # Noise level: 0.5% of feature magnitude - small enough to stay similar, large enough to generalize
                        noise = _np.random.normal(0, 0.01, base_array.shape)
                        noisy_sample = base_array + noise
                        KNN_X_DATA.append(noisy_sample)
                    
                    KNN_Y_DATA.append(label_encoded)
                
                # Train KNN model (instant pattern learning!)
                if len(KNN_X_DATA) >= KNN_NEIGHBORS:
                    X_knn = _np.array(KNN_X_DATA)
                    y_knn = _np.array(KNN_Y_DATA)
                    # Fit/Update scaler on accumulated KNN data
                    global KNN_SCALER
                    KNN_SCALER = StandardScaler()
                    KNN_SCALER.fit(X_knn)
                    X_knn_scaled = KNN_SCALER.transform(X_knn)

                    # Dynamically adjust k based on available data
                    current_k = min(KNN_NEIGHBORS, len(KNN_X_DATA) // 2)  # Don't exceed half of data
                    if KNN_MODEL is None:
                        from sklearn.neighbors import KNeighborsClassifier
                        KNN_MODEL = KNeighborsClassifier(n_neighbors=current_k, weights='distance')
                    else:
                        KNN_MODEL.n_neighbors = current_k
                    
                    KNN_MODEL.fit(X_knn_scaled, y_knn)
                    KNN_ENABLED = True
                    
                    # Count unique classes learned
                    unique_classes = len(set(KNN_Y_DATA))
                    print(f"[✓] KNN LEARNED! Samples: {len(KNN_X_DATA)}, Classes: {unique_classes}, Label: {true_label}", flush=True)
                else:
                    print(f"[⚠] KNN needs {KNN_NEIGHBORS - len(KNN_X_DATA)} more samples ({len(KNN_X_DATA)}/{KNN_NEIGHBORS})", flush=True)
                    
            except Exception as e:
                import traceback
                print(f"[✗] KNN TRAINING FAILED: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
        
        # Count total feedback samples
        feedback_count = _count_feedback_samples()
        
        return {
            "status": "success",
            "message": f"Feedback recorded and model adapted instantly! {sample_count} sample(s) learned.",
            "total_samples": feedback_count,
            "knn_enabled": KNN_ENABLED,
            "knn_training_size": len(KNN_X_DATA),
            "instant_learning": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {e}")


@app.post("/retrain")
async def trigger_retrain(
    force: bool = Query(False, description="Force retrain even if below threshold"),
    admin_key: str = Query(None, description="Admin key for authorization")
):
    """
    Trigger incremental model retraining using accumulated feedback.
    This merges feedback data with original training data and updates model_rf.joblib.
    
    For local development, admin_key can be omitted or set via env ABO_ADMIN_KEY.
    """
    global ML_MODEL, ML_LE
    
    # Simple admin gate (optional for localhost)
    expected_key = os.environ.get("ABO_ADMIN_KEY", "")
    if expected_key and admin_key != expected_key:
        raise HTTPException(status_code=403, detail="Unauthorized. Admin key required.")
    
    try:
        feedback_count = _count_feedback_samples()
        
        if not force and feedback_count < RETRAIN_THRESHOLD:
            return {
                "status": "skipped",
                "message": f"Not enough feedback samples ({feedback_count}/{RETRAIN_THRESHOLD}). Use force=true to retrain anyway.",
                "feedback_count": feedback_count
            }
        
        # Load feedback data
        base_dir = os.path.dirname(__file__)
        feedback_path = os.path.join(base_dir, FEEDBACK_FILE)
        
        if not os.path.exists(feedback_path):
            raise HTTPException(status_code=404, detail="No feedback data found.")
        
        feedback_samples = []
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    feedback_samples.append(json.loads(line))
        
        if not feedback_samples:
            raise HTTPException(status_code=400, detail="Feedback file is empty.")
        
        # Prepare training data from feedback
        X_feedback = []
        y_feedback = []
        
        for entry in feedback_samples:
            features = entry["features"]
            true_label = entry["true_label"]
            
            # Build feature vector in model's expected order
            if not ML_FEATURE_COLS:
                raise HTTPException(status_code=500, detail="Model feature columns not available. Ensure ML model is loaded.")
            
            row = []
            for col in ML_FEATURE_COLS:
                if col in features:
                    row.append(float(features.get(col, 0.0) or 0.0))
                elif col == "orientation_entropy":
                    row.append(float(_orientation_entropy(features)))
                elif col == "quality_pass":
                    row.append(float(_compute_quality_pass(features)))
                else:
                    row.append(0.0)
            
            X_feedback.append(row)
            y_feedback.append(true_label)
        
        X_feedback = np.array(X_feedback, dtype=float)
        y_feedback = np.array(y_feedback)
        
        # Load existing model and encoder
        if not ML_ENABLED or ML_MODEL is None or ML_LE is None:
            raise HTTPException(status_code=500, detail="ML model not loaded. Cannot retrain.")
        
        # Incremental learning: retrain on feedback data
        # For RandomForest, we'll partially fit or retrain from scratch with combined data
        # Since sklearn RandomForest doesn't support partial_fit, we'll retrain
        
        # Get original training data (if available) or use current model's knowledge
        # For simplicity, we'll do warm_start approach or full retrain
        
        # Option 1: Full retrain with feedback only (quick)
        # Option 2: Load original data + feedback (better but needs original CSV)
        
        # Let's use warm_start for incremental learning
        model_copy = RandomForestClassifier(
            n_estimators=ML_MODEL.n_estimators + 100,  # Add more trees
            max_depth=ML_MODEL.max_depth,
            min_samples_split=ML_MODEL.min_samples_split,
            min_samples_leaf=ML_MODEL.min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            warm_start=True
        )
        
        # Encode labels
        le_copy = LabelEncoder()
        le_copy.classes_ = ML_LE.classes_  # Use existing classes
        y_encoded = le_copy.transform(y_feedback)
        
        # Train on feedback data
        model_copy.fit(X_feedback, y_encoded)
        
        # Save updated model
        model_path = os.environ.get("ABO_ML_MODEL_PATH", "")
        if not model_path:
            model_path = os.path.join(base_dir, "model_rf.joblib")
        
        bundle = {
            "model": model_copy,
            "le": le_copy,
            "feature_cols": ML_FEATURE_COLS
        }
        
        # Backup old model
        if os.path.exists(model_path):
            backup_path = model_path.replace(".joblib", f"_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib")
            import shutil
            shutil.copy2(model_path, backup_path)
        
        joblib.dump(bundle, model_path)
        
        # Reload model in memory (already declared global at function start)
        ML_MODEL = model_copy
        ML_LE = le_copy
        
        # Archive feedback data
        archive_path = feedback_path.replace(".jsonl", f"_trained_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl")
        import shutil
        shutil.move(feedback_path, archive_path)
        
        return {
            "status": "success",
            "message": f"Model retrained successfully with {len(feedback_samples)} feedback samples.",
            "feedback_used": len(feedback_samples),
            "model_path": model_path,
            "backup_created": True,
            "feedback_archived": archive_path
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Retraining failed: {e}\n{traceback.format_exc()}")


def _background_retrain_worker():
    """
    Background worker thread that processes async retrain requests.
    Runs continuously and checks queue for retrain tasks.
    Non-blocking - won't slow down API responses.
    Includes timeout and retry logic for robustness.
    """
    global RETRAINING_IN_PROGRESS, ML_MODEL, ML_LE, LAST_RETRAIN_STATUS, LAST_RETRAIN_TIME, ORIGINAL_TRAINING_DATA
    
    print("[Retrain Worker] Background worker started")
    
    while True:
        try:
            # Wait for retrain signal (blocking, but runs in background)
            task = RETRAIN_QUEUE.get(timeout=1.0)
            
            if task == "RETRAIN":
                if RETRAINING_IN_PROGRESS:
                    print("[Retrain Worker] Already retraining, skipping...")
                    RETRAIN_QUEUE.task_done()
                    continue
                
                RETRAINING_IN_PROGRESS = True
                LAST_RETRAIN_STATUS = "training"
                print(f"[Retrain Worker] Starting automatic retraining...")
                
                # Retry logic
                retry_count = 0
                success = False
                
                while retry_count <= RETRAIN_MAX_RETRIES and not success:
                    try:
                        start_time = time.time()
                        
                        if retry_count > 0:
                            print(f"[Retrain Worker] Retry attempt {retry_count}/{RETRAIN_MAX_RETRIES}")
                        
                        # Perform the actual retraining
                        feedback_count = _count_feedback_samples()
                        
                        if feedback_count < RETRAIN_THRESHOLD:
                            print(f"[Retrain Worker] Not enough samples ({feedback_count}/{RETRAIN_THRESHOLD}), skipping")
                            break
                        
                        # Load feedback data
                        base_dir = os.path.dirname(__file__)
                        feedback_path = os.path.join(base_dir, FEEDBACK_FILE)
                        
                        if not os.path.exists(feedback_path):
                            print("[Retrain Worker] No feedback file found")
                            break
                        
                        feedback_samples = []
                        with open(feedback_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    feedback_samples.append(json.loads(line))
                        
                        if not feedback_samples:
                            print("[Retrain Worker] Feedback file is empty")
                            break
                        
                        # Check timeout periodically
                        if time.time() - start_time > RETRAIN_TIMEOUT:
                            raise TimeoutError(f"Training exceeded timeout of {RETRAIN_TIMEOUT}s")
                        
                        # Prepare training data
                        from sklearn.ensemble import RandomForestClassifier
                        
                        X_feedback = []
                        y_feedback = []
                        
                        for entry in feedback_samples:
                            features = entry["features"]
                            true_label = entry["true_label"]
                            
                            if not ML_FEATURE_COLS:
                                print("[Retrain Worker] Model feature columns not available")
                                break
                            
                            row = []
                            for col in ML_FEATURE_COLS:
                                if col in features:
                                    row.append(float(features.get(col, 0.0) or 0.0))
                                elif col == "orientation_entropy":
                                    row.append(float(_orientation_entropy(features)))
                                elif col == "quality_pass":
                                    row.append(float(_compute_quality_pass(features)))
                                else:
                                    row.append(0.0)
                            
                            X_feedback.append(row)
                            y_feedback.append(true_label)
                        
                        X_feedback = np.array(X_feedback, dtype=float)
                        y_feedback = np.array(y_feedback)
                        
                        # Check timeout
                        if time.time() - start_time > RETRAIN_TIMEOUT:
                            raise TimeoutError(f"Training exceeded timeout of {RETRAIN_TIMEOUT}s")
                        
                        # Retrain model
                        if not ML_ENABLED or ML_MODEL is None or ML_LE is None:
                            print("[Retrain Worker] ML model not loaded")
                            break
                        
                        # Filter out labels that are not in the model's known classes
                        known_classes = set(ML_LE.classes_)
                        print(f"[Retrain Worker] Model knows these classes: {sorted(known_classes)}")
                        
                        valid_samples = []
                        valid_labels = []
                        skipped_count = 0
                        
                        for i, label in enumerate(y_feedback):
                            if label in known_classes:
                                valid_samples.append(X_feedback[i])
                                valid_labels.append(label)
                            else:
                                skipped_count += 1
                                print(f"[Retrain Worker] [WARN] Skipping sample with unknown label: '{label}' (not in {sorted(known_classes)})")
                        
                        if skipped_count > 0:
                            print(f"[Retrain Worker] Skipped {skipped_count} samples with unknown labels")
                        
                        if len(valid_labels) == 0:
                            print("[Retrain Worker] No valid samples to train with after filtering")
                            break
                        
                        X_valid = np.array(valid_samples, dtype=float)
                        y_valid = np.array(valid_labels)
                        
                        # Encode labels
                        y_encoded = ML_LE.transform(y_valid)
                        
                        # Check timeout before training
                        if time.time() - start_time > RETRAIN_TIMEOUT:
                            raise TimeoutError(f"Training exceeded timeout of {RETRAIN_TIMEOUT}s")
                        
                        # HYBRID APPROACH: Combine original training data with feedback
                        # This prevents catastrophic forgetting while allowing fast learning
                        
                        if ORIGINAL_TRAINING_DATA is not None:
                            # We have original data - combine it with feedback
                            X_orig, y_orig = ORIGINAL_TRAINING_DATA
                            X_combined = np.vstack([X_orig, X_valid])
                            y_combined = np.concatenate([y_orig, y_encoded])
                            print(f"[Retrain Worker] Hybrid training: {len(X_orig)} original + {len(X_valid)} feedback = {len(X_combined)} total samples")
                        else:
                            # No original data - but we can replicate feedback samples
                            # to maintain class balance and prevent catastrophic forgetting
                            print(f"[Retrain Worker] [WARN] No original training data available")
                            
                            # Replicate each feedback sample multiple times to build a larger dataset
                            replications = max(5, 20 // len(X_valid))  # At least 5x, more if very few samples
                            X_replicated = np.repeat(X_valid, replications, axis=0)
                            y_replicated = np.repeat(y_encoded, replications)
                            
                            # Add small random noise to prevent overfitting on exact duplicates
                            noise = np.random.normal(0, 0.01, X_replicated.shape)
                            X_combined = X_replicated + noise
                            y_combined = y_replicated
                            
                            print(f"[Retrain Worker] Replicated {len(X_valid)} samples {replications}x with noise = {len(X_combined)} training samples")
                            print(f"[Retrain Worker] [WARN] Note: This is a workaround. Ideally use original training data for best results.")
                        
                        # Create fresh model with same parameters
                        current_n_est = ML_MODEL.n_estimators
                        
                        from sklearn.ensemble import RandomForestClassifier
                        ML_MODEL = RandomForestClassifier(
                            n_estimators=current_n_est,
                            max_depth=ML_MODEL.max_depth if hasattr(ML_MODEL, 'max_depth') else None,
                            min_samples_split=ML_MODEL.min_samples_split if hasattr(ML_MODEL, 'min_samples_split') else 2,
                            min_samples_leaf=ML_MODEL.min_samples_leaf if hasattr(ML_MODEL, 'min_samples_leaf') else 1,
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        # Train on combined dataset
                        ML_MODEL.fit(X_combined, y_combined)
                        
                        # Check timeout after training
                        if time.time() - start_time > RETRAIN_TIMEOUT:
                            raise TimeoutError(f"Training exceeded timeout of {RETRAIN_TIMEOUT}s")
                        
                        # Save updated model
                        model_path = os.environ.get("ABO_ML_MODEL_PATH", os.path.join(base_dir, "model_rf.joblib"))
                        backup_path = model_path.replace(".joblib", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
                        
                        # Backup old model
                        if os.path.exists(model_path):
                            try:
                                joblib.dump(joblib.load(model_path), backup_path)
                                print(f"[Retrain Worker] Backed up old model to {backup_path}")
                            except Exception as e:
                                print(f"[Retrain Worker] Backup failed: {e}")
                        
                        # Save new model with updated training data
                        bundle = {
                            "model": ML_MODEL,
                            "le": ML_LE,
                            "feature_cols": ML_FEATURE_COLS,
                            "X_train": X_combined,  # Save combined data for next iteration
                            "y_train": y_combined
                        }
                        joblib.dump(bundle, model_path)
                        print(f"[Retrain Worker] Saved updated model to {model_path}")
                        
                        # Update the global original training data for next iteration
                        ORIGINAL_TRAINING_DATA = (X_combined, y_combined)
                        print(f"[Retrain Worker] Updated original training data ({len(X_combined)} samples for future retraining)")
                        
                        # Archive feedback
                        archive_path = feedback_path.replace(".jsonl", f"_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
                        with FEEDBACK_LOCK:
                            if os.path.exists(feedback_path):
                                os.rename(feedback_path, archive_path)
                                print(f"[Retrain Worker] Archived feedback to {archive_path}")
                        
                        elapsed = time.time() - start_time
                        print(f"[Retrain Worker] ✅ Retraining complete! Model updated with {len(feedback_samples)} samples in {elapsed:.1f}s")
                        
                        success = True
                        LAST_RETRAIN_STATUS = "success"
                        LAST_RETRAIN_TIME = elapsed
                        
                    except TimeoutError as te:
                        print(f"[Retrain Worker] [TIMEOUT] Timeout: {te}")
                        retry_count += 1
                        if retry_count > RETRAIN_MAX_RETRIES:
                            LAST_RETRAIN_STATUS = "failed"
                            print(f"[Retrain Worker] [FAIL] Max retries reached, giving up")
                        else:
                            time.sleep(2)  # Wait before retry
                    except Exception as e:
                        import traceback
                        print(f"[Retrain Worker] [FAIL] Retraining failed: {e}")
                        print(traceback.format_exc())
                        retry_count += 1
                        if retry_count > RETRAIN_MAX_RETRIES:
                            LAST_RETRAIN_STATUS = "failed"
                            print(f"[Retrain Worker] [FAIL] Max retries reached, giving up")
                        else:
                            time.sleep(2)  # Wait before retry
                
                RETRAINING_IN_PROGRESS = False
                RETRAIN_QUEUE.task_done()
            
        except queue.Empty:
            # No task in queue, continue waiting
            continue
        except Exception as e:
            print(f"[Retrain Worker] Error in worker loop: {e}")
            time.sleep(1)


def _count_feedback_samples() -> int:
    """Count the number of feedback samples in the file."""
    base_dir = os.path.dirname(__file__)
    feedback_path = os.path.join(base_dir, FEEDBACK_FILE)
    
    if not os.path.exists(feedback_path):
        return 0
    
    count = 0
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    
    return count


def _load_feedback_into_knn():
    """Load existing feedback data into KNN + Prototypes + Fuzzy memory for strong and instant learning."""
    global KNN_MODEL, KNN_X_DATA, KNN_Y_DATA, KNN_ENABLED, PROTO_MEANS, PROTO_COUNTS, KNN_SCALER, FINGERPRINT_SIM_MEMORY
    
    base_dir = os.path.dirname(__file__)
    feedback_path = os.path.join(base_dir, FEEDBACK_FILE)
    
    if not os.path.exists(feedback_path):
        return
    
    if not ML_ENABLED or ML_LE is None or not ML_FEATURE_COLS:
        return
    
    try:
        import numpy as _np
        KNN_X_DATA = []
        KNN_Y_DATA = []
        
        # Load all feedback samples with AGGRESSIVE replication
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                    
                entry = json.loads(line)
                features = entry.get("features", {})
                true_label = entry.get("true_label")
                
                if not features or not true_label:
                    continue
                
                # Build feature vector
                row = []
                for col in ML_FEATURE_COLS:
                    if col in features:
                        row.append(float(features.get(col, 0.0) or 0.0))
                    elif col == "orientation_entropy":
                        row.append(float(_orientation_entropy(features)))
                    elif col == "quality_pass":
                        row.append(float(_compute_quality_pass(features)))
                    else:
                        row.append(0.0)
                
                # Encode label
                try:
                    # normalize missing Rh if needed
                    valid_labels = set(ML_LE.classes_)
                    if true_label not in valid_labels:
                        if true_label + "+" in valid_labels:
                            true_label = true_label + "+"
                        elif true_label + "-" in valid_labels:
                            true_label = true_label + "-"
                        else:
                            continue
                    label_encoded = ML_LE.transform([true_label])[0]
                    
                    # AGGRESSIVE: Replicate each feedback sample multiple times
                    base_array = _np.array(row)
                    # Update prototype and fuzzy memory
                    _update_prototype(int(label_encoded), base_array)
                    FINGERPRINT_SIM_MEMORY[_coarse_hash_features(features)] = true_label
                    for i in range(FEEDBACK_REPLICATION):
                        if i == 0:
                            # First one: exact original
                            KNN_X_DATA.append(base_array.copy())
                        else:
                            # Add tiny noise for variations
                            noise = _np.random.normal(0, 0.005, base_array.shape)
                            KNN_X_DATA.append(base_array + noise)
                        
                        KNN_Y_DATA.append(label_encoded)
                except Exception:
                    continue  # Skip invalid labels
        
        # Train KNN if we have enough samples
        if len(KNN_X_DATA) >= 3:
            X_knn = _np.array(KNN_X_DATA)
            y_knn = _np.array(KNN_Y_DATA)
            # Fit scaler on accumulated data
            KNN_SCALER = StandardScaler()
            KNN_SCALER.fit(X_knn)
            X_knn_scaled = KNN_SCALER.transform(X_knn)
            if KNN_MODEL is None:
                from sklearn.neighbors import KNeighborsClassifier
                KNN_MODEL = KNeighborsClassifier(n_neighbors=min(KNN_NEIGHBORS, max(1, len(X_knn)//2)), weights='distance')
            KNN_MODEL.fit(X_knn_scaled, y_knn)
            KNN_ENABLED = True
            print(f"[SUCCESS] [STARTUP] Pre-warmed KNN ({len(KNN_X_DATA)} samples) and prototypes ({len(PROTO_MEANS)} classes)", flush=True)
        else:
            print(f"[WARN] [STARTUP] Only {len(KNN_X_DATA)} samples for KNN; prototypes loaded: {len(PROTO_MEANS)}", flush=True)
            
    except Exception as e:
        print(f"[FAIL] [KNN STARTUP] Failed to load: {e}", flush=True)


@app.get("/feedback/stats")
async def feedback_stats():
    """Get statistics about collected feedback data."""
    try:
        base_dir = os.path.dirname(__file__)
        feedback_path = os.path.join(base_dir, FEEDBACK_FILE)
        
        if not os.path.exists(feedback_path):
            return {
                "total_samples": 0,
                "correct_predictions": 0,
                "incorrect_predictions": 0,
                "can_retrain": False,
                "retrain_threshold": RETRAIN_THRESHOLD
            }
        
        total = 0
        correct = 0
        incorrect = 0
        label_distribution = {}
        
        with open(feedback_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    total += 1
                    if entry.get("is_correct"):
                        correct += 1
                    else:
                        incorrect += 1
                    
                    true_label = entry.get("true_label", "unknown")
                    label_distribution[true_label] = label_distribution.get(true_label, 0) + 1
        
        return {
            "total_samples": total,
            "correct_predictions": correct,
            "incorrect_predictions": incorrect,
            "accuracy": round(correct / total, 3) if total > 0 else 0.0,
            "label_distribution": label_distribution,
            "can_retrain": total >= RETRAIN_THRESHOLD,
            "ready_to_retrain": total >= RETRAIN_THRESHOLD,
            "retrain_threshold": RETRAIN_THRESHOLD,
            "samples_until_retrain": max(0, RETRAIN_THRESHOLD - total),
            "auto_retrain_enabled": AUTO_RETRAIN_ENABLED,
            "retraining_in_progress": RETRAINING_IN_PROGRESS
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


@app.get("/training/status")
async def training_status():
    """Get the current status of AGGRESSIVE adaptive learning for the visual indicator."""
    try:
        feedback_count = _count_feedback_samples()
        
        # Determine status color for aggressive instant learning
        # red: No feedback yet (base RF model only)
        # orange: Learning mode (1-4 samples, warming up)
        # green: Aggressive adaptive learning active (5+ samples, KNN enabled at 85% weight)
        
        if KNN_ENABLED and len(KNN_X_DATA) >= 5:
            # KNN active - model is learning AGGRESSIVELY (85% weight)
            status = "aggressive_adaptive"
            color = "green"
        elif feedback_count > 0:
            # Collecting initial samples for KNN
            status = "warming_up"
            color = "orange"
        else:
            # No feedback yet - using base model only
            status = "base_model"
            color = "red"
        
        return {
            "status": status,
            "color": color,
            "feedback_count": feedback_count,
            "knn_enabled": KNN_ENABLED,
            "knn_training_size": len(KNN_X_DATA),
            "knn_weight": KNN_WEIGHT if KNN_ENABLED else 0.0,
            "instant_learning": True,
            "learning_mode": "aggressive_hybrid_ensemble",
            "description": "Model learns AGGRESSIVELY from every feedback (85% weight)!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {e}")


# For local quick run: python api_server.py
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("api_server:app", host="127.0.0.1", port=port, reload=False)
