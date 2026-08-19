import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

SERVICES_DIR = os.path.dirname(__file__)


def _get_model_path(filename):
    return os.path.join(SERVICES_DIR, filename)


# ---------- Head Pose helpers ----------
def _estimate_head_pose(face_landmarks, img_w, img_h):
    """
    Estimate head yaw and pitch from specific face landmarks.
    Returns a qualitative label: 'Forward', 'Left', 'Right', 'Up', 'Down'
    """
    # Key landmark indices for pose estimation
    # Nose tip=1, Left eye inner=133, Right eye inner=362, Chin=152, Forehead=10
    try:
        nose = face_landmarks[1]
        left_eye = face_landmarks[133]
        right_eye = face_landmarks[362]
        chin = face_landmarks[152]
        forehead = face_landmarks[10]

        # Yaw: compare nose x to midpoint of eyes
        eye_mid_x = (left_eye.x + right_eye.x) / 2
        yaw_offset = nose.x - eye_mid_x  # positive = turned right

        # Pitch: compare nose y to midpoint of forehead-chin
        vertical_mid = (forehead.y + chin.y) / 2
        pitch_offset = nose.y - vertical_mid  # positive = looking down

        # Thresholds
        YAW_THRESH = 0.03
        PITCH_THRESH = 0.03

        if abs(yaw_offset) < YAW_THRESH and abs(pitch_offset) < PITCH_THRESH:
            return "Forward"
        elif yaw_offset > YAW_THRESH:
            return "Right"
        elif yaw_offset < -YAW_THRESH:
            return "Left"
        elif pitch_offset > PITCH_THRESH:
            return "Down"
        else:
            return "Up"
    except Exception:
        return "Unknown"


# ---------- Iris Gaze Tracking ----------
def _detect_iris_gaze(face_landmarks):
    """
    Detect if the person is looking at the camera using iris landmark positions.

    MediaPipe Face Mesh provides 478 landmarks:
      - Left iris center:  473
      - Right iris center: 468
      - Left eye corners:  33 (outer), 133 (inner)
      - Right eye corners: 362 (inner), 263 (outer)
      - Left eye top/bottom eyelid:  159, 145
      - Right eye top/bottom eyelid: 386, 374

    Logic:
      For each eye, compute the horizontal ratio:
        ratio = (iris_x - eye_inner_x) / (eye_outer_x - eye_inner_x)
      A ratio near 0.5 means the iris is centered => looking at camera.
      Also check vertical ratio to see if looking up/down.

    Returns:
        (bool: is_gazing, float: left_h_ratio, float: right_h_ratio)
    """
    try:
        # Left eye: outer=33, inner=133, iris=468, top=159, bottom=145
        left_outer  = face_landmarks[33]
        left_inner  = face_landmarks[133]
        left_iris   = face_landmarks[468]   # 468 = left iris center
        left_top    = face_landmarks[159]
        left_bottom = face_landmarks[145]

        # Right eye: inner=362, outer=263, iris=473, top=386, bottom=374
        right_inner  = face_landmarks[362]
        right_outer  = face_landmarks[263]
        right_iris   = face_landmarks[473]  # 473 = right iris center
        right_top    = face_landmarks[386]
        right_bottom = face_landmarks[374]

        # --- Horizontal gaze ratio ---
        # Left eye: iris relative to [outer(33) ... inner(133)]
        left_eye_width = abs(left_inner.x - left_outer.x)
        if left_eye_width > 0:
            left_h_ratio = (left_iris.x - left_outer.x) / left_eye_width
        else:
            return False, 0.5, 0.5

        # Right eye: iris relative to [inner(362) ... outer(263)]
        right_eye_width = abs(right_outer.x - right_inner.x)
        if right_eye_width > 0:
            right_h_ratio = (right_iris.x - right_inner.x) / right_eye_width
        else:
            return False, 0.5, 0.5

        # --- Vertical gaze ratio ---
        left_eye_height = abs(left_bottom.y - left_top.y)
        if left_eye_height > 0:
            left_v_ratio = (left_iris.y - left_top.y) / left_eye_height
        else:
            left_v_ratio = 0.5

        right_eye_height = abs(right_bottom.y - right_top.y)
        if right_eye_height > 0:
            right_v_ratio = (right_iris.y - right_top.y) / right_eye_height
        else:
            right_v_ratio = 0.5

        # --- Decision ---
        # Center zone thresholds — symmetric around 0.5 (iris centered in eye)
        H_CENTER_LOW  = 0.35
        H_CENTER_HIGH = 0.65
        V_CENTER_LOW  = 0.30
        V_CENTER_HIGH = 0.70

        left_looking  = H_CENTER_LOW < left_h_ratio  < H_CENTER_HIGH and V_CENTER_LOW < left_v_ratio  < V_CENTER_HIGH
        right_looking = H_CENTER_LOW < right_h_ratio < H_CENTER_HIGH and V_CENTER_LOW < right_v_ratio < V_CENTER_HIGH

        # Both eyes must be gazing forward
        is_gazing = left_looking and right_looking
        return is_gazing, left_h_ratio, right_h_ratio

    except Exception:
        return False, 0.5, 0.5


# ---------- Smile detection helper ----------
def _detect_smile(face_landmarks):
    """
    Detect smile using lip landmarks.
    Compares mouth width to mouth height ratio.
    Landmarks: 61 (left mouth), 291 (right mouth), 0 (top lip), 17 (bottom lip)
    """
    try:
        left_mouth = face_landmarks[61]
        right_mouth = face_landmarks[291]
        top_lip = face_landmarks[0]
        bottom_lip = face_landmarks[17]

        mouth_width = abs(right_mouth.x - left_mouth.x)
        mouth_height = abs(bottom_lip.y - top_lip.y)

        ratio = mouth_width / mouth_height if mouth_height > 0 else 0
        return ratio > 2.8  # Empirical threshold
    except Exception:
        return False


# ---------- Facial expression helper ----------
def _detect_expressions(face_landmarks):
    """
    Detect basic expressions from landmarks.
    Returns a dict of detected expressions.
    """
    expressions = {}
    try:
        # Brow raise: compare brow landmarks to eye landmarks
        # Left brow: 70, Left eye top: 159
        left_brow = face_landmarks[70]
        left_eye_top = face_landmarks[159]
        right_brow = face_landmarks[300]
        right_eye_top = face_landmarks[386]

        left_brow_raise = left_eye_top.y - left_brow.y
        right_brow_raise = right_eye_top.y - right_brow.y
        avg_brow_raise = (left_brow_raise + right_brow_raise) / 2
        expressions["brow_raised"] = avg_brow_raise > 0.04

        # Eye openness: top eyelid (159) vs bottom eyelid (145)
        left_eye_open = abs(face_landmarks[159].y - face_landmarks[145].y)
        right_eye_open = abs(face_landmarks[386].y - face_landmarks[374].y)
        avg_eye_open = (left_eye_open + right_eye_open) / 2
        expressions["wide_eyes"] = avg_eye_open > 0.025

    except Exception:
        expressions["brow_raised"] = False
        expressions["wide_eyes"] = False

    return expressions


# ---------- Posture helper ----------
def _analyze_posture(pose_landmarks):
    """
    Analyze shoulder alignment and lean.
    Landmarks: LEFT_SHOULDER=11, RIGHT_SHOULDER=12, NOSE=0
    Returns: 'Upright', 'Leaning Left', 'Leaning Right', 'Slouching'
    """
    try:
        left_shoulder = pose_landmarks[11]
        right_shoulder = pose_landmarks[12]
        nose = pose_landmarks[0]

        # Shoulder tilt
        shoulder_diff_y = abs(left_shoulder.y - right_shoulder.y)
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
        nose_offset = nose.x - shoulder_mid_x

        TILT_THRESH = 0.04
        LEAN_THRESH = 0.08

        if shoulder_diff_y > TILT_THRESH:
            return "Slouching / Tilted"
        elif nose_offset > LEAN_THRESH:
            return "Leaning Right"
        elif nose_offset < -LEAN_THRESH:
            return "Leaning Left"
        else:
            return "Upright"
    except Exception:
        return "Unknown"


def analyze_video(video_path):
    print("Loading MediaPipe Vision models...")

    # --- Setup Face Detector ---
    face_det_model = _get_model_path('blaze_face_short_range.tflite')
    face_det_options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=face_det_model),
        min_detection_confidence=0.6
    )

    # --- Setup Face Landmarker ---
    face_lm_model = _get_model_path('face_landmarker.task')
    face_lm_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=face_lm_model),
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- Setup Pose Landmarker ---
    pose_model = _get_model_path('pose_landmarker.task')
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model),
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # --- Setup Hand Landmarker ---
    hand_model = _get_model_path('hand_landmarker.task')
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model),
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps if fps > 0 else 30

    # Counters
    frames_processed = 0
    face_detected_frames = 0
    iris_gaze_frames = 0  # frames where iris gaze is directed at camera

    head_pose_counts = {"Forward": 0, "Left": 0, "Right": 0, "Up": 0, "Down": 0, "Unknown": 0}
    smile_frames = 0
    brow_raised_frames = 0
    wide_eyes_frames = 0

    posture_counts = {"Upright": 0, "Leaning Left": 0, "Leaning Right": 0, "Slouching / Tilted": 0, "Unknown": 0}
    gesture_active_frames = 0  # frames where hands detected

    with vision.FaceDetector.create_from_options(face_det_options) as face_detector, \
         vision.FaceLandmarker.create_from_options(face_lm_options) as face_landmarker, \
         vision.PoseLandmarker.create_from_options(pose_options) as pose_landmarker, \
         vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker:

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if i % frame_interval != 0:
                continue

            frames_processed += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # --- Eye Contact / Face Detection ---
            face_det_result = face_detector.detect(mp_image)
            if face_det_result.detections:
                face_detected_frames += 1

            # --- Face Landmarks: Iris Gaze, Head Pose, Smile, Expressions ---
            face_lm_result = face_landmarker.detect(mp_image)
            if face_lm_result.face_landmarks:
                lms = face_lm_result.face_landmarks[0]
                h, w = frame.shape[:2]

                # Iris-based eye contact (primary metric)
                is_gazing, _, _ = _detect_iris_gaze(lms)
                if is_gazing:
                    iris_gaze_frames += 1

                # Head pose (still tracked for direction metrics)
                pose_label = _estimate_head_pose(lms, w, h)
                head_pose_counts[pose_label] = head_pose_counts.get(pose_label, 0) + 1

                if _detect_smile(lms):
                    smile_frames += 1

                expressions = _detect_expressions(lms)
                if expressions.get("brow_raised"):
                    brow_raised_frames += 1
                if expressions.get("wide_eyes"):
                    wide_eyes_frames += 1

            # --- Body Posture ---
            pose_result = pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks:
                posture_label = _analyze_posture(pose_result.pose_landmarks[0])
                posture_counts[posture_label] = posture_counts.get(posture_label, 0) + 1

            # --- Hand Gestures ---
            hand_result = hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks:
                gesture_active_frames += 1

    cap.release()

    # --- Compute Final Scores ---
    def pct(count):
        return round((count / frames_processed) * 100) if frames_processed > 0 else 0

    # Dominant head pose
    dominant_head_pose = max(head_pose_counts, key=head_pose_counts.get)

    # Dominant posture
    dominant_posture = max(posture_counts, key=posture_counts.get)

    return {
        # Eye Contact (Iris-based: how often both irises are centered in the eye socket)
        "eye_contact_pct": pct(iris_gaze_frames),
        "frames_sampled": frames_processed,

        # Head Pose (still tracked for direction awareness)
        "head_pose_dominant": dominant_head_pose,
        "head_pose_forward_pct": pct(head_pose_counts.get("Forward", 0)),

        # Smile & Expressions
        "smile_pct": pct(smile_frames),
        "brow_raised_pct": pct(brow_raised_frames),
        "wide_eyes_pct": pct(wide_eyes_frames),

        # Body Posture
        "posture_dominant": dominant_posture,
        "upright_pct": pct(posture_counts.get("Upright", 0)),

        # Hand Gestures
        "gesture_active_pct": pct(gesture_active_frames)
    }