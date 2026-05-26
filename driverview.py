"""
╔══════════════════════════════════════════════════════════════════╗
║           DASHCAM PRO v4 - UNIFIED ALERT SYSTEM                  ║
║   Integrated g2.py (Hardware/Sensors/OOP) + phone.py (Hands)     ║
╚══════════════════════════════════════════════════════════════════╝
"""
import sys
import cv2
import time
import math
import threading
import queue
import numpy as np
import pyttsx3
import mediapipe as mp
from scipy.spatial import distance as scipy_dist
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("DashcamPro")

# --- Configurations ---
EAR_THRESH       = 0.22
DROWSY_TIME_SEC  = 4.0
MAR_THRESH       = 1.05
MAR_TIME_SEC     = 1.2
YAW_THRESH       = 35.0
YAW_TIME_SEC     = 3.0

class VoiceEngine:
    def __init__(self):
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import os
        while True:
            text = self._q.get()
            if text is None: break
            try:
                os.system(
                    f'powershell -c "Add-Type -AssemblyName System.Speech; '
                    f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\');"'
                )
            except Exception as e:
                log.error("TTS error: %s", e)

    def speak(self, text: str):
        while not self._q.empty():
            try: self._q.get_nowait()
            except queue.Empty: break
        self._q.put(text)

    def stop(self):
        self._q.put(None)

class AlertEngine:
    COOLDOWN = 5.0
    def __init__(self, voice: VoiceEngine):
        self.voice = voice
        self._last = {}
        self._lock = threading.Lock()
    
    def trigger(self, msg: str, key: str, severity: str = "warning"):
        now = time.time()
        with self._lock:
            if now - self._last.get(key, 0) < self.COOLDOWN:
                return
            self._last[key] = now
        
        log.warning("[%s] %s", severity.upper(), msg)
        self.voice.speak(msg)
        AlertEngine.active_msg = msg
        AlertEngine.active_severity = severity
        AlertEngine.active_until = time.time() + 3.0

    active_msg = ""
    active_severity = "warning"
    active_until = 0.0

    @staticmethod
    def draw_overlay(frame):
        if time.time() >= AlertEngine.active_until:
            return frame
        h, w = frame.shape[:2]
        sev = AlertEngine.active_severity
        color = (0, 0, 200) if sev == "warning" else (0, 0, 230) if sev == "critical" else (0, 140, 0)
        panel = frame.copy()
        cv2.rectangle(panel, (0, h - 50), (w, h), color, -1)
        cv2.addWeighted(panel, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, AlertEngine.active_msg, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

def eye_aspect_ratio(pts):
    A = scipy_dist.euclidean(pts[1], pts[5])
    B = scipy_dist.euclidean(pts[2], pts[4])
    C = scipy_dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def mouth_aspect_ratio(pts):
    A = scipy_dist.euclidean(pts[1], pts[7])
    B = scipy_dist.euclidean(pts[2], pts[6])
    C = scipy_dist.euclidean(pts[3], pts[5])
    D = scipy_dist.euclidean(pts[0], pts[4])
    return (A + B + C) / (2.0 * D + 1e-6)

class FaceAnalyser:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    MOUTH = [61, 81, 13, 311, 308, 402, 14, 178]
    NOSE = 1

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self._result = None

    def process(self, rgb_frame):
        self._result = self.face_mesh.process(rgb_frame)

    def _lm(self):
        if self._result and self._result.multi_face_landmarks:
            return self._result.multi_face_landmarks[0].landmark
        return None  
    def get_eye_landmarks(self, frame):
        lm = self._lm()
        if not lm: return None, None
        h, w = frame.shape[:2]
        return [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.LEFT_EYE], \
               [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.RIGHT_EYE]

    def get_mouth_landmarks(self, frame):
        lm = self._lm()
        if not lm: return None
        h, w = frame.shape[:2]
        return [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.MOUTH]

    def get_face_center_and_head_pose(self, frame):
        lm = self._lm()
        if not lm: return None, None
        h, w = frame.shape[:2]
        nose_y = int(lm[self.NOSE].y * h)
        nose_x = int(lm[self.NOSE].x * w)
        
        le, re = self.get_eye_landmarks(frame)
        if le and re:
            lec = np.mean(le, axis=0)
            rec = np.mean(re, axis=0)
            eye_center_y = int((lec[1] + rec[1]) / 2)
            face_center_x = int((lec[0] + rec[0]) / 2)
            face_center_y = int((lec[1] + rec[1]) / 2)
            
            vertical_diff = nose_y - eye_center_y
            eye_dist = scipy_dist.euclidean(lec, rec)
            
            dl = abs(nose_x - lec[0])
            dr = abs(nose_x - rec[0])
            yaw_ratio = dl / (dl + dr + 1e-6)
            pitch_ratio = vertical_diff / (eye_dist + 1e-6)
            
            head_pose = None
            if yaw_ratio > 0.75:
                head_pose = "RIGHT"
            elif yaw_ratio < 0.25:
                head_pose = "LEFT"
            elif pitch_ratio > 0.85:
                head_pose = "DOWN"
            elif pitch_ratio < 0.2:
                head_pose = "UP"
                
            return (face_center_x, face_center_y), head_pose
        return (nose_x, nose_y), None

class AttentionTracker:
    def __init__(self):
        self.score = 0.0

    def update(self, is_safe):
        if is_safe: self.score -= 0.08
        else: self.score += 0.01
        self.score = max(0.0, min(self.score, 100.0))

    def penalize(self, val):
        self.score = min(100.0, self.score + val)

    def attention(self): return int(100 - self.score)
    def state(self):
        att = self.attention()
        if att > 85: return "FOCUSED", (0, 255, 0)
        elif att > 60: return "SLIGHT FATIGUE", (0, 255, 255)
        else: return "HIGH RISK", (0, 0, 255)

class EventTracker:
    def __init__(self, time_limit):
        self.start = None
        self.alerted = False
        self.time_limit = time_limit
        self.last_true_time = 0

    def update(self, condition):
        now = time.time()
        if condition:
            self.last_true_time = now
            if not self.start:
                self.start = now
                self.alerted = False
            if now - self.start >= self.time_limit and not self.alerted:
                self.alerted = True
                return True
        else:
            if now - self.last_true_time > 0.5:
                self.start = None
                self.alerted = False
        return False
        
    def duration(self):
        return time.time() - self.start if self.start else 0.0

class DashcamProV4:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)
        self.cap.set(4, 480)
        
        self.voice = VoiceEngine()
        self.alert = AlertEngine(self.voice)
        
        self.face = FaceAnalyser()
        self.attention = AttentionTracker()
        
        self.tr_drowsy = EventTracker(DROWSY_TIME_SEC)
        self.tr_danger = EventTracker(DROWSY_TIME_SEC * 2)
        self.tr_head   = EventTracker(YAW_TIME_SEC)
        
        self.last_yawn = 0

    def run(self):
        log.info("Started Unified Dashcam. Press ESC to exit.")
        while True:
            ret, frame = self.cap.read()
            if not ret: continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            self.face.process(rgb)
            face_center, head_pose = self.face.get_face_center_and_head_pose(frame)
            
            ear_val = 0.35
            mar_val = 0.0
            
            le, re = self.face.get_eye_landmarks(frame)
            if le and re:
                ear_val = (eye_aspect_ratio(le) + eye_aspect_ratio(re)) / 2.0
                ec = (0, 0, 255) if ear_val < EAR_THRESH else (0, 255, 100)
                cv2.polylines(frame, [np.array(le)], True, ec, 1)
                cv2.polylines(frame, [np.array(re)], True, ec, 1)
                
            mouth = self.face.get_mouth_landmarks(frame)
            if mouth:
                mar_val = mouth_aspect_ratio(mouth)
                mc = (0, 165, 255) if mar_val > MAR_THRESH else (0, 255, 100)
                cv2.polylines(frame, [np.array(mouth)], True, mc, 1)

            # Evaluate Conditions
            is_drowsy = ear_val < EAR_THRESH
            is_yawning = mar_val > MAR_THRESH
            is_distracted_pose = head_pose is not None

            # Trackers
            if self.tr_drowsy.update(is_drowsy):
                self.alert.trigger("Drowsiness Detected", "drowsy", "warning")
                self.attention.penalize(35)
                
            if self.tr_danger.update(is_drowsy):
                self.alert.trigger("Wake up driver!", "danger_sleep", "critical")
                self.attention.penalize(120)

            if is_yawning:
                if time.time() - self.last_yawn > 5:
                    self.alert.trigger("Yawning", "yawn", "warning")
                    self.attention.penalize(15)
                    self.last_yawn = time.time()

            if self.tr_head.update(is_distracted_pose):
                self.alert.trigger("focus on driving", "head_pose", "warning")
                self.attention.penalize(8)
                
            safe = not is_drowsy and not is_yawning and not is_distracted_pose
            self.attention.update(safe)

            if self.attention.attention() <= 40:
                if time.time() - getattr(self, 'last_low_att', 0) > 15:
                    self.alert.trigger("low attention. wake up", "low_att", "critical")
                    self.last_low_att = time.time()

            # HUD Rendering
            self._draw_hud(frame, ear_val, mar_val, head_pose)
            frame = AlertEngine.draw_overlay(frame)

            cv2.imshow("UNIFIED DASHCAM v4", frame)
            if cv2.waitKey(1) & 0xFF == 27: # ESC
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        self.voice.stop()

    def _draw_hud(self, frame, ear, mar, head_pose):
        # Attention
        st, stc = self.attention.state()
        cv2.putText(frame, st, (380, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, stc, 3)

        # Left Info Panel
        panel = frame.copy()
        cv2.rectangle(panel, (0, 0), (220, 200), (0, 0, 0), -1)
        cv2.addWeighted(panel, 0.45, frame, 0.55, 0, frame)

        def put(t, y, c): cv2.putText(frame, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
        
        ear_c = (0, 0, 255) if ear < EAR_THRESH else (0, 255, 100)
        put(f"EAR: {ear:.3f} [{self.tr_drowsy.duration():.1f}s]", 30, ear_c)
        
        mar_c = (0, 165, 255) if mar > MAR_THRESH else (0, 255, 100)
        put(f"MAR: {mar:.3f}", 70, mar_c)
        
        hd_c = (0, 0, 255) if head_pose else (0, 255, 100)
        put(f"Head: {head_pose}" if head_pose else "Head: STRAIGHT", 110, hd_c)

if __name__ == "__main__":
    DashcamProV4().run()