import cv2
import mediapipe as mp
import math
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

class HandVolumeController:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.setup_audio()
        self.vol_history = []
        self.smoothing_factor = 7
        self.calibration_mode = True
        self.max_distance = 150
        self.min_distance = 30

    def setup_audio(self):
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            vol_range = self.volume.GetVolumeRange()
            self.min_vol = vol_range[0]
            self.max_vol = vol_range[1]
        except Exception as e:
            print(f"Audio setup failed: {e}")
            self.volume = None

    def calculate_distance(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
        
        distance = math.hypot(x2 - x1, y2 - y1)
        
        return (x1, y1), (x2, y2), distance

    def smooth_volume(self, vol):
        self.vol_history.append(vol)
        if len(self.vol_history) > self.smoothing_factor:
            self.vol_history.pop(0)
        return np.mean(self.vol_history)

    def map_volume(self, distance):
        distance = max(self.min_distance, min(self.max_distance, distance))
        normalized = (distance - self.min_distance) / (self.max_distance - self.min_distance)
        vol = self.min_vol + (self.max_vol - self.min_vol) * normalized
        return self.smooth_volume(vol)

    def draw_interface(self, frame, thumb_pos, index_pos, distance, current_vol):
        cv2.circle(frame, thumb_pos, 12, (0, 255, 255), -1)
        cv2.circle(frame, index_pos, 12, (0, 255, 255), -1)
        cv2.line(frame, thumb_pos, index_pos, (0, 255, 255), 3)
        
        vol_percent = int((current_vol - self.min_vol) / (self.max_vol - self.min_vol) * 100)
        
        cv2.putText(frame, f'Distance: {int(distance)}', (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.rectangle(frame, (20, 60), (220, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 60), (20 + vol_percent * 2, 80), (0, 255, 0), -1)
        cv2.putText(frame, f'Volume: {vol_percent}%', (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.calibration_mode:
            cv2.putText(frame, 'CALIBRATION MODE - Move fingers to max/min', (20, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = self.hands.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                thumb_pos, index_pos, distance = self.calculate_distance(hand_landmarks, frame.shape)
                
                if self.calibration_mode:
                    self.max_distance = max(self.max_distance, distance)
                    self.min_distance = min(self.min_distance, distance)
                
                if self.volume:
                    vol_level = self.map_volume(distance)
                    self.volume.SetMasterVolumeLevel(vol_level, None)
                
                self.draw_interface(frame, thumb_pos, index_pos, distance, 
                                  vol_level if self.volume else self.min_vol)
        
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Hand Volume Controller Started")
        print("Press 'c' to toggle calibration mode")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = self.process_frame(frame)
            cv2.imshow('Hand Volume Controller', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.calibration_mode = not self.calibration_mode
                print(f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandVolumeController()
    controller.run()