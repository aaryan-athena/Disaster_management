from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple


class PoseDetector:
    """MediaPipe Pose detector for body pose estimation."""
    
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Pose detector.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect(self, image_bgr: np.ndarray) -> Optional[Dict]:
        """
        Detect pose in the image.
        
        Args:
            image_bgr: Input image in BGR format
            
        Returns:
            Dictionary containing pose landmarks and pose state, or None if no pose detected
        """
        if image_bgr is None or image_bgr.size == 0:
            return None
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Get image dimensions
        h, w = image_bgr.shape[:2]
        
        # Extract key landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get key points
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Calculate pose state
        pose_state = self._classify_pose(
            nose, left_shoulder, right_shoulder,
            left_hip, right_hip, left_knee, right_knee,
            left_ankle, right_ankle
        )
        
        # Check if face is visible (nose visibility)
        face_visible = nose.visibility > 0.5
        
        # Calculate bounding box for the person
        x_coords = [lm.x for lm in landmarks if lm.visibility > 0.5]
        y_coords = [lm.y for lm in landmarks if lm.visibility > 0.5]
        
        if x_coords and y_coords:
            x_min = int(min(x_coords) * w)
            x_max = int(max(x_coords) * w)
            y_min = int(min(y_coords) * h)
            y_max = int(max(y_coords) * h)
            
            # Add padding
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        else:
            bbox = None
        
        return {
            'landmarks': results.pose_landmarks,
            'pose_state': pose_state,
            'face_visible': face_visible,
            'bbox': bbox,
            'nose': (int(nose.x * w), int(nose.y * h)) if nose.visibility > 0.5 else None
        }
    
    def _classify_pose(
        self,
        nose, left_shoulder, right_shoulder,
        left_hip, right_hip, left_knee, right_knee,
        left_ankle, right_ankle
    ) -> str:
        """
        Classify the pose as standing, sitting, or fallen.
        
        Args:
            Various landmark points
            
        Returns:
            Pose state: "standing", "sitting", or "fallen"
        """
        # Calculate average positions
        shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        knee_y = (left_knee.y + right_knee.y) / 2
        ankle_y = (left_ankle.y + right_ankle.y) / 2
        
        # Calculate body angle (vertical alignment)
        shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        hip_x = (left_hip.x + right_hip.x) / 2
        
        # Vertical distance from shoulders to hips (normalized)
        torso_height = hip_y - shoulder_y  # Positive means hips below shoulders
        
        # Vertical distance from hips to knees
        hip_to_knee = knee_y - hip_y
        
        # Vertical distance from knees to ankles
        knee_to_ankle = ankle_y - knee_y
        
        # Total leg length
        leg_height = ankle_y - hip_y
        
        # Horizontal deviation (how much the body is tilted)
        horizontal_deviation = abs(shoulder_x - hip_x)
        
        # Check visibility of key points
        lower_body_visible = (
            left_knee.visibility > 0.5 and right_knee.visibility > 0.5 and
            left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5
        )
        
        # Classification logic
        # 1. Check if fallen (body horizontal or severely tilted)
        if horizontal_deviation > 0.2:
            return "fallen"
        
        # 2. Check if torso is horizontal (fallen)
        if torso_height < 0.05:  # Shoulders and hips at same level
            return "fallen"
        
        # 3. If lower body not visible, likely sitting or upper body only
        if not lower_body_visible:
            return "sitting"
        
        # 4. Analyze leg configuration
        # Standing: legs extended, knees below hips, ankles below knees
        # Sitting: knees at or above hip level, legs bent
        
        # Calculate the ratio of hip-to-knee vs knee-to-ankle
        if abs(leg_height) > 0.01:
            knee_position_ratio = hip_to_knee / leg_height
        else:
            knee_position_ratio = 0
        
        # For standing: knees should be roughly midway or lower (ratio ~0.4-0.6)
        # For sitting: knees are higher up or legs are compressed (ratio > 0.7 or leg_height small)
        
        # Check if legs are extended (standing indicator)
        if leg_height > 0.3 and torso_height > 0.1:
            # Long legs visible, good torso height
            if knee_position_ratio < 0.65:
                # Knees are in lower portion of legs - standing
                return "standing"
            else:
                # Knees are high up - sitting
                return "sitting"
        elif leg_height > 0.2 and torso_height > 0.08:
            # Moderate leg length
            if knee_position_ratio < 0.6:
                return "standing"
            else:
                return "sitting"
        elif leg_height < 0.2:
            # Short leg height - likely sitting with legs compressed
            return "sitting"
        else:
            # Default to sitting if unclear
            return "sitting"
    
    def draw_pose(self, image_bgr: np.ndarray, pose_result: Dict) -> np.ndarray:
        """
        Draw pose landmarks on the image.
        
        Args:
            image_bgr: Input image in BGR format
            pose_result: Result from detect() method
            
        Returns:
            Image with pose landmarks drawn
        """
        if pose_result is None or pose_result['landmarks'] is None:
            return image_bgr
        
        # Draw pose landmarks
        self.mp_drawing.draw_landmarks(
            image_bgr,
            pose_result['landmarks'],
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
        
        return image_bgr
    
    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
