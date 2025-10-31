"""
Temporal smoothing and debouncing for keyword spotting
Reduces false positives by requiring consistent detections
"""

import collections
from typing import Tuple, Optional


class TemporalSmoother:
    """
    Temporal smoothing filter for keyword detection
    Requires multiple consecutive detections before confirming
    """
    
    def __init__(self, 
                 window_size: int = 3,
                 min_confidence: float = 0.7,
                 debounce_ms: int = 500):
        """
        Args:
            window_size: Number of consecutive positive detections required
            min_confidence: Minimum confidence threshold
            debounce_ms: Minimum time between detections (milliseconds)
        """
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.debounce_ms = debounce_ms / 1000.0  # Convert to seconds
        
        # Detection history (sliding window)
        self.detection_window = collections.deque(maxlen=window_size)
        self.confidence_window = collections.deque(maxlen=window_size)
        
        # Last confirmed detection time
        self.last_detection_time = None
        
        # Running average confidence
        self.running_confidence = 0.0
        self.confidence_alpha = 0.3  # Exponential moving average factor
        
    def update(self, is_hello: bool, confidence: float, current_time: float) -> Tuple[bool, float]:
        """
        Update smoother with new detection result
        
        Args:
            is_hello: Raw detection result
            confidence: Confidence score
            current_time: Current timestamp (seconds)
        
        Returns:
            (smoothed_is_hello, smoothed_confidence) tuple
        """
        # Add to window
        self.detection_window.append(1 if is_hello else 0)
        self.confidence_window.append(confidence)
        
        # Update running confidence (exponential moving average)
        if self.running_confidence == 0.0:
            self.running_confidence = confidence
        else:
            self.running_confidence = (
                self.confidence_alpha * confidence + 
                (1 - self.confidence_alpha) * self.running_confidence
            )
        
        # Check if window is full
        if len(self.detection_window) < self.window_size:
            return False, self.running_confidence
        
        # Check debouncing (time between detections)
        if self.last_detection_time is not None:
            time_since_last = current_time - self.last_detection_time
            if time_since_last < self.debounce_ms:
                return False, self.running_confidence
        
        # Require majority of positive detections in window
        positive_count = sum(self.detection_window)
        required_positive = (self.window_size + 1) // 2  # Majority
        
        # Average confidence in window
        avg_confidence = sum(self.confidence_window) / len(self.confidence_window)
        
        # Confirm detection if:
        # 1. Majority of window is positive
        # 2. Average confidence meets threshold
        if positive_count >= required_positive and avg_confidence >= self.min_confidence:
            self.last_detection_time = current_time
            return True, avg_confidence
        
        return False, avg_confidence
    
    def reset(self):
        """Reset smoother state"""
        self.detection_window.clear()
        self.confidence_window.clear()
        self.last_detection_time = None
        self.running_confidence = 0.0


class AdvancedSmoother:
    """
    Advanced temporal smoother with adaptive thresholding
    Adjusts sensitivity based on noise level
    """
    
    def __init__(self,
                 window_size: int = 5,
                 min_confidence: float = 0.75,
                 debounce_ms: int = 800,
                 adaptive_threshold: bool = True):
        """
        Args:
            window_size: Detection window size
            min_confidence: Base confidence threshold
            debounce_ms: Debounce time in milliseconds
            adaptive_threshold: Enable adaptive threshold adjustment
        """
        self.window_size = window_size
        self.base_threshold = min_confidence
        self.current_threshold = min_confidence
        self.debounce_ms = debounce_ms / 1000.0
        self.adaptive_threshold = adaptive_threshold
        
        # Detection history
        self.detection_history = collections.deque(maxlen=window_size * 2)
        self.confidence_history = collections.deque(maxlen=window_size * 2)
        self.last_detection_time = None
        
        # Adaptive threshold parameters
        self.false_positive_count = 0
        self.consecutive_negatives = 0
        
    def update(self, is_hello: bool, confidence: float, current_time: float) -> Tuple[bool, float]:
        """Update with adaptive thresholding"""
        self.detection_history.append(is_hello)
        self.confidence_history.append(confidence)
        
        # Adaptive threshold adjustment
        if self.adaptive_threshold:
            if not is_hello:
                self.consecutive_negatives += 1
                if self.consecutive_negatives > 10:
                    # Lower threshold slightly after many negatives
                    self.current_threshold = max(
                        self.base_threshold - 0.05,
                        self.current_threshold - 0.01
                    )
            else:
                self.consecutive_negatives = 0
                if confidence > 0.9:
                    # High confidence, might be noise - raise threshold
                    self.current_threshold = min(
                        self.base_threshold + 0.1,
                        self.current_threshold + 0.02
                    )
        
        # Check debouncing
        if self.last_detection_time is not None:
            time_since_last = current_time - self.last_detection_time
            if time_since_last < self.debounce_ms:
                return False, sum(self.confidence_history) / len(self.confidence_history)
        
        # Require sustained detection pattern
        recent_window = list(self.detection_history)[-self.window_size:]
        recent_confidences = list(self.confidence_history)[-self.window_size:]
        
        if len(recent_window) < self.window_size:
            return False, sum(recent_confidences) / len(recent_confidences) if recent_confidences else 0.0
        
        # Check for sustained positive pattern
        positive_ratio = sum(recent_window) / len(recent_window)
        avg_confidence = sum(recent_confidences) / len(recent_confidences)
        
        # Require high positive ratio AND high average confidence
        if positive_ratio >= 0.7 and avg_confidence >= self.current_threshold:
            self.last_detection_time = current_time
            self.false_positive_count = 0
            return True, avg_confidence
        
        # Track potential false positives
        if is_hello and avg_confidence < self.current_threshold:
            self.false_positive_count += 1
        
        return False, avg_confidence
    
    def reset(self):
        """Reset smoother"""
        self.detection_history.clear()
        self.confidence_history.clear()
        self.last_detection_time = None
        self.current_threshold = self.base_threshold
        self.false_positive_count = 0
        self.consecutive_negatives = 0

