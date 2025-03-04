# congestion_evaluator.py
import time


class CongestionEvaluator:
    def __init__(self, threshold=50):
        self.congestion_time = 0
        self.vehicle_count = 0
        self.average_speed = 0
        self.frame_rate = 0
        self.congestion_value = 0
        self.threshold = threshold
        self.total_congestion_time = 0
        self.congestion_count = 0
        self.total_congestion_value = 0
        self.congestion_history = []

    def update_congestion_value(self, vehicle_count, average_speed, frame_rate):
        self.vehicle_count = vehicle_count
        self.average_speed = average_speed
        self.frame_rate = frame_rate
        self.congestion_value = self.calculate_congestion_value()
        self.congestion_time = time.time()
        self.total_congestion_value += self.congestion_value
        self.congestion_count += 1
        self.congestion_history.append(self.congestion_value)
        if self.congestion_value > self.threshold:
            self.total_congestion_time += 1

    def calculate_congestion_value(self):
        if self.vehicle_count == 0:
            return 0
        return (self.vehicle_count / self.average_speed) * self.frame_rate

    def get_congestion_data(self):
        congestion_level = "Severe Congestion" if self.congestion_value > self.threshold else "Normal"
        average_congestion_value = self.total_congestion_value / self.congestion_count if self.congestion_count > 0 else 0
        total_congestion_time_minutes = self.total_congestion_time / 60
        return {
            "congestion_time": self.congestion_time,
            "vehicle_count": self.vehicle_count,
            "average_speed": self.average_speed,
            "frame_rate": self.frame_rate,
            "congestion_value": self.congestion_value,
            "congestion_level": congestion_level,
            "total_congestion_time_minutes": total_congestion_time_minutes,
            "average_congestion_value": average_congestion_value,
            "latest_congestion_level": congestion_level
        }

    def predict_congestion(self, future_frames=10):
        if len(self.congestion_history) < future_frames:
            return {
            "predicted_value": None,
            "predicted_level": "can't predict"
        }
        recent_values = self.congestion_history[-future_frames:]
        predicted_value = sum(recent_values) / len(recent_values)
        predicted_level = "Severe Congestion" if predicted_value > self.threshold else "Normal"
        return {
        "predicted_value": predicted_value,
        "predicted_level": predicted_level
    }