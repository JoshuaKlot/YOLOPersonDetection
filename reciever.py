import socket
import threading
import struct
import cv2
import numpy as np
import imagehash
from PIL import Image
from collections import deque
import hashlib
from datetime import datetime, timedelta

# Threshold for detecting repeated frames (i.e., looped video)
REPEAT_THRESHOLD = 5
HASH_SIMILARITY_THRESHOLD = 5

# Color timestamp settings
TIMESTAMP_START_X = 10
TIMESTAMP_START_Y = 10
TIMESTAMP_BAR_WIDTH = 10
TIMESTAMP_BAR_HEIGHT = 15
TIMESTAMP_TOLERANCE_SECONDS = 3

# Load YOLO classes from coco.names (if needed)
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except:
    classes = []

class SequenceLoopDetector:
    """Detect repeated (looped) footage by fingerprinting short sequences"""
    def __init__(self, window_size=8, hash_size=8, repeat_threshold=3, interval_tolerance=2, max_fingerprints=500):
        self.window_size = int(window_size)
        self.hash_size = int(hash_size)
        self.repeat_threshold = int(repeat_threshold)
        self.interval_tolerance = int(interval_tolerance)
        self.max_fingerprints = int(max_fingerprints)

        self.window = deque(maxlen=self.window_size)
        self.window_flipped = deque(maxlen=self.window_size)
        self.fingerprints = {}
        self.frame_index = 0

    def _phash_hex(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        h = imagehash.phash(pil, hash_size=self.hash_size)
        return h.__str__()

    def add_frame(self, frame):
        """Add a frame to the sliding window. Returns (loop_detected: bool, info: dict)"""
        try:
            h = self._phash_hex(frame)
        except Exception:
            return False, {}

        flipped = cv2.flip(frame, 1)
        try:
            hf = self._phash_hex(flipped)
        except Exception:
            hf = h

        self.window.append(h)
        self.window_flipped.append(hf)
        self.frame_index += 1

        if len(self.window) < self.window_size:
            return False, {}

        concat = ''.join(self.window)
        fingerprint = hashlib.sha1(concat.encode('utf-8')).hexdigest()
        concat_f = ''.join(self.window_flipped)
        fingerprint_flipped = hashlib.sha1(concat_f.encode('utf-8')).hexdigest()

        detected = False
        info = {}

        for fp in (fingerprint, fingerprint_flipped):
            lst = self.fingerprints.get(fp)
            if lst is None:
                self.fingerprints[fp] = [self.frame_index]
            else:
                lst.append(self.frame_index)
                if len(lst) > self.max_fingerprints:
                    lst.pop(0)

                if len(lst) >= self.repeat_threshold:
                    diffs = [j - i for i, j in zip(lst[-self.repeat_threshold:-1], lst[-self.repeat_threshold+1:])]
                    if diffs:
                        maxd = max(diffs)
                        mind = min(diffs)
                        if maxd - mind <= self.interval_tolerance:
                            detected = True
                            info = {
                                'fingerprint': fp,
                                'periods': diffs,
                                'last_indices': lst[-self.repeat_threshold:]
                            }
                            break

        if len(self.fingerprints) > self.max_fingerprints:
            keys = list(self.fingerprints.keys())[: len(self.fingerprints) - self.max_fingerprints // 2]
            for k in keys:
                self.fingerprints.pop(k, None)

        return detected, info

def decode_color_timestamp(frame, start_x=10, start_y=10, bar_width=10, bar_height=15, num_chars=19):
    """Extract color-encoded timestamp from frame.
    Returns decoded timestamp string or None if extraction fails.
    """
    # Define the same color map used for encoding
    color_map = {
        '0': (255, 0, 0),      # Blue
        '1': (0, 255, 0),      # Green
        '2': (0, 0, 255),      # Red
        '3': (255, 255, 0),    # Cyan
        '4': (255, 0, 255),    # Magenta
        '5': (0, 255, 255),    # Yellow
        '6': (128, 0, 255),    # Purple
        '7': (255, 128, 0),    # Orange
        '8': (0, 128, 255),    # Light Blue
        '9': (128, 255, 0),    # Lime
        '-': (255, 255, 255),  # White
        ' ': (128, 128, 128),  # Gray
        ':': (200, 200, 200),  # Light Gray
    }
    
    # Invert the map for decoding
    reverse_map = {v: k for k, v in color_map.items()}
    
    decoded_chars = []
    
    for i in range(num_chars):
        x1 = start_x + i * bar_width
        x2 = x1 + bar_width
        y1 = start_y
        y2 = start_y + bar_height
        
        # Check bounds
        if x2 >= frame.shape[1] or y2 >= frame.shape[0]:
            return None
        
        # Extract the color bar region
        roi = frame[y1:y2, x1:x2]
        
        # Get average color in the region
        avg_color = cv2.mean(roi)[:3]  # BGR
        avg_color = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
        
        # Find closest matching color with tolerance
        best_match = None
        best_distance = float('inf')
        
        for target_color, char in reverse_map.items():
            # Calculate Euclidean distance in color space
            distance = sum((a - b) ** 2 for a, b in zip(avg_color, target_color)) ** 0.5
            if distance < best_distance:
                best_distance = distance
                best_match = char
        
        # If color is too far from any expected color, fail
        if best_distance > 50:  # Tolerance threshold
            return None
        
        decoded_chars.append(best_match)
    
    return ''.join(decoded_chars)

def verify_timestamp(timestamp_str, tolerance_seconds=3):
    """Check if decoded timestamp is close to current time"""
    try:
        ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        diff = abs((ts - now).total_seconds())
        return diff <= tolerance_seconds, diff
    except:
        return False, None

def handle_client(conn, addr, window_id):
    print(f"[INFO] Connected to sender at {addr}")
    
    conn_file = conn.makefile('rb')
    # Loop detection disabled (SequenceLoopDetector instantiation commented out)
    # detector = SequenceLoopDetector(window_size=8, hash_size=8, repeat_threshold=3, interval_tolerance=2)
    repeat_count = 0
    tamper_count = 0
    valid_timestamp_count = 0
    # Per-connection mirror toggle (press 'm' to toggle)
    mirror_stream = False

    try:
        while True:
            packed_len = conn_file.read(struct.calcsize('<L'))
            if not packed_len:
                break

            frame_size = struct.unpack('<L', packed_len)[0]
            if frame_size == 0:
                break

            frame_data = conn_file.read(frame_size)
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # Loop detection disabled
            # loop_detected, info = detector.add_frame(frame)
            # if loop_detected:
            #     repeat_count += 1
            # else:
            #     repeat_count = 0

            # Decode color timestamp
            decoded_ts = decode_color_timestamp(frame, TIMESTAMP_START_X, TIMESTAMP_START_Y, 
                                                TIMESTAMP_BAR_WIDTH, TIMESTAMP_BAR_HEIGHT)
            
            timestamp_valid = False
            if decoded_ts:
                is_valid, time_diff = verify_timestamp(decoded_ts, TIMESTAMP_TOLERANCE_SECONDS)
                if is_valid:
                    timestamp_valid = True
                    valid_timestamp_count += 1
                    cv2.putText(frame, f"TS OK: {decoded_ts} ({time_diff:.1f}s diff)", 
                               (20, frame.shape[0] - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    tamper_count += 1
                    cv2.putText(frame, f"TAMPERED: Old timestamp {decoded_ts} ({time_diff:.1f}s diff)", 
                               (20, frame.shape[0] - 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(f"[WARN] {addr} - Timestamp out of range: {decoded_ts} (diff: {time_diff:.1f}s)")
            else:
                tamper_count += 1
                cv2.putText(frame, "TAMPERED: Cannot decode timestamp", 
                           (20, frame.shape[0] - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"[WARN] {addr} - Failed to decode color timestamp")

            # Loop detection overlay disabled
            # if repeat_count >= REPEAT_THRESHOLD:
            #     cv2.putText(frame, "Loop Detected", (20, 40),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Apply mirror toggle if enabled
            if mirror_stream:
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, "MIRROR", (frame.shape[1]-110, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show statistics
            cv2.putText(frame, f"Valid: {valid_timestamp_count} | Tampered: {tamper_count}", 
                       (20, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Display the video frame
            cv2.imshow(f"Stream from {addr[0]}:{addr[1]}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mirror_stream = not mirror_stream
                print(f"[INFO] {addr} - Mirror stream {'ON' if mirror_stream else 'OFF'}")

    except Exception as e:
        print(f"[ERROR] {addr} - {e}")
    finally:
        conn_file.close()
        conn.close()
        cv2.destroyWindow(f"Stream from {addr[0]}:{addr[1]}")

# Main server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999))  # Listen on all network interfaces
server_socket.listen(5)  # Allow up to 5 pending connections
print("[INFO] Receiver server started, waiting for senders...")
print("[INFO] Color-encoded timestamp detection enabled")

window_counter = 0

try:
    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr, window_counter), daemon=True).start()
        window_counter += 1
except KeyboardInterrupt:
    print("[INFO] Shutting down server.")
finally:
    server_socket.close()
    cv2.destroyAllWindows()