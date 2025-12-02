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

# Timestamp verification settings
TIMESTAMP_TOLERANCE_SECONDS = 3  # Allow up to 3 seconds difference
TIMESTAMP_WARNING_SECONDS = 10   # Warn if difference is significant

# Load YOLO classes from coco.names (if needed)
try:
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
except:
    classes = []

def decode_timestamp_steganography(frame):
    """
    Decode timestamp from the least significant bits of pixel values.
    
    Args:
        frame: BGR image with encoded timestamp
    
    Returns:
        Decoded timestamp string or None if decoding fails
    """
    try:
        height, width, channels = frame.shape
        
        # Use same deterministic pattern as encoding
        np.random.seed(42)
        total_pixels = height * width * channels
        
        # First, read the length (16 bits)
        pixel_positions = np.random.permutation(total_pixels)
        
        length_bits = ''
        for i in range(16):
            pos = pixel_positions[i]
            channel = pos % channels
            pixel_pos = pos // channels
            row = pixel_pos // width
            col = pixel_pos % width
            
            # Extract LSB
            pixel_val = frame[row, col, channel]
            length_bits += str(pixel_val & 1)
        
        # Convert length from binary
        message_length = int(length_bits, 2)
        
        if message_length <= 0 or message_length > 100:  # Sanity check
            return None
        
        # Read the message bits
        message_bits = ''
        for i in range(16, 16 + message_length * 8):
            pos = pixel_positions[i]
            channel = pos % channels
            pixel_pos = pos // channels
            row = pixel_pos // width
            col = pixel_pos % width
            
            pixel_val = frame[row, col, channel]
            message_bits += str(pixel_val & 1)
        
        # Convert bits to bytes
        timestamp_bytes = bytearray()
        for i in range(0, len(message_bits), 8):
            byte = message_bits[i:i+8]
            timestamp_bytes.append(int(byte, 2))
        
        return timestamp_bytes.decode('utf-8')
    except Exception as e:
        return None

def verify_timestamp(timestamp_str, tolerance_seconds=3, warning_seconds=10):
    """
    Check if decoded timestamp is close to current time.
    
    Returns:
        (status, diff_seconds, message)
        status: 'valid', 'warning', 'tampered', or 'error'
    """
    try:
        ts = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        now = datetime.now()
        diff = (now - ts).total_seconds()  # Positive if timestamp is in the past
        abs_diff = abs(diff)
        
        if abs_diff <= tolerance_seconds:
            return 'valid', diff, f"Timestamp OK ({abs_diff:.1f}s)"
        elif abs_diff <= warning_seconds:
            if diff > 0:
                return 'warning', diff, f"WARNING: Timestamp {abs_diff:.1f}s OLD"
            else:
                return 'warning', diff, f"WARNING: Timestamp {abs_diff:.1f}s in FUTURE"
        else:
            if diff > 0:
                return 'tampered', diff, f"TAMPERED: Timestamp {abs_diff:.1f}s OLD (possible replay attack)"
            else:
                return 'tampered', diff, f"TAMPERED: Timestamp {abs_diff:.1f}s in FUTURE"
    except Exception as e:
        return 'error', None, f"Invalid timestamp format: {timestamp_str}"

def compute_frame_hash(frame):
    """
    Compute a hash of the frame for simple duplicate detection.
    
    Args:
        frame: BGR image frame
    
    Returns:
        Hash string
    """
    # Resize to smaller size for faster hashing
    small_frame = cv2.resize(frame, (64, 64))
    # Convert to grayscale
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    # Compute hash
    frame_hash = hashlib.md5(gray.tobytes()).hexdigest()
    return frame_hash

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

def handle_client(conn, addr, window_id):
    print(f"[INFO] Connected to sender at {addr}")
    
    conn_file = conn.makefile('rb')
    # Sequence-based loop detection
    #detector = SequenceLoopDetector(window_size=8, hash_size=8, repeat_threshold=3, interval_tolerance=2)
    #sequence_repeat_count = 0
    
    # Simple frame hash loop detection
    seen_hashes = set()
    frame_repeat_count = 0
    max_seen_hashes = 1000  # Limit memory usage
    
    # Timestamp tracking
    tamper_count = 0
    warning_count = 0
    valid_timestamp_count = 0
    decode_fail_count = 0
    
    # Statistics
    total_frames = 0
    last_valid_timestamp = None
    simple_loop_detections = 0

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
            
            total_frames += 1

            # Simple frame hash-based loop detection
            frame_hash = compute_frame_hash(frame)
            if frame_hash in seen_hashes:
                frame_repeat_count += 1
            else:
                seen_hashes.add(frame_hash)
                frame_repeat_count = 0
                
                # Limit memory usage by clearing old hashes
                if len(seen_hashes) > max_seen_hashes:
                    seen_hashes.clear()
            
            # Track simple loop detections
            if frame_repeat_count >= REPEAT_THRESHOLD:
                simple_loop_detections += 1

            # Sequence-based loop detection
            # sequence_loop_detected, info = detector.add_frame(frame)
            # if sequence_loop_detected:
            #     sequence_repeat_count += 1
            # else:
            #     sequence_repeat_count = 0

            # Decode steganographic timestamp
            decoded_ts = decode_timestamp_steganography(frame)
            
            if decoded_ts:
                status, time_diff, message = verify_timestamp(
                    decoded_ts, 
                    TIMESTAMP_TOLERANCE_SECONDS, 
                    TIMESTAMP_WARNING_SECONDS
                )
                
                if status == 'valid':
                    valid_timestamp_count += 1
                    last_valid_timestamp = decoded_ts
                    cv2.putText(frame, f"{decoded_ts} ({abs(time_diff):.1f}s)", 
                               (20, frame.shape[0] - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                
                elif status == 'warning':
                    warning_count += 1
                    cv2.putText(frame, f"{message}", 
                               (20, frame.shape[0] - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    print(f"[WARN] {addr} - {message}")
                    
                    # Draw warning border
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                                 (0, 165, 255), 3)
                
                elif status == 'tampered':
                    tamper_count += 1
                    cv2.putText(frame, f"{message}", 
                               (20, frame.shape[0] - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print(f"[ALERT] {addr} - {message}")
                    
                    # Draw red border for tamper alert
                    cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                                 (0, 0, 255), 5)
                
                else:  # error
                    decode_fail_count += 1
                    cv2.putText(frame, f"{message}", 
                               (20, frame.shape[0] - 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                decode_fail_count += 1
                cv2.putText(frame, "Cannot decode timestamp (possible tampering)", 
                           (20, frame.shape[0] - 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"[WARN] {addr} - Failed to decode steganographic timestamp")
                
                # Draw red border
                cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), 
                             (0, 0, 255), 3)

            # Frame hash loop detection overlay
            if frame_repeat_count >= REPEAT_THRESHOLD:
                cv2.putText(frame, "LOOP DETECTED (Frame Hash)", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if frame_repeat_count == REPEAT_THRESHOLD:  # Only print once per loop
                    print(f"[ALERT] {addr} - Simple frame hash loop detected")
            
            # Sequence loop detection overlay (optional - can enable if desired)
            # if sequence_repeat_count >= REPEAT_THRESHOLD:
            #     cv2.putText(frame, "SEQUENCE LOOP DETECTED", (20, 75),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            #     if sequence_repeat_count == REPEAT_THRESHOLD:
            #         print(f"[ALERT] {addr} - Sequence loop detected")

            # Show comprehensive statistics
            stats_y = frame.shape[0] - 50
            cv2.putText(frame, f"Frames: {total_frames} | Valid: {valid_timestamp_count} | Warn: {warning_count} | Tamper: {tamper_count} | Fail: {decode_fail_count}", 
                       (20, stats_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Show current system time for reference
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, f"System Time: {current_time}", 
                       (20, stats_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Display the video frame
            cv2.imshow(f"Stream from {addr[0]}:{addr[1]}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Print statistics
                print(f"\n[STATS] {addr} Statistics:")
                print(f"  Total frames: {total_frames}")
                print(f"  Valid timestamps: {valid_timestamp_count} ({100*valid_timestamp_count/max(1,total_frames):.1f}%)")
                print(f"  Warnings: {warning_count} ({100*warning_count/max(1,total_frames):.1f}%)")
                print(f"  Tampered: {tamper_count} ({100*tamper_count/max(1,total_frames):.1f}%)")
                print(f"  Decode failures: {decode_fail_count} ({100*decode_fail_count/max(1,total_frames):.1f}%)")
                print(f"  Simple loop detections: {simple_loop_detections}")
                if last_valid_timestamp:
                    print(f"  Last valid timestamp: {last_valid_timestamp}")

    except Exception as e:
        print(f"[ERROR] {addr} - {e}")
    finally:
        conn_file.close()
        conn.close()
        cv2.destroyWindow(f"Stream from {addr[0]}:{addr[1]}")
        print(f"[INFO] {addr} - Connection closed")

# Main server setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('0.0.0.0', 9999))  # Listen on all network interfaces
server_socket.listen(5)  # Allow up to 5 pending connections
print("[INFO] Receiver server started, waiting for senders...")
print("[INFO] Steganographic timestamp detection enabled")
print(f"[INFO] Timestamp tolerance: {TIMESTAMP_TOLERANCE_SECONDS}s (valid)")
print(f"[INFO] Warning threshold: {TIMESTAMP_WARNING_SECONDS}s (suspicious)")
print("[INFO] Frame hash loop detection enabled")
print("\n[INFO] Controls:")
print("  Q - Quit stream")
print("  S - Show detailed statistics")

window_counter = 0

try:
    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr, window_counter), daemon=True).start()
        window_counter += 1
except KeyboardInterrupt:
    print("\n[INFO] Shutting down server.")
finally:
    server_socket.close()
    cv2.destroyAllWindows()