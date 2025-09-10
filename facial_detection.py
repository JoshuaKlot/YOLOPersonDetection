import cv2
import socket
import struct
import time
import numpy as np
import os
from pathlib import Path
import threading
from queue import Queue

def load_face_database(database_path, max_faces_per_person=5):
    """Load faces from database folder and train the recognizer - optimized for Pi"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    training_faces = []
    training_labels = []

    person_names = {}
    label_counter = 0
    
    print(f"[INFO] Loading face database from: {database_path}")
    
    if not os.path.exists(database_path):
        print(f"[ERROR] Database path does not exist: {database_path}")
        return None, {}, False
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Walk through database directory
    for person_folder in os.listdir(database_path):
        person_path = os.path.join(database_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
            
        person_faces = []
        print(f"[INFO] Processing person: {person_folder}")
        
        # Load limited number of images per person for performance
        image_files = [f for f in os.listdir(person_path) 
                      if Path(f).suffix.lower() in image_extensions]
        image_files = image_files[:max_faces_per_person]  # Limit for Pi performance
        print(f"[INFO] Num Image found: {image_files.count}")
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            
            try:
                # Load and process image
                img = cv2.imread(image_path)
                print(f"[INFO] Image found: {image_path}")
                if img is None:
                    continue
                
                # Resize image if too large (saves processing time)
                height, width = img.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Use more lenient detection parameters for Pi
                faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    # Extract and resize face ROI - smaller size for Pi
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (80, 80))  # Smaller than original
                    
                    training_faces.append(roi_gray)
                    training_labels.append(label_counter)
                    person_faces.append(roi_gray)
                    
            except Exception as e:
                print(f"[WARNING] Error processing {image_path}: {e}")
                continue
        
        if person_faces:
            person_names[label_counter] = person_folder
            label_counter += 1
            print(f"[INFO] Loaded {len(person_faces)} faces for {person_folder}")
        else:
            print(f"[WARNING] No faces found for {person_folder}")
    
    if not training_faces:
        print("[ERROR] No faces found in database!")
        return None, {}, False
    
    print(f"[INFO] Training recognizer with {len(training_faces)} faces from {len(person_names)} people...")
    
    try:
        face_recognizer.train(training_faces, np.array(training_labels))
        print("[INFO] Face recognition training completed!")
        return face_recognizer, person_names, True
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        return None, {}, False

# Get database path from user
database_path = input("Enter the path to facial database folder: ").strip()
if not database_path:
    database_path = "./face_database"  # Default path

# Load face database with Pi optimizations
print("[INFO] Loading database (limited to 5 faces per person for Pi performance)...")
face_recognizer, person_names, database_loaded = load_face_database(database_path, max_faces_per_person=5)

if not database_loaded:
    print("[ERROR] Failed to load face database. Exiting...")
    exit()

# Ask the user for the receiver IP
RECEIVER_IP = input("Enter the receiver IP address: ").strip()
RECEIVER_PORT = 9999
print(f"[INFO] Connecting to receiver at {RECEIVER_IP}:{RECEIVER_PORT}...")

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((RECEIVER_IP, RECEIVER_PORT))
conn_file = client_socket.makefile('wb')

# Initialize camera with Pi-optimized settings
cam = cv2.VideoCapture(0)
# Lower resolution for better Pi performance
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)   # Reduced from 640
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)  # Reduced from 480
cam.set(cv2.CAP_PROP_FPS, 15)            # Lower FPS for Pi

if not cam.isOpened():
    print("[ERROR] Camera not available")
    conn_file.close()
    client_socket.close()
    exit()

# Initialize face detection with Pi-optimized parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n[INFO] Loaded people in database:", list(person_names.values()))
print("\n[INFO] Pi Optimizations active:")
print("  - Reduced resolution: 480x360")
print("  - Lower FPS: 15")
print("  - Face recognition every 3rd frame")
print("  - Limited database size")
print("\n[INFO] Controls:")
print("  SPACE - Start/stop recording")
print("  L - Play recorded loop manually")
print("  A - Toggle auto-loop when known face detected")
print("  S - Show database statistics")
print("  + - Increase confidence threshold (stricter)")
print("  - - Decrease confidence threshold (looser)")
print("  Q - Quit")

# Variables with Pi optimizations
recording = False
recorded_frames = []
loop_index = 0
inserting_loop = False
auto_loop_enabled = False
face_detected_counter = 0
FACE_DETECTION_THRESHOLD = 3  # Reduced for faster response on Pi
CONFIDENCE_THRESHOLD = 120     # Higher threshold for Pi (more lenient)
frame_counter = 0
FACE_RECOGNITION_INTERVAL = 3  # Only do face recognition every 3rd frame

# Performance monitoring
last_fps_time = time.time()
fps_counter = 0
current_fps = 0

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        frame_counter += 1
        
        # FPS calculation
        fps_counter += 1
        if time.time() - last_fps_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            last_fps_time = time.time()
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only do face detection/recognition every few frames to save CPU
        known_face_detected = False
        if frame_counter % FACE_RECOGNITION_INTERVAL == 0:
            # Detect faces with Pi-optimized parameters
            faces = face_cascade.detectMultiScale(
                gray, 1.2, 3, 
                minSize=(40, 40),      # Minimum face size
                maxSize=(200, 200)     # Maximum face size
            )
            
            # Draw rectangles around faces and recognize them
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Try to recognize the face
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (80, 80))  # Smaller for Pi
                
                label, confidence = face_recognizer.predict(roi_gray)
                
                # Check if confidence is good enough
                if confidence < CONFIDENCE_THRESHOLD:
                    person_name = person_names.get(label, f"Person_{label}")
                    cv2.putText(frame, f"{person_name} ({confidence:.0f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    known_face_detected = True
                else:
                    cv2.putText(frame, f"Unknown ({confidence:.0f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Handle automatic loop triggering
        if auto_loop_enabled and known_face_detected and recorded_frames:
            face_detected_counter += 1
            if face_detected_counter >= FACE_DETECTION_THRESHOLD and not inserting_loop:
                print("[INFO] Known face detected! Starting automatic loop playback...")
                inserting_loop = True
                loop_index = 0
                face_detected_counter = 0
        else:
            if not known_face_detected:
                face_detected_counter = 0
        
        # Display status with FPS
        status_text = []
        status_text.append(f"FPS: {current_fps}")
        if recording:
            status_text.append("REC")
        status_text.append(f"DB: {len(person_names)}")
        if auto_loop_enabled:
            status_text.append("AUTO")
        if inserting_loop:
            status_text.append("LOOP")
            
        cv2.putText(frame, " | ".join(status_text), (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show confidence threshold
        cv2.putText(frame, f"Conf Threshold: {CONFIDENCE_THRESHOLD}", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("Sender", frame)
        
        # Handle keyboard input
        if key == ord(' '):
            # Start/Stop recording
            if not recording:
                print("[INFO] Recording started. Press SPACE to stop.")
                recording = True
                recorded_frames.clear()
            else:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured.")
                recording = False
                loop_index = 0
                
        elif key == ord('l') and recorded_frames:
            # Manual loop playback
            print("[INFO] Playing back recorded footage...")
            inserting_loop = True
            loop_index = 0
            
        elif key == ord('a'):
            # Toggle auto-loop
            auto_loop_enabled = not auto_loop_enabled
            print(f"[INFO] Auto-loop {'enabled' if auto_loop_enabled else 'disabled'}")
            
        elif key == ord('+') or key == ord('='):
            # Increase confidence threshold (stricter)
            CONFIDENCE_THRESHOLD = min(150, CONFIDENCE_THRESHOLD + 10)
            print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD} (stricter)")
            
        elif key == ord('-'):
            # Decrease confidence threshold (more lenient)
            CONFIDENCE_THRESHOLD = max(30, CONFIDENCE_THRESHOLD - 10)
            print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD} (more lenient)")
            
        elif key == ord('s'):
            # Show database statistics
            print("\n[INFO] System Statistics:")
            print(f"  Current FPS: {current_fps}")
            print(f"  Total people in DB: {len(person_names)}")
            for label, name in person_names.items():
                print(f"  Label {label}: {name}")
            print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
            print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
            print(f"  Face recognition interval: every {FACE_RECOGNITION_INTERVAL} frames")
            
        elif key == ord('q'):
            break
        
        # Record live frames if recording is on
        if recording:
            recorded_frames.append(frame.copy())
        
        # Decide which frame to send
        if inserting_loop and recorded_frames:
            source_frame = recorded_frames[loop_index]
            loop_index += 1
            if loop_index >= len(recorded_frames):
                inserting_loop = False
                print("[INFO] Loop playback finished. Returning to live feed.")
        else:
            source_frame = frame
        
        # Encode and send the frame
        result, encoded = cv2.imencode('.jpg', source_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Lower quality for Pi
        frame_bytes = encoded.tobytes()
        conn_file.write(struct.pack('<L', len(frame_bytes)))
        conn_file.write(frame_bytes)
        conn_file.flush()
        
        time.sleep(0.05)  # Slower frame rate for Pi (~20 FPS max)
    
    # Send end of stream signal
    conn_file.write(struct.pack('<L', 0))
    
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
    conn_file.close()
    client_socket.close()
