import cv2
import socket
import struct
import time
import numpy as np
import os
from pathlib import Path
import threading
from queue import Queue

def augment_face_data(face_roi, num_augmentations=3):
    """Create augmented versions of face data to improve training"""
    augmented_faces = [face_roi]  # Original face
    
    for i in range(num_augmentations):
        # Random rotation (-15 to +15 degrees)
        angle = np.random.uniform(-15, 15)
        rows, cols = face_roi.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(face_roi, rotation_matrix, (cols, rows))
        augmented_faces.append(rotated)
        
        # Slight brightness adjustment
        brightness_factor = np.random.uniform(0.8, 1.2)
        bright_adjusted = cv2.convertScaleAbs(face_roi, alpha=brightness_factor, beta=0)
        augmented_faces.append(bright_adjusted)
        
        # Gaussian blur for slight smoothing
        blurred = cv2.GaussianBlur(face_roi, (3, 3), 0.5)
        augmented_faces.append(blurred)
        
        # Histogram equalization for lighting variations
        equalized = cv2.equalizeHist(face_roi)
        augmented_faces.append(equalized)
    
    return augmented_faces

def preprocess_face(face_roi):
    """Enhanced preprocessing for better recognition"""
    # Histogram equalization for consistent lighting
    face_roi = cv2.equalizeHist(face_roi)
    
    # Gaussian blur to reduce noise
    face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0.5)
    
    # Resize to consistent size
    face_roi = cv2.resize(face_roi, (100, 100))  # Larger than original for better features
    
    return face_roi

def load_face_database(database_path, max_faces_per_person=5, use_augmentation=True):
    """Load faces from database folder and train the recognizer - with improvements for small datasets"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Try different recognizers - EigenFaces often works better with small datasets
    # Uncomment one of these alternatives:
    # face_recognizer = cv2.face.EigenFaceRecognizer_create()  # Often better for small datasets
    # face_recognizer = cv2.face.FisherFaceRecognizer_create() # Good for small datasets with multiple classes
    
    # LBPH with optimized parameters for small datasets
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2,        # Increased radius for more features
        neighbors=10,    # More neighbors for robustness
        grid_x=8,       # Standard grid
        grid_y=8,       # Standard grid
        threshold=100.0  # Lower initial threshold
    )
    
    training_faces = []
    training_labels = []
    person_names = {}
    label_counter = 0
    
    print(f"[INFO] Loading face database from: {database_path}")
    print(f"[INFO] Data augmentation: {'ON' if use_augmentation else 'OFF'}")
    
    if not os.path.exists(database_path):
        print(f"[ERROR] Database path does not exist: {database_path}")
        return None, {}, False
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for person_folder in os.listdir(database_path):
        person_path = os.path.join(database_path, person_folder)
        
        if not os.path.isdir(person_path):
            continue
            
        person_faces = []
        print(f"[INFO] Processing person: {person_folder}")
        
        image_files = [f for f in os.listdir(person_path) 
                      if Path(f).suffix.lower() in image_extensions]
        image_files = image_files[:max_faces_per_person]
        
        for image_file in image_files:
            image_path = os.path.join(person_path, image_file)
            
            try:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                
                # Resize if too large
                height, width = img.shape[:2]
                if width > 640:
                    scale = 640 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # More aggressive face detection for training
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1,     # More sensitive
                    minNeighbors=3,      # Less strict
                    minSize=(30, 30),
                    maxSize=(300, 300)
                )
                
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    # Enhanced preprocessing
                    roi_processed = preprocess_face(roi_gray)
                    
                    if use_augmentation:
                        # Create augmented versions
                        augmented_faces = augment_face_data(roi_processed, num_augmentations=2)
                        for aug_face in augmented_faces:
                            training_faces.append(aug_face)
                            training_labels.append(label_counter)
                            person_faces.append(aug_face)
                    else:
                        training_faces.append(roi_processed)
                        training_labels.append(label_counter)
                        person_faces.append(roi_processed)
                    
            except Exception as e:
                print(f"[WARNING] Error processing {image_path}: {e}")
                continue
        
        if person_faces:
            person_names[label_counter] = person_folder
            label_counter += 1
            print(f"[INFO] Loaded {len(person_faces)} face samples for {person_folder}")
        else:
            print(f"[WARNING] No faces found for {person_folder}")
    
    if not training_faces:
        print("[ERROR] No faces found in database!")
        return None, {}, False
    
    print(f"[INFO] Training recognizer with {len(training_faces)} face samples from {len(person_names)} people...")
    
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
    database_path = "./face_database"

# Ask about data augmentation
use_aug = input("Use data augmentation? (y/n, default=y): ").strip().lower()
use_augmentation = use_aug != 'n'

# Load face database with improvements
print("[INFO] Loading database with enhanced preprocessing...")
face_recognizer, person_names, database_loaded = load_face_database(
    database_path, 
    max_faces_per_person=5, 
    use_augmentation=use_augmentation
)

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
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cam.set(cv2.CAP_PROP_FPS, 15)

if not cam.isOpened():
    print("[ERROR] Camera not available")
    conn_file.close()
    client_socket.close()
    exit()

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n[INFO] Loaded people in database:", list(person_names.values()))
print("\n[INFO] Enhanced Recognition Features:")
print("  - Data augmentation for small datasets")
print("  - Improved preprocessing (histogram equalization, denoising)")
print("  - Optimized LBPH parameters")
print("  - Better confidence thresholds")
print("\n[INFO] Controls:")
print("  SPACE - Start/stop recording")
print("  L - Play recorded loop manually")
print("  A - Toggle auto-loop when known face detected")
print("  S - Show database statistics")
print("  + - Increase confidence threshold (stricter)")
print("  - - Decrease confidence threshold (looser)")
print("  Q - Quit")

# Variables - adjusted for better recognition with small datasets
recording = False
recorded_frames = []
loop_index = 0
inserting_loop = False
auto_loop_enabled = False
face_detected_counter = 0
FACE_DETECTION_THRESHOLD = 3
CONFIDENCE_THRESHOLD = 80  # Lower threshold to start with (more lenient)
frame_counter = 0
FACE_RECOGNITION_INTERVAL = 1

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
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        known_face_detected = False
        if frame_counter % FACE_RECOGNITION_INTERVAL == 0:
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray, 1.2, 3, 
                minSize=(40, 40),
                maxSize=(200, 200)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract and preprocess face for recognition
                roi_gray = gray[y:y+h, x:x+w]
                roi_processed = preprocess_face(roi_gray)
                
                label, confidence = face_recognizer.predict(roi_processed)
                
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
        
        # Show confidence threshold
        cv2.putText(frame, f"Conf Threshold: {CONFIDENCE_THRESHOLD}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Show FPS
        cv2.putText(frame, f"FPS: {current_fps}", (10, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        key = cv2.waitKey(1) & 0xFF
        cv2.imshow("Sender", frame)
        
        # Handle keyboard input
        if key == ord(' '):
            if not recording:
                print("[INFO] Recording started. Press SPACE to stop.")
                recording = True
                recorded_frames.clear()
            else:
                print(f"[INFO] Recording stopped. {len(recorded_frames)} frames captured.")
                recording = False
                loop_index = 0
                
        elif key == ord('l') and recorded_frames:
            print("[INFO] Playing back recorded footage...")
            inserting_loop = True
            loop_index = 0
            
        elif key == ord('a'):
            auto_loop_enabled = not auto_loop_enabled
            print(f"[INFO] Auto-loop {'enabled' if auto_loop_enabled else 'disabled'}")
            
        elif key == ord('+') or key == ord('='):
            CONFIDENCE_THRESHOLD = min(150, CONFIDENCE_THRESHOLD + 10)
            print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD} (stricter)")
            
        elif key == ord('-'):
            CONFIDENCE_THRESHOLD = max(30, CONFIDENCE_THRESHOLD - 10)
            print(f"[INFO] Confidence threshold: {CONFIDENCE_THRESHOLD} (more lenient)")
            
        elif key == ord('s'):
            print("\n[INFO] Enhanced System Statistics:")
            print(f"  Current FPS: {current_fps}")
            print(f"  Total people in DB: {len(person_names)}")
            print(f"  Data augmentation: {'ON' if use_augmentation else 'OFF'}")
            for label, name in person_names.items():
                print(f"  Label {label}: {name}")
            print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
            print(f"  Face size: 100x100 pixels")
            print(f"  Recognition interval: every {FACE_RECOGNITION_INTERVAL} frames")
            
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
        result, encoded = cv2.imencode('.jpg', source_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = encoded.tobytes()
        conn_file.write(struct.pack('<L', len(frame_bytes)))
        conn_file.write(frame_bytes)
        conn_file.flush()
        
        time.sleep(0.05)
    
    # Send end of stream signal
    conn_file.write(struct.pack('<L', 0))
    
except Exception as e:
    print(f"[ERROR] {e}")
finally:
    cam.release()
    cv2.destroyAllWindows()
    conn_file.close()
    client_socket.close()
