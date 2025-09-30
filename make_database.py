import cv2
import os
import time
from pathlib import Path

def create_face_database_entry(database_path="./face_database", num_images=50, capture_interval=0.3):
    """
    Capture face images to build a database entry for a person.
    
    Args:
        database_path: Root folder for the face database
        num_images: Number of images to capture per person
        capture_interval: Seconds between each capture
    """
    
    # Initialize face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Create database directory if it doesn't exist
    os.makedirs(database_path, exist_ok=True)
    
    print("\n" + "="*60)
    print("FACE DATABASE BUILDER")
    print("="*60)
    print(f"\nDatabase location: {os.path.abspath(database_path)}")
    print(f"Images to capture per person: {num_images}")
    print(f"Capture interval: {capture_interval}s")
    
    # Get person's name
    person_name = input("\nEnter person's name (or 'quit' to exit): ").strip()
    
    if person_name.lower() == 'quit' or not person_name:
        print("Exiting...")
        return
    
    # Create person's folder
    person_folder = os.path.join(database_path, person_name)
    os.makedirs(person_folder, exist_ok=True)
    
    # Check existing images
    existing_images = [f for f in os.listdir(person_folder) if f.endswith(('.jpg', '.png'))]
    start_index = len(existing_images)
    
    if start_index > 0:
        print(f"\n[INFO] Found {start_index} existing images for {person_name}")
        print(f"[INFO] New images will start at index {start_index}")
    
    # Initialize camera
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cam.isOpened():
        print("[ERROR] Cannot access camera!")
        return
    
    print("\n" + "-"*60)
    print("INSTRUCTIONS:")
    print("-"*60)
    print("1. Position your face in the green box")
    print("2. Move your head slightly (left, right, up, down)")
    print("3. Try different angles and expressions")
    print("4. Images will be captured automatically")
    print("\nPress 'q' to quit early")
    print("Press SPACE to start capturing")
    print("-"*60)
    
    capturing = False
    images_captured = 0
    last_capture_time = 0
    
    # Warmup frames
    for _ in range(10):
        cam.read()
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read from camera")
            break
        
        # Create display frame
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        # Draw face detection boxes
        face_detected = len(faces) > 0
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if face_detected else (0, 0, 255)
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Show capture zone indicator
            if capturing:
                cv2.putText(display_frame, "CAPTURING", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status display
        status_text = f"Captured: {images_captured}/{num_images}"
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if not capturing:
            cv2.putText(display_frame, "Press SPACE to start", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif not face_detected:
            cv2.putText(display_frame, "No face detected!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Face detected - Capturing...", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Progress bar
        bar_width = 600
        bar_height = 30
        bar_x = 20
        bar_y = display_frame.shape[0] - 50
        
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        progress = int((images_captured / num_images) * bar_width)
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + progress, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Show frame
        cv2.imshow("Face Database Builder", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') and not capturing:
            capturing = True
            print(f"\n[INFO] Starting capture for {person_name}...")
            
        elif key == ord('q'):
            print("\n[INFO] Capture cancelled by user")
            break
        
        # Automatic capture logic
        if capturing and face_detected and images_captured < num_images:
            current_time = time.time()
            
            if current_time - last_capture_time >= capture_interval:
                # Save the face region
                for (x, y, w, h) in faces:
                    # Extract face with some padding
                    padding = 20
                    y1 = max(0, y - padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    x1 = max(0, x - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    
                    face_img = frame[y1:y2, x1:x2]
                    
                    # Save image
                    img_index = start_index + images_captured
                    img_path = os.path.join(person_folder, f"{person_name}_{img_index:03d}.jpg")
                    cv2.imwrite(img_path, face_img)
                    
                    images_captured += 1
                    last_capture_time = current_time
                    
                    print(f"[{images_captured}/{num_images}] Captured image: {os.path.basename(img_path)}")
                    
                    # Only capture one face per frame
                    break
        
        # Check if done
        if images_captured >= num_images:
            print(f"\n[SUCCESS] Captured {images_captured} images for {person_name}!")
            print(f"[INFO] Images saved to: {person_folder}")
            print("\nPress any key to continue or 'q' to quit...")
            cv2.waitKey(0)
            break
    
    # Cleanup
    cam.release()
    cv2.destroyAllWindows()
    
    # Ask if user wants to add another person
    print("\n" + "="*60)
    another = input("Add another person? (y/n): ").strip().lower()
    if another == 'y':
        create_face_database_entry(database_path, num_images, capture_interval)


if __name__ == "__main__":
    # Configuration
    DATABASE_PATH = input("Enter database path (press Enter for './face_database'): ").strip()
    if not DATABASE_PATH:
        DATABASE_PATH = "./face_database"
    
    NUM_IMAGES = input("Number of images to capture per person (press Enter for 50): ").strip()
    NUM_IMAGES = int(NUM_IMAGES) if NUM_IMAGES.isdigit() else 50
    
    CAPTURE_INTERVAL = input("Seconds between captures (press Enter for 0.3): ").strip()
    CAPTURE_INTERVAL = float(CAPTURE_INTERVAL) if CAPTURE_INTERVAL else 0.3
    
    print("\n[INFO] Starting Face Database Builder...")
    print(f"[INFO] Database: {DATABASE_PATH}")
    print(f"[INFO] Images per person: {NUM_IMAGES}")
    print(f"[INFO] Capture interval: {CAPTURE_INTERVAL}s")
    
    create_face_database_entry(DATABASE_PATH, NUM_IMAGES, CAPTURE_INTERVAL)
    
    print("\n[INFO] Database builder finished!")
    print(f"[INFO] Your database is located at: {os.path.abspath(DATABASE_PATH)}")