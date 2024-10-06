import cv2
from Cocoa import NSArray
from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
import os
import time
import threading
import requests

def list_cameras():
    devices = AVCaptureDevice.devicesWithMediaType_(AVMediaTypeVideo)
    cameras = []
    for device in devices:
        cameras.append(device.localizedName())
    return cameras

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_recording(path, frame):
    cv2.imwrite(path, frame)

    print(f"Saved recording: {path}")

    # url = 'http://127.0.0.1:5000/vision/post_jpg'
    # data = {
    #     'vision_string': f"{path.split('/')[-1]}",
    #     'timestamp' : time.time()
    # }
    # response = requests.post(url, json=data)

    return

if __name__ == "__main__":
    cameras = list_cameras()
    camera_idx = 0
    for i, camera in enumerate(cameras):
        if "LRCP" in camera:
            camera_idx = 0
            break
    
    save_dir = "../../data/dataset/up"
    interval = 1
    ensure_dir(save_dir)
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"Camera {camera_idx} could not be opened.")
        assert(False)

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to grab frame from camera {camera_idx}.")
                break

            filename = os.path.join(save_dir, f"image_{idx}.jpg")

            thread = threading.Thread(target=save_recording, 
                                  args=(filename, frame))
            thread.start()

            idx += 1
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[INFO/vid_record.py] Terminated by user.")
    finally:
        cap.release()

        for thread in threading.enumerate():
            if thread != threading.current_thread():
                thread.join()

        print("[INFO/vid_record.py] All threads stopped.")