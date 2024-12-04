from flask import Flask, render_template, Response
import cv2
import torch
import threading
import logging
from colorama import Fore, Style

# Set up Flask application
app = Flask(__name__)

# Set up logging with color support
class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "DEBUG": Fore.CYAN
        }
        reset = Style.RESET_ALL
        log_fmt = f"{level_color.get(record.levelname, '')}%(asctime)s - %(levelname)s - %(message)s{reset}"
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Set up logging
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])

# Load YOLOv5 model with the specified path
logging.info("Loading YOLOv5 model...")
model = torch.hub.load('C:/Users/gabka/Desktop/UWear/yolov5', 'custom', path='C:/Users/gabka/Desktop/UWear/yolov5/best.pt', source='local')
logging.info("YOLOv5 model loaded successfully.")

# Function to handle video streaming and inference
def generate_frames():
    cap = cv2.VideoCapture(0)  # Test with different indices (0, 1, 2) if necessary
    if not cap.isOpened():
        logging.error("Unable to access the camera.")
        return  # Stop streaming if camera cannot be accessed

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame.")
            break

        # Run inference
        results = model(frame)

        # Determine legality of clothing
        detections = results.pandas().xyxy[0]  # DataFrame of detections
        message = "Detection Results: "
        for index, row in detections.iterrows():
            item = row['name']
            # Append messages based on detection
            if item == 'Skirt':
                message += "Skirt detected (illegal wear). "
            elif item in ['Trousers', 'valid_top']:
                message += f"{item} detected (legal wear). "
            elif item == 'cropped_top':
                message += "Cropped top detected (illegal wear). "
            elif item == 'ripped_pants':
                message += "Ripped pants detected (illegal wear). "
            elif item == 'shorts':
                message += "Shorts detected (illegal wear). "
            elif item == 'sleeveless':
                message += "Sleeveless top detected (illegal wear). "

        logging.info(f"Detections: {message.strip()}")

        # Annotate frame
        annotated_frame = results.render()[0]
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# Flask route to stream video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Flask route to render the main page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
