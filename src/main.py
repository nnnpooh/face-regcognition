# import dependencies
import io
import os
from base64 import b64decode, b64encode

import cv2
import face_recognition

import numpy as np
import PIL

# from google.colab.output import eval_js
# from IPython.display import Image, Javascript, display
from ultralytics import YOLO, settings
from utils import take_photo


# function to convert the JavaScript object into an OpenCV image
def js_to_image(js_reply):
    """
    Params:
            js_reply: JavaScript object containing image from webcam
    Returns:
            img: OpenCV BGR image
    """
    # decode base64 image
    image_bytes = b64decode(js_reply.split(",")[1])
    # convert bytes to numpy array
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # decode numpy array into OpenCV BGR image
    img = cv2.imdecode(jpg_as_np, flags=1)

    return img


# function to convert OpenCV Rectangle bounding box image into base64 byte string to be overlayed on video stream
def bbox_to_bytes(bbox_array):
    """
    Params:
            bbox_array: Numpy array (pixels) containing rectangle to overlay on video stream.
    Returns:
          bytes: Base64 image byte string
    """
    # convert array into PIL image
    bbox_PIL = PIL.Image.fromarray(bbox_array, "RGBA")
    iobuf = io.BytesIO()
    # format bbox into png for return
    bbox_PIL.save(iobuf, format="png")
    # format return string
    bbox_bytes = "data:image/png;base64,{}".format(
        (str(b64encode(iobuf.getvalue()), "utf-8"))
    )

    return bbox_bytes


# def take_photo_backup(filename="photo.jpg", quality=0.8):
#     js = Javascript("""
#     async function takePhoto(quality) {
#       const div = document.createElement('div');
#       const capture = document.createElement('button');
#       capture.textContent = 'Capture';
#       div.appendChild(capture);

#       const video = document.createElement('video');
#       video.style.display = 'block';
#       const stream = await navigator.mediaDevices.getUserMedia({video: true});

#       document.body.appendChild(div);
#       div.appendChild(video);
#       video.srcObject = stream;
#       await video.play();

#       // Resize the output to fit the video element.
#       google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

#       // Wait for Capture to be clicked.
#       await new Promise((resolve) => capture.onclick = resolve);

#       const canvas = document.createElement('canvas');
#       canvas.width = video.videoWidth;
#       canvas.height = video.videoHeight;
#       canvas.getContext('2d').drawImage(video, 0, 0);
#       stream.getVideoTracks()[0].stop();
#       div.remove();
#       return canvas.toDataURL('image/jpeg', quality);
#     }
#     """)
#     display(js)

#     # get photo data
#     data = eval_js("takePhoto({})".format(quality))
#     # get OpenCV format image
#     img = js_to_image(data)
#     # grayscale img
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # save image
#     cv2.imwrite(filename, img)

#     return filename


# Update multiple settings
settings.update({"runs_dir": os.getcwd(), "tensorboard": False})

Candidate_class = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


try:
    # Take a photo
    filename = take_photo("photo.jpg")
    # filename = "photo.jpg"

    # Load the known face for comparison
    known_image = face_recognition.load_image_file("me.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize YOLO model for human detection
    model = YOLO("yolov8n.pt")

    # Read the captured frame
    frame = cv2.imread(filename)

    # Detect humans using YOLO
    results = model(frame, classes=[0])
    num_humans = len(results[0].boxes)

    # Convert frame to RGB for face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Create a copy of the frame to annotate
    annotated_frame = frame.copy()

    # Text to display
    detection_text = f"Humans Detected: {num_humans}"

    # Flag to track if your face is found
    me_found = False

    # Iterate through YOLO detection boxes
    for box in results[0].boxes:
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Check if this box contains a face
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            # Check if the face is within the YOLO detection box
            if left >= x1 and right <= x2 and top >= y1 and bottom <= y2:
                # Compare the face with the known face
                matches = face_recognition.compare_faces(
                    [known_face_encoding], face_encoding
                )

                if matches[0]:
                    # Draw rectangle for this box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Put "Me" text on the frame
                    cv2.putText(
                        annotated_frame,
                        "Me",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )

                    me_found = True
                    break

        # If your face is found, stop searching
        if me_found:
            break

    # If no face match found, draw first human detection box
    if not me_found and num_humans > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (217, 22, 86), 2)
        cv2.putText(
            annotated_frame,
            "Stranger",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (217, 22, 86),
            2,
        )

    # Display the annotated frame
    # _, buffer = cv2.imencode(".jpg", annotated_frame)
    cv2.imshow("Image", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # display(Image(data=buffer.tobytes()))


except Exception as err:
    # Handle potential errors
    print(f"An error occurred: {str(err)}")
