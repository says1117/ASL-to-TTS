import cv2                                  # OpenCV for video capture and display
import mediapipe as mp                      # MediaPipe for hand landmark detection
from mediapipe.tasks import python          # MediaPipe Tasks API for hand landmark detection
from mediapipe.tasks.python import vision
import numpy as np                          # NumPy for array manipulation

def normalize_hand(landmarks_list: list[float], is_left_hand: bool = False) -> list[float]:
    arr = np.array(landmarks_list, dtype=np.float32)

    # Wrist coordinates live at indices 0, 1, 2
    wrist = arr[0:3]

    # Reshape flat 63-float list into (21, 3) so each row is one landmark.
    # Subtracting wrist broadcasts across all 21 rows simultaneously.
    landmarks_matrix = arr.reshape(21, 3)
    normalized = landmarks_matrix - wrist

    # If the hand is a left hand, negate all x values (column 0) so it looks
    # like a right hand to the model. Training data is right-hand only, so
    # mirroring left-hand input makes both hands use the same coordinate space.
    if is_left_hand:
        normalized[:, 0] *= -1

    # Flatten back to 63 floats for the rest of the pipeline
    return normalized.flatten().tolist()


DrawingUtils = mp.tasks.vision.drawing_utils
DrawingStyles = mp.tasks.vision.drawing_styles
HandConnections = mp.tasks.vision.HandLandmarksConnections

# Function to draw hand landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    if not detection_result.hand_landmarks: # If no hand landmarks are detected, return the original image
        return rgb_image

    annotated_image = np.copy(rgb_image)    # Set annoated image to the original image, so we can draw landmarks over it
    hand_landmarks_list = detection_result.hand_landmarks   # Get the list of hand landmarks from the detection result

    for hand_landmarks in hand_landmarks_list:  # Loop through each detected landmark in hand_landmarks_list
        DrawingUtils.draw_landmarks(    # Draw the landmarks and connections on the annotated image
            annotated_image,
            hand_landmarks,
            HandConnections.HAND_CONNECTIONS,
            DrawingStyles.get_default_hand_landmarks_style(),
            DrawingStyles.get_default_hand_connections_style())
    
    return annotated_image

class HandTracker:
    def __init__(self):
        # Initialize Hand Landmarker Settings
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)  # Flip the camera
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert color from BGR to RGB (MediaPipe uses RGB format)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)   # Convert RGB frame to a MediaPipe Image format for processing
        
        # Detect Landmarks from mp_image
        # Returns list of hand landmarks and their connections to detection_result
        detection_result = self.detector.detect(mp_image)

        # Extracts raw X, Y, Z values for the classifier
        landmark_array = []
        if detection_result.hand_landmarks: # If hand landmarks are detected
            first_hand = detection_result.hand_landmarks[0] # Get all 21 joints in the first detected hand
            for joint in first_hand:    # Loop through each joint
                landmark_array.extend([joint.x, joint.y, joint.z])  # Appends x, y, z values

        # Annotate frame with detected landmarks
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

        # Display the annotated frame to camera in a window
        display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR) # Convert annotated image back to BGR format for OpenCV display

        # Normalize landmarks before returning so both training and live
        # inference see wrist-relative coordinates. An empty list means
        # no hand was detected — skip normalization in that case.
        if landmark_array:
            # MediaPipe reports handedness in the flipped frame, so "Left" here
            # means the hand appears on the left side of the mirrored image.
            # Test live to confirm which label corresponds to your dominant hand.
            handedness_label = detection_result.handedness[0][0].category_name
            is_left = (handedness_label == "Left")
            landmark_array = normalize_hand(landmark_array, is_left_hand=is_left)

        # Hand the data back to main.py!
        return landmark_array, display_frame