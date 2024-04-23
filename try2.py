import cv2
import numpy as np
import onnxruntime as ort
from deep_sort_realtime.deepsort_tracker import DeepSort

class OnnxYoloDetector():
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name
        self.class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']

    def detect_objects(self, frame):
        preprocessed_frame = self.preprocess_input(frame.copy())
        ort_inputs = {self.input_name: preprocessed_frame}
        ort_outs = self.sess.run(None, ort_inputs)
        return self.postprocess_output(ort_outs, frame.shape)

    def preprocess_input(self, frame):
        img_size = 320  # Assuming your model expects 320x320 input
        input_img = cv2.resize(frame, (320, 320))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_array = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_array

    def postprocess_output(self, inference_output, frame_shape):
        outputs = np.array(inference_output[0])
        outputs = np.transpose(outputs, (0, 2, 1))
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.2, 0.3, 0.5)

        filtered_boxes = []
        filtered_scores = []
        filtered_class_ids = []

        if result_boxes.any():
            for i in result_boxes.flatten():
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[i])
                filtered_class_ids.append(class_ids[i])

        return filtered_boxes, filtered_scores, filtered_class_ids

# Initialize ONNX YOLO detector
model_path = 'torchserve/models/trained_model_10epoch.onnx'
onnx_yolo_detector = OnnxYoloDetector(model_path)

# Initialize DeepSort tracker
object_tracker = DeepSort()

# Replace 'path/to/video.mp4' with the path to your video file
cap = cv2.VideoCapture('producer/videos/video1.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

f_skip = 3
f_ct = -1

while True:
    ret, frame = cap.read()
    if ret:
        up_scale_height = frame.shape[1]/320
        up_scale_width = frame.shape[0]/320
        if not ret:
            print("No frame received from video capture. Exiting...")
            break

        f_ct+=1
        if f_ct%f_skip!=0:
            continue

        # Detect objects in the frame using ONNX YOLO detector
        boxes, scores, class_ids = onnx_yolo_detector.detect_objects(frame)

        # Perform object tracking using DeepSort
        detections = [(box, score, onnx_yolo_detector.class_names[class_id]) for box, score, class_id in zip(boxes, scores, class_ids)]
        tracked_objects = object_tracker.update_tracks(detections, frame=frame)

        # Draw tracked objects on frame
        for obj in tracked_objects:
            print(obj.det_class)
            box = obj.to_tlwh()  # Get bounding box in (top_left_x, top_left_y, width, height) format
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(box[0]*up_scale_height), int(box[1]*up_scale_width)), (int((box[2]+box[0])*up_scale_height), int((box[3]+box[1])*up_scale_width)), (255, 0, 0), 2)

            cv2.putText(frame, str(obj.track_id), (int(box[0]*up_scale_height), int((box[1] - 10)*up_scale_width)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Quit if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
