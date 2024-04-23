import cv2
import numpy as np
import onnxruntime as ort
import time

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

        # Scale input pixel values to 0 to 1
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

        # filtered_boxes = []
        # filtered_scores = []
        # filtered_class_ids = []
        # print(result_boxes)
        # # if len(result_boxes) > 0:
        # #     for box in result_boxes:
        # #         index = box[0]
        # #         filtered_boxes.append(boxes[])
        # #         filtered_scores.append(scores[index])
        # #         filtered_class_ids.append(class_ids[index])

        # return filtered_boxes, filtered_scores, filtered_class_ids








# Replace 'path/to/your/model.onnx' with the path to your ONNX model
model_path = 'torchserve/models/trained_model_60epoch.onnx'

# Initialize ONNX YOLO detector
onnx_yolo_detector = OnnxYoloDetector(model_path)

# Replace 'path/to/video.mp4' with the path to your video file
cap = cv2.VideoCapture('producer/videos/video1.mp4')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
while True:
    ret, frame = cap.read()
    # frame = cv2.resize(frame,(320,320))
    up_scale_width = frame.shape[0]/320
    up_scale_height = frame.shape[1]/320
    # print(up_scale_height,up_scale_width)
    if not ret:
        print("No frame received from video capture. Exiting...")
        break

    # Detect objects in the frame using ONNX YOLO detector
    boxes, scores, class_ids = onnx_yolo_detector.detect_objects(frame)
    # boxes, scores, class_ids = onnx_yolo_detector.apply_nms(boxes, scores, class_ids)

    # print(boxes)

    # Draw detections on frame  
    for box, score, class_id in zip(boxes, scores, class_ids):
        if len(box)==4:
            # print(box)
            cv2.rectangle(frame, (int(box[0]*up_scale_height), int(box[1]*up_scale_width)), (int((box[2]+box[0])*up_scale_height), int((box[3]+box[1])*up_scale_width)), (255, 0, 0), 2)
            label = f"{onnx_yolo_detector.class_names[class_id]}: {score:.2f}"
            cv2.putText(frame, label, (int(box[0]*up_scale_height),int( box[1]*up_scale_width) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
