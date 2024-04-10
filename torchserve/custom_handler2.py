import numpy as np
import base64
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import onnxruntime as ort

class ModelHandler():
    img_size = 320

    def __init__(self):
        # super().__init__()
        self.object_tracker = DeepSort()
        self.frame_skip = 3
        self.frame_count = -1
        self.class_names = ['bicycle', 'bus', 'car', 'motorbike', 'person']
        self.onnx_yolo_detector = self.load_onnx_yolo_detector('models/trained_model_10epoch.onnx')

    def load_onnx_yolo_detector(self, model_path):
        return OnnxYoloDetector(model_path)

    def handle(self, data):
        self.frame = data
        model_input = self.preprocess(data.copy())
        model_output = self.inference(model_input)
        return self.postprocess(model_output,data)

    def preprocess(self, data):
        # images = []
        # for row in data:
        #     image = row.get("data") or row.get("body")
        #     if isinstance(image, str):
        #         image = base64.b64decode(image)

        #     if isinstance(image, (bytearray, bytes)):
        #         image = np.frombuffer(image, dtype=np.uint8)
        #     else:
        #         image = row

        #     images.append(image)
        # out = np.array(images)
        # out = cv2.imdecode(out, cv2.IMREAD_COLOR)

        input_img = cv2.resize(data, (320, 320))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_array = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_array

    def inference(self, data):
        ort_inputs = {self.onnx_yolo_detector.input_name: data}
        ort_outs = self.onnx_yolo_detector.sess.run(None, ort_inputs)
        return ort_outs

    def postprocess(self, inference_output,frame):
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

       

        detections= []
        detect = [(box, score, self.class_names[class_id]) for box, score, class_id in zip(filtered_boxes, filtered_scores, filtered_class_ids)]

        tracked_objects = self.object_tracker.update_tracks(detect,frame=self.frame)
        for obj in tracked_objects:
            box = obj.to_tlwh()  # Replace with actual bounding box attribute name
            detection = {
                'class_id': self.class_names.index(obj.det_class),
                'class_name': obj.det_class,
                'confidence': obj.det_conf,  # Replace with actual confidence score attribute name
                'box': [c.item() for c in box],
                'scale': self.img_size / 320,
                'track_id': obj.track_id  # Assuming a track_id attribute exists
            }
            detections.append(detection)
            # print(obj.track_id
            # print(obj.det_class,obj.class_name)
            # obj.detection['track_id'] = obj.track_id

        return [detections]
    
class OnnxYoloDetector():
    def __init__(self, model_path):
        self.sess = ort.InferenceSession(model_path)
        self.input_name = self.sess.get_inputs()[0].name

    def detect_objects(self, frame):
        preprocessed_frame = self.preprocess_input(frame.copy())
        ort_inputs = {self.input_name: preprocessed_frame}
        ort_outs = self.sess.run(None, ort_inputs)
        return self.postprocess_output(ort_outs, frame.shape)

    def preprocess_input(self, frame):
        img_size = 320
        input_img = cv2.resize(frame, (320, 320))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_array = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_array

    def postprocess_output(self, inference_output, frame_shape):
        outputs = np.array(inference_output[0])
        outputs = np.transpose(outputs, (0, 2, 1))
        rows = outputs.shape[1]
        return rows


# Initialize model handler
model_handler = ModelHandler()

# Replace 'path/to/your/model.onnx' with the path to your ONNX model file
# model_handler.initialize()

# Replace 'path/to/video.mp4' with the path to your video file
cap = cv2.VideoCapture('C:/Users/opdar/BVM/Mtech/iiitb/2nd Sem/SPE/End_project/Object_Counting_1/producer/videos/video0.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received from video capture. Exiqting...")
        break
    up_scale_height = frame.shape[1]/320
    up_scale_width = frame.shape[0]/320
    # Perform object detection and tracking
    predictions = model_handler.handle(frame)

    # Display the resulting frame with object detection and tracking
    for obj in predictions[0]:
        # print(obj)
        box = obj['box']
        cv2.rectangle(frame, (int(box[0]*up_scale_height), int(box[1]*up_scale_width)), (int((box[2]+box[0])*up_scale_height), int((box[3]+box[1])*up_scale_width)), (255, 0, 0), 2)

        cv2.putText(frame, str(obj['track_id']), (int(box[0]*up_scale_height), int((box[1] - 10)*up_scale_width)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    # Quit if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
