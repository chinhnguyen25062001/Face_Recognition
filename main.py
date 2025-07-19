import cv2
import numpy as np
import argparse
import torch
from torchvision import transforms

from face_alignment import norm_crop
from face_recognition.model import iresnet_inference
from face_recognition.utils import compare_encodings, read_features

from ultralytics import YOLO

def mapping_bbox(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (tuple): The first bounding box (x_min, y_min, x_max, y_max).
        box2 (tuple): The second bounding box (x_min, y_min, x_max, y_max).

    Returns:
        float: The IoU score.
    """
    # Calculate the intersection area
    x_min_inter = max(box1[0], box2[0])
    y_min_inter = max(box1[1], box2[1])
    x_max_inter = min(box1[2], box2[2])
    y_max_inter = min(box1[3], box2[3])

    intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
        0, y_max_inter - y_min_inter + 1
    )

    # Calculate the area of each bounding box
    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def save_video(frames, output_path, fps):
    if not frames:
        print("No frames to write.")
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"[INFO] Saved output video to: {output_path}")

class FaceRecognition:
    def __init__(self, det_model_path: str, recog_model_path: str, feature_path: str, device: str) -> None:
        self.recog_model_path = recog_model_path
        self.det_model_path = det_model_path
        self.feature_path = feature_path
        self.device = device

        self.det_model = YOLO(self.det_model_path)
        self.det_conf = 0.25
        self.det_iou = 0.65

        self.recognizer = iresnet_inference(model_name="r100", path=self.recog_model_path, device=self.device)
        self.images_names, self.images_embs = read_features(feature_path=self.feature_path)

        self.previous_bboxes = []
        self.previous_names = []

    def __detect(self, input_image: np.ndarray):
        output = self.det_model.predict(
            input_image,
            conf=self.det_conf,
            iou=self.det_iou,
            device=self.device,
            agnostic_nms=True,
            verbose=False,
            classes = [0]
            )
        result = output[0]
        bb_results = result.boxes.xyxy.detach().cpu().numpy().tolist()
        landmarks = result.keypoints.xy.detach().cpu().numpy()
        landmarks = landmarks.astype(int)
        return bb_results, landmarks

    @torch.no_grad()
    def __get_feature(self, face_image):
        """
        Extract features from a face image.

        Args:
            face_image: The input face image.

        Returns:
            numpy.ndarray: The extracted features.
        """
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Convert to RGB
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Preprocess image (BGR)

        face_image = face_preprocess(face_image).unsqueeze(0).to(device)

        # Inference to get feature
        emb_img_face = self.recognizer(face_image).cpu().numpy()

        # Convert to array
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)

        return images_emb

    def __recognition(self, face_image):
        """
        Recognize a face image.

        Args:
            face_image: The input face image.

        Returns:
            tuple: A tuple containing the recognition score and name.
        """
        # Get feature from face
        query_emb = self.__get_feature(face_image)

        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min]
        score = score[0]

        return score, name

    def __check_input(self, source):
        if source.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')):
            return 'image'
        elif source.lower().endswith((".mp4", ".avi", ".webm")):
            return 'video'
        elif source == 'camera':
            return 'camera'
        return 'error'

    def inference(self, source):
        
        type_data = self.__check_input(source)
        if type_data == 'image':
            input_image = cv2.imread(source)
            bb_results, landmarks = self.__detect(input_image)

            output_image = input_image.copy()
            for j in range(len(bb_results)):
                face_alignment = norm_crop(img=input_image, landmark=landmarks[j])
                score, name = self.__recognition(face_image=face_alignment)
                if name is not None:
                    if score < 0.35:
                        caption = "UN_KNOWN"
                    else:
                        caption = f"{name}:{score:.2f}"
                        # caption = name
                xmin,ymin,xmax,ymax = list(map(int, bb_results[j]))
                cap_pos = (xmin, ymin - 5)
                color = (0,255,0) if caption != 'UN_KNOWN' else (255,0,0)
                output_image = cv2.rectangle(output_image, (xmin,ymin), (xmax,ymax), color, 1)
                output_image = cv2.putText(output_image, caption, cap_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1, cv2.LINE_AA)
            return output_image
        elif type_data == 'video':
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print("Error: Could not open video.")
                return
            list_output_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Stop if video ends
                bb_results, landmarks = self.__detect(frame)
                output_image = frame.copy()
                current_bboxes = []
                current_names = []
                for j in range(len(bb_results)):
                    current_bbox = bb_results[j]
                    caption = "UN_KNOWN"
                    match_found = False
                    for k in range(len(self.previous_bboxes)):
                        prev_bbox = self.previous_bboxes[k]
                        iou = mapping_bbox(current_bbox, prev_bbox)
                        if iou > 0.5:
                            caption = self.previous_names[k]
                            match_found = True
                            break
                    if not match_found or caption == "UN_KNOWN":
                        face_alignment = norm_crop(img=frame, landmark=landmarks[j])

                        score, name = self.__recognition(face_image=face_alignment)
                        if name is not None:
                            if score < 0.35:
                                caption = "UN_KNOWN"
                            else:
                                # caption = f"{name}:{score:.2f}"
                                caption = name
                    current_bboxes.append(current_bbox)
                    current_names.append(caption)
                    xmin,ymin,xmax,ymax = list(map(int, bb_results[j]))
                    cap_pos = (xmin, ymin - 5)
                    color = (0,255,0) if caption != 'UN_KNOWN' else (255,0,0)
                    output_image = cv2.rectangle(output_image, (xmin,ymin), (xmax,ymax), color, 1)
                    output_image = cv2.putText(output_image, caption, cap_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                self.previous_bboxes = current_bboxes
                self.previous_names = current_names
                list_output_frames.append(output_image)

            cap.release()
            return list_output_frames
        elif type_data == 'camera':
            cap = cv2.VideoCapture(0)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # Stop if video ends
                
                bb_results, landmarks = self.__detect(frame)
                output_image = frame.copy()
                current_bboxes = []
                current_names = []
                for j in range(len(bb_results)):
                    current_bbox = bb_results[j]
                    caption = "UN_KNOWN"
                    match_found = False
                    for k in range(len(self.previous_bboxes)):
                        prev_bbox = self.previous_bboxes[k]
                        iou = mapping_bbox(current_bbox, prev_bbox)
                        if iou > 0.5:
                            caption = self.previous_names[k]
                            match_found = True
                            break
                    if not match_found or caption == "UN_KNOWN":
                        face_alignment = norm_crop(img=frame, landmark=landmarks[j])

                        score, name = self.__recognition(face_image=face_alignment)
                        if name is not None:
                            if score < 0.35:
                                caption = "UN_KNOWN"
                            else:
                                caption = f"{name}:{score:.2f}"
                                # caption = name
                    current_bboxes.append(current_bbox)
                    current_names.append(caption)
                    xmin,ymin,xmax,ymax = list(map(int, bb_results[j]))
                    cap_pos = (xmin, ymin - 5)
                    color = (0,255,0) if caption != 'UN_KNOWN' else (255,0,0)
                    output_image = cv2.rectangle(output_image, (xmin,ymin), (xmax,ymax), color, 1)
                    output_image = cv2.putText(output_image, caption, cap_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                self.previous_bboxes = current_bboxes
                self.previous_names = current_names
                # Display the captured frame
                cv2.imshow('Camera', output_image)
                # Press 'q' to exit the loop
                if cv2.waitKey(1) == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_model', type=str, default='face_detection/weights/yolov8n-face.pt', help='Path to detection model')
    parser.add_argument('--recog_model', type=str, default='face_recognition/weights/arcface_r100.pth', help='Path to recognition model')
    parser.add_argument('--features', type=str, default='./datasets/face_features/feature', help='Path to stored features')
    parser.add_argument('--input', type=str, default='input_video.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to output video')
    parser.add_argument('--fps', type=int, default=30, help='Output video FPS')
    args = parser.parse_args()

    # Initializing Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceRecognition(
        det_model_path=args.det_model,
        recog_model_path=args.recog_model,
        feature_path=args.features,
        device=device
    )

    # Video Inference
    print("[INFO] Running inference...")
    output_frames = model.inference(source=args.input)

    print("[INFO] Saving output video...")
    save_video(output_frames, args.output, args.fps)

    # # Camera Inference
    # model.inference(source='camera')
