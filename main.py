import torch
import cv2 as cv

import main
from sort import *


class CoinDetection:
    """
    This is main class of the coin detection model
    It includes multi object tracking and display results on the screen
    All methods will be upgrade in the future
    """

    def initialize_model(self):
        """
        This is method that initialize the model
        """
        try:
            model = torch.hub.load('WongKinYiu/', 'custom',
                                   path_or_model='best.pt',
                                   source="local")  # Main branch CUSTOM model of WongKinYiu YOLO7
        except FileNotFoundError:
            from git.repo.base import Repo
            print("Yolov7 is not installed. Cloning yolov7 from github")
            Repo.clone_from("https://github.com/WongKinYiu/yolov7", "WongKinYiu/")
            print("Done")
            model = torch.hub.load('WongKinYiu/', 'custom',
                                   path_or_model='best.pt', source="local")

        return model

    def cuda_available(self):
        """
        if cuda available use gpu('cuda:0')
        else use cpu('cpu')
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        return device

    def get_writer(self):
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 30
        dimension = (1280, 720)
        writer = cv.VideoWriter('final_result.mp4', fourcc, fps, dimension)

        return writer

    def video_detections(self, thickness, media):
        """
        Multi detection for video files
        """
        video = media
        model = self.initialize_model()
        model.to(self.cuda_available())
        writer = self.get_writer()
        mot_tracker = Sort()
        video = cv.VideoCapture(video)
        while video.isOpened():
            ret, frame = video.read()  # getting frames from video stream
            if not ret:  # Check for existing frames
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            detections = model(frame)  # Getting detections in Tensorflow tensor
            normalized_detections = [t.cpu().numpy() for t in
                                     detections.xyxy[0]]  # Getting bbox coordinates from tensor

            # Part of the code that preparing data for tracker
            if not normalized_detections:
                mot_data = np.empty((0, 5))
            else:
                mot_data = np.array(normalized_detections)
            track_bbs_ids = mot_tracker.update(mot_data)

            for subject in track_bbs_ids:  # (x1, y1, x2, y2, id)
                start_point = (int(subject[0]), int(subject[1]))
                end_point = (int(subject[2]), int(subject[3]))
                width = (int(subject[3]) - int(subject[1]))  # coin width ||| 55px = 2cm
                ratio_px_mm = 0.275
                # 20mm / 55px
                mm = width / ratio_px_mm
                cm = mm / 100
                id = subject[4]
                cv.rectangle(frame, start_point, end_point, (0, 0, 255), thickness)
                cv.putText(frame, f"{round(cm, 2)} CM", start_point, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
                cv.putText(frame, str(int(id)), (int(subject[2]), int(subject[1])), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (255, 0, 0), thickness)

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

    def image_detections(self, thickness, media):
        """
        Multi detections for image
        """
        mot_tracker = Sort()
        model = self.initialize_model()
        model.to(self.cuda_available())
        img = cv.imread(media)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        detections = model(img)
        normalized_detections = [t.cpu().numpy() for t in
                                 detections.xyxy[0]]  # Getting bbox coordinates from tensor

        mot_data = np.array(normalized_detections)
        track_bbs_ids = mot_tracker.update(mot_data)

        for subject in track_bbs_ids:  # (x1, y1, x2, y2, id)
            start_point = (int(subject[0]), int(subject[1]))
            end_point = (int(subject[2]), int(subject[3]))
            width = (int(subject[3]) - int(subject[1]))  # coin width ||| 55px = 2cm
            ratio_px_mm = 0.275
            # 20mm / 55px
            mm = width / ratio_px_mm
            cm = mm / 100
            id = subject[4]
            cv.rectangle(img, start_point, end_point, (0, 0, 255), thickness)
            cv.putText(img, f"{round(cm, 2)} CM", start_point, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            cv.putText(img, str(int(id)), (int(subject[2]), int(subject[1])), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       (255, 0, 0), thickness)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imshow("results", img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def main(self):
        __pictureExtensionType__ = ["png", "jpg", "jpeg"]
        __videoExtensionType__ = ["mkv", "mp4", "avi", "webm"]

        thickness = int(input("Write line thickness from 1 to 3"
                              " (2 strongly recommend): "))  # Thickness of the all bbox lines and displayed numbers
        media = input("Enter file name: ")
        media_ext = media.split('.')[-1]

        if media_ext in __videoExtensionType__:
            self.video_detections(thickness, media)
        if media_ext in __pictureExtensionType__:
            self.image_detections(thickness, media)


# TODO:
# Допилить детекцию чтобы она различала номиналы по цветам (попробовать через hsv)


if __name__ == '__main__':
    cls = CoinDetection()
    cls.main()
