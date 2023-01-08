import torch
import cv2 as cv
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

        model = torch.hub.load('WongKinYiu/yolov7', 'custom',
                               path_or_model='best.pt')  # Main branch CUSTOM model of WongKinYiu YOLO7

        return model

    def get_video(self):
        """
        This method is preparing video writer
        Default file extension is 'mp4'
        """
        video = cv.VideoCapture('test.mp4')

        return video

    def cuda_available(self, model):
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

    def coin_detection(self):
        mot_tracker = Sort()  # Initialize multi object tracker
        thickness = 2  # Thickness of the all bbox lines and displayed numbers
        model = self.initialize_model()
        device = self.cuda_available(model)
        model.to(device)
        video = self.get_video()
        writer = self.get_writer()

        while video.isOpened():
            ret, frame = video.read()  # getting frames from video stream
            if not ret:  # Check for existing frames
                break
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            detections = model(frame)  # Getting detections in Tensorflow tensor
            normalized_detections = [t.cpu().numpy() for t in
                                     detections.xyxy[0]]  # Getting bbox coordinates from tensor

            # Part of the code that preparing data for tracker
            if normalized_detections == []:
                mot_data = np.empty((0, 5))
            else:
                mot_data = np.array(normalized_detections)
            track_bbs_ids = mot_tracker.update(mot_data)

            for subject in track_bbs_ids:  # (x1, y1, x2, y2. id)
                start_point = (int(subject[0]), int(subject[1]))
                end_point = (int(subject[2]), int(subject[3]))
                diametr = int(subject[3]) - int(subject[1])  # coin diametr
                id = subject[4]
                cv.rectangle(frame, start_point, end_point, (0, 0, 255), thickness)
                cv.putText(frame, str(int(diametr)), start_point, cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0),
                           thickness)
                cv.putText(frame, str(int(id)), (int(subject[2]), int(subject[1])), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           (255, 0, 0), thickness)

            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()


# TODO:
# Попробовать убрать проверку на пустой массив с боксами
# На выводе выходит битый видос, нужен фикс
# В телеге скинул тестовый видос, он будет прогоняться как тест модели
# Допилить детекцию чтобы она различала номиналы по цветам (мб попробовать через яркость, либо же через hsv)
#

if __name__ == '__main__':
    cls = CoinDetection()
    cls.coin_detection()
