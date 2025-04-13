import sys
from PyQt5.QtWidgets import QApplication, QWidget,QGridLayout,QPushButton,QLabel,QFileDialog,QSlider,QProgressBar
from PyQt5.QtGui import QIcon,QImage,QPixmap,QFont
from PyQt5.QtCore import QRect,Qt,QCoreApplication
from video_controller import video_controller
import cv2
import matplotlib
matplotlib.use('Agg')
import sys
import yaml
from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def relative_kp(kp_source, kp_driving, kp_driving_initial):

    source_area = ConvexHull(kp_source['fg_kp'][0].data.cpu().numpy()).volume
    driving_area = ConvexHull(kp_driving_initial['fg_kp'][0].data.cpu().numpy()).volume
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_new = {k: v for k, v in kp_driving.items()}

    kp_value_diff = (kp_driving['fg_kp'] - kp_driving_initial['fg_kp'])
    kp_value_diff *= adapt_movement_scale
    kp_new['fg_kp'] = kp_value_diff + kp_source['fg_kp']

    return kp_new
def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                   **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])

    kp_detector.to(device)
    dense_motion_network.to(device)
    inpainting.to(device)


    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])


    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()

    return inpainting, kp_detector, dense_motion_network
def make_animation(source_image, driving_video, inpainting_network, kp_detector, dense_motion_network, device, bar,mode = 'relative'):


    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        source = source.to(device)
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).to(device)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])
        bar.setRange(0,driving.shape[2]-1)


        for frame_idx in tqdm(range(driving.shape[2])):
            bar.setValue(frame_idx)
            driving_frame = driving[:, :, frame_idx]
            driving_frame = driving_frame.to(device)
            kp_driving = kp_detector(driving_frame)
            if mode == 'standard':
                kp_norm = kp_driving
            elif mode=='relative':
                kp_norm = relative_kp(kp_source=kp_source, kp_driving=kp_driving,
                                    kp_driving_initial=kp_driving_initial)

            dense_motion = dense_motion_network(source_image=source, kp_driving=kp_norm,
                                                    kp_source=kp_source, bg_param = None,
                                                    dropout_flag = False)
            out = inpainting_network(source,dense_motion)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])


    return predictions


def find_best_frame(source, driving, cpu):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device= device)
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num


class Myapp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle("LifeLike")
        self.setWindowIcon(QIcon('happy.ico'))
        self.resize(1576,600)
        self.s_label= QLabel()
        self.d_label = QLabel()
        self.p_label = QLabel()
        self.c_label = QLabel()
        self.s_buttom = QPushButton('Source Image', self)
        self.d_buttom = QPushButton('Driving video', self)
        self.p_buttom = QPushButton('Prediction', self)
        self.play_buttom = QPushButton(self)
        self.stop_buttom = QPushButton("||",self)
        self.slider = QSlider(self)
        self.slider.setGeometry(QRect(720, 572, 640, 22))
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setObjectName("slider_videoframe")
        self.label_framecnt = QLabel(self)
        self.label_framecnt.setGeometry(QRect(1380, 570, 171, 22))
        self.label_framecnt.setObjectName("label_framecnt")


        style = '''
            QProgressBar {
                border: 2px solid #000;
                border-radius: 5px;
                text-align:center;
                height: 20px;
                width:200px;
            }
            QProgressBar::chunk {
                background: #09c;
                width:1px;
            }
            QPushButton {
            background-color: #FFFFFF;
            color: #000000;
            border-style: outset;
            padding: 1.5px;
            font: bold 20px;
            border-width: 2px;
            border-radius: 5px;
            border-color: #000000;
            }
            QPushButton:hover {
                background-color: lightgreen;
            }
            QSlider::groove:horizontal {
            background-color: gray;
            border: 1px solid;
            height: 10px;
            margin: 0px;
            }
            QSlider::handle:horizontal {
            background-color: #1E90FF;
            border: 1px  solid;
            height: 10px;
            width: 10px;
            margin: -15px 0px;
            }
            QLabel{font-size: 12pt;
                   width: 10px;}'''
        styles = '''
                    
                    QPushButton {
                    background-color: #FFFFFF;
                    color: #000000;
                    border-style: outset;
                    padding: 1.5px;
                    font: bold 20px;
                    border-width: 2px;
                    border-radius: 5px;
                    border-color: #000000;
                    }
                    QPushButton:hover {
                        background-color: red;
                    }'''
        self.bar = QProgressBar(self)
        self.bar.setGeometry(QRect(30, 570, 480, 21))
        self.bar.setStyleSheet(style)

        self.s_buttom.setStyleSheet(style)
        self.d_buttom.setStyleSheet(style)
        self.p_buttom.setStyleSheet(style)

        self.play_buttom.setFont(QFont('Webdings'))
        self.play_buttom.setText('4')
        self.play_buttom.setStyleSheet(style)
        self.slider.setStyleSheet(style)

        self.label_framecnt.setStyleSheet(style)

        self.stop_buttom.setStyleSheet(styles)






        layout = QGridLayout(self)
        layout2 = QGridLayout(self)
        layout.addWidget(self.s_buttom, 0, 0)
        layout.addWidget(self.d_buttom, 0, 1)
        layout.addWidget(self.p_buttom, 0, 2)
        layout.addWidget(self.s_label,1, 0)
        layout.addWidget(self.d_label, 1, 1)
        layout.addWidget(self.p_label, 1, 2)
        layout.addWidget(self.c_label, 2, 0)


        self.setLayout(layout)
        #self.setGeometry(300, 300, 400, 100)
        self.play_buttom.setGeometry(535, 567, 80, 30)
        self.stop_buttom.setGeometry(625,567,80,30)

        self.s_buttom.clicked.connect(self.s_open)
        self.d_buttom.clicked.connect(self.d_open)
        self.p_buttom.clicked.connect(self.p_open)

        #self.replay_buttom.clicked.connect(self.replay)
        #self.play_buttom.clicked.connect(self.play1)
        self.sourceimage = ""
        self.drivingvideo = ""
        self.resultvideo= ""



    def s_open(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', '*.png *.jpg *.bmp')
        if filename == "":
            return

        self.img = cv2.imread(filename, -1)
        self.img = cv2.resize(self.img,(512, 512))

        height, width, channel = self.img.shape
        if channel==4:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2BGR)
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.s_label.setPixmap(QPixmap.fromImage(self.qImg))
        self.sourceimage = filename

        return self.sourceimage



    def d_open(self):
        filename1, _ = QFileDialog.getOpenFileName(self, 'Open Video', '', '*.mp4')
        if filename1 == "":
            return
        self.drivingvideo = filename1
        if self.resultvideo == "":
            cap = cv2.VideoCapture(filename1)
            cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS)
            ret, frame = cap.read()
            frame = cv2.resize(frame, (512, 512))
            height, width, channel = frame.shape
            bytesPerline = 3 * width
            self.qImg = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
            self.d_label.setPixmap(QPixmap.fromImage(self.qImg))

            cap.release()
        else:
            self.video_controller = video_controller(video_path=self.resultvideo, driving_path=self.drivingvideo,ui=self.window())

        return self.drivingvideo


    def p_open(self):
        if self.sourceimage == "" or self.drivingvideo == "":
            return
        config = "./config/vox-256.yaml"
        #config = "./config/vox15.yaml"
        #checkpoints = "./checkpoints/vox.pth.tar"
        #checkpoints = "./checkpoints/22266/00000099-checkpoint.pth.tar"
        #checkpoints = "./checkpoints/new/00000099-checkpoint.pth.tar" #84/24/06  #10/75/42/45
        checkpoints = "./checkpoints/10/00000099-checkpoint.pth.tar"
        #checkpoints = "./checkpoints/22266/00000099-checkpoint.pth.tar"
        #checkpoints = "./checkpoints/base/00000099-checkpoint.pth.tar"
        result_video = './result.mp4'
        result_video1 = './rresult.mp4'
        self.resultvideo = result_video
        img_shape = (256, 256)
        type = lambda x: list(map(int, x.split(',')))
        mode = 'relative'
        #mode = 'standard'
        find_best_frame = "True"
        source_image = imageio.imread(self.sourceimage)
        reader = imageio.get_reader(self.drivingvideo)
        fps = reader.get_meta_data()['fps']
        driving_video = []
        try:
            for im in reader:
                driving_video.append(im)
        except RuntimeError:
            pass
        reader.close()
        if (torch.cuda.is_available() == False):
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0')
        source_image = resize(source_image, img_shape)[..., :3]
        driving_video = [resize(frame, img_shape)[..., :3] for frame in driving_video]
        inpainting, kp_detector, dense_motion_network = load_checkpoints(config_path=config,
                                                                                      checkpoint_path=checkpoints,
                                                                                      device=device)

        if (find_best_frame == False):
            i = find_best_frame(source_image, driving_video, device)
            print("Best frame: " + str(i))
            driving_forward = driving_video[i:]
            driving_backward = driving_video[:(i + 1)][::-1]
            predictions_forward = make_animation(source_image, driving_forward, inpainting, kp_detector,
                                                 dense_motion_network, device=device, mode=mode)
            predictions_backward = make_animation(source_image, driving_backward, inpainting, kp_detector,
                                                  dense_motion_network, device=device, mode=mode)
            predictions = predictions_backward[::-1] + predictions_forward[1:]
        else:
            predictions = make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network,
                                         device=device, bar = self.bar,mode=mode)

        imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
        imageio.mimsave('c333273.mp4', [img_as_ubyte(frame) for frame in predictions], fps=fps)
        imageio.imsave('last.png',img_as_ubyte(predictions[-1]))

        self.video_controller = video_controller(video_path="result.mp4",driving_path=self.drivingvideo,ui=self.window())
        self.play_buttom.clicked.connect(self.video_controller.play)
        self.stop_buttom.clicked.connect(self.video_controller.pause)

        return self.resultvideo



if __name__ == '__main__':
    app = QApplication(sys.argv)


    window = Myapp()

    window.show()
    app.exec()

