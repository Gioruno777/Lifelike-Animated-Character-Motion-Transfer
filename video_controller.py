from PyQt5 import QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from opencv_engine import opencv_engine
import cv2

# videoplayer_state_dict = {
#  "stop":0,
#  "play":1,
#  "pause":2
# }

class video_controller(object):
    def __init__(self, video_path,driving_path, ui):
        self.video_path = video_path
        self.driving_path = driving_path
        self.ui = ui
        self.qpixmap_fix_width = 512 # 16x9 = 1920x1080 = 1280x720 = 800x450
        self.qpixmap_fix_height = 512
        self.current_frame_no = 0
        self.videoplayer_state = "pause"
        self.init_video_info()
        self.set_video_player()

    def init_video_info(self):
        videoinfo = opencv_engine.getvideoinfo(self.video_path)
        self.vc = videoinfo["vc"]
        self.video_fps = 27#videoinfo["fps"]
        self.video_total_frame_count = videoinfo["frame_count"]
        self.video_width = videoinfo["width"]
        self.video_height = videoinfo["height"]
        self.ui.slider.setRange(0, self.video_total_frame_count-1)
        self.ui.slider.valueChanged.connect(self.getslidervalue)
        drivinginfo = opencv_engine.getvideoinfo(self.driving_path)
        self.vcd = drivinginfo["vc"]
        self.video_widthd = drivinginfo["width"]
        self.video_heightd= drivinginfo["height"]



    def set_video_player(self):
        self.timer=QTimer() # init QTimer
        self.timer.timeout.connect(self.timer_timeout_job) # when timeout, do run one
        self.timer.start(self.video_fps) # start Timer, here we set '1000ms//Nfps' while timeout one time
        #self.timer.start(1) # but if CPU can not decode as fast as fps, we set 1 (need decode time)

    def set_current_frame_no(self, frame_no):
        self.vcd.set(1, frame_no)
        self.vc.set(1, frame_no) # bottleneck

        #@WongWongTimer
    def __get_next_frame(self):
        ret, framed = self.vcd.read()
        ret, frame = self.vc.read()
        self.ui.label_framecnt.setText(f"Frame number: {self.current_frame_no}/{self.video_total_frame_count-1}")
        self.setslidervalue(self.current_frame_no)
        return framed,frame

    def __update_label_frame(self, framed,frame):
        framed = cv2.resize(framed,(256,256))
        bytesPerlined = 3 * 256
        qimgd = QImage(framed,256, 256, bytesPerlined, QImage.Format_RGB888).rgbSwapped()
        self.qpixmapd = QPixmap.fromImage(qimgd)
        self.qpixmapd = self.qpixmapd.scaledToWidth(self.qpixmap_fix_width)
        self.qpixmapd = self.qpixmapd.scaledToHeight(self.qpixmap_fix_height)
        self.ui.d_label.setPixmap(self.qpixmapd)

        bytesPerline = 3 * self.video_width
        qimg = QImage(frame,self.video_width, self.video_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)
        self.qpixmap = self.qpixmap.scaledToWidth(self.qpixmap_fix_width)
        self.qpixmap = self.qpixmap.scaledToHeight(self.qpixmap_fix_height)
        self.ui.p_label.setPixmap(self.qpixmap)

        # self.ui.label_videoframe.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop) # up and left
        self.ui.p_label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter) # Center


    def play(self):
        self.videoplayer_state = "play"

    def stop(self):
        self.videoplayer_state = "stop"

    def pause(self):
        self.videoplayer_state = "pause"

    def timer_timeout_job(self):
        if (self.videoplayer_state == "play"):
            if self.current_frame_no >= self.video_total_frame_count-1:
                self.videoplayer_state = "pause"
                self.current_frame_no = 0 # auto replay
                self.set_current_frame_no(self.current_frame_no)
            else:
                self.current_frame_no += 1

        if (self.videoplayer_state == "stop"):
            self.current_frame_no = 0
            self.set_current_frame_no(self.current_frame_no)

        if (self.videoplayer_state == "pause"):
            self.current_frame_no = self.current_frame_no
            self.set_current_frame_no(self.current_frame_no)

        framed,frame = self.__get_next_frame()
        self.__update_label_frame(framed,frame)

    def getslidervalue(self):
        self.current_frame_no = self.ui.slider.value()
        self.set_current_frame_no(self.current_frame_no)

    def setslidervalue(self, value):
        self.ui.slider.setValue(self.current_frame_no)