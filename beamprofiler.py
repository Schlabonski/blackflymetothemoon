import PyCapture2 as pc2
import numpy as np
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg

cols = 1920
rows = 1200

qualitative_colors = [(228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0)]

class MainWindow(QtGui.QMainWindow):
    """Docstring for MainWindow. """

    def __init__(self, parent=None):
        """TODO: to be defined1. """
        super(MainWindow, self).__init__(parent)

        ## Switch to using white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # initialize the main window
        self.setWindowTitle('BlackFly me to the moon')
        self.win = pg.GraphicsWindow()
        self.init_cam()
        win = self.win
        win.resize(1000,600)
        self.setCentralWidget(win)
        imageview = win.addViewBox(row=0, col=0, lockAspect=True)
        self.create_camera_menu()

        # add image plot
        img = pg.ImageItem(border='w')
        imageview.addItem(img)
        im_data = self.get_image()
        self.im_data = im_data
        img.setImage(im_data)
        self.img = img

        timer = QtCore.QTimer(parent=win)
        timer.timeout.connect(self.updateImage)
        timer.start(50)

        # add a region of interest to the image
        roi = pg.RectROI([0, 0], [cols, rows], pen=qualitative_colors[0])
        roi.addScaleHandle(pos=(0,0), center=(1,1))
        imageview.addItem(roi)
        self.roi = roi
        roi.sigRegionChanged.connect(self.update_roi)
        
        # create integration limits for the profiles
        x_integration_limits = [0, cols]
        y_integration_limits = [0, rows]
        self.x_int_lims = x_integration_limits
        self.y_int_lims = y_integration_limits

        # add row profile plot
        rp = self.calculate_row_profile()
        rp_plot = win.addPlot(row=0, col=1)
        self.rp_curve = rp_plot.plot(x=rp, y=np.arange(rows),pen=qualitative_colors[1])

        timer.timeout.connect(self.updateRowprofile)

        # add columnn profile
        cp = self.calculate_col_profile()
        cp_plot = win.addPlot(row=1, col=0)
        self.cp_curve = cp_plot.plot(y=cp, pen=qualitative_colors[1])

        timer.timeout.connect(self.updateColprofile)
    
    def create_camera_menu(self):
        # add a menu bar to the window for advanced blackfly control
        menubar = self.menuBar()
        blackfly_conf_menu = menubar.addMenu('&Camera settings')

        # define action to set resolution
        resolution_action = QtGui.QAction('&Video Mode', self)
        resolution_action.triggered.connect(self.set_video_mode)
        blackfly_conf_menu.addAction(resolution_action)

    def set_video_mode(self):
        self.popup = QtGui.QInputDialog()
        vmfr = self.cam.getVideoModeAndFrameRate()
        output = self.popup.getItem(self, 'Video mode', 'Select a video mode', self.video_modes,
                current=vmfr[0])
        print(output)
        

    def im_to_array(self, im):
        imarr = np.array(im.getData())
        imarr = np.reshape(imarr, (im.getRows(), im.getCols()))
        imarr = imarr.T
        imarr = imarr[::,::-1]
        return imarr.astype(np.float64)

    def init_cam(self):
        # 1. Start the bus interface
        bus = pc2.BusManager()
        bus.forceAllIPAddressesAutomatically() # necessary due to my networking inabilities

        # 2. Connect to the camera and start capturin
        cam = pc2.Camera()
        cam.connect(bus.getCameraFromIndex(0))
        cam.startCapture()
        self.cam = cam

        # 3. collect framerates, videomodes, ...
        fr = pc2.FRAMERATE
        self.framerates = [f for f in dir(fr) if not f.startswith('__')]
        
        vm = pc2.VIDEO_MODE
        self.video_modes = [v for v in dir(vm) if not v.startswith('__')]

    def get_image(self):
        """Returns an image.
        :returns: np.ndarray

        """
        cam = self.cam
        im = cam.retrieveBuffer()
        a = self.im_to_array(im) # for numpy manipulation of data
        return a

    def updateImage(self):
        im_data = self.get_image()
        self.im_data = im_data
        self.img.setImage(im_data)

    def update_roi(self):
        x_integration_limits = self.x_int_lims
        y_integration_limits = self.y_int_lims
        boundaries = self.roi.parentBounds().getCoords()
        x_integration_limits[0] = int(max(boundaries[0], 0))
        x_integration_limits[1] = int(min(boundaries[2], cols))
        y_integration_limits[0] = int(max(boundaries[1], 0))
        y_integration_limits[1] = int(min(boundaries[3], rows))

    # add row profile
    def calculate_row_profile(self):
        x_integration_limits = self.x_int_lims
        im_data = self.im_data
        return np.sum(im_data[x_integration_limits[0]:x_integration_limits[1],::], axis=0)

    def updateRowprofile(self):
        rp = self.calculate_row_profile()
        self.rp_curve.setData(x=rp, y=np.arange(rows))

    # add column profile
    def calculate_col_profile(self):
        y_integration_limits = self.y_int_lims
        im_data = self.im_data
        return np.sum(im_data[::,y_integration_limits[0]:y_integration_limits[1]], axis=1)

    def updateColprofile(self):
        cp = self.calculate_col_profile()
        self.cp_curve.setData(y=cp)


def main():
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
