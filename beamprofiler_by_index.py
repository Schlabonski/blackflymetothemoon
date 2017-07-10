#! /usr/bin/python3

import sys
import time
import logging

import PyCapture2 as pc2
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph.dockarea import  DockArea, Dock

import pyximport; pyximport.install(setup_args={"script_args":["--compiler=unix"],
					"include_dirs":np.get_include()}, reload_support=True)
from gaussfit import gaussian_fit

import cProfile

def profile(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__
        logging.debug('%s called' % datafn)
        return func(*args, **kwargs)

    return wrapper

qualitative_colors = [(228,26,28),(55,126,184),(77,175,74),(152,78,163),(255,127,0)]
grey = (211,211,211)
camera_index = int(sys.argv[1])
camera_serial_numbers = [16292944, 16302806]

class MainWindow(QtGui.QMainWindow):
    """Docstring for MainWindow. """

    cols = 1920
    rows = 1200
    cols_array = np.arange(cols)
    rows_array = np.arange(rows)
    acquisition_timer_interval = 1000

    def __init__(self, parent=None):
        """TODO: to be defined1. """
        super(MainWindow, self).__init__(parent)

        # set state parameters
        self.capturing = True
        self.gaussian = False
        self.gaussian_inited = False
        self.acquisition_inited = False
        self.substract_background = False
        self.fit_roi_only = False

	## Initialize cam
        self.init_cam()

        ## Switch to using white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # initialize the main window
        self.setWindowTitle('BlackFly me to the moon {0}'.format(camera_serial_numbers[camera_index]))
        self.win = pg.GraphicsWindow()
        win = self.win
        win.resize(1000,600)
        self.setCentralWidget(win)

        # add layout 
        self._add_window_layout()

        # create menu bar
        self.create_camera_menu()

        # add image plot
        imageview = pg.ViewBox(lockAspect=True)
        imageview_layout = pg.GraphicsLayoutWidget()
        imageview_layout.addItem(imageview)
        self.vlayout.addWidget(imageview_layout, 0, 0)
        self.imageview = imageview
        img = pg.ImageItem(view=imageview, border='w')
        imageview.addItem(img)
        self.im_data = np.zeros((self.cols, self.rows))
        im_data = self.get_image()
        self.im_data = im_data
        img.setImage(im_data)
        self.img = img

        # main data acquisition timer
        acquisition_timer = QtCore.QTimer(parent=win)
        self.acquisition_timer = acquisition_timer
        acquisition_timer.timeout.connect(self.updateImage)
        acquisition_timer.start(self.acquisition_timer_interval)
        self.acquisition_inited = True


        # add a region of interest to the image
        roi = pg.RectROI([0, 0], [self.cols, self.rows], pen=qualitative_colors[0])
        roi.addScaleHandle(pos=(0,0), center=(1,1))
        imageview.addItem(roi)
        self.roi = roi
        roi.sigRegionChanged.connect(self.update_roi)
        
        # create integration limits for the profiles
        x_integration_limits = [0, self.cols]
        y_integration_limits = [0, self.rows]
        self.x_int_lims = x_integration_limits
        self.y_int_lims = y_integration_limits

        # add row profile plot
        rp = self.calculate_row_profile()
        rp_plot = pg.PlotWidget()
        self.vlayout.addWidget(rp_plot, 0, 1)
        rp_plot.hideAxis('bottom')
        self.rp_curve = rp_plot.plot(x=rp, y=np.arange(self.rows),pen=qualitative_colors[1])
        self.rp_plot = rp_plot

        acquisition_timer.timeout.connect(self.updateRowprofile)

        # add columnn profile
        cp = self.calculate_col_profile()
        cp_plot = pg.PlotWidget()
        self.vlayout.addWidget(cp_plot, 1, 0)
        cp_plot.hideAxis('left')
        self.cp_curve = cp_plot.plot(y=cp, pen=qualitative_colors[1])
        self.cp_plot = cp_plot

        acquisition_timer.timeout.connect(self.updateColprofile)

        # add functional buttons
        self._add_btn_layout()
        self._add_capture_toggle_btn()
        self._add_reset_roi_btn()
        self._add_gaussfit_btn()
        self._add_background_save_btn()
        self._add_background_substraction_checkbox()

    def _add_window_layout(self):
        vlayout = QtGui.QGridLayout(self.win)
        self.vlayout = vlayout

    def _add_btn_layout(self):
        # add a DockArea for buttons
        area = DockArea()
        dock = Dock("Functions")
        area.addDock(dock)
        area.show()

        self.vlayout.addWidget(area, 1, 1)
        btn_layout = pg.LayoutWidget()
        dock.addWidget(btn_layout)
        self.btn_layout = btn_layout

    def _add_capture_toggle_btn(self):
        btn = QtGui.QPushButton('Start/Stop')
        btn.clicked.connect(self.toggle_capture)
        self.btn_layout.addWidget(btn)
        self.btn_layout.nextRow()
        self.btn_toggle_capture = btn

    def _add_reset_roi_btn(self):
        btn = QtGui.QPushButton('Reset ROI')
        self.btn_layout.addWidget(btn)
        self.btn_layout.nextRow()
        btn.clicked.connect(self.reset_ROI)
        self.btn_reset_roi = btn

    def _add_gaussfit_btn(self):
        btn = QtGui.QPushButton('Gaussian Fit ON/OFF')
        self.btn_layout.addWidget(btn)
        self.btn_layout.nextRow()
        btn.clicked.connect(self.toggle_gaussian)
        btn.clicked.connect(self.updateRowprofile)
        btn.clicked.connect(self.updateColprofile)
        self.btn_gaussian = btn
    
    def _add_background_save_btn(self):
        btn = QtGui.QPushButton('Save background')
        self.btn_layout.addWidget(btn)
        self.btn_layout.nextRow()
        btn.clicked.connect(self._save_background)
        self.btn_save_background = btn

    def _add_background_substraction_checkbox(self):
        chbox = QtGui.QCheckBox('Substract background')
        self.btn_layout.addWidget(chbox)
        self.btn_layout.nextRow()
        chbox.stateChanged.connect(self._toggle_substraction)
        self.chbox_susbtract_background = chbox

    def _save_background(self):
        """Saves the current im_data to a .npy file for substraction.

        """
        np.save("background_{0}.npy".format(camera_serial_numbers[camera_index]), self.im_data)
        if self.substract_background:
            self.background_array = self.im_data
    
    def _toggle_substraction(self):
        if not self.substract_background:
            self.background_array = np.load("background_{0}.npy".format(camera_serial_numbers[camera_index]))
            self.substract_background = True

        else:
            self.substract_background = False

    def toggle_gaussian(self):
        self.gaussian = not self.gaussian
        if self.gaussian:
            cp = self.calculate_col_profile()
            print(self.x_int_lims)
            if not self.fit_roi_only:
                y_fit = gaussian_fit(self.cols_array, cp)
                cp_gaussian = self.cp_plot.plot(x=self.cols_array, y=y_fit[0])
            else:
                x0 = self.x_int_lims[0]
                x1 = self.x_int_lims[1]
                y_fit = gaussian_fit(self.cols_array[x0:x1], cp[x0:x1])
                cp_gaussian = self.cp_plot.plot(x=self.cols_array[x0:x1], y=y_fit[0])
            self.cp_gaussian = cp_gaussian
            
            rp = self.calculate_row_profile()
            if not self.fit_roi_only:
                x_fit = gaussian_fit(self.rows_array, rp)
                rp_gaussian = self.rp_plot.plot(x=x_fit[0], y=self.rows_array)
            else:
                y0 = self.y_int_lims[0]
                y1 = self.y_int_lims[1]
                x_fit = gaussian_fit(self.rows_array[y0:y1], rp[y0:y1])
                rp_gaussian = self.rp_plot.plot(x=x_fit[0][y0:y1], y=self.rows_array)

            self.rp_gaussian = rp_gaussian

        else:
            self.cp_gaussian.setData([], [])
            self.rp_gaussian.setData([], [])

    def reset_ROI(self):
        self.roi.setSize([self.cols, self.rows])
        self.roi.setPos([0,0])
    
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
        self.rows = im.getRows()
        self.cols = im.getCols()
        self.rows_array = np.arange(self.rows)
        self.cols_array = np.arange(self.cols)
        imarr = np.reshape(imarr, (self.rows, self.cols))
        imarr = imarr.T
        imarr = imarr[::,::-1]
        imarr = imarr.astype(np.float64)
        if self.substract_background:
            imarr -= self.background_array
        return imarr

    def init_cam(self):
        # 1. Start the bus interface
        # 2. Connect to the camera and start capturin
        # often the camera throws a bus master failure in the first and a isochronous start failure
        # in the second connection attempt. These are supposed to be caught in the following.
        for i in range(100):
            try:
                bus = pc2.BusManager()
                bus.forceAllIPAddressesAutomatically() # necessary due to my networking inabilities
                cam = pc2.GigECamera()
                cam.connect(bus.getCameraFromSerialNumber(camera_serial_numbers[camera_index]))
                # set up camera for GigE use
                gigeconf = cam.getGigEConfig()
                gigeconf.enablePacketResend = True
                cam.setGigEConfig(gigeconf)

                # set up configuration of camera
                conf = cam.getConfiguration()
                conf.numBuffers = 4
                conf.grabTimeout = self.acquisition_timer_interval
                conf.grabMode = 1 # BUFFER_FRAMES grab mode, see docs
                cam.setConfiguration(conf)
                
                # start streaming
                cam.startCapture()
                time.sleep(.1)
                print("Started stream for cam {0}".format(camera_serial_numbers[camera_index]))

            except pc2.Fc2error as e:
                try:
                    cam.disconnect()
                except:
                    pass

                del cam, bus
                print(e)
                print("Retrying...")
            else:
                break

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
        logging.debug("Trying to retrieve Buffer.")
        try:
            im = cam.retrieveBuffer()
            logging.debug("Buffer retrieved. Converting to array.")
            a = self.im_to_array(im) # for numpy manipulation of data
            logging.debug("Array conversion done.")
            return a
        except pc2.Fc2error as e:
            logging.error(e)
            if 'Timeout error' in e.__str__() and self.acquisition_inited: # this can cause the program to freeze
                logging.info("Restarting stream.")
                self.toggle_capture()
                self.toggle_capture()
            else: # could e.g. be image inconsistency, no restart needed 
                pass
            print(e.__str__())
            return self.im_data # don't update but don't confuse types either

    def updateImage(self):
        im_data = self.get_image()
        self.im_data = im_data
        self.img.setImage(im_data)

    def update_roi(self):
        x_integration_limits = self.x_int_lims
        y_integration_limits = self.y_int_lims
        boundaries = self.roi.parentBounds().getCoords()
        x_integration_limits[0] = int(max(boundaries[0], 0))
        x_integration_limits[1] = int(min(boundaries[2], self.cols))
        y_integration_limits[0] = int(max(boundaries[1], 0))
        y_integration_limits[1] = int(min(boundaries[3], self.rows))

    # add row profile
    def calculate_row_profile(self):
        x_integration_limits = self.x_int_lims
        im_data = self.im_data
        return np.sum(im_data[x_integration_limits[0]:x_integration_limits[1],::], axis=0)

    def updateRowprofile(self):
        rp = self.calculate_row_profile()
        self.rp_curve.setData(x=rp, y=self.rows_array)
        if self.gaussian:
            x_fit = gaussian_fit(self.rows_array, rp)
            #print("Row Mean: ", x_fit[2][0])
            if not x_fit[1]: # not estimate
                self.rp_gaussian.setPen(qualitative_colors[0])
            else:
                self.rp_gaussian.setPen(grey)
            self.rp_gaussian.setData(x=x_fit[0], y=self.rows_array)

    def calculate_col_profile(self):
        y_integration_limits = self.y_int_lims
        im_data = self.im_data
        return np.sum(im_data[::,y_integration_limits[0]:y_integration_limits[1]], axis=1)

    def updateColprofile(self):
        cp = self.calculate_col_profile()
        self.cp_curve.setData(y=cp)
        if self.gaussian:
            y_fit = gaussian_fit(self.cols_array, cp)
            print("Col Mean: ", y_fit[2][0])
            if not y_fit[1]: # not estimate
                self.cp_gaussian.setPen(qualitative_colors[0])
            else:
                self.cp_gaussian.setPen(grey)
            self.cp_gaussian.setData(y=y_fit[0])

    def toggle_capture(self):
        #TODO: Write a method that toggles the capture and processing of images.
        # Can this be done by disconnecting or stopping the main data acquisition timer?
        if self.capturing:
            self.acquisition_timer.stop()
            self.cam.stopCapture()
        else:
            self.cam.startCapture()
            self.acquisition_timer.start(self.acquisition_timer_interval)

        self.capturing = not self.capturing

def main():
    logging.basicConfig(filename='beamprofiler.log',
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
    logging.info('Started.')
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
