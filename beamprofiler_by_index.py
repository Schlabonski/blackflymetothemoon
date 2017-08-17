#! /usr/bin/python3

import datetime
import sys
import os
import time
import logging

import PyCapture2 as pc2
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import Qt
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
try:
    camera_index = int(sys.argv[1])
except:
    pass
camera_serial_numbers = [16292944, 16302806]

class CameraWindow(QtGui.QMainWindow):
    """Docstring for MainWindow. """

    cols = 1920
    rows = 1200
    cols_array = np.arange(cols)
    rows_array = np.arange(rows)
    acquisition_timer_interval = 1000

    def __init__(self, cam=None, parent=None, serial=None):
        """TODO: to be defined1. """
        super(CameraWindow, self).__init__(parent)

        # set state parameters
        self.capturing = True
        self.gaussian = False
        self.gaussian_inited = False
        self.acquisition_inited = False
        self.substract_background = False
        self.adapt_levels_boolean = False
        self.fit_roi_only = True

	## Initialize cam
        if cam is None:
            self.init_cam()
            self.serial = camera_serial_numbers[camera_index]
        else:
            self.cam = cam
            self.serial = serial

        ## Switch to using white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        # initialize the main window
        self.setWindowTitle('BlackFly me to the moon {0}'.format(serial))
        self.win = pg.GraphicsWindow()
        win = self.win
        win.resize(1000,600)
        self.setCentralWidget(win)

        # add layout 
        self._add_window_layout()

        # create menu bar
        #self.create_camera_menu()

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
        #rp_plot.hideAxis('bottom')
        self.rp_curve = rp_plot.plot(x=rp, y=np.arange(self.rows),pen=qualitative_colors[1])
        self.rp_plot = rp_plot

        acquisition_timer.timeout.connect(self.updateRowprofile)

        # add columnn profile
        cp = self.calculate_col_profile()
        cp_plot = pg.PlotWidget()
        self.vlayout.addWidget(cp_plot, 1, 0)
        #cp_plot.hideAxis('left')
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
        self._add_dump_data_btn()
        self._add_image_level_dialogue()

        # create a popup window that shows the results of the gauss fit in
        # a scroll plot
        self.scwin = ScrollWindow(parent=self)
        self.acquisition_timer.timeout.connect(self.scwin.updatePlot)

    def closeEvent(self, event):
        event.accept()
        try:
            self.cam.stopCapture()
            self.cam.disconnect()
            del cam
        except:
            pass
        self.scwin.deleteLater()
        self.deleteLater()

    def _add_window_layout(self):
        vlayout = QtGui.QGridLayout(self.win)
        vlayout.setColumnStretch(0, 2)
        vlayout.setColumnStretch(1, 1)
        vlayout.setRowStretch(0,2)
        vlayout.setRowStretch(1,1)
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
        self.chbox_substract_background = chbox

    def _add_dump_data_btn(self):
        btn = QtGui.QPushButton('Save data')
        self.btn_layout.addWidget(btn)
        self.btn_layout.nextRow()
        btn.clicked.connect(self._dump_data)
        self.btn_dump_data = btn

    def _add_image_level_dialogue(self):
        """
        Adds a checkbox and two sliders two control the black and white level
        of the image.

        :return: None
        """
        # add a checkbox to toggle automatic level setting
        # add a layout for the checkbox. This Layout will have three components
        # The Checkbox itseld and the two boundaries

        # The Checkbox
        chbox_layout = self.btn_layout.addLayout()
        self.btn_layout.nextRow()
        chbox = QtGui.QCheckBox('User custom image levels with Range')
        chbox_layout.addWidget(chbox)
        chbox_layout.nextCol()
        chbox.stateChanged.connect(self._toggle_adapt_levels)
        self.chbox_adapt_levels = chbox

        # Lower Boundary
        self.label_black = QtGui.QLabel('0')
        chbox_layout.addWidget(self.label_black)
        chbox_layout.nextCol()

        # Upper Boundary
        self.label_white = QtGui.QLabel('255')
        chbox_layout.addWidget(self.label_white)

        # add a slider that regulates the black value
        # The new Layer consists of three parts
        # The lower Boundary, the slider and the upper boundary

        # Layout and lower boundary
        sl_min_layout = self.btn_layout.addLayout()
        self.btn_layout.nextRow()
        self.label_min_1 = QtGui.QLabel('0')
        sl_min_layout.addWidget(self.label_min_1)
        sl_min_layout.nextCol()

        # The slider
        lvl_sl_min = QtGui.QSlider(Qt.Horizontal)
        lvl_sl_min.setMinimum(0)
        lvl_sl_min.setMaximum(255)
        lvl_sl_min.setValue(0)
        lvl_sl_min.valueChanged.connect(self._adapt_levels)
        self.lvl_sl_min = lvl_sl_min
        sl_min_layout.addWidget(lvl_sl_min)
        sl_min_layout.nextCol()

        # The upper boundary
        self.label_max_1 = QtGui.QLabel('255')
        sl_min_layout.addWidget(self.label_max_1)

        # add a slider to control white level
        # The new Layer consists of three parts
        # The lower Boundary, the slider and the upper boundary

        # Layout and lower boundary
        sl_max_layout = self.btn_layout.addLayout()
        self.btn_layout.nextRow()
        self.label_min_2 = QtGui.QLabel('0')
        sl_max_layout.addWidget(self.label_min_2)
        sl_max_layout.nextCol()

        # The slider
        lvl_sl_max = QtGui.QSlider(Qt.Horizontal)
        lvl_sl_max.setMinimum(0)
        lvl_sl_max.setMaximum(255)
        lvl_sl_max.setValue(255)
        lvl_sl_max.valueChanged.connect(self._adapt_levels)
        self.lvl_sl_max = lvl_sl_max
        sl_max_layout.addWidget(lvl_sl_max)
        sl_max_layout.nextCol()

        # Upper boundary
        self.label_max_2 = QtGui.QLabel('255')
        sl_max_layout.addWidget(self.label_max_2)


    def _save_background(self):
        """Saves the current im_data to a .npy file for substraction.

        """
        np.save("background_{0}.npy".format(self.serial), self.im_data)
        if self.substract_background:
            self.background_array = self.im_data
    
    def _toggle_substraction(self):
        if not self.substract_background:
            self.background_array = np.load("background_{0}.npy".format(self.serial))
            self.substract_background = True

        else:
            self.substract_background = False

    def _toggle_adapt_levels(self):
        """ Toggles between automatic&userdefined black and white values of
        self.img"""
        self.adapt_levels_boolean = not self.adapt_levels_boolean
        if self.adapt_levels_boolean:
            self._adapt_levels()

    def _adapt_levels(self):
        """ Adapts the black and white levels of the image. """

        if self.adapt_levels_boolean:
            # set image levels to slider values
            self.img.setLevels([self.lvl_sl_min.value(), self.lvl_sl_max.value()])

        # do not allow black level slider to be greater than white level slider
        self.lvl_sl_max.setMinimum(self.lvl_sl_min.value())
        self.lvl_sl_min.setMaximum(self.lvl_sl_max.value())
        self.label_max_1.setText(str(self.lvl_sl_max.value()))
        self.label_min_2.setText(str(self.lvl_sl_min.value()))
        self.label_black.setText(str(self.lvl_sl_min.value()))
        self.label_white.setText(str(self.lvl_sl_max.value()))

    def toggle_gaussian(self):
        self.gaussian = not self.gaussian
        if self.gaussian:
            cp = self.calculate_col_profile()
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
                rp_gaussian = self.rp_plot.plot(x=x_fit[0], y=self.rows_array[y0:y1])

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
            print("detect bus")
            try:
                print("init cam")
                bus = pc2.BusManager()

                # test code
                cams=bus.discoverGigECameras(10)
                print(cams)
                # end test code
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
                print("Retrying 1...")
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
        self.img.setImage(im_data, autoLevels=not self.adapt_levels_boolean)
        if not self.adapt_levels_boolean:
            self.label_black.setText(str(self.img.levels[0]))
            self.label_white.setText(str(self.img.levels[1]))

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
        integral = np.sum(im_data[x_integration_limits[0]:x_integration_limits[1],::], axis=0)
        length = x_integration_limits[1] - x_integration_limits[0]
        normalized = integral / length
        return normalized

    def updateRowprofile(self):
        rp = self.calculate_row_profile()
        self.rp_data = rp
        self.rp_curve.setData(x=rp, y=self.rows_array)
        if self.gaussian:
            #x_fit = gaussian_fit(self.rows_array, rp)
            #print("Row Mean: ", x_fit[2][0])
            if not self.fit_roi_only:
                x_fit = gaussian_fit(self.rows_array, rp)
                if not x_fit[1]: # not estimate
                    self.rp_gaussian.setPen(qualitative_colors[0])
                else:
                    self.rp_gaussian.setPen(grey)
                self.rp_gaussian.setData(x=x_fit[0], y=self.rows_array)
            else:
                y0 = self.y_int_lims[0]
                y1 = self.y_int_lims[1]
                x_fit = gaussian_fit(self.rows_array[y0:y1], rp[y0:y1])
                if not x_fit[1]: # not estimate
                    self.rp_gaussian.setPen(qualitative_colors[0])
                else:
                    self.rp_gaussian.setPen(grey)
                self.rp_gaussian.setData(x=x_fit[0], y=self.rows_array[y0:y1])
            #self.rp_gaussian.setData(x=x_fit[0], y=self.rows_array)
            self.scwin.feedData(np.array([x_fit[2][0], x_fit[2][1]]), data_slice=[2,4])

    def calculate_col_profile(self):
        y_integration_limits = self.y_int_lims
        im_data = self.im_data
        integral = np.sum(im_data[::,y_integration_limits[0]:y_integration_limits[1]], axis=1)
        length = y_integration_limits[1] - y_integration_limits[0]
        normalized = integral / length
        return normalized

    def updateColprofile(self):
        cp = self.calculate_col_profile()
        self.cp_data = cp
        self.cp_curve.setData(y=cp)
        if self.gaussian:
            #y_fit = gaussian_fit(self.cols_array, cp)
            if not self.fit_roi_only:
                y_fit = gaussian_fit(self.cols_array, cp)
                if not y_fit[1]: # not estimate
                    self.cp_gaussian.setPen(qualitative_colors[0])
                else:
                    self.cp_gaussian.setPen(grey)
                self.cp_gaussian.setData(y=y_fit[0])
            else:
                x0 = self.x_int_lims[0]
                x1 = self.x_int_lims[1]
                y_fit = gaussian_fit(self.cols_array[x0:x1], cp[x0:x1])
                if not y_fit[1]: # not estimate
                    self.cp_gaussian.setPen(qualitative_colors[0])
                else:
                    self.cp_gaussian.setPen(grey)
                self.cp_gaussian.setData(x=self.cols_array[x0:x1], y=y_fit[0])
            #self.cp_gaussian.setData(y=y_fit[0])
            self.scwin.feedData(np.array([y_fit[2][0], y_fit[2][1]]), data_slice=[0,2], count=False)

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

    def _dump_data(self):
        """Stores current data with timestampt in subdirectory.

        """
        data = {}
        # add image array
        data["im_array"] = self.im_data

        # add row and column profiles
        data["row_profile"] = self.rp_data
        data["col_profile"] = self.cp_data

        # add region of interest info
        roi_pos = self.roi.pos()
        data["roi_pos"] = np.array([roi_pos.x(), roi_pos.y()])
        roi_size = self.roi.size()
        data["roi_size"] = np.array([roi_size.x(), roi_size.y()])
        
        data_root = "./data"
        if not os.path.exists(data_root):
            os.makedirs(data_root)

        timestamp = time.time()
        st = datetime.datetime.fromtimestamp(timestamp).strftime('_%Y_%m_%d_%H_%M_%S')

        for name in data.keys():
            arr = data[name]
            filename = data_root + '/' + name + st
            np.save(filename, arr)

class ScrollWindow(QtGui.QMainWindow):

    x = np.arange(100) # static x axis, store 200 points for every parameter

    def __init__(self, parent=None):
        """TODO: to be defined1. """
        super(ScrollWindow, self).__init__(parent)
        
        # initialize the window 
        win = pg.GraphicsWindow()
        win.resize(1000,600)
        self.setCentralWidget = win
        self.win = win

        # initialize two plots to show the evolution of e.g.
        # mean and stdev of row and column fits
        lineplots = []
        title = ["Column fit", "Row fit"]
        for i in range(2):
            p_mean = win.addPlot(title="{0} Mean".format(title[i]))
            p_std = win.addPlot(title="{0} Standard Deviation".format(title[i]))
            for p in [p_mean, p_std]:
                p.setDownsampling(mode='peak')
                p.setClipToView(True)
                p.setLimits(xMax=max(self.x))
                p.addLegend()
            pen0 = pg.mkPen(qualitative_colors[0], width=3)
            pen1 = pg.mkPen(qualitative_colors[1], width=3)
            l0 = p_mean.plot(pen=pen0, name='Mean')
            l1 = p_std.plot(pen=pen1, name='Standard deviation')
            lineplots.append(l0)
            lineplots.append(l1)

            win.nextRow()

        self.lineplots = lineplots

        # describe how the plotted parameters behave, i.e. how many they are
        # and initialize structure to store them
        self.n_pars = 4
        data_array = np.zeros((self.n_pars, len(self.x)))
        self.data_array = data_array
        self.feed_count = 0

    def feedData(self, data_chunk, data_slice, count=True):
        """Feeds a new chunk of data to the data array.
        The chunk is pushed into data_array.

        :data_chunk: np.ndarray of length n_pars

        """
        if self.feed_count < len(self.x) and count:
            self.feed_count += 1
        n0 = data_slice[0]
        n1 = data_slice[1]
        data_array = self.data_array
        data_array[n0:n1] = np.roll(data_array[n0:n1], shift=-1, axis=1)
        data_array[n0:n1, -1] = data_chunk
        self.data_array = data_array

    def updatePlot(self):
        indx_feed_count = len(self.x) - self.feed_count
        for i, l in enumerate(self.lineplots):
            l.setData(x=self.x[:self.feed_count], y=self.data_array[i, indx_feed_count:])

class MainWindow(QtGui.QMainWindow):

    """The MainWindow should contain a dialog from which one
    can choose which camera stream to start. The camera object will 
    then be passed to a CameraWindow for display."""

    acquisition_timer_interval = 1000
    def __init__(self, parent=None):
        """Initialize the main window and its dialog. """

        super(MainWindow, self).__init__(parent)
        
        self.setWindowTitle("Launch control center")
        # initialize the gige bus to manage the cameras
        self.bus, serial_nums = self.init_gige_bus()

        # create a dropdown menu for camera selection
        comboBox = QtGui.QComboBox(self)
        for serial in serial_nums:
            comboBox.addItem(str(serial))
        comboBox.move(40, 10)
        self.comboBox = comboBox

        # create a button to start stream
        startButton = QtGui.QPushButton("Start stream", self)
        startButton.clicked.connect(self.start_stream)
        startButton.move(40, 50)
        self.camera_windows = []


    def init_gige_bus(self):
        """Initializes a GigE bus interface and finds all attached cameras.
        :returns: pc2.BusManager

        """
        bus = pc2.BusManager()
        # test code
        cams=bus.discoverGigECameras(10)
        print(cams)
        # end test code
        bus.forceAllIPAddressesAutomatically()
        n_cams = bus.getNumOfCameras()
        serial_nums = []
        for i in range(n_cams):
            serial_nums.append(bus.getCameraSerialNumberFromIndex(i))
        return bus, serial_nums

    def initialize_cam(self, serial):
        """
        Initializes a camera for a given serial number and bus.
        The bus should be initialized first.
        :returns: GigECamera object
        """
        # sometimes the cam has a hard time starting. We catch some
        # exceptions and just try again.
        for i in range(20):
            cam = pc2.GigECamera()
            try:
                cam.connect(self.bus.getCameraFromSerialNumber(serial))
                print("connected:")
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
                print("Started stream for cam {0}".format(serial))
                return cam

            except pc2.Fc2error as e:
                try:
                    cam.disconnect()
                except:
                    pass

                del cam
                print(e)
                print("Retrying 2...")
            else:
                break
            time.sleep(.1)
        print("No connection to camera established.")

    def start_stream(self):
        serial = int(self.comboBox.currentText())
        cam = self.initialize_cam(serial)
        if cam is not None:
            camWin = CameraWindow(cam=cam, parent=self, serial=serial)
            camWin.show()
            self.camera_windows.append(camWin)

def main():
    logging.basicConfig(filename='beamprofiler.log',
                                filemode='w',
                                format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                                datefmt='%H:%M:%S',
                                level=logging.DEBUG)
    logging.info('Started.')
    app = QtGui.QApplication(sys.argv)
    #window = CameraWindow()
    window = MainWindow()
    window.show()
    app.exec_()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        main()
