import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

cols = 1920
rows = 1080

def get_image():
    """Returns an image.
    :returns: np.ndarray

    """
    return np.random.rand(cols, rows)

# init app and window
app = QtGui.QApplication([])
win = pg.GraphicsWindow(title='BlackFly Mushroom')
win.resize(1000,600)
imageview = win.addViewBox(row=0, col=0, lockAspect=True)

# add image plot
img = pg.ImageItem(border='w')
imageview.addItem(img)
im_data = get_image()
img.setImage(im_data)

def updateImage():
    global im_data
    im_data = get_image()
    img.setImage(im_data)

timer = QtCore.QTimer()
timer.timeout.connect(updateImage)
timer.start(50)

# add row profile
def calculate_row_profile():
    global im_data
    return np.sum(im_data, axis=0)

rp = calculate_row_profile()
rp_plot = win.addPlot(row=0, col=1)
rp_curve = rp_plot.plot(x=calculate_row_profile(), y=np.arange(rows))

def updateRowprofile():
    rp = calculate_row_profile()
    rp_curve.setData(x=rp, y=np.arange(rows))

timer.timeout.connect(updateRowprofile)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
