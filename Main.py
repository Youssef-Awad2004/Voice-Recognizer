from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic

import Interface
import sys

class UI(QMainWindow):
    def __init__(self):
        super(UI,self).__init__()
        uic.loadUi("FingerPrint.ui",self)
        Interface.initConnectors(self)
        self.show()

    def update_progress_bar(self,value):
        self.progress.setValue(value)


app=QApplication(sys.argv)
UIWindow=UI()
app.exec_() 