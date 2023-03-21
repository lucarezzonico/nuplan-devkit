import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QObject, QThread, QMutex, pyqtSignal, QTimer, QSettings
from multiprocessing import Manager
from src.utils.logger import CLOG
from src.gui_setup.MainWindow import Ui_MainWindow
import yaml
from task import Task

def runGUI(guiState, silence):
    app = QApplication(sys.argv)
    mainwindow = MainWindow(guiState, silence)
    mainwindow.cl.log('Starting up')
    # widget = QStackedWidget()
    # widget.addWidget(mainwindow)
    # widget.show()
    mainwindow.show()

    if not app.exec_():
        mainwindow.shutdown()
        sys.exit(0)

def guiStateInit(guiState):
    guiState['shutdown'] = False
    

class MainWindow(QMainWindow):
    def __init__(self, guiState, silence=False, parent=None):
        super(MainWindow, self).__init__(parent)
        # UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
                
        # Push Buttons
        self.ui.pb_run.clicked.connect((self.pb_run_pushed))
        
        # add comboBox item
        self.ui.cb_task

        # Settings
        self.loadSettings()
        
        # COMMUNICATION BETWEEN THREADS
        self.guiState = guiState
        self.cl = CLOG(processName="GUI", timed=True, silence=silence)
        
        self.task = Task() # Trainin, Simulation, Network Visualization,
        
    def pb_run_pushed(self):
        self.ui.cb_task.currentText()
        print(self.ui.cb_task.currentText())
        
    def loadSettings(self):
        self.setting_window = QSettings('Metar App', 'Window Size')
        self.setting_variables = QSettings('Metar App', 'Variables')
        
        height = self.setting_window.value('window_height')
        width = self.setting_window.value('window_width')
        print(height, width)
        
        # First time open the app with default values (except), then open the app with the last used 
        try:
            self.ui.cb_task.setCurrentIndex(int(self.setting_variables.value('cb_task')))

        except:
            pass
        
    def saveSettings(self):
        self.setting_variables.setValue('cb_task', self.ui.cb_task.currentIndex())

    def shutdown(self):
        self.saveSettings()
        self.guiState['shutdown'] = True
        self.cl.log('Graceful shutdown')


if __name__ == '__main__':
    guiState = dict()
    guiStateInit(guiState)
    
    runGUI(guiState, silence=False)