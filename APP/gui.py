import os
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtCore import Qt, QObject, QThread, QMutex, pyqtSignal, QTimer, QSettings
from multiprocessing import Manager
from src.utils.logger import CLOG
from src.gui_setup.MainWindow import Ui_MainWindow
import yaml
from task import Task

def runGUI(guiState, soundState, silence):
    app = QApplication(sys.argv)
    mainwindow = MainWindow(guiState, soundState, silence)
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
    guiState['getReportClicked'] = False
    guiState['metarReport'] = ""


class MainWindow(QMainWindow):
    def __init__(self, guiState, silence=False, parent=None):
        super(MainWindow, self).__init__(parent)
        # UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
                
        # Push Buttons
        self.ui.run.clicked.connect((self.getReport))
        
        # report history folder
        self.abs_dirname = os.path.dirname(os.path.abspath(__file__))
        
        # add comboBox item
        self.ui.cb_task

        # Settings
        self.loadSettings()
        
        # COMMUNICATION BETWEEN THREADS
        self.guiState = guiState
        self.cl = CLOG(processName="GUI", timed=True, silence=silence)
        
        self.task = Task() # Trainin, Simulation, Network Visualization,
        
    def browseFolder(self):
        self.history_folder = QFileDialog.getExistingDirectory(None, 'Select a folder:', self.abs_dirname, QFileDialog.ShowDirsOnly)
        self.ui.lineEdit_Config_history_path.setText(self.history_folder)

    def getReport(self):
        self.guiState['getReportClicked'] = True

        # Run Report Computation
        self.guiState['metarReport'] = metar.get_report_from_gui(type_of_report=self.ui.comboBox_Config_type_of_report.currentData(), station_identifier=self.ui.lineEdit_Config_station_identifier.text(), station_altitude=self.ui.spinBox_Config_station_altitude.value(),
                                                                force=self.ui.spinBox_Wind_force.value(), direction_left=self.ui.spinBox_Wind_direction_left.value(), direction_right=self.ui.spinBox_Wind_direction_right.value(), gusts=self.ui.spinBox_Wind_gusts.value(),
                                                                prevailing_distance=self.ui.spinBox_Visibility_prevailing_distance.value(), smallest_distance=self.ui.spinBox_Visibility_smallest_distance.value(),
                                                                smallest_direction=self.ui.comboBox_Visibility_smallest_direction.currentData(),
                                                                weather_phenomena=self.ui.lineEdit_Weather_phenomena.text(),
                                                                cloud_list=[{"crowd": str(self.ui.comboBox_Clouds_layer1_crowd.currentData()), "height": self.ui.spinBox_Clouds_layer1_height.value(), "thundercloud": self.ui.comboBox_Clouds_layer1_thundercloud.currentText()},
                                                                            {"crowd": str(self.ui.comboBox_Clouds_layer2_crowd.currentData()), "height": self.ui.spinBox_Clouds_layer2_height.value(), "thundercloud": self.ui.comboBox_Clouds_layer2_thundercloud.currentText()},
                                                                            {"crowd": str(self.ui.comboBox_Clouds_layer3_crowd.currentData()), "height": self.ui.spinBox_Clouds_layer3_height.value(), "thundercloud": self.ui.comboBox_Clouds_layer3_thundercloud.currentText()}],
                                                                dry_temperature=self.ui.doubleSpinBox_Temperature_and_DewPoint_dry_temperature.value(), wet_temperature=self.ui.doubleSpinBox_Temperature_and_DewPoint_wet_temperature.value(),
                                                                QFE=self.ui.spinBox_AirPressure_QFE.value()
                                                                )
                                                    
        self.updateReport()
        self.saveReport()
        
        self.soundState['Lana'] = True
        
        self.guiState['getReportClicked'] = False
            
    def updateReport(self):
        self.ui.lineEdit_METAR_report.setText(self.guiState['metarReport'])

    def saveReport(self):
        # Compute Report History
        if os.path.exists(self.ui.lineEdit_Config_history_path.text()) and len(self.ui.lineEdit_Config_history_path.text())>0:
            text_file = open(self.ui.lineEdit_Config_history_path.text() + "/report_history.txt", "a")
            text_file.write(metar.get_Date_and_Time() + ": " + self.guiState['metarReport'] + "\n")
            text_file.close()
        
    def loadSettings(self):
        self.setting_window = QSettings('Metar App', 'Window Size')
        self.setting_variables = QSettings('Metar App', 'Variables')
        
        height = self.setting_window.value('window_height')
        width = self.setting_window.value('window_width')
        print(height, width)
        
        # First time open the app with default values (except), then open the app with the last used 
        try:
            self.ui.comboBox_Config_type_of_report.setCurrentIndex(int(self.setting_variables.value('Configurations/type of report')))
            self.ui.lineEdit_Config_history_path.setText(self.setting_variables.value('Configurations/report history path'))
            self.ui.lineEdit_Config_station_identifier.setText(self.setting_variables.value('Configurations/station identifier'))
            self.ui.spinBox_Config_station_altitude.setValue(int(self.setting_variables.value('Configurations/station altitude')))
            self.ui.spinBox_Visibility_prevailing_distance.setValue(int(self.setting_variables.value('Observation/Visibility/prevailing distance')))
            self.ui.spinBox_Visibility_smallest_distance.setValue(int(self.setting_variables.value('Observation/Visibility/smallest distance')))
            self.ui.comboBox_Visibility_smallest_direction.setCurrentIndex(int(self.setting_variables.value('Observation/Visibility/smallest direction')))            
            self.ui.comboBox_Clouds_layer1_crowd.setCurrentIndex(int(self.setting_variables.value('Observation/Clouds/layer1 crowd')))
            self.ui.spinBox_Clouds_layer1_height.setValue(int(self.setting_variables.value('Observation/Clouds/layer1 height')))
            self.ui.comboBox_Clouds_layer1_thundercloud.setCurrentIndex(int(self.setting_variables.value('Observation/Clouds/layer1 thundercloud')))
            self.ui.comboBox_Clouds_layer2_crowd.setCurrentIndex(int(self.setting_variables.value('Observation/Clouds/layer2 crowd')))
            self.ui.spinBox_Clouds_layer2_height.setValue(int(self.setting_variables.value('Observation/Clouds/layer2 height')))
            self.ui.comboBox_Clouds_layer2_thundercloud.setCurrentIndex(int(self.setting_variables.value('Observation/Clouds/layer2 thundercloud')))
            self.ui.comboBox_Clouds_layer3_crowd.setCurrentIndex(int(self.setting_variables.value('Observation/Clouds/layer3 crowd')))
            self.ui.spinBox_Clouds_layer3_height.setValue(int(self.setting_variables.value('Observation/Clouds/layer3 height')))
            self.ui.comboBox_Clouds_layer3_thundercloud.setCurrentIndex(int(self.setting_variables.value('Observation/Clouds/layer3 thundercloud')))
            self.ui.lineEdit_Weather_phenomena.setText(self.setting_variables.value('Observation/Weather/phenomena'))
            self.ui.spinBox_Wind_force.setValue(int(self.setting_variables.value('Measurements/Wind/force')))
            self.ui.spinBox_Wind_direction_left.setValue(int(self.setting_variables.value('Measurements/Wind/direction left limit')))
            self.ui.spinBox_Wind_direction_right.setValue(int(self.setting_variables.value('Measurements/Wind/direction right limit')))
            self.ui.spinBox_Wind_gusts.setValue(int(self.setting_variables.value('Measurements/Wind/gusts')))
            self.ui.doubleSpinBox_Temperature_and_DewPoint_dry_temperature.setValue(float(self.setting_variables.value('Measurements/Temperature and Dew Point/dry temperature')))
            self.ui.doubleSpinBox_Temperature_and_DewPoint_wet_temperature.setValue(float(self.setting_variables.value('Measurements/Temperature and Dew Point/wet temperature')))
            self.ui.spinBox_AirPressure_QFE.setValue(int(self.setting_variables.value('Measurements/Air Pressure/QFE')))
            self.ui.lineEdit_METAR_report.setText(self.setting_variables.value('METAR Report'))
            
        except:
            pass
        
    def saveSettings(self):
        self.setting_window.setValue('window_height', self.rect().height())
        self.setting_window.setValue('window_width', self.rect().width())

        self.setting_variables.setValue('Configurations/type of report', self.ui.comboBox_Config_type_of_report.currentIndex())
        self.setting_variables.setValue('Configurations/report history path', self.ui.lineEdit_Config_history_path.text())
        self.setting_variables.setValue('Configurations/station identifier', self.ui.lineEdit_Config_station_identifier.text())
        self.setting_variables.setValue('Configurations/station altitude', self.ui.spinBox_Config_station_altitude.value())
        self.setting_variables.setValue('Observation/Visibility/prevailing distance', self.ui.spinBox_Visibility_prevailing_distance.value())
        self.setting_variables.setValue('Observation/Visibility/smallest distance', self.ui.spinBox_Visibility_smallest_distance.value())
        self.setting_variables.setValue('Observation/Visibility/smallest direction', self.ui.comboBox_Visibility_smallest_direction.currentIndex())
        self.setting_variables.setValue('Observation/Clouds/layer1 crowd', self.ui.comboBox_Clouds_layer1_crowd.currentIndex())
        self.setting_variables.setValue('Observation/Clouds/layer1 height', self.ui.spinBox_Clouds_layer1_height.value())
        self.setting_variables.setValue('Observation/Clouds/layer1 thundercloud', self.ui.comboBox_Clouds_layer1_thundercloud.currentIndex())
        self.setting_variables.setValue('Observation/Clouds/layer2 crowd', self.ui.comboBox_Clouds_layer2_crowd.currentIndex())
        self.setting_variables.setValue('Observation/Clouds/layer2 height', self.ui.spinBox_Clouds_layer2_height.value())
        self.setting_variables.setValue('Observation/Clouds/layer2 thundercloud', self.ui.comboBox_Clouds_layer2_thundercloud.currentIndex())
        self.setting_variables.setValue('Observation/Clouds/layer3 crowd', self.ui.comboBox_Clouds_layer3_crowd.currentIndex())
        self.setting_variables.setValue('Observation/Clouds/layer3 height', self.ui.spinBox_Clouds_layer3_height.value())
        self.setting_variables.setValue('Observation/Clouds/layer3 thundercloud', self.ui.comboBox_Clouds_layer3_thundercloud.currentIndex())
        self.setting_variables.setValue('Observation/Weather/phenomena', self.ui.lineEdit_Weather_phenomena.text())
        self.setting_variables.setValue('Measurements/Wind/force', self.ui.spinBox_Wind_force.value())
        self.setting_variables.setValue('Measurements/Wind/direction left limit', self.ui.spinBox_Wind_direction_left.value())
        self.setting_variables.setValue('Measurements/Wind/direction right limit', self.ui.spinBox_Wind_direction_right.value())
        self.setting_variables.setValue('Measurements/Wind/gusts', self.ui.spinBox_Wind_gusts.value())
        self.setting_variables.setValue('Measurements/Temperature and Dew Point/dry temperature', self.ui.doubleSpinBox_Temperature_and_DewPoint_dry_temperature.value())
        self.setting_variables.setValue('Measurements/Temperature and Dew Point/wet temperature', self.ui.doubleSpinBox_Temperature_and_DewPoint_wet_temperature.value())
        self.setting_variables.setValue('Measurements/Air Pressure/QFE', self.ui.spinBox_AirPressure_QFE.value())
        self.setting_variables.setValue('METAR Report', self.ui.lineEdit_METAR_report.text())
        
    def shutdown(self):
        self.saveSettings()
        self.guiState['shutdown'] = True
        self.cl.log('Graceful shutdown')


if __name__ == '__main__':
    guiState = Manager().dict()
    guiStateInit(guiState)
        
    runGUI(guiState, silence=False)