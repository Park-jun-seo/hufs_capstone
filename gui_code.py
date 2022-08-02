import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator ,MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import uic


from PyQt5.QtCore import *
from PyQt5 import QtCore, QtMultimedia
import librosa
import librosa.display

import queue, os, threading

import pyaudio
import wave
import time
from PyQt5.QtMultimedia import QSound
from PyQt5.QtCore import QCoreApplication
form_class = uic.loadUiType("untitled.ui")[0]


class tiThread(QThread):
    
 
    def __init__(self, parent):
        super().__init__(parent)
        self.ti=0    
        self.parent = parent 
        self.parent.pushButton_4.setDisabled(True)
        self.parent.pushButton_5.clicked.connect(self.tistop1)

        self.parent.timerVar = QTimer()
        self.parent.timerVar.setInterval(1000)
        self.parent.timerVar.timeout.connect(self.printTime)
        self.parent.timerVar.start()
        


    def printTime(self):
        self.ti+=1
        self.parent.lcdNumber.display(self.ti)
        

    def tistop1(self):
        self.parent.label_2.setText('예측 중...')
        self.parent.pushButton_4.setDisabled(False)
        self.parent.pushButton_5.setDisabled(True)
        self.parent.timerVar.stop()
        
        


class reThread(QThread):
    
    #초기화 메서드 구현    
    def __init__(self, parent): 
        super().__init__(parent)
        self.MIC_DEVICE_ID = 1
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000     
        self.SAMPLE_SIZE = 2  
        self.target = 'abc.wav'
        self.tog = False
        self.frames = []
        self.parent = parent
        self.p = pyaudio.PyAudio()
        
        
        
        self.stream = self.p.open(input_device_index=self.MIC_DEVICE_ID,
                    format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK)


        self.parent.timerVar2 = QTimer()
        self.parent.pushButton_5.clicked.connect(self.tistop)
        self.parent.timerVar2.setInterval(1)
        self.parent.timerVar2.timeout.connect(self.record)
        self.parent.timerVar2.start()


    def tistop(self):
        self.tog = True
        self.parent.timerVar2.stop()
        self.parent.pushButton_4.setDisabled(False)
        self.parent.pushButton_6.setDisabled(False)

        self.record_data()
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.plt_g()
        
        self.pyto_def()
        
        

        
    def record(self):      
        data = self.stream.read(self.CHUNK)
        self.frames.append(data)
        if self.tog == True:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

 
        
    def record_data(self):
        wf = wave.open(self.target, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.SAMPLE_SIZE)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self.frames))
 
        if isinstance(self.target, str):
            wf.close()

    def plt_g(self):
  
        self.ax = self.fig.add_subplot(111)
        for i in reversed(range(self.parent.graph.count())): 
            self.parent.graph.itemAt(i).widget().deleteLater()

       

        self.ax.get_yaxis().set_visible(False)

   
        self.fig.subplots_adjust(0.05, 0.25, 0.95, 1)
        self.parent.graph.addWidget(self.canvas)

        audio_path = 'abc.wav'
        y, sr = librosa.load(audio_path, sr=16000)
        librosa.display.waveshow(y, sr,x_axis='s',alpha=0.5,ax=self.ax ) 
        self.canvas.draw()

        

    def pyto_def(self):
        
        import pyto
        urr = self.parent.plainTextEdit.toPlainText()
        #print(urr)
        #st = pyto.pre_S('G:\\내 드라이브\\캡스톤프로젝트\\ver.3\\model\\first_para_1000.tar')
       
        st = pyto.pre_S(urr)
        self.parent.textEdit_2.clear()
        self.parent.textEdit_2.setFontPointSize(28)
        ans = ['시','청','의','창1','살1','은','외','창2','살2','이','다']
        #st=['청', '의', '창', '살', '은', '외', '창', '살', '이', '다']
        #a=0
        b=0
        
        for j in st:
            if (j =='창' or j=='살') and b<5:
                st[b] =j+'1'
            elif (j =='창' or j=='살') and b>5:
                st[b] =j+'2'
            b+=1
        a_sub_b = [x for x in ans if x not in st]
        for j in ans:
            if j in a_sub_b:
                self.parent.textEdit_2.setTextColor(QColor(255, 0, 0, 255))
                self.parent.textEdit_2.insertPlainText(''.join([i for i in j if not i.isdigit()]))
            else:
                self.parent.textEdit_2.setTextColor(QColor(0, 0, 255, 255))
                self.parent.textEdit_2.insertPlainText(''.join([i for i in j if not i.isdigit()]))
        self.parent.listWidget.clear()
        for j in a_sub_b:
            self.parent.listWidget.addItem(''.join([i for i in j if not i.isdigit()]))
 
        '''
        for j in  ans:
            if st[a] == j:
                self.parent.textEdit_2.setTextColor(QColor(0, 0, 255, 255))
                self.parent.textEdit_2.insertPlainText(j)
            elif st[a] != j:
                self.parent.textEdit_2.setTextColor(QColor(255, 0, 0, 255))
                self.parent.textEdit_2.insertPlainText(j)
            a+=1
            if len(st)<a:
                break'''
        #self.parent.textEdit_2.insertPlainText(''.join(a_sub_b))
        self.parent.textEdit_2.setAlignment(Qt.AlignCenter)
        self.parent.label_2.setText('예측완료...')
        #self.parent.textEdit_2.insertPlainText(''.join(a_sub_b))
        #self.parent.textEdit_2.setAlignment(Qt.AlignCenter)
        #print(st)
      
        



class WindowClass(QMainWindow, form_class) :

    def __init__(self) :
        super().__init__()
        self.q = queue.Queue()
        self.ecorder = False
        self.recording = False
        #self.setFixedSize(757, 444)
        
        
        
        self.setupUi(self)

        self.textEdit.setReadOnly(True)
        self.textEdit_2.setReadOnly(True)
        self.pushButton_5.setDisabled(True)
        self.pushButton_6.setDisabled(True)
        self.lcdNumber.setSegmentStyle(2)
        self.pushButton_4.setStyleSheet('QPushButton {color: red;}')
        self.pushButton_6.setStyleSheet('QPushButton {color: green;}')
        
        self.pushButton_4.clicked.connect(self.re_start)
        self.pushButton_2.clicked.connect(self.audio)
        self.pushButton_3.clicked.connect(self.ans_img)
        self.pushButton_6.clicked.connect(self.reaudio)
        '''
        qPixmapVar = QPixmap()
        qPixmapVar.load('은.jpg')
        self.label_3.setScaledContents(True)
        self.label_3.setPixmap(qPixmapVar)'''
       
     
        self.textEdit_2.clear()
    
    def audio(self): 
        self.chunk = 1024
        self.path = 'as.wav'

        with wave.open(self.path, 'rb') as f:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                            channels = f.getnchannels(),
                            rate = f.getframerate(),
                            output= True)

            data = f.readframes(self.chunk)
            while data:
                stream.write(data)
                data = f.readframes(self.chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()

    def reaudio(self): 
        self.chunk = 1024
        self.path = 'abc.wav'

        with wave.open(self.path, 'rb') as f:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                            channels = f.getnchannels(),
                            rate = f.getframerate(),
                            output= True)

            data = f.readframes(self.chunk)
            while data:
                stream.write(data)
                data = f.readframes(self.chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()
    

   

    def re_start(self):
        time.sleep(0.5)
        x = tiThread(self)
        x2 = reThread(self)
        x2.start()
        x.start()
        for i in reversed(range(self.graph.count())): 
            self.graph.itemAt(i).widget().deleteLater()
        self.textEdit_2.clear()
        self.lcdNumber.display(0)
        self.label_2.setText('녹음중...')     
        
        self.pushButton_4.setDisabled(True)
        self.pushButton_5.setDisabled(False)
        self.pushButton_6.setDisabled(True)

    def ans_img(self):
        self.chunk = 1024
        
        path=''
        try:
            path = self.listWidget.selectedItems()

        except:
            self.label_2.setText('다시 선택하십시오...')
        if path !='':
            qPixmapVar = QPixmap()
            qPixmapVar.load(path[0].text()+'.jpg')
            self.label_3.setScaledContents(True)
            self.label_3.setPixmap(qPixmapVar)
            with wave.open(path[0].text()+'.wav', 'rb') as f:
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                            channels = f.getnchannels(),
                            rate = f.getframerate(),
                            output= True)

                data = f.readframes(self.chunk)
                while data:
                    stream.write(data)
                    data = f.readframes(self.chunk)
                stream.stop_stream()
                stream.close()
                p.terminate()
    
        
   
    


if __name__ == "__main__" :
 
    app = QApplication(sys.argv) 

  
    myWindow = WindowClass() 

    myWindow.show()
    app.exec_()




