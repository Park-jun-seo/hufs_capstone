import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

import os
import matplotlib
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available() # GPU를 사용가능하면 True, 아니라면 False를 리턴
device = torch.device("cuda" if USE_CUDA else "cpu") # GPU 사용 가능하면 사용하고 아니면 CPU 사용
print("다음 기기로 학습합니다:", device)

ch = {'시':0,'청':3,'의':6,'창':1,'살1':9,'은':5,'외':2,'창':10,'살':8,'이':4,'다':7,'*':11,'-':12}
num = {0:'시',3:'청',6:'의',1:'창',9:'살1',5:'은',2:'외',10:'창',8:'살',4:'이',7:'다',11:'*',12:'-'}

def tr(s):
  p=[]
  pp=[]
  for j in  s:
    p=[]
    for i in  j[0]:
      if i in ch:
        p.append(ch[i])
      else:
        p.append(11)
    pp.append(torch.tensor(p))
  return pp

def va_mfcc(s):
  import librosa
  import librosa.display
  import numpy as np
  import random
  frame_length = 0.1
  frame_stride = 0.1  
  pp=[]

  y, sr = librosa.load(s, sr=16000)
  input_nfft = int(round(sr*frame_length))
  input_stride = int(round(sr*frame_stride))
  MFCCs = librosa.feature.mfcc(y, sr, n_fft=input_nfft , hop_length=input_stride , n_mfcc=128)
  pp.append(torch.tensor(MFCCs))
  return pp

def timesmh(ti):
  h = 0
  m = 0
  s = ti*0.01

  m = s//60
  s %=60 
  h = m//60
  m %=60
  da = "{0:d} hour {1:d} min {2:0.2f} sec".format(int(h),int(m),s)
  return da 
  
def tr_num(s):
  p=[]
  s=s.tolist()
  for j in  s:
    p.append(num[j])
  return p

def decode(a):
  ss=tr_num(a)
  return ss

def tw(arr):
  sent=[]
  pre=''
  for i in arr:
    if pre=='':
      pre=i
      sent.append(i)
      continue
    if pre==i and i!='*' :
      continue
    elif pre==i and i=='*' :
      sent.append(i)
      continue
    elif pre!=i:
      pre=i
      sent.append(i)
  while '-' in sent:
      sent.remove('-')
  return sent

#pp=['시', '*', '의', '*', '*', '은', '외', '창', '살', '이', '이', '*', '이', '다']

#print(tw(tw(pp)))

class GRU_seq(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers) : 
        super(GRU_seq, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.con1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bat2d1 = nn.BatchNorm2d(32)
        self.hta1 =  nn.Hardtanh(0, 20, inplace=True)
        
        
        self.gru1 = nn.GRU(input_size, hidden_size, num_layers,bidirectional = True,batch_first=True)
        self.gru2 = nn.GRU(hidden_size*2, hidden_size, num_layers,bidirectional = True,batch_first=True)

        self.bat_r = nn.BatchNorm1d(hidden_size*2)
  
    

        self.cone = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bat2de = nn.BatchNorm2d(1)
        self.htae =  nn.Hardtanh(0, 20, inplace=True)
 

        self.layer_1 = nn.Linear(hidden_size*2, num_classes)

        self.bat_r = nn.BatchNorm1d(hidden_size*2)

        self.bat1 = nn.BatchNorm1d(num_classes)

    def forward(self,x):
        x = self.con1(x)
        x = self.bat2d1(x)
        x = self.hta1(x)

        x = x.squeeze(0)

        h_0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) #Hidden State
    
        output,hn = self.gru1(x,h_0)
        output,hn = self.gru2(output,h_0)
    


        
        output = self.cone(output.unsqueeze(0))
        output = self.bat2de(output)
        output = self.htae(output)
       
        
        sen = []
        for i in range(output[0].size(0)):
          #tt = self.bat1(output[0][i]) #pre-processing for first layer       
          tt = self.layer_1(output[0][i]) # first layer
          tt = self.bat1(tt)
          sen.append(tt)
        output = torch.stack(sen).unsqueeze(0)
      
        return output

def pre_S(mod):
  net1 = GRU_seq(13,128,64,1).to(device) 

  learning_rate = 0.001
  optimizer = torch.optim.Adam(net1.parameters(), lr=learning_rate)

  ###############################모델 불러오기#######################################
  checkpoint = torch.load(mod)

  net1.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoc_h = checkpoint['epoch']
  loss = checkpoint['loss']

  net1.eval()
  #print(epoc_h)
  ur = "abc.wav"


  tex = va_mfcc(ur)[0].to(device)

  pre_label = net1(tex.transpose(0, 1).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
  print(tw(decode(pre_label.argmax(dim=1))))
  
  return tw(decode(pre_label.argmax(dim=1)))
  #print(tw(decode(pre_label.argmax(dim=1))))


