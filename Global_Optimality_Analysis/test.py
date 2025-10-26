import numpy as np
import torch
from MLP_Training.training import DataGenerator, GenerateDataset, train_MLP, CalB_MX, CalBr
from MyCase.PrepareData import prepare_data, calculate_PTDF, calculate_Bkr, CalBMX
from MyCase.MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88




MyCase = Data4Case39()
Name = MyCase.name
Case = MyCase.Case
Exit_Brh = MyCase.Exit_Brh
Exit_Gen = MyCase.Exit_Gen

T = 24
SCR0 = 3
MaxGenNum = 3
r = 0.05
n = 20
D = 365
DailyLoadRate = np.array([
    1.4794, 1.3847, 1.2558, 1.1566, 1.0889, 1.0464, 1.0217, 1.0033, 
    0.9796, 0.9798, 0.9993, 1.1088, 1.3298, 1.4862, 1.5312, 1.5535, 
    1.5766, 1.5292, 1.5056, 1.4927, 1.5508, 1.6042, 1.5598, 1.5300
]) * 1.3

data = prepare_data(Case, T, MaxGenNum, r, n, D, SCR0, DailyLoadRate, Exit_Brh, Exit_Gen, Name)

Dataset = GenerateDataset(data, length=1000, batch_size=32)
Data = next(iter(Dataset))
x1, x2, y, gSCR = Data
print("Generator Status Batch Shape:", x1.shape)
print("Branch Status Batch Shape:", x2.shape)
print("Bkr Batch Shape:", y.shape)
print("gSCR Batch Shape:", gSCR.shape)

model, params = train_MLP(data, epochs=100, batch_size=32, learning_rate=0.001, train_size=10000)
w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, w_fc4, b_fc4, w_fc5, b_fc5 = params
np.savez(r"d:\SynologyDrive\Code_Python\TEP_IEEE39\Global_Optimality_Analysis-V(1)\model_params.npz",
         w_fc1=w_fc1, b_fc1=b_fc1,
         w_fc2=w_fc2, b_fc2=b_fc2,
         w_fc3=w_fc3, b_fc3=b_fc3,
         w_fc4=w_fc4, b_fc4=b_fc4,
         w_fc5=w_fc5, b_fc5=b_fc5)
print("Saved params to model_params.npz")

d = np.load(r"d:\SynologyDrive\Code_Python\TEP_IEEE39\Global_Optimality_Analysis-V(1)\model_params.npz")
w_fc1 = d['w_fc1']; b_fc1 = d['b_fc1']
w_fc2 = d['w_fc2']; b_fc2 = d['b_fc2']
w_fc3 = d['w_fc3']; b_fc3 = d['b_fc3']
w_fc4 = d['w_fc4']; b_fc4 = d['b_fc4']
w_fc5 = d['w_fc5']; b_fc5 = d['b_fc5']
print(w_fc1.shape)
print(b_fc1.shape)
print(w_fc2.shape)
print(b_fc2.shape)
print(w_fc3.shape)
print(b_fc3.shape)
print(w_fc4.shape)
print(b_fc4.shape)
print(w_fc5.shape)
print(b_fc5.shape)
print('*****************************************************************************')
GenStatus = np.ones((data['NGen'], 1))
BrhStatus = np.ones((data['NBrh'], 1))
HVDC_Cap = np.ones((data['NHVDC'], 1)) * 7
a = w_fc1 @ GenStatus + b_fc1
b = w_fc2 @ BrhStatus + b_fc2
c = w_fc3 @ HVDC_Cap + b_fc3
d = w_fc4 @ (a + b + c) + b_fc4
e = w_fc5 @ d + b_fc5
print(a.shape)
print(b.shape)
print(c.shape)
print(d.shape)
print(e.shape)
print('*****************************************************************************')
f = w_fc5 @ ((w_fc4 @ ((w_fc1 @ GenStatus + b_fc1) + (w_fc2 @ BrhStatus + b_fc2) + w_fc3 @ HVDC_Cap + b_fc3) + b_fc4)) + b_fc5
print(f)
print('*****************************************************************************')
NBus = data['NBus']
NBrh = data['NBrh']
HVDC_Bus = data['BusHVDC']
Load_Bus = data['BusLoad']
B_Brh_ini = data['B_Brh_ini']
B_Gen_ini = data['B_Gen_ini']
BusAdjList = data['BusAdjList']
HVDC_Cap_Min = data['HVDC_Cap_MIN']
HVDC_Cap_Max = data['HVDC_Cap_MAX']
MX_NBus_NGen = data['MX_NBus_NGen']
MX_NBus_NBus_R = data['MX_NBus_NBus_R']
MX_NBus_NBus_C = data['MX_NBus_NBus_C']
MXH_NBus_NHVDC = data['MXH_NBus_NHVDC']
MXH_NHVDC_NBus = data['MXH_NHVDC_NBus']

GenStatus = data['Exit_Gen'].reshape(-1, 1)
BrhStatus = data['Exit_Brh'].reshape(-1, 1)

gSCR1_List = []
gSCR2_List = []
gSCR3_List = []
for i in range(400):
    HVDC_Cap = np.ones((data['NHVDC'], 1)) * 2 +  20 * i / 1000
    B_MX = CalB_MX(BusAdjList, MX_NBus_NGen, BrhStatus, GenStatus, B_Brh_ini, B_Gen_ini, NBus, NBrh)
    IfINV, Br = CalBr(B_MX, HVDC_Bus, Load_Bus)
    gSCR1 = np.min(np.linalg.eigvals(np.linalg.inv(np.diag(HVDC_Cap.flatten())) @ Br))
    gSCR1_List.append(gSCR1)
    BkrList, Gen_B_MX, Bus_B_MX = calculate_Bkr(Brh_Status=BrhStatus, UC_Status=GenStatus, 
                                                B_Brh_ini=B_Brh_ini, B_Gen_ini=B_Gen_ini, BusAdjList=BusAdjList, 
                                                NBus=NBus, NBrh=NBrh, MX_NBus_NGen=MX_NBus_NGen, 
                                                MX_NBus_NBus_R=MX_NBus_NBus_R, MX_NBus_NBus_C=MX_NBus_NBus_C)
    gSCR2 = np.min(np.linalg.eigvals(BkrList[0] - SCR0 * (MXH_NBus_NHVDC @ np.diag(HVDC_Cap.flatten()) @ MXH_NHVDC_NBus)))
    gSCR2_List.append(gSCR2)

    gSCR3 = np.min(w_fc5 @ ((w_fc4 @ ((w_fc1 @ GenStatus + b_fc1) + (w_fc2 @ BrhStatus + b_fc2) + w_fc3 @ HVDC_Cap + b_fc3) + b_fc4)) + b_fc5)
    gSCR3_List.append(gSCR3)
    print(f"Iteration {i+1}: gSCR1: {np.real(gSCR1):.6f}, gSCR2: {np.real(gSCR2):.6f}, gSCR3: {np.real(gSCR3):.6f}")








def if_pos2neg(gSCR_List, value=0):
    for i in range(1, len(gSCR_List)):
        if gSCR_List[i-1] > value and gSCR_List[i] <= value:
            return i
    return i

idx1 = if_pos2neg(gSCR1_List, value=3)
idx2 = if_pos2neg(gSCR2_List, value=0)
idx3 = if_pos2neg(gSCR3_List, value=3)
print(f"gSCR1 crosses 3 at index: {idx1}, HVDC Cap: {2 + 20 * idx1 / 1000:.4f}")
print(f"gSCR2 crosses 0 at index: {idx2}, HVDC Cap: {2 + 20 * idx2 / 1000:.4f}")
print(f"gSCR3 crosses 3 at index: {idx3}, HVDC Cap: {2 + 20 * idx3 / 1000:.4f}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


config = {
    'font.family': 'Times New Roman',
    'font.size': 18,
    'font.weight': 'normal',
    'mathtext.fontset': 'stix',
}
rcParams.update(config)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['axes.unicode_minus'] = False
x1 = np.array([0, idx1, idx2, idx3, 400])
x2 = ['0', f'{idx1}', f'{idx2}', f'{idx3}', '400']
y1 = np.array([10, 3, 0, -10])
y2 = ['10', '3', '0', '-10']
plt.figure(figsize=(12, 4))
plt.plot(np.array(gSCR1_List), label='gSCR 1', linestyle='solid')
plt.plot(np.array(gSCR2_List), label='gSCR 2', linestyle='dashed')
plt.plot(np.array(gSCR3_List), label='gSCR 3', linestyle='dotted')
plt.xticks(x1, x2)
plt.yticks(y1, y2)
plt.xlabel('Steps')
plt.ylabel('gSCR Value')
plt.grid(False)
ax = plt.gca()
ax.axhline(3, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.axhline(0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(idx1, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(idx2, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax.axvline(idx3, color='gray', linestyle='--', alpha=0.7, linewidth=1)
plt.legend()
plt.savefig(fname=f'F.svg', dpi=600, bbox_inches='tight')
plt.show()

