import time
from xml.parsers.expat import model
import cvxpy as cp
import numpy as np
import sys
sys.path.append(r'd:\SynologyDrive\Code_Python\TEP_IEEE39\Global_Optimality_Analysis-V(1)')
from MyCase.MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88
from MyCase.PrepareData import prepare_data, calculate_PTDF, calculate_Bkr

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



class MLP4gSCRN(torch.nn.Module):
    def __init__(self, input_size_1, input_size_2, input_size_3, hidden_size, output_size):
        super(MLP4gSCRN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size_1, hidden_size)
        self.fc2 = torch.nn.Linear(input_size_2, hidden_size)
        self.fc3 = torch.nn.Linear(input_size_3, hidden_size)
        self.fc4 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc5 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, GenStatus, BrhStatus, HVDC_Cap):
        x1 = self.fc1(GenStatus)
        x2 = self.fc2(BrhStatus)
        x3 = self.fc3(HVDC_Cap)
        x = x1 + x2 + x3
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
def CalB_MX(BusAdjList, MX_NBus_NGen, BrhStatus, GenStatus, B_Brh_ini, B_Gen_ini, NBus, NBrh):
    Gen_B_MX = np.diag((MX_NBus_NGen @ (GenStatus * B_Gen_ini)).flatten())
    Brh_B_MX = np.zeros([NBus, NBus])
    for i in range(NBrh):
        f_node = BusAdjList[i, 0] - 1
        t_node = BusAdjList[i, 1] - 1
        Brh_B_MX[f_node, t_node] = - B_Brh_ini[i] * BrhStatus[i, 0]
        Brh_B_MX[t_node, f_node] = - B_Brh_ini[i] * BrhStatus[i, 0]
    for i in range(NBus):
        Brh_B_MX[i, i] = - Brh_B_MX[i, :] @ np.ones(shape=[NBus, 1])
    B_MX = Brh_B_MX + Gen_B_MX
    return B_MX
   

def CalBr(B_MX, HVDC_Bus, Load_Bus):
    B_HH = np.zeros(shape=[len(HVDC_Bus), len(HVDC_Bus)])
    B_HL = np.zeros(shape=[len(HVDC_Bus), len(Load_Bus)])
    B_LL = np.zeros(shape=[len(Load_Bus), len(Load_Bus)])
    for i in range(len(HVDC_Bus)):
        for j in range(len(HVDC_Bus)):
            B_HH[i, j] = B_MX[HVDC_Bus[i] - 1, HVDC_Bus[j] - 1]

    for i in range(len(HVDC_Bus)):
        for j in range(len(Load_Bus)):
            B_HL[i, j] = B_MX[HVDC_Bus[i] - 1, Load_Bus[j] - 1]

    for i in range(len(Load_Bus)):
        for j in range(len(Load_Bus)):
            B_LL[i, j] = B_MX[Load_Bus[i] - 1, Load_Bus[j] - 1]
    try:
        Br = B_HH - B_HL @ (np.linalg.inv(B_LL)) @ B_HL.T
        return 1, Br
    except np.linalg.LinAlgError:
        return 0, None


def DataGenerator(data, length):
    NBrh = data['NBrh']
    NBus = data['NBus']
    NGen = data['NGen']
    ExitBrh = data['Exit_Brh']
    ExitGen = data['Exit_Gen']
    HVDC_Bus = data['BusHVDC']
    Load_Bus = data['BusLoad']
    B_Brh_ini = data['B_Brh_ini']
    B_Gen_ini = data['B_Gen_ini']
    BusAdjList = data['BusAdjList']
    MX_NBus_NGen = data['MX_NBus_NGen']
    HVDC_Cap_MAX = data['HVDC_Cap_MAX'] 
    HVDC_Cap_MIN = data['HVDC_Cap_MIN'] 
    for i in range(length):
        GenStatus = np.zeros((NGen))
        BrhStatus = np.zeros((NBrh))
        HVDC_Cap = np.zeros((len(HVDC_Bus)))
        for b in  range(NBrh):
            if ExitBrh[b] == 0:
                ExitBrh[b] = np.random.choice([0, 1, 2, 3])
            elif ExitBrh[b] == 1:
                ExitBrh[b] = np.random.choice([1, 2, 3])
            elif ExitBrh[b] == 2:
                ExitBrh[b] = np.random.choice([2, 3])
            elif ExitBrh[b] == 3:
                ExitBrh[b] = 3
        for g in range(NGen):
            if ExitGen[g] == 0:
                ExitGen[g] = np.random.choice([0, 1])
            elif ExitGen[g] == 1:
                ExitGen[g] = 1
        for h in range(len(HVDC_Bus)):
            HVDC_Cap[h] = HVDC_Cap_MIN[h] + (HVDC_Cap_MAX[h] - HVDC_Cap_MIN[h]) * np.random.rand()
        GenStatus = ExitGen.reshape(-1, 1)
        BrhStatus = ExitBrh.reshape(-1, 1)
        B_MX = CalB_MX(BusAdjList, MX_NBus_NGen, BrhStatus, GenStatus, B_Brh_ini, B_Gen_ini, NBus, NBrh)
        IfINV, Br = CalBr(B_MX, HVDC_Bus, Load_Bus)
        gSCR = np.linalg.eigvals(np.linalg.inv(np.diag(HVDC_Cap)) @ Br)
        # print(gSCR)
        if IfINV == 1:
            yield GenStatus, BrhStatus, HVDC_Cap, gSCR
        else:
            continue


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def GenerateDataset(data, length, batch_size):
    Dataset = []
    for gen, brh, HVDC_Cap, gSCR in DataGenerator(data=data, length=length):
        x1 = torch.FloatTensor(gen)
        x2 = torch.FloatTensor(brh)
        x3 = torch.FloatTensor(HVDC_Cap)
        y = torch.FloatTensor(gSCR)
        Dataset.append((x1, x2, x3, y))
    Dataset = MyDataset(Dataset)
    Dataset = DataLoader(dataset=Dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    return Dataset


def train_MLP(data, hidden_size=128, learning_rate=0.001, epochs=50, batch_size=64, train_size=10000, device='cuda'):
    NGen = data['NGen']
    NBrh = data['NBrh']
    NHVDC = data['NHVDC']
    output_size = NHVDC

    model = MLP4gSCRN(input_size_1=NGen, input_size_2=NBrh, input_size_3=NHVDC, 
                      hidden_size=hidden_size, output_size=output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss().to(device)

    Dataset = GenerateDataset(data, length=train_size, batch_size=batch_size)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (x1, x2, x3, y) in enumerate(Dataset):
            optimizer.zero_grad()
            x1 = x1.view(x1.size(0), -1).float().to(device)
            x2 = x2.view(x2.size(0), -1).float().to(device)
            x3 = x3.view(x3.size(0), -1).float().to(device)
            y = y.view(y.size(0), -1).float().to(device)
            outputs = model(x1, x2, x3)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(Dataset):.12f}')
    w_fc1 = model.fc1.weight.detach().cpu().numpy()   
    b_fc1 = model.fc1.bias.detach().cpu().numpy().reshape(-1, 1)
    w_fc2 = model.fc2.weight.detach().cpu().numpy()
    b_fc2 = model.fc2.bias.detach().cpu().numpy().reshape(-1, 1)
    w_fc3 = model.fc3.weight.detach().cpu().numpy()
    b_fc3 = model.fc3.bias.detach().cpu().numpy().reshape(-1, 1)
    w_fc4 = model.fc4.weight.detach().cpu().numpy()
    b_fc4 = model.fc4.bias.detach().cpu().numpy().reshape(-1, 1)
    w_fc5 = model.fc5.weight.detach().cpu().numpy()
    b_fc5 = model.fc5.bias.detach().cpu().numpy().reshape(-1, 1)
    return model, [w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, w_fc4, b_fc4, w_fc5, b_fc5]




    