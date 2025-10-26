import math
from typing import Any
import numpy as np
from numpy import ndarray, dtype, floating
import time
import itertools

def CalBMX(BusAdjList, Brh_Status, B_Brh_ini, NBus, NBrh):
    B = np.zeros([NBus, NBus])
    for i in range(NBrh):
        f_node = BusAdjList[i, 0] - 1
        t_node = BusAdjList[i, 1] - 1
        B[f_node, t_node] = - B_Brh_ini[i] * Brh_Status[i, 0]
        B[t_node, f_node] = - B_Brh_ini[i] * Brh_Status[i, 0]
    for i in range(NBus):
        B[i, i] = - B[i, :] @ np.ones(shape=[NBus, 1])
    return B

def calculate_PTDF(BusAdjList, Brh_Status, B_Brh_ini, NBus, NBrh):
    B_Brh = CalBMX(BusAdjList=BusAdjList, Brh_Status=Brh_Status, B_Brh_ini=B_Brh_ini, NBus=NBus, NBrh=NBrh)
    B_Brh_ = np.linalg.inv(B_Brh[1:, 1:])
    B_ = np.zeros((NBus, NBus))
    B_[1:, 1:] = B_Brh_
    B2 = np.zeros(shape=[NBrh, NBus])
    for i in range(NBrh):
        f_node = BusAdjList[i, 0] - 1
        t_node = BusAdjList[i, 1] - 1
        B2[i, f_node] = -B_Brh_ini[i] * Brh_Status[i, 0]
        B2[i, t_node] = B_Brh_ini[i] * Brh_Status[i, 0]
    PTDF = B2 @ B_
    return PTDF


def calculate_Bkr(Brh_Status, UC_Status, B_Brh_ini, B_Gen_ini, BusAdjList, NBus, NBrh, MX_NBus_NGen, MX_NBus_NBus_R, MX_NBus_NBus_C):
    Bus_B_MX = CalBMX(BusAdjList=BusAdjList, Brh_Status=Brh_Status, B_Brh_ini=B_Brh_ini, NBus=NBus, NBrh=NBrh)
    BkrList = []
    for t in range(UC_Status.shape[1]):
        uc_status = UC_Status[:, t].reshape(-1, 1)
        Gen_B_MX = np.diag((MX_NBus_NGen @ (uc_status * B_Gen_ini)).flatten())
        Bkr = MX_NBus_NBus_R @ (Bus_B_MX + Gen_B_MX) @ MX_NBus_NBus_C
        BkrList.append(Bkr)
    return BkrList, Gen_B_MX, Bus_B_MX


def prepare_data(Case, T, MaxGenNum, r, n, D, SCR0, DailyLoadRate, Exit_Brh, Exit_Gen, Name=None):
    crf = (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    baseMVA = Case['baseMVA']
    NBus = Case['bus'].shape[0]
    NBrh = Case['brh'].shape[0]
    NGen = Case['gen'].shape[0]
    NHVDC = Case['HVDC'].shape[0]


    BusAll = Case['bus'][:, 0].astype(int).tolist()
    BusGen = list(dict.fromkeys(Case['gen'][:, 0].astype(int).tolist()))
    BusHVDC = Case['HVDC'][:, 0].astype(int).tolist()
    BusLoad = [y for y in BusAll if y not in BusHVDC]
    NumGenType = np.max(Case['gen'][:, -1]).astype(int)

    MXH_NBus_NHVDC = np.zeros(shape=(NBus, NHVDC), dtype=int)
    MXH_NHVDC_NBus = np.zeros(shape=(NHVDC, NBus), dtype=int)
    MXH_NBus_NHVDC[0: NHVDC, 0: NHVDC] = np.eye(NHVDC, dtype=int)
    MXH_NHVDC_NBus[0: NHVDC, 0: NHVDC] = np.eye(NHVDC, dtype=int)

    MX_NBus_NBus_R = np.zeros(shape=(NBus, NBus), dtype=int)
    MX_NBus_NBus_C = np.zeros(shape=(NBus, NBus), dtype=int)
    MX_NBus_NBus_R[np.array(BusAll, dtype=int) - 1, np.array(BusHVDC + BusLoad, dtype=int) - 1] = 1
    MX_NBus_NBus_C[np.array(BusHVDC + BusLoad, dtype=int) - 1, np.array(BusAll, dtype=int) - 1] = 1

    GenDiag = np.diag([-1] * T)
    for i in range(T - 1):
        j = i + 1
        GenDiag[j, i] = 1
    GenDiag = GenDiag[:, 0:T - 1]

    Add_Gen_Cost = Case['gen'][:, 10].reshape(-1, 1) * 1e6 * crf 
    Add_Brh_Cost = Case['brh'][:, 6].reshape(-1, 1) * 1e4 * crf 

    P_D = DailyLoadRate * Case['bus'][:, 2].reshape(-1, 1) / baseMVA
    P_LS_cost = 1000 * np.ones(shape=(NBus, 1)) @ DailyLoadRate.reshape(1, -1)


    B_Gen_ini = 1 / Case['gen'][:, -2].reshape(-1, 1)
    B_Brh_ini = 1 / Case['brh'][:, 3].reshape(-1, 1)

    
    MX_NBus_NHVDC = np.zeros(shape=(NBus, NHVDC))
    for i in range(NHVDC):
        bus = int(Case['HVDC'][i, 0] - 1)
        MX_NBus_NHVDC[bus, i] = 1


    BusAdjList = Case['brh'][:, 0: 2].astype(int)

    AdjMX = np.zeros(shape=[NBrh, NBus], dtype=int)
    for i in range(NBrh):
        f_node = BusAdjList[i, 0] - 1
        t_node = BusAdjList[i, 1] - 1
        AdjMX[i, f_node] = 1
        AdjMX[i, t_node] = -1
    AdjMX = AdjMX.T


    MX_NBus_NGen = np.zeros(shape=(NBus, NGen), dtype=int)
    for i in range(len(BusGen)):
        BusIdx = BusGen[i] - 1
        NGenPerBus = NumGenType * MaxGenNum
        MX_NBus_NGen[BusIdx, i * NGenPerBus: (i + 1) * NGenPerBus] = 1


    Gen_P_cost = Case['gen'][:, -3].reshape(-1, 1)
    Gen_PMAX_ini = Case['gen'][:, 8].reshape(-1, 1) / baseMVA
    Gen_PMIN_ini = Case['gen'][:, 9].reshape(-1, 1) / baseMVA
    Gen_RU = Gen_PMAX_ini * 0.4
    Gen_RD = Gen_PMAX_ini * -0.4

    Gen_SU_cost = Case['gen'][:, -6].reshape(-1, 1)
    Gen_SD_cost = Case['gen'][:, -5].reshape(-1, 1)
    Gen_SU_Time = Case['gen'][:, -7].reshape(-1, 1).astype(int)
    Gen_SD_Time = Case['gen'][:, -8].reshape(-1, 1).astype(int)


    HVDC_Cap_MAX = np.array(Case['HVDC'][:, 8]).reshape(-1, 1) / baseMVA
    HVDC_Cap_MIN = np.array(Case['HVDC'][:, 9]).reshape(-1, 1) / baseMVA
    HVDC_Cap_cost = 2000 * np.array(Case['HVDC'][:, -3]).reshape(-1, 1)
    HVDC_P_cost = np.array(Case['HVDC'][:, -3]).reshape(-1, 1)

    Brh_MAX_ini = Case['brh'][:, 5].reshape(-1, 1) / baseMVA

    BusAdjSet = set()
    for row in BusAdjList:
        f, t = row
        BusAdjSet.add((min(f, t), max(f, t)))

    all_possible_pairs = set(itertools.combinations(range(1, NBus + 1), 2))
    BusAdj_complement_pairs = all_possible_pairs - BusAdjSet

    MaxBrh = 3

    return {
        'crf': crf, 'baseMVA': baseMVA, 'NBus': NBus, 'NBrh': NBrh, 'NGen': NGen, 'NHVDC': NHVDC,
        'BusAll': BusAll, 'BusGen': BusGen, 'BusHVDC': BusHVDC, 'BusLoad': BusLoad, 'NumGenType': NumGenType,
        'MXH_NBus_NHVDC': MXH_NBus_NHVDC, 'MXH_NHVDC_NBus': MXH_NHVDC_NBus,
        'MX_NBus_NBus_R': MX_NBus_NBus_R, 'MX_NBus_NBus_C': MX_NBus_NBus_C,
        'Gen_SU_cost': Gen_SU_cost, 'Gen_SD_cost': Gen_SD_cost, 'Gen_SU_Time': Gen_SU_Time, 'Gen_SD_Time': Gen_SD_Time,
        'GenDiag': GenDiag, 'Add_Gen_Cost': Add_Gen_Cost, 'Add_Brh_Cost': Add_Brh_Cost,
        'P_D': P_D, 'P_LS_cost': P_LS_cost, 'B_Gen_ini': B_Gen_ini, 'B_Brh_ini': B_Brh_ini,
        'MX_NBus_NHVDC': MX_NBus_NHVDC, 'BusAdjList': BusAdjList, 'AdjMX': AdjMX, 'MX_NBus_NGen': MX_NBus_NGen,
        'Gen_P_cost': Gen_P_cost, 'Gen_PMAX_ini': Gen_PMAX_ini, 'Gen_PMIN_ini': Gen_PMIN_ini,
        'Gen_RU': Gen_RU, 'Gen_RD': Gen_RD, 'HVDC_Cap_MAX': HVDC_Cap_MAX, 'HVDC_Cap_MIN': HVDC_Cap_MIN,
        'HVDC_Cap_cost': HVDC_Cap_cost, 'HVDC_P_cost': HVDC_P_cost, 'Brh_MAX_ini': Brh_MAX_ini,
        'BusAdj_complement_pairs': BusAdj_complement_pairs, 'Exit_Brh': Exit_Brh, 'Exit_Gen': Exit_Gen, 'D': D, 'SCR0': SCR0, 'T': T, 'Name': Name
        }

def CalBkr(Bus_B_MX, HVDC_Bus, Load_Bus):
    B_HH = np.zeros(shape=[len(HVDC_Bus), len(HVDC_Bus)])
    B_HL = np.zeros(shape=[len(HVDC_Bus), len(Load_Bus)])
    B_LL = np.zeros(shape=[len(Load_Bus), len(Load_Bus)])
    for i in range(len(HVDC_Bus)):
        for j in range(len(HVDC_Bus)):
            B_HH[i, j] = Bus_B_MX[HVDC_Bus[i] - 1, HVDC_Bus[j] - 1]

    for i in range(len(HVDC_Bus)):
        for j in range(len(Load_Bus)):
            B_HL[i, j] = Bus_B_MX[HVDC_Bus[i] - 1, Load_Bus[j] - 1]

    for i in range(len(Load_Bus)):
        for j in range(len(Load_Bus)):
            B_LL[i, j] = Bus_B_MX[Load_Bus[i] - 1, Load_Bus[j] - 1]
    Bkr = B_HH - B_HL @ (np.linalg.inv(B_LL)) @ B_HL.T
    return Bkr

if __name__ == "__main__":
    import sys
    sys.path.append(r'd:\SynologyDrive\Code_Python\TEP_IEEE39')
    from MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88
    MyCase = Data4Case39()
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

    data = prepare_data(Case, T, MaxGenNum, r, n, D, SCR0, DailyLoadRate, Exit_Brh, Exit_Gen)
    BusAdjList, Brh_Status, B_Brh_ini, NBus, NBrh = data['BusAdjList'], np.ones((data['NBrh'], 1)), data['B_Brh_ini'], data['NBus'], data['NBrh']
    Bus_B_MX = CalBMX(BusAdjList=BusAdjList, Brh_Status=Brh_Status, B_Brh_ini=B_Brh_ini, NBus=NBus, NBrh=NBrh)
    Gen_B_MX = np.diag((data['MX_NBus_NGen'] @ (data['Exit_Gen'] * data['B_Gen_ini'])).flatten())
    HVDC_Bus, Load_Bus = data['BusHVDC'], data['BusLoad']
    Bkr = CalBkr(Bus_B_MX + Gen_B_MX, HVDC_Bus, Load_Bus)
    print(Bkr)


    
