import time
import julia
import cvxpy as cp
import numpy as np
import sys
import os
sys.path.append(r'd:\SynologyDrive\Code_Python\TEP_IEEE39')
from MyCase.MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88
from MyCase.PrepareData import prepare_data, calculate_PTDF, calculate_Bkr
from MLP_Training.training import DataGenerator, GenerateDataset, train_MLP, CalB_MX, CalBr

def to_numpy(x):
    import numpy as _np
    try:
        arr = _np.array(x)
        return arr.astype(float)
    except Exception:
        try:
            return _np.array([float(v) for v in x])
        except Exception:
            return _np.array([[float(v) for v in row] for row in x])
        

if __name__ == "__main__":
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
    ]) * 1.7


    data = prepare_data(Case, T, MaxGenNum, r, n, D, SCR0, DailyLoadRate, Exit_Brh, Exit_Gen)
    Exit_Brh = data['Exit_Brh']
    Exit_Gen = data['Exit_Gen']
    B_Brh_ini = data['B_Brh_ini']
    B_Gen_ini = data['B_Gen_ini']
    BusAdjList = data['BusAdjList']
    BusAdj_complement_pairs = data['BusAdj_complement_pairs']
    MX_NBus_NGen = data['MX_NBus_NGen']
    MX_NBus_NBus_R = data['MX_NBus_NBus_R']
    MX_NBus_NBus_C = data['MX_NBus_NBus_C']
    Add_Brh_Cost = data['Add_Brh_Cost']
    Add_Gen_Cost = data['Add_Gen_Cost']
    Gen_PMAX_ini = data['Gen_PMAX_ini']
    Gen_PMIN_ini = data['Gen_PMIN_ini']
    Brh_MAX_ini = data['Brh_MAX_ini']
    NBus = data['NBus']
    NBrh = data['NBrh']
    NGen = data['NGen']
    NHVDC = data['NHVDC']
    HVDC_Bus = data['BusHVDC']
    Load_Bus = data['BusLoad']
    T = data['T']
    D = data['D']
    P_D = data['P_D']
    SCR0 = data['SCR0']
    baseMVA = data['baseMVA']
    MX_NBus_NHVDC = data['MX_NBus_NHVDC']
    MXH_NBus_NHVDC = data['MXH_NBus_NHVDC']
    MXH_NHVDC_NBus = data['MXH_NHVDC_NBus']
    P_LS_cost = data['P_LS_cost']
    Gen_P_cost = data['Gen_P_cost']
    HVDC_P_cost = data['HVDC_P_cost']
    HVDC_Cap_cost = data['HVDC_Cap_cost']
    Gen_RU = data['Gen_RU']
    Gen_RD = data['Gen_RD']
    GenDiag = data['GenDiag']
    HVDC_Cap_MAX = data['HVDC_Cap_MAX']
    Gen_SU_cost = data['Gen_SU_cost']
    Gen_SD_cost = data['Gen_SD_cost']
    Gen_SU_Time = data['Gen_SU_Time']
    Gen_SD_Time = data['Gen_SD_Time']
    ##########################################################################################################################
    path = "model_params.npz"
    if os.path.exists(path):
        d = np.load(path)
        w_fc1 = d['w_fc1']; b_fc1 = d['b_fc1']
        w_fc2 = d['w_fc2']; b_fc2 = d['b_fc2']
        w_fc3 = d['w_fc3']; b_fc3 = d['b_fc3']
        w_fc4 = d['w_fc4']; b_fc4 = d['b_fc4']
        w_fc5 = d['w_fc5']; b_fc5 = d['b_fc5']
        print("Loaded params from model_params.npz")
    else:
        model, params = train_MLP(data, epochs=100, batch_size=32, learning_rate=0.001, train_size=10000, hidden_size=256)
        w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, w_fc4, b_fc4, w_fc5, b_fc5 = params
        np.savez(path,
                w_fc1=w_fc1, b_fc1=b_fc1,
                w_fc2=w_fc2, b_fc2=b_fc2,
                w_fc3=w_fc3, b_fc3=b_fc3,
                w_fc4=w_fc4, b_fc4=b_fc4,
                w_fc5=w_fc5, b_fc5=b_fc5)
        print("Saved params to model_params.npz")

    data = [Exit_Brh, Exit_Gen, B_Brh_ini, B_Gen_ini, BusAdjList, BusAdj_complement_pairs, MX_NBus_NGen,
            MX_NBus_NBus_R, MX_NBus_NBus_C, Add_Brh_Cost, Add_Gen_Cost, Gen_PMAX_ini, Gen_PMIN_ini,
            Brh_MAX_ini, NBus, NBrh, NGen, NHVDC, T, D, P_D, SCR0, baseMVA, MX_NBus_NHVDC,
            MXH_NBus_NHVDC, MXH_NHVDC_NBus, P_LS_cost, Gen_P_cost, HVDC_P_cost, HVDC_Cap_cost,
            Gen_RU, Gen_RD, GenDiag, HVDC_Cap_MAX, Gen_SU_cost, Gen_SD_cost, Gen_SU_Time, Gen_SD_Time,
            w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3, w_fc4, b_fc4, w_fc5, b_fc5]
    ##########################################################################################################################
    jl = julia.Julia()
    jl.include('Solve_MILP.jl')
    Gen_Status, Brh_Status, UC_Status, HVDC_Cap = jl.SolveMILP(data)
    Gen_Status = to_numpy(Gen_Status)
    Brh_Status = to_numpy(Brh_Status)
    UC_Status = to_numpy(UC_Status)
    HVDC_Cap = to_numpy(HVDC_Cap)

    print("Generator Status:", Gen_Status.shape)
    print("Branch Status:", Brh_Status.shape)
    print("Unit Commitment Status:", UC_Status.shape)
    print("HVDC Capacities:", HVDC_Cap.shape)

    for t in range(T):
        GenStatus = UC_Status[:, t].reshape(-1, 1)
        BrhStatus = Brh_Status.reshape(-1, 1)
        HVDC_Cap = HVDC_Cap.reshape(-1, 1)
        B_MX = CalB_MX(BusAdjList, MX_NBus_NGen, BrhStatus, GenStatus, B_Brh_ini, B_Gen_ini, NBus, NBrh)
        IfINV, Br = CalBr(B_MX, HVDC_Bus, Load_Bus)
        gSCR = np.linalg.eigvals(np.linalg.inv(np.diag(HVDC_Cap.flatten())) @ Br)
        print('   Actual gSCR:', gSCR)
        f = w_fc5 @ ((w_fc4 @ ((w_fc1 @ GenStatus + b_fc1) + (w_fc2 @ BrhStatus + b_fc2) + w_fc3 @ HVDC_Cap + b_fc3) + b_fc4)) + b_fc5
        print('Predicted gSCR:', f.flatten())
