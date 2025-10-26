import time
import julia
import cvxpy as cp
import numpy as np
import sys
sys.path.append(r'd:\SynologyDrive\Code_Python\TEP_IEEE39')
from MyCase.MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88
from MyCase.PrepareData import prepare_data, calculate_PTDF, calculate_Bkr

if __name__ == "__main__":
    MyCase = Data4Case88()
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


    data = [Exit_Brh, Exit_Gen, B_Brh_ini, B_Gen_ini, BusAdjList, BusAdj_complement_pairs, MX_NBus_NGen,
            MX_NBus_NBus_R, MX_NBus_NBus_C, Add_Brh_Cost, Add_Gen_Cost, Gen_PMAX_ini, Gen_PMIN_ini,
            Brh_MAX_ini, NBus, NBrh, NGen, NHVDC, T, D, P_D, SCR0, baseMVA, MX_NBus_NHVDC,
            MXH_NBus_NHVDC, MXH_NHVDC_NBus, P_LS_cost, Gen_P_cost, HVDC_P_cost, HVDC_Cap_cost,
            Gen_RU, Gen_RD, GenDiag, HVDC_Cap_MAX, Gen_SU_cost, Gen_SD_cost, Gen_SU_Time, Gen_SD_Time]


    ##########################################################################################################################
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f"开始时间: {start_time_str}")

    with open('Julia_start_time.txt', 'w') as f:
        f.write(f"开始时间: {start_time_str}\n")

    jl = julia.Julia()
    jl.include('Solve_MISDP.jl')
    X = jl.SolveMISDP(data)
    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f"结束时间: {end_time_str}")
    print(f"Julia总运行时间: {end_time - start_time:.2f} 秒")

    with open('Julia_start_time.txt', 'a') as f:
        f.write(f"结束时间: {end_time_str}\n")
        f.write(f"Julia总运行时间: {end_time - start_time:.2f} 秒\n")
