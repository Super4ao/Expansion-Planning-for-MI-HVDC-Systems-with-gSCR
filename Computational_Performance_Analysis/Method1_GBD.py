import time
import cvxpy as cp
import numpy as np
import sys
sys.path.append(r'd:\SynologyDrive\Code_Python\TEP_IEEE39')
from MyCase.MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88
from MyCase.PrepareData import prepare_data, calculate_PTDF, calculate_Bkr




def solve_slack_MISDP_problem(data):
    NBus = data['NBus']
    NBrh = data['NBrh']
    NGen = data['NGen']
    NHVDC = data['NHVDC']
    T = data['T']
    D = data['D']
    P_D = data['P_D']
    SCR0 = data['SCR0']
    baseMVA = data['baseMVA']
    AdjMX = data['AdjMX']
    MX_NBus_NGen = data['MX_NBus_NGen']
    MX_NBus_NHVDC = data['MX_NBus_NHVDC']
    MXH_NBus_NHVDC = data['MXH_NBus_NHVDC']
    MXH_NHVDC_NBus = data['MXH_NHVDC_NBus']
    P_LS_cost = data['P_LS_cost']
    Gen_P_cost = data['Gen_P_cost']
    HVDC_P_cost = data['HVDC_P_cost']
    HVDC_Cap_cost = data['HVDC_Cap_cost']
    Brh_MAX_ini = data['Brh_MAX_ini']
    Gen_RU = data['Gen_RU']
    Gen_RD = data['Gen_RD']
    GenDiag = data['GenDiag']
    Gen_PMAX_ini = data['Gen_PMAX_ini']
    Gen_PMIN_ini = data['Gen_PMIN_ini']
    HVDC_Cap_MAX = data['HVDC_Cap_MAX']
    Add_Brh_Cost = data['Add_Brh_Cost']
    Add_Gen_Cost = data['Add_Gen_Cost']
    B_Brh_ini = data['B_Brh_ini']
    B_Gen_ini = data['B_Gen_ini']
    BusAdjList = data['BusAdjList']
    BusAdj_complement_pairs = data['BusAdj_complement_pairs']
    MX_NBus_NBus_R = data['MX_NBus_NBus_R']
    MX_NBus_NBus_C = data['MX_NBus_NBus_C']
    Gen_SU_cost = data['Gen_SU_cost']
    Gen_SD_cost = data['Gen_SD_cost']
    Gen_SU_Time = data['Gen_SU_Time']
    Gen_SD_Time = data['Gen_SD_Time']

    epsilon = 1e-4
    identity_matrix = np.eye(NBus)

    Brh_P = cp.Variable((NBrh, T))
    Gen_P = cp.Variable((NGen, T))
    ls_Pos = cp.Variable((NBus, T))
    ls_Neg = cp.Variable((NBus, T))
    HVDC_P = cp.Variable((NHVDC, T))
    HVDC_Cap = cp.Variable((NHVDC, 1))
    Gen_B_MX = cp.Variable((NBus, T))
    Bus_B_MX = cp.Variable((NBus, NBus))
    Brh_Status = cp.Variable((NBrh, 1))
    Gen_Status = cp.Variable((NGen, 1))
    UC_Status = cp.Variable((NGen, T))
    cost_SU = cp.Variable((NGen, T-1))
    cost_SD = cp.Variable((NGen, T-1))
    constraints = []
    constraints += [Brh_Status >= Exit_Brh, Brh_Status <= 3, Gen_Status >= Exit_Gen, Gen_Status <= 1]
    constraints += [Bus_B_MX[f_node - 1, t_node - 1] == 0 for f_node, t_node in BusAdj_complement_pairs]
    constraints += [Bus_B_MX[t_node - 1, f_node - 1] == 0 for f_node, t_node in BusAdj_complement_pairs]
    constraints += [Bus_B_MX[BusAdjList[i, 0] - 1, BusAdjList[i, 1] - 1] == -Brh_Status[i, 0] * B_Brh_ini[i, 0] for i in range(NBrh)]
    constraints += [Bus_B_MX[BusAdjList[i, 1] - 1, BusAdjList[i, 0] - 1] == -Brh_Status[i, 0] * B_Brh_ini[i, 0] for i in range(NBrh)]
    constraints += [Bus_B_MX[i, i] == -cp.sum([Bus_B_MX[i, j] for j in range(NBus) if j != i]) for i in range(NBus)]
    constraints += [ls_Neg >= 0, ls_Pos >= 0, ls_Pos <= 0.1 * P_D]
    constraints += [HVDC_P >= 0, HVDC_P <= HVDC_Cap @ np.ones((1, T)), HVDC_Cap >= 0, HVDC_Cap <= HVDC_Cap_MAX]
    constraints += [Brh_P >= -cp.multiply(Brh_MAX_ini, Brh_Status) @ np.ones((1, T))]
    constraints += [Brh_P <=  cp.multiply(Brh_MAX_ini, Brh_Status) @ np.ones((1, T))]
    constraints += [Gen_P >= cp.multiply(Gen_PMIN_ini, Gen_Status) @ np.ones((1, T))]
    constraints += [Gen_P <= cp.multiply(Gen_PMAX_ini, Gen_Status) @ np.ones((1, T))]
    constraints += [Gen_P @ GenDiag >= Gen_RD @ np.ones((1, T - 1))]
    constraints += [Gen_P @ GenDiag <= Gen_RU @ np.ones((1, T - 1))]
    constraints += [MX_NBus_NGen @ Gen_P + MX_NBus_NHVDC @ HVDC_P + ls_Pos - ls_Neg - P_D == AdjMX @ Brh_P]
    constraints += [cp.sum((MX_NBus_NGen @ Gen_P + MX_NBus_NHVDC @ HVDC_P + ls_Pos - ls_Neg), axis=0) == np.sum(P_D, axis=0)]
    constraints += [0 <= UC_Status, UC_Status <= Gen_Status @ np.ones((1, T))]
    for t in range(T):
        constraints += [Gen_B_MX[:, t].reshape((NBus, 1)) == MX_NBus_NGen @ cp.multiply(UC_Status[:, t].reshape((NGen, 1)), B_Gen_ini)]
        constraints += [(MX_NBus_NBus_R @ (Bus_B_MX + cp.diag(Gen_B_MX[:, t])) @ MX_NBus_NBus_C - SCR0 * (MXH_NBus_NHVDC @ cp.diag(HVDC_Cap.flatten()) @ MXH_NHVDC_NBus)) >> epsilon * identity_matrix]

    for i in range(NGen):
        Min_Up_Time = Gen_SU_Time[i, 0]
        Min_Dn_Time = Gen_SD_Time[i, 0]
        for t in range(1, T):
            for k in range(t + 1, min(t + Min_Up_Time, T)):
                constraints += [UC_Status[i, k] >= UC_Status[i, t] - UC_Status[i, t - 1]]
            for k in range(t + 1, min(t + Min_Dn_Time, T)):
                constraints += [1 - UC_Status[i, k] >= UC_Status[i, t - 1] - UC_Status[i, t]]
        for t in range(T - 1):
            constraints += [cost_SU[i, t] >= Gen_SU_cost[i, 0] * (UC_Status[i, t + 1] - UC_Status[i, t])]
            constraints += [cost_SD[i, t] >= Gen_SD_cost[i, 0] * (UC_Status[i, t] - UC_Status[i, t + 1])]

    constraints += [cost_SU >= 0]
    constraints += [cost_SD >= 0]
    Cost_SU = cp.sum(cost_SU)
    Cost_SD = cp.sum(cost_SD)
    Cost_GP = cp.sum(cp.multiply(Gen_P, Gen_P_cost @ np.ones((1, T))))
    Cost_HP = cp.sum(cp.multiply(HVDC_P, HVDC_P_cost @ np.ones((1, T))))
    Cost_LSP = cp.sum(cp.multiply(ls_Pos, P_LS_cost))
    Cost_LSN = cp.sum(cp.multiply(ls_Neg, P_LS_cost))
    Cost_HCap = cp.sum(cp.multiply(HVDC_Cap, HVDC_Cap_cost))
    Cost_ABrh = cp.sum(cp.multiply(Add_Brh_Cost, (Brh_Status - Exit_Brh)))
    Cost_AGen = cp.sum(cp.multiply(Add_Gen_Cost, (Gen_Status - Exit_Gen)))
    obj = D * baseMVA * (Cost_GP + Cost_HP + Cost_LSP + Cost_LSN + Cost_SU + Cost_SD) + Cost_HCap + Cost_ABrh + Cost_AGen
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)
    return np.around(Brh_Status.value), np.around(Gen_Status.value), np.round(UC_Status.value)



def solve_feasible_sub_problem(Brh_Statu_Star, UC_Statu_Star, BkrList_Star, PTDF_Star, data):
    NBus = data['NBus']
    NBrh = data['NBrh']
    NGen = data['NGen']
    NHVDC = data['NHVDC']
    T = data['T']
    D = data['D']
    P_D = data['P_D']
    SCR0 = data['SCR0']
    MX_NBus_NGen = data['MX_NBus_NGen']
    MX_NBus_NHVDC = data['MX_NBus_NHVDC']
    MXH_NBus_NHVDC = data['MXH_NBus_NHVDC']
    MXH_NHVDC_NBus = data['MXH_NHVDC_NBus']
    Brh_MAX_ini = data['Brh_MAX_ini']
    Gen_RU = data['Gen_RU']
    Gen_RD = data['Gen_RD']
    GenDiag = data['GenDiag']
    Gen_PMAX_ini = data['Gen_PMAX_ini']
    Gen_PMIN_ini = data['Gen_PMIN_ini']
    HVDC_Cap_MAX = data['HVDC_Cap_MAX']

    Brh_P = cp.Variable((NBrh, T))
    Gen_P = cp.Variable((NGen, T))
    ls_Pos = cp.Variable((NBus, T))
    ls_Neg = cp.Variable((NBus, T))
    HVDC_P = cp.Variable((NHVDC, T))
    HVDC_Cap = cp.Variable((NHVDC, 1))

    epsilon = 1e-3
    identity_matrix = np.eye(NBus)
    # Slack variables
    s_scr = cp.Variable((NBus, T))
    s_gen_lb = cp.Variable((NGen, T))
    s_gen_ub = cp.Variable((NGen, T))
    s_brh_lb = cp.Variable((NBrh, T))
    s_brh_ub = cp.Variable((NBrh, T))

    constraints = [
        Brh_P >= -cp.multiply(Brh_MAX_ini, Brh_Statu_Star) @ np.ones((1, T)) - s_brh_lb,
        Brh_P <=  cp.multiply(Brh_MAX_ini, Brh_Statu_Star) @ np.ones((1, T)) + s_brh_ub,
        Gen_P >=  cp.multiply(Gen_PMIN_ini @ np.ones((1, T)), UC_Statu_Star) - s_gen_lb,
        Gen_P <=  cp.multiply(Gen_PMAX_ini @ np.ones((1, T)), UC_Statu_Star) + s_gen_ub,
        Gen_P @ GenDiag >= Gen_RD @ np.ones((1, T - 1)),
        Gen_P @ GenDiag <= Gen_RU @ np.ones((1, T - 1)),
        PTDF_Star @ (MX_NBus_NGen @ Gen_P + MX_NBus_NHVDC @ HVDC_P + ls_Pos - ls_Neg - P_D) == Brh_P,
        cp.sum((MX_NBus_NGen @ Gen_P + MX_NBus_NHVDC @ HVDC_P + ls_Pos - ls_Neg), axis=0) == np.sum(P_D, axis=0),
        ls_Neg >= 0,
        ls_Pos >= 0,
        ls_Pos <= 0.1 * P_D,
        HVDC_Cap >= 0,
        HVDC_Cap <= HVDC_Cap_MAX,
        HVDC_P >= 0,
        HVDC_P <= HVDC_Cap @ np.ones((1, T)),
        s_scr >= 0,
        s_gen_lb >= 0,
        s_gen_ub >= 0,
        s_brh_lb >= 0,
        s_brh_ub >= 0
    ]
    scr_cons = []
    for t in range(T):
        scr_cons += [(BkrList_Star[t] + cp.diag(s_scr[:, t]) - SCR0 * (MXH_NBus_NHVDC @ cp.diag(HVDC_Cap.flatten()) @ MXH_NHVDC_NBus)) >> epsilon * identity_matrix]

    obj = cp.sum(s_scr) + cp.sum(s_gen_lb) + cp.sum(s_gen_ub) + cp.sum(s_brh_lb) + cp.sum(s_brh_ub)
    prob = cp.Problem(cp.Minimize(obj), constraints + scr_cons)
    prob.solve(solver=cp.MOSEK, verbose=False)

    lambda_Brh_LB = constraints[0].dual_value
    lambda_Brh_UB = constraints[1].dual_value
    lambda_Gen_LB = constraints[2].dual_value
    lambda_Gen_UB = constraints[3].dual_value
    lambda_scr_list = []
    for t in range(T):
        lambda_scr_list.append(scr_cons[t].dual_value)
    s_brh_lb_val = s_brh_lb.value
    s_brh_ub_val = s_brh_ub.value
    s_gen_lb_val = s_gen_lb.value
    s_gen_ub_val = s_gen_ub.value
    s_scr_val = s_scr.value
    DualValue = (lambda_Brh_LB, lambda_Brh_UB, lambda_Gen_LB, lambda_Gen_UB, lambda_scr_list)

    return prob.status, prob.value, DualValue

def solve_sub_problem(Brh_Statu_Star, UC_Statu_Star, BkrList_Star, PTDF_Star, data):
    NBus = data['NBus']
    NBrh = data['NBrh']
    NGen = data['NGen']
    NHVDC = data['NHVDC']
    T = data['T']
    D = data['D']
    P_D = data['P_D']
    SCR0 = data['SCR0']
    baseMVA = data['baseMVA']
    MX_NBus_NGen = data['MX_NBus_NGen']
    MX_NBus_NHVDC = data['MX_NBus_NHVDC']
    MXH_NBus_NHVDC = data['MXH_NBus_NHVDC']
    MXH_NHVDC_NBus = data['MXH_NHVDC_NBus']
    P_LS_cost = data['P_LS_cost']
    Gen_P_cost = data['Gen_P_cost']
    HVDC_P_cost = data['HVDC_P_cost']
    HVDC_Cap_cost = data['HVDC_Cap_cost']
    Brh_MAX_ini = data['Brh_MAX_ini']
    Gen_RU = data['Gen_RU']
    Gen_RD = data['Gen_RD']
    GenDiag = data['GenDiag']
    Gen_PMAX_ini = data['Gen_PMAX_ini']
    Gen_PMIN_ini = data['Gen_PMIN_ini']
    HVDC_Cap_MAX = data['HVDC_Cap_MAX']
    
    Brh_P = cp.Variable((NBrh, T))
    Gen_P = cp.Variable((NGen, T))
    ls_Pos = cp.Variable((NBus, T))
    ls_Neg = cp.Variable((NBus, T))
    HVDC_P = cp.Variable((NHVDC, T))
    HVDC_Cap = cp.Variable((NHVDC, 1))

    epsilon = 1e-3
    identity_matrix = np.eye(NBus)

    constraints = [
        Brh_P >= -cp.multiply(Brh_MAX_ini, Brh_Statu_Star) @ np.ones((1, T)),
        Brh_P <=  cp.multiply(Brh_MAX_ini, Brh_Statu_Star) @ np.ones((1, T)),
        Gen_P >=  cp.multiply(Gen_PMIN_ini @ np.ones((1, T)), UC_Statu_Star),
        Gen_P <=  cp.multiply(Gen_PMAX_ini @ np.ones((1, T)), UC_Statu_Star),
        Gen_P @ GenDiag >= Gen_RD @ np.ones((1, T - 1)),
        Gen_P @ GenDiag <= Gen_RU @ np.ones((1, T - 1)),
        PTDF_Star @ (MX_NBus_NGen @ Gen_P + MX_NBus_NHVDC @ HVDC_P + ls_Pos - ls_Neg - P_D) == Brh_P,
        cp.sum((MX_NBus_NGen @ Gen_P + MX_NBus_NHVDC @ HVDC_P + ls_Pos - ls_Neg), axis=0) == np.sum(P_D, axis=0),
        ls_Neg >= 0,
        ls_Pos >= 0,
        ls_Pos <= 0.1 * P_D,
        HVDC_Cap >= 0,
        HVDC_Cap <= HVDC_Cap_MAX,
        HVDC_P >= 0,
        HVDC_P <= HVDC_Cap @ np.ones((1, T))
    ]
    scr_cons = []
    for t in range(T):
        scr_cons += [(BkrList_Star[t] - SCR0 * (MXH_NBus_NHVDC @ cp.diag(HVDC_Cap.flatten()) @ MXH_NHVDC_NBus)) >> epsilon * identity_matrix]

    Cost_GP = baseMVA * cp.sum(cp.multiply(Gen_P, Gen_P_cost @ np.ones((1, T))))
    Cost_HP = baseMVA * cp.sum(cp.multiply(HVDC_P, HVDC_P_cost @ np.ones((1, T))))
    Cost_LSP = baseMVA * cp.sum(cp.multiply(ls_Pos, P_LS_cost))
    Cost_LSN = baseMVA * cp.sum(cp.multiply(ls_Neg, P_LS_cost))
    Cost_HCap = baseMVA * cp.sum(cp.multiply(HVDC_Cap, HVDC_Cap_cost))
    obj = D * (Cost_GP + Cost_HP + Cost_LSP + Cost_LSN) + Cost_HCap
    prob = cp.Problem(cp.Minimize(obj), constraints + scr_cons)
    try:
        prob.solve(solver=cp.MOSEK, mosek_params={'MSK_IPAR_NUM_THREADS': 120})
    except Exception as e:
        print(f"子问题求解失败: {e}")
        return cp.INFEASIBLE, float('inf'), None, None
    if prob.status == cp.OPTIMAL:
        print(np.round(np.sum(ls_Pos.value), 2))
        print(np.round(np.sum(ls_Neg.value), 2))
        print(np.round(np.sum(Gen_P.value), 2))
        print(np.round(np.sum(HVDC_P.value), 2))
        print(np.round(np.sum(P_D), 2))
        lambda_Brh_LB = constraints[0].dual_value
        lambda_Brh_UB = constraints[1].dual_value
        lambda_Gen_LB = constraints[2].dual_value
        lambda_Gen_UB = constraints[3].dual_value
        lambda_scr_list = []
        for t in range(T):
            lambda_scr_list.append(scr_cons[t].dual_value)
        DualValue = (lambda_Brh_LB, lambda_Brh_UB, lambda_Gen_LB, lambda_Gen_UB, lambda_scr_list)
        PrimalValue = (Brh_P.value, Gen_P.value, ls_Pos.value, ls_Neg.value, HVDC_P.value, HVDC_Cap.value)
        return prob.status, prob.value, DualValue, PrimalValue
    else:
        return prob.status, float('inf'), None, None


def solve_master_problem(data):
    NBus = data['NBus']
    NBrh = data['NBrh']
    NGen = data['NGen']
    baseMVA = data['baseMVA']
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
    P_LS_cost = data['P_LS_cost']
    Gen_P_cost = data['Gen_P_cost']
    HVDC_P_cost = data['HVDC_P_cost']
    HVDC_Cap_cost = data['HVDC_Cap_cost']
    Gen_SU_cost = data['Gen_SU_cost']
    Gen_SD_cost = data['Gen_SD_cost']
    Gen_SU_Time = data['Gen_SU_Time']
    Gen_SD_Time = data['Gen_SD_Time']
    eta = cp.Variable()
    Gen_B_MX = cp.Variable((NBus, T))
    Bus_B_MX = cp.Variable((NBus, NBus))
    Brh_Status = cp.Variable((NBrh, 1), integer=True)
    Gen_Status = cp.Variable((NGen, 1), integer=True)
    UC_Status = cp.Variable((NGen, T), integer=True)
    cost_SU = cp.Variable((NGen, T-1))
    cost_SD = cp.Variable((NGen, T-1))
    BKR_List = []
    for t in range(T):
        BKR_List.append(cp.Variable((NBus, NBus)))
    constraints = []
    constraints += [eta >= 0, Brh_Status >= Exit_Brh, Brh_Status <= 3, Gen_Status >= Exit_Gen, Gen_Status <= 1]
    constraints += [Bus_B_MX[f_node - 1, t_node - 1] == 0 for f_node, t_node in BusAdj_complement_pairs]
    constraints += [Bus_B_MX[t_node - 1, f_node - 1] == 0 for f_node, t_node in BusAdj_complement_pairs]
    constraints += [Bus_B_MX[BusAdjList[i, 0] - 1, BusAdjList[i, 1] - 1] == -Brh_Status[i, 0] * B_Brh_ini[i, 0] for i in range(NBrh)]
    constraints += [Bus_B_MX[BusAdjList[i, 1] - 1, BusAdjList[i, 0] - 1] == -Brh_Status[i, 0] * B_Brh_ini[i, 0] for i in range(NBrh)]
    constraints += [Bus_B_MX[i, i] == -cp.sum([Bus_B_MX[i, j] for j in range(NBus) if j != i]) for i in range(NBus)]
    constraints += [0 <= UC_Status, UC_Status <= Gen_Status @ np.ones((1, T))]
    constraints += [UC_Status[:, 0] == Gen_Status[:, 0]]
    for t in range(T):
        constraints += [Gen_B_MX[:, t].reshape((NBus, 1)) == MX_NBus_NGen @ cp.multiply(UC_Status[:, t].reshape((NGen, 1)), B_Gen_ini)]
        constraints += [BKR_List[t] == MX_NBus_NBus_R @ (Bus_B_MX + cp.diag(Gen_B_MX[:, t])) @ MX_NBus_NBus_C]
    for i in range(NGen):
        Min_Up_Time = Gen_SU_Time[i, 0]
        Min_Dn_Time = Gen_SD_Time[i, 0]
        for t in range(1, T):
            for k in range(t + 1, min(t + Min_Up_Time, T)):
                constraints += [UC_Status[i, k] >= UC_Status[i, t] - UC_Status[i, t - 1]]
            for k in range(t + 1, min(t + Min_Dn_Time, T)):
                constraints += [1 - UC_Status[i, k] >= UC_Status[i, t - 1] - UC_Status[i, t]]
        for t in range(T - 1):
            constraints += [cost_SU[i, t] >= Gen_SU_cost[i, 0] * (UC_Status[i, t + 1] - UC_Status[i, t])]
            constraints += [cost_SD[i, t] >= Gen_SD_cost[i, 0] * (UC_Status[i, t] - UC_Status[i, t + 1])]
    constraints += [cost_SU >= 0]
    constraints += [cost_SD >= 0]
    Cost_SU = cp.sum(cost_SU)
    Cost_SD = cp.sum(cost_SD)
    Cost_ABrh = cp.sum(cp.multiply(Add_Brh_Cost, (Brh_Status - Exit_Brh)))
    Cost_AGen = cp.sum(cp.multiply(Add_Gen_Cost, (Gen_Status - Exit_Gen)))
    obj = Cost_ABrh + Cost_AGen + eta + D * (Cost_SU + Cost_SD)
    ##########################################################################################################################
    Brh_Status_star, Gen_Status_star, UC_Status_star = solve_slack_MISDP_problem(data)  
    # Brh_Status_star = Exit_Brh
    # Gen_Status_star = Exit_Gen
    # UC_Status_star = Gen_Status_star @ np.ones((1, T))
    Bkr_List_star, _, _ = calculate_Bkr(Brh_Status_star, UC_Status_star, B_Brh_ini, B_Gen_ini, BusAdjList, NBus, NBrh, MX_NBus_NGen, MX_NBus_NBus_R, MX_NBus_NBus_C)
    PTDF_Star = calculate_PTDF(BusAdjList=BusAdjList, Brh_Status=Brh_Status_star, B_Brh_ini=B_Brh_ini, NBus=NBus, NBrh=NBrh)
    upper_bound = float('inf')
    lower_bound = -float('inf')
    ##########################################################################################################################
    for iteration in range(5000):
        print(f"==================================== Round {iteration + 1} ====================================")
        SubStatus, PrimalObj, DualValue, PrimalValue = solve_sub_problem(Brh_Status_star, UC_Status_star, Bkr_List_star, PTDF_Star, data)
        if SubStatus == cp.OPTIMAL:
            lambda_Brh_LB, lambda_Brh_UB, lambda_Gen_LB, lambda_Gen_UB, lambda_scr_list = DualValue
            Brh_P_star, Gen_P_star, ls_Pos_star, ls_Neg_star, HVDC_P_star, HVDC_Cap_star = PrimalValue
            cut = (eta >= PrimalObj +
                cp.sum(cp.multiply(cp.sum(lambda_Brh_LB, axis=1), cp.multiply(-Brh_MAX_ini.flatten(), (Brh_Status - Brh_Status_star).flatten()))) +
                cp.sum(cp.multiply(cp.sum(lambda_Brh_UB, axis=1), cp.multiply(-Brh_MAX_ini.flatten(), (Brh_Status - Brh_Status_star).flatten()))) +  
                cp.sum(cp.multiply(lambda_Gen_LB, cp.multiply( Gen_PMIN_ini @ np.ones((1, T)), (UC_Status - UC_Status_star)))) +
                cp.sum(cp.multiply(lambda_Gen_UB, cp.multiply(-Gen_PMAX_ini @ np.ones((1, T)), (UC_Status - UC_Status_star)))) +  
                cp.sum(cp.hstack([cp.trace(lambda_scr_list[t].T @ (BKR_List[t] - Bkr_List_star[t])) for t in range(T)])))
            constraints.append(cut)
            add_brh_cost = np.sum(data['Add_Brh_Cost'] * (Brh_Status_star - data['Exit_Brh']))
            add_gen_cost = np.sum(data['Add_Gen_Cost'] * (Gen_Status_star - data['Exit_Gen']))
            current_total_cost = PrimalObj + add_brh_cost + add_gen_cost
            upper_bound = min(upper_bound, current_total_cost)

        else:
            print(f"子问题{SubStatus}，添加不可行割")
            FeasStatus, FeasObj, DualValue_feas = solve_feasible_sub_problem(Brh_Status_star, UC_Status_star, Bkr_List_star, PTDF_Star, data)
            lambda_Brh_LB, lambda_Brh_UB, lambda_Gen_LB, lambda_Gen_UB, lambda_scr_list = DualValue_feas
            cut = (0 >= FeasObj +
                cp.sum(cp.multiply(cp.sum(lambda_Brh_LB, axis=1), cp.multiply(-Brh_MAX_ini.flatten(), (Brh_Status - Brh_Status_star).flatten()))) +
                cp.sum(cp.multiply(cp.sum(lambda_Brh_UB, axis=1), cp.multiply(-Brh_MAX_ini.flatten(), (Brh_Status - Brh_Status_star).flatten()))) +  
                cp.sum(cp.multiply(lambda_Gen_LB, cp.multiply( Gen_PMIN_ini @ np.ones((1, T)), (UC_Status - UC_Status_star)))) +
                cp.sum(cp.multiply(lambda_Gen_UB, cp.multiply(-Gen_PMAX_ini @ np.ones((1, T)), (UC_Status - UC_Status_star)))))
            constraints.append(cut)  
    
        ##########################################################################################################################
        Masterprob = cp.Problem(cp.Minimize(obj), constraints)  
        Masterprob.solve(solver=cp.GUROBI, verbose=False, Threads=120)
        print("Master Problem Status:", Masterprob.status)
        lower_bound = Masterprob.value
        Bkr_List_star = [BKR_List[t].value for t in range(T)]
        Brh_Status_star = Brh_Status.value
        Gen_Status_star = Gen_Status.value
        UC_Status_star = UC_Status.value
        PTDF_Star = calculate_PTDF(BusAdjList=BusAdjList, Brh_Status=Brh_Status_star, B_Brh_ini=B_Brh_ini, NBus=NBus, NBrh=NBrh)
        ##########################################################################################################################
        relative_gap = (upper_bound - lower_bound) / (abs(upper_bound))
        print("当前上界 (Upper Bound):", upper_bound)
        print("当前下界 (Lower Bound):", lower_bound)
        print(f"收敛差距: |UB - LB| = {abs(upper_bound - lower_bound)}")
        print(f"相对间隙: |UB - LB| / |UB| = {relative_gap:.6%}") # 以百分比形式打印
        tol = 0.025
        if relative_gap < tol:
            print("收敛了！")
            print(f"最终最优值: {upper_bound}") 
            Cost_GP_star = baseMVA * np.sum(Gen_P_star * (Gen_P_cost @ np.ones((1, T))))
            Cost_HP_star = baseMVA * np.sum(HVDC_P_star * (HVDC_P_cost @ np.ones((1, T))))
            Cost_LSP_star = baseMVA * np.sum(ls_Pos_star * P_LS_cost)
            Cost_LSN_star = baseMVA * np.sum(ls_Neg_star * P_LS_cost)
            Cost_HCap_star = baseMVA * np.sum(HVDC_Cap_star * HVDC_Cap_cost)
            print('TotalBrh:', np.sum(Brh_Status_star))
            print('TotalGen:', np.sum(Gen_Status_star))
            print('Cost_GP_star:', D * Cost_GP_star)
            print('Cost_HP_star:', D * Cost_HP_star)
            print('Cost_LSP_star:', D * Cost_LSP_star)
            print('Cost_LSN_star:', D * Cost_LSN_star)
            print('Cost_HCap_star:', Cost_HCap_star)
            print('Add_Brh_Cost:', add_brh_cost)
            print('Add_Gen_Cost:', add_gen_cost)
            break
    return Gen_Status_star, Brh_Status_star, upper_bound


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
    ##########################################################################################################################
    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f"开始时间: {start_time_str}")

    with open('benders_start_time.txt', 'w') as f:
        f.write(f"开始时间: {start_time_str}\n")

    Gen_Statu_star, Brh_Statu_star, GlobalObj = solve_master_problem(data=data)
    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f"结束时间: {end_time_str}")
    print(f"Benders分解总运行时间: {end_time - start_time:.2f} 秒")

    with open('benders_start_time.txt', 'a') as f:
        f.write(f"结束时间: {end_time_str}\n")
        f.write(f"Benders分解总运行时间: {end_time - start_time:.2f} 秒\n")
