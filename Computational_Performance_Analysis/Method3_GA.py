import concurrent
import time
import random
import numpy as np
import cvxpy as cp
import sys
import concurrent.futures

sys.path.append(r'd:\SynologyDrive\Code_Python\TEP_IEEE39')
from MyCase.MyCases import Data4Case9, Data4Case24, Data4Case39, Data4Case88
from MyCase.PrepareData import prepare_data, calculate_PTDF, calculate_Bkr


def solve_sub_problem_ga(Brh_Statu_Star, UC_Star, Bkr_Star, PTDF_Star, data, return_solution=False):
    # print('Solving Sub-Problem GA')
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
    Gen_PMAX_ini = data['Gen_PMAX_ini'].reshape((-1, 1))
    Gen_PMIN_ini = data['Gen_PMIN_ini'].reshape((-1, 1))
    HVDC_Cap_MAX = data['HVDC_Cap_MAX']

    # 变量
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
        Gen_P >=  cp.multiply(Gen_PMIN_ini @ np.ones((1, T)), UC_Star),
        Gen_P <=  cp.multiply(Gen_PMAX_ini @ np.ones((1, T)), UC_Star),
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
        scr_cons += [(Bkr_Star[t] - SCR0 * (MXH_NBus_NHVDC @ cp.diag(HVDC_Cap.flatten()) @ MXH_NHVDC_NBus)) >> epsilon * identity_matrix]

    Cost_GP = baseMVA * cp.sum(cp.multiply(Gen_P, Gen_P_cost @ np.ones((1, T))))
    Cost_HP = baseMVA * cp.sum(cp.multiply(HVDC_P, HVDC_P_cost @ np.ones((1, T))))
    Cost_LSP = baseMVA * cp.sum(cp.multiply(ls_Pos, P_LS_cost))
    Cost_LSN = baseMVA * cp.sum(cp.multiply(ls_Neg, P_LS_cost))
    Cost_HCap = baseMVA * cp.sum(cp.multiply(HVDC_Cap, HVDC_Cap_cost))
    obj = D * (Cost_GP + Cost_HP + Cost_LSP + Cost_LSN) + Cost_HCap

    prob = cp.Problem(cp.Minimize(obj), constraints + scr_cons)
    try:
        prob.solve(solver=cp.MOSEK, verbose=False, mosek_params={'MSK_IPAR_NUM_THREADS': 4})
    except Exception:
        return cp.INFEASIBLE, float('inf') if not return_solution else (cp.INFEASIBLE, float('inf'), None)

    if prob.status == cp.OPTIMAL:
        if return_solution:
            sol = {
                'Gen_P': Gen_P.value,
                'Brh_P': Brh_P.value,
                'HVDC_P': HVDC_P.value,
                'HVDC_Cap': HVDC_Cap.value,
                'ls_Pos': ls_Pos.value,
                'ls_Neg': ls_Neg.value,
            }
            return prob.status, prob.value, sol
        return prob.status, prob.value
    else:
        return (prob.status, float('inf')) if not return_solution else (prob.status, float('inf'), None)


def evaluate_candidate(brh_vec, uc_mat, data, bigM=1e12):

    Brh_Statu = np.array(brh_vec, dtype=float).reshape((-1, 1))
    UC_Star = np.array(uc_mat, dtype=float)
    NGen, T = UC_Star.shape

    Exit_Gen = data['Exit_Gen'].astype(int).flatten()
    Gen_Statu = np.maximum(np.max(UC_Star, axis=1).astype(int), Exit_Gen).reshape((-1, 1))

    add_brh_cost = float(np.sum(data['Add_Brh_Cost'] * (Brh_Statu - data['Exit_Brh'])))
    add_gen_cost = float(np.sum(data['Add_Gen_Cost'].reshape((-1, 1)) * (Gen_Statu - data['Exit_Gen'].reshape((-1, 1)))))
    Bkr_star, _, _ = calculate_Bkr(
        Brh_Statu, UC_Star,
        data['B_Brh_ini'], data['B_Gen_ini'], data['BusAdjList'],
        data['NBus'], data['NBrh'],
        data['MX_NBus_NGen'], data['MX_NBus_NBus_R'], data['MX_NBus_NBus_C']
    )
    PTDF_star = calculate_PTDF(
        BusAdjList=data['BusAdjList'],
        Brh_Status=Brh_Statu,
        B_Brh_ini=data['B_Brh_ini'],
        NBus=data['NBus'],
        NBrh=data['NBrh']
    )
    status, op_cost = solve_sub_problem_ga(Brh_Statu, UC_Star, Bkr_star, PTDF_star, data)
    if status != cp.OPTIMAL:
        return np.inf
    baseMVA = data['baseMVA']
    D = data['D']
    Gen_SU_cost = data['Gen_SU_cost'] 
    Gen_SD_cost = data['Gen_SD_cost'] 
    if T >= 2:
        delta = UC_Star[:, 1:] - UC_Star[:, :-1]                
        su = np.maximum(0.0, delta)                            
        sd = np.maximum(0.0, -delta)                            
        su_cost = float(np.sum(Gen_SU_cost * su))
        sd_cost = float(np.sum(Gen_SD_cost * sd))
        susd_cost_annual = baseMVA * D * (su_cost + sd_cost)
    else:
        susd_cost_annual = 0.0

    return float(op_cost + susd_cost_annual + add_brh_cost + add_gen_cost)

def _init_ga_worker(shared_data):
    global _shared_ga_data
    _shared_ga_data = shared_data


def _worker_evaluate(individual):
    brh, gen = individual
    return evaluate_candidate(brh, gen, _shared_ga_data)

def evaluate_population(population, data, use_parallel=False, max_workers=None):
    if not use_parallel:
        return [evaluate_candidate(brh, gen, data) for brh, gen in population]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=_init_ga_worker,
        initargs=(data,)
    ) as executor:
        return list(executor.map(_worker_evaluate, population))
    
    
def ga_solve(data,
             pop_size=40,
             generations=60,
             cx_prob=0.8,
             mut_prob=0.2,
             elitism=2,
             patience=20,
             use_parallel=False,
             max_workers=None):
    
    elitism = int(elitism * pop_size) if elitism < 1 else elitism

    NBrh = data['NBrh']
    NGen = data['NGen']
    T = data['T']
    Exit_Brh = data['Exit_Brh'].astype(int).flatten()

    brh_lb = Exit_Brh                  
    brh_ub = np.full(NBrh, 3, dtype=int)
    uc_lb = 0
    uc_ub = 1

    def random_individual():
        brh = np.array([random.randint(int(brh_lb[i]), int(brh_ub[i])) for i in range(NBrh)], dtype=int)
        uc = np.random.randint(uc_lb, uc_ub + 1, size=(NGen, T), dtype=int)
        return brh, uc

    def repair(brh, uc):
        brh = np.minimum(np.maximum(brh, brh_lb), brh_ub).astype(int)
        uc = np.minimum(np.maximum(uc, uc_lb), uc_ub).astype(int)
        return brh, uc

    def crossover(p1, p2):
        brh1, uc1 = p1
        brh2, uc2 = p2
        child1_b, child1_uc = brh1.copy(), uc1.copy()
        child2_b, child2_uc = brh2.copy(), uc2.copy()
        if random.random() < cx_prob:
            mask_b = np.random.rand(NBrh) < 0.5
            child1_b[mask_b], child2_b[mask_b] = brh2[mask_b], brh1[mask_b]
            mask_uc = np.random.rand(NGen, T) < 0.5
            tmp = child1_uc.copy()
            child1_uc[mask_uc] = child2_uc[mask_uc]
            child2_uc[mask_uc] = tmp[mask_uc]
        return (child1_b, child1_uc), (child2_b, child2_uc)

    def mutate(ind):
        brh, uc = ind
        for i in range(NBrh):
            if random.random() < mut_prob:
                brh[i] = random.randint(int(brh_lb[i]), int(brh_ub[i]))
        flip_mask = np.random.rand(NGen, T) < mut_prob
        uc = uc.copy()
        uc[flip_mask] = 1 - uc[flip_mask]
        return repair(brh, uc)

    def roulette_probs(fitness):
        fitness = np.asarray(fitness, dtype=float)
        finite_mask = np.isfinite(fitness)
        worst_fill = np.max(fitness[finite_mask]) if np.any(finite_mask) else 1.0
        fitness = np.where(finite_mask, fitness, worst_fill + 1e6)
        order = np.argsort(fitness)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(fitness))
        scores = (len(fitness) - ranks).astype(float)
        probs = scores / np.sum(scores)
        return probs
    
    population = [random_individual() for _ in range(pop_size)]
    fitness = evaluate_population(population, data, use_parallel, max_workers)
    best_idx = int(np.argmin(fitness))
    best_fit = fitness[best_idx]
    best_ind = (population[best_idx][0].copy(), population[best_idx][1].copy())

    print(f"[GA] init best = {best_fit:.4f}")
    no_improve = 0

    for gen_no in range(1, generations + 1):
        new_pop = []
        elite_indices = np.argsort(fitness)[:elitism]
        for ei in elite_indices:
            new_pop.append((population[ei][0].copy(), population[ei][1].copy()))
        probs = roulette_probs(fitness)
        while len(new_pop) < pop_size:
            i1 = np.random.choice(pop_size, p=probs)
            i2 = np.random.choice(pop_size, p=probs)
            c1, c2 = crossover(population[i1], population[i2])
            c1 = mutate(c1)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                c2 = mutate(c2)
                new_pop.append(c2)

        population = new_pop
        fitness = evaluate_population(population, data, use_parallel, max_workers)
        cur_best_idx = int(np.argmin(fitness))
        cur_best_fit = fitness[cur_best_idx]

        if cur_best_fit < best_fit:
            best_fit = cur_best_fit
            best_ind = (population[cur_best_idx][0].copy(), population[cur_best_idx][1].copy())
            no_improve = 0
        else:
            no_improve += 1

        print(f"[GA] gen {gen_no:03d} best = {best_fit:.4f} (no_improve={no_improve})")

        if no_improve >= patience:
            print(f"[GA] early stop: no improvement in {patience} generations")
            break

    return best_ind, best_fit


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

    start_time = time.time()
    start_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))
    print(f"开始时间: {start_time_str}")
    with open('ga_start_time.txt', 'w', encoding='utf-8') as f:
        f.write(f"开始时间: {start_time_str}\n")

    # 运行 GA（生成 Brh_Status 与 UC_Status）
    (best_brh, best_uc), best_value = ga_solve(
        data,
        pop_size=50,
        generations=1,
        cx_prob=0.6,
        mut_prob=0.01,
        elitism=0.2,
        patience=100,
        use_parallel=False,  
        max_workers=50       
    )

    # 由 UC 生成 Gen_Status（不低于 Exit_Gen）
    Exit_Gen_vec = data['Exit_Gen'].astype(int).flatten()
    best_gen_status = np.maximum(np.max(best_uc, axis=1).astype(int), Exit_Gen_vec).reshape((-1, 1))

    # 最终运行结果：与 Method1 一致，Bkr/PTDF 用 UC（日内时变）
    Brh_Statu = best_brh.reshape((-1, 1)).astype(float)
    Bkr_star, _, _ = calculate_Bkr(
        Brh_Statu, best_uc,
        data['B_Brh_ini'], data['B_Gen_ini'], data['BusAdjList'],
        data['NBus'], data['NBrh'],
        data['MX_NBus_NGen'], data['MX_NBus_NBus_R'], data['MX_NBus_NBus_C']
    )
    PTDF_star = calculate_PTDF(
        BusAdjList=data['BusAdjList'],
        Brh_Status=Brh_Statu,
        B_Brh_ini=data['B_Brh_ini'],
        NBus=data['NBus'],
        NBrh=data['NBrh']
    )
    status, op_cost, sol = solve_sub_problem_ga(Brh_Statu, best_uc, Bkr_star, PTDF_star, data, return_solution=True)

    end_time = time.time()
    end_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))
    print(f"结束时间: {end_time_str}")
    print(f"GA总运行时间: {end_time - start_time:.2f} 秒")
    print("最优目标值(投资+运营):", best_value)
    print("最优线路状态合计(回路数):", int(np.sum(best_brh)))
    print("最优机组投资状态合计:", int(np.sum(best_gen_status)))
    print("UC启用总小时数:", int(np.sum(best_uc)))
    if sol is not None:
        ls_total = float(np.sum(sol['ls_Pos']) + np.sum(sol['ls_Neg']))
        print(f"运营求解状态: {status}, 年运营成本: {op_cost:.4f}, 负荷调节总量: {ls_total:.4f}")

    with open('ga_start_time.txt', 'a', encoding='utf-8') as f:
        f.write(f"结束时间: {end_time_str}\n")
        f.write(f"GA总运行时间: {end_time - start_time:.2f} 秒\n")
        f.write(f"最优目标值(投资+运营): {best_value}\n")
        f.write(f"最优线路状态合计(回路数): {int(np.sum(best_brh))}\n")
        f.write(f"最优机组投资状态合计: {int(np.sum(best_gen_status))}\n")
        f.write(f"UC启用总小时数: {int(np.sum(best_uc))}\n")
        if sol is not None:
            f.write(f"运营求解状态: {status}, 年运营成本: {op_cost:.4f}\n")
            f.write(f"负荷调节总量: {float(np.sum(sol['ls_Pos']) + np.sum(sol['ls_Neg'])):.4f}\n")