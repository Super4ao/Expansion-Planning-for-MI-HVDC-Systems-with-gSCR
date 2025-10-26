using JuMP, Pajarito, Mosek, Gurobi, MosekTools
using LinearAlgebra
import MathOptInterface as MOI


function SolveMISDP(data)
    (Exit_Brh, Exit_Gen, B_Brh_ini, B_Gen_ini, BusAdjList, BusAdj_complement_pairs, MX_NBus_NGen,
     MX_NBus_NBus_R, MX_NBus_NBus_C, Add_Brh_Cost, Add_Gen_Cost, Gen_PMAX_ini, Gen_PMIN_ini,
     Brh_MAX_ini, NBus, NBrh, NGen, NHVDC, T, D, P_D, SCR0, baseMVA, MX_NBus_NHVDC,
     MXH_NBus_NHVDC, MXH_NHVDC_NBus, P_LS_cost, Gen_P_cost, HVDC_P_cost, HVDC_Cap_cost,
     Gen_RU, Gen_RD, GenDiag, HVDC_Cap_MAX, Gen_SU_cost, Gen_SD_cost, Gen_SU_Time, Gen_SD_Time) = data
    Exit_Brh = vec(Exit_Brh)
    Exit_Gen = vec(Exit_Gen)
    B_Gen_ini = vec(B_Gen_ini)
    B_Brh_ini = vec(B_Brh_ini)
    HVDC_Cap_MAX = vec(HVDC_Cap_MAX)
    Add_Gen_Cost = vec(Add_Gen_Cost)
    Add_Brh_Cost = vec(Add_Brh_Cost)
    Gen_SU_cost = vec(Gen_SU_cost)
    Gen_SD_cost = vec(Gen_SD_cost)
    Gen_SU_Time = vec(Gen_SU_Time)
    Gen_SD_Time = vec(Gen_SD_Time)
    model = Model(
        optimizer_with_attributes(
            Pajarito.Optimizer,
            "tol_rel_gap" => 0.00,
            "oa_solver" => optimizer_with_attributes(Gurobi.Optimizer, "Threads" => 240, "OutputFlag" => 1),
            "conic_solver" => optimizer_with_attributes(Mosek.Optimizer, "NUM_THREADS" => 240, MOI.Silent() => false)
        )
    )
    @variable(model, Brh_P[1:NBrh, 1:T])
    @variable(model, Gen_P[1:NGen, 1:T])
    @variable(model, ls_Pos[1:NBus, 1:T])
    @variable(model, ls_Neg[1:NBus, 1:T])
    @variable(model, HVDC_P[1:NHVDC, 1:T])
    @variable(model, HVDC_Cap[1:NHVDC])
    @variable(model, Gen_Status[1:NGen], Int)
    @variable(model, Brh_Status[1:NBrh], Int)
    BigM = 1e12
    dthetaU = abs.(Brh_MAX_ini ./ B_Brh_ini)   
    dthetaL = -dthetaU
    GenMax = 1
    BrhMax = 3
    @constraint(model, Gen_Status .>= Exit_Gen)  
    @constraint(model, Gen_Status .<= GenMax)
    @constraint(model, Brh_Status .>= Exit_Brh)
    @constraint(model, Brh_Status .<= BrhMax)   
    ref_bus = 1
    @variable(model, theta[1:NBus, 1:T])
    @variable(model, dtheta[1:NBrh, 1:T])
    @variable(model, z[1:NBrh, 1:BrhMax], Bin)       
    @variable(model, Wk[1:NBrh, 1:BrhMax, 1:T])       
    @constraint(model, [t=1:T], theta[ref_bus, t] == 0)
    @constraint(model, [i=1:NBrh, t=1:T], dthetaL[i] <= dtheta[i, t] <= dthetaU[i])
    for i = 1:NBrh, t = 1:T, k = 1:BrhMax
        @constraint(model,  Wk[i, k, t] <=  dthetaU[i] * z[i, k])
        @constraint(model,  Wk[i, k, t] >=  dthetaL[i] * z[i, k])
        @constraint(model,  Wk[i, k, t] - dtheta[i, t] <=  dthetaU[i] * (1 - z[i, k]))
        @constraint(model,  Wk[i, k, t] - dtheta[i, t] >=  dthetaL[i] * (1 - z[i, k]))
    end
    for i = 1:NBrh
        @constraint(model, Exit_Brh[i] + sum(z[i, k] for k in 1:BrhMax) == Brh_Status[i])
    end
    for i = 1:NBrh, t = 1:T
        f_node = BusAdjList[i, 1]
        t_node = BusAdjList[i, 2]
        @constraint(model, dtheta[i, t] == theta[f_node, t] - theta[t_node, t])
        @constraint(model, Brh_P[i, t] == B_Brh_ini[i] * (Exit_Brh[i] * dtheta[i, t] + sum(Wk[i, k, t] for k in 1:BrhMax)))
    end
    @constraint(model, Brh_P .>= -(Brh_MAX_ini .* Brh_Status) * ones(1, T))
    @constraint(model, Brh_P .<=  (Brh_MAX_ini .* Brh_Status) * ones(1, T))
    AdjMX = zeros(Float64, NBus, NBrh)
    for i = 1:NBrh
        f_node = BusAdjList[i, 1]
        t_node = BusAdjList[i, 2]
        AdjMX[f_node, i] =  1.0
        AdjMX[t_node, i] = -1.0
    end
    @constraint(model, MX_NBus_NGen * Gen_P + ls_Pos - ls_Neg + MX_NBus_NHVDC * HVDC_P - AdjMX * Brh_P .== P_D)
    @constraint(model, sum(Gen_P, dims=1) + sum(HVDC_P, dims=1) + sum(ls_Pos, dims=1) - sum(ls_Neg, dims=1) .== sum(P_D, dims=1))
    @constraint(model, Gen_P .>= Gen_PMIN_ini .* (Gen_Status * ones(1, T)))
    @constraint(model, Gen_P .<= Gen_PMAX_ini .* (Gen_Status * ones(1, T)))
    @constraint(model, Gen_P * GenDiag .>= Gen_RD .* (Gen_Status * ones(1, T - 1)))
    @constraint(model, Gen_P * GenDiag .<= Gen_RU .* (Gen_Status * ones(1, T - 1)))
    @constraint(model, HVDC_P .>= 0)
    @constraint(model, HVDC_P .<= HVDC_Cap * ones(1, T))
    @constraint(model, HVDC_Cap .>= 0)
    @constraint(model, HVDC_Cap .<= HVDC_Cap_MAX)
    @constraint(model, ls_Neg .>= 0)
    @constraint(model, ls_Pos .>= 0)
    @constraint(model, ls_Pos .<= 0.1 * P_D)
    @variable(model, Gen_B_MX[1:NBus, 1:T])
    @variable(model, Bus_B_MX[1:NBus, 1:NBus, 1:T])
    @variable(model, UC_Status[1:NGen, 1:T], Int)
    @constraint(model, UC_Status .>= 0)
    @constraint(model, UC_Status .<= Gen_Status * ones(1, T))
    @constraint(model, Gen_B_MX == MX_NBus_NGen * (UC_Status .* (B_Gen_ini * ones(1, T))))
    for t = 1:T
        for (f_node, t_node) in BusAdj_complement_pairs
            @constraint(model, Bus_B_MX[f_node, t_node, t] == 0)
            @constraint(model, Bus_B_MX[t_node, f_node, t] == 0)
        end

        for i = 1: NBrh
            f_node = BusAdjList[i, 1]
            t_node = BusAdjList[i, 2]
            @constraint(model, Bus_B_MX[f_node, t_node, t] == -B_Brh_ini[i] * Brh_Status[i])
            @constraint(model, Bus_B_MX[t_node, f_node, t] == -B_Brh_ini[i] * Brh_Status[i])           
        end

        for i = 1:NBus
            @constraint(model, Bus_B_MX[i, i, t] == Gen_B_MX[i, t] - sum(Bus_B_MX[i, j, t] for j = 1:NBus if j != i))
        end
    end
    @variable(model, Bkr[1:NBus, 1:NBus, 1:T])
    @variable(model, SCRMX[1:NBus, 1:NBus, 1:T])
    for t = 1:T
        @constraint(model, Bkr[:, :, t] == MX_NBus_NBus_R * Bus_B_MX[:, :, t] * MX_NBus_NBus_C)
        @constraint(model, SCRMX[:, :, t] == Bkr[:, :, t] - SCR0 * (MXH_NBus_NHVDC * Diagonal(HVDC_Cap) * MXH_NHVDC_NBus))
        @constraint(model, SCRMX[:, :, t] in PSDCone())
    end
    @variable(model, cost_SU[1:NGen, 1:T-1])
    @variable(model, cost_SD[1:NGen, 1:T-1])
    @constraint(model, cost_SU .>= 0)
    @constraint(model, cost_SD .>= 0)
    for i = 1: NGen
        for t = 2: T
            for j = t: min(T, t + Gen_SU_Time[i, 1] - 1)
                @constraint(model, UC_Status[i, j] .>= UC_Status[i, t] - UC_Status[i, t - 1])
            end

            for j = t: min(T, t + Gen_SD_Time[i, 1] - 1)
                @constraint(model, UC_Status[i, j] .<= 1 - (UC_Status[i, t - 1] - UC_Status[i, t]))
            end
        end

        for t = 1: T - 1
            @constraint(model, cost_SU[i, t] .>= Gen_SU_cost[i, 1] * (UC_Status[i, t + 1] - UC_Status[i, t]))
            @constraint(model, cost_SD[i, t] .>= Gen_SD_cost[i, 1] * (UC_Status[i, t] - UC_Status[i, t + 1]))
        end
    end
    Cost_SU = sum(cost_SU)
    Cost_SD = sum(cost_SD)
    Cost_GP = baseMVA * sum(Gen_P_cost .* Gen_P)
    Cost_HP = baseMVA * sum(HVDC_P_cost .* HVDC_P)
    Cost_LSP = baseMVA * sum(ls_Pos .* P_LS_cost)
    Cost_LSN = baseMVA * sum(ls_Neg .* P_LS_cost)
    Cost_HCap = baseMVA * sum(HVDC_Cap_cost .* HVDC_Cap)
    Cost_AGen = sum(Add_Gen_Cost .* (Gen_Status .- Exit_Gen))
    Cost_ABrh = sum(Add_Brh_Cost .* (Brh_Status .- Exit_Brh))
    obj = D * (Cost_GP + Cost_HP + Cost_LSP + Cost_LSN + Cost_SU + Cost_SD) + Cost_HCap + Cost_AGen + Cost_ABrh
    @objective(model, Min, obj)
    optimize!(model)
    println(objective_value(model))
    println(sum(value.(ls_Pos)))
    println(sum(value.(ls_Neg)))
    println(sum(value.(Gen_P)))
    println(sum(value.(HVDC_P)))
    println(sum(value.(P_D)))
    println(sum(value.(Gen_Status)))
    println(sum(value.(Brh_Status)))
    println(value.(Gen_Status))
    println(value.(Brh_Status))
    println("================================================================")
    println(D * value.(Cost_SU))
    println(D * value.(Cost_SD))
    println(D * value.(Cost_GP))
    println(D * value.(Cost_HP))
    println(D * value.(Cost_LSP))
    println(D * value.(Cost_LSN))
    println(value.(Cost_HCap))
    println(value.(Cost_AGen))
    println(value.(Cost_ABrh))
    return 0

end
