# models/ato_model.py

import gurobipy as gp
from gurobipy import GRB

def solve_ato(demands, probabilities, C, P, T, L, G, verbose=False):
    """
    Risolve il problema Assemble-To-Order con domanda stocastica.

    Args:
        demands: lista di vettori d_j^{(s)} (es: [[100, 50], [90, 60], ...])
        probabilities: lista π_s
        C: costi componenti (es: [1, 1, 3])
        P: prezzi prodotti finali (es: [6, 8.5])
        T: tempo produzione componenti per macchina (es: [0.5, 0.25, 0.25])
        L: disponibilità macchina (es: 6.0)
        G: matrice gozinto G_ij (es: [[1,1], [1,1], [0,1]])
        verbose: True per log gurobi

    Returns:
        dict con 'x', 'y', 'objective'
    """
    n_scenarios = len(demands)
    I = len(C)  # componenti
    J = len(P)  # prodotti

    model = gp.Model("ATO")
    if not verbose:
        model.setParam("OutputFlag", 0)

    # Variabili di primo stadio
    x = model.addVars(I, vtype=GRB.CONTINUOUS, name="x")

    # Variabili di secondo stadio
    y = model.addVars(n_scenarios, J, vtype=GRB.CONTINUOUS, name="y")

    # Obiettivo
    expected_revenue = gp.quicksum(probabilities[s] * gp.quicksum(P[j] * y[s, j] for j in range(J)) for s in range(n_scenarios))
    total_cost = gp.quicksum(C[i] * x[i] for i in range(I))
    model.setObjective(expected_revenue - total_cost, GRB.MAXIMIZE)

    # Vincoli macchina
    model.addConstr(gp.quicksum(T[i] * x[i] for i in range(I)) <= L, name="capacity")

    # Vincoli su ogni scenario
    for s in range(n_scenarios):
        for j in range(J):
            model.addConstr(y[s, j] <= demands[s][j], name=f"demand_s{s}_j{j}")
        for i in range(I):
            model.addConstr(gp.quicksum(G[i][j] * y[s, j] for j in range(J)) <= x[i], name=f"gozinto_s{s}_i{i}")

    model.optimize()

    return {
        "x": [x[i].X for i in range(I)],
        "y": [[y[s, j].X for j in range(J)] for s in range(n_scenarios)],
        "objective": model.ObjVal,
        "model": model
    }
