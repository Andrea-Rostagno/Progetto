# models/newsvendor_model.py
import gurobipy as gp
from gurobipy import GRB

def solve_newsvendor(demands, probabilities, cost=1, selling_price=10, verbose=False):
    """
    Solves the newsvendor problem given demand scenarios and probabilities.

    Parameters:
        demands (list of int): possible demand values
        probabilities (list of float): associated probabilities
        cost (float): unit cost
        selling_price (float): unit selling price

    Returns:
        dict: {'x_opt': ..., 'objective': ..., 'model': m}
    """
    m = gp.Model("newsvendor")
    if not verbose:
        m.setParam("OutputFlag", 0)

    n_scenarios = len(demands)
    scenarios = range(n_scenarios)

    x = m.addVar(vtype=GRB.INTEGER, lb=0, name="X")  # number of newspapers to buy
    y = m.addVars(n_scenarios, vtype=GRB.INTEGER, lb=0, name="Y")  # newspapers sold per scenario

    expected_profit = sum(probabilities[s] * y[s] for s in scenarios)
    m.setObjective(selling_price * expected_profit - cost * x, GRB.MAXIMIZE)

    for s in scenarios:
        m.addConstr(y[s] <= x)
        m.addConstr(y[s] <= demands[s])

    m.optimize()

    return {
        'x_opt': x.X,
        'objective': m.ObjVal,
        'model': m
    }
