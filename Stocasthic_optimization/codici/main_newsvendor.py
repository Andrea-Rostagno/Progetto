import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
from scenario_tree import *
from models.newsvendor_model import solve_newsvendor


class EasyStochasticModel(StochModel):
    def __init__(self, sim_setting):
        self.averages = sim_setting['averages']
        self.dim_obs = len(sim_setting['averages'])
        self.cov_matrix = np.diag(sim_setting.get("variances", [400]))

        np.random.seed(sim_setting.get("seed", 42))

    def simulate_one_time_step(self, parent_node, n_children):
        probs = np.ones(n_children)/n_children
        obs = np.random.multivariate_normal(
            mean=self.averages,
            cov=self.cov_matrix,
            size=n_children
        ).T # obs.shape = (len_vector, n_children)
        return probs, obs 

sim_setting = {
    'averages': [25] * 20,  # 20 scenari iniziali
    'variances': [200] * 20,
    'seed': 123
}

easy_model = EasyStochasticModel(sim_setting)

scen_tree = ScenarioTree(
    name="std_MC_newsvendor_tree",
    branching_factors=[50], #massimo 50 se no scoppia
    len_vector=20,
    initial_value=[0],
    stoch_model=easy_model,
)

scen_tree.plot()

################################################################################################

# --- Parametri iniziali 20 scenari---
n_sets = 20           # numero di set/scenari paralleli
n_scenarios = 50      # numero di scenari per ogni set
confidence_level = 0.95
width = 10.0

results = []

for j in range(n_sets):
    demands = []
    probs = []
    for node_id in scen_tree.leaves:
        node = scen_tree.nodes[node_id]
        demand = float(node['obs'][j])         # prendi il j-esimo scenario!
        demand = max(0, round(demand))         # valori interi e >= 0
        demands.append(demand)
        probs.append(node['path_prob'])

    demands_agg, probs_agg = scen_tree.aggregate_discrete_demands(demands, probs)
    
    # print(f"\n--- SET {j+1} ---")
    # print("Domande distinte e probabilità:")
    # for d, p in zip(demands_agg, probs_agg):
    #     print(f"d = {d}, π = {p}")
    # print(f"Somma totale delle probabilità: {sum(probs_agg):.2f}")

    result = solve_newsvendor(demands_agg, probs_agg)
    
    # salva risultati
    results.append(result['objective'])

    # print("📦 Quantità ottimale di giornali da ordinare:", int(result['x_opt']))
    # print("💰 Profitto atteso massimo:", f"{result['objective']:.2f}€")
    # print(f"⏱️ Tempo ottimizzazione: {end-start:.4f} s")

# ---- Alla fine: statistiche su tutti i set ----s
results = np.array(results)
print("\n===========================")
print(f"Statistiche sui 20 set:")
print(f"Media profitto atteso: {np.mean(results):.2f}€")
print(f"Deviazione standard:  {np.std(results):.2f}€")
print("===========================")

z = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for 95% confidence interval
lower_bound = np.mean(results) - z * np.std(results) / np.sqrt(n_sets)
upper_bound = np.mean(results) + z * np.std(results) / np.sqrt(n_sets)

# Display the results
print(f"Estimated profit: {np.mean(results):.2f}€")
print(f"95% confidence interval: ({lower_bound:.2f}, {upper_bound:.2f})")
actual_width = upper_bound - lower_bound
print(f"actual_width: {actual_width:.2f}")

#####################################################################

# --- Riduzione degli scenari per avere un intervallo di confidenza di 10€ ---
new_num_set =  int((np.std(results) * 2 * z/ width)**2)
print(f"new_num_set: {new_num_set}")

sim_setting = {
    'averages': [25] * new_num_set,
    'variances': [200] * new_num_set,
    'seed': 123
}

easy_model = EasyStochasticModel(sim_setting)

scen_tree = ScenarioTree(
    name="std_MC_newsvendor_tree",
    branching_factors=[50], #massimo 50 se no scoppia
    len_vector=new_num_set,
    initial_value=[0],
    stoch_model=easy_model,
)

timing_results = {}
results = []
times = []

for j in range(new_num_set):
    demands = []
    probs = []
    for node_id in scen_tree.leaves:
        node = scen_tree.nodes[node_id]
        demand = float(node['obs'][j])         # prendi il j-esimo scenario!
        demand = max(0, round(demand))         # valori interi e >= 0
        demands.append(demand)
        probs.append(node['path_prob'])

    demands_agg, probs_agg = scen_tree.aggregate_discrete_demands(demands, probs)
    
    # print(f"\n--- SET {j+1} ---")
    # print("Domande distinte e probabilità:")
    # for d, p in zip(demands_agg, probs_agg):
    #     print(f"d = {d}, π = {p}")
    # print(f"Somma totale delle probabilità: {sum(probs_agg):.2f}")

    start = time.perf_counter()
    result = solve_newsvendor(demands_agg, probs_agg)
    end = time.perf_counter()
    times.append(end - start)
    
    # salva risultati
    results.append(result['objective'])

    # print("📦 Quantità ottimale di giornali da ordinare:", int(result['x_opt']))
    # print("💰 Profitto atteso massimo:", f"{result['objective']:.2f}€")
    # print(f"⏱️ Tempo ottimizzazione: {end-start:.4f} s")

# ---- Alla fine: statistiche su tutti i set ----s
results = np.array(results)
mean_time = np.mean(times)
timing_results['Full_Solution'] = mean_time
print("\n===========================")
print(f"Statistiche sui nuovi " f"{new_num_set} set:")
print(f"Media profitto atteso: {np.mean(results):.2f}€")
print(f"Deviazione standard:  {np.std(results):.2f}€")
print("===========================")

lower_bound = np.mean(results) - z * np.std(results) / np.sqrt(new_num_set)
upper_bound = np.mean(results) + z * np.std(results) / np.sqrt(new_num_set)

# Display the results
print(f"Estimated profit: {np.mean(results):.2f}€")
print(f"95% confidence interval: ({lower_bound:.2f}, {upper_bound:.2f})")
actual_width = upper_bound - lower_bound
print(f"actual_width: {actual_width:.2f}")

##################################################################################################

k_min = 1
k_max = 15
all_means, all_stds, all_times_red, all_times_solve = [], [], [], []
sse = np.zeros((k_max, new_num_set))
print("\n===========================")
print(f"Riduzione degli scenari via KMeans (k={k_min}-{k_max})")

for k in range(k_min, k_max+1):
    profits_k = []
    times_k = []
    times_k_solve = []
    
    for j in range(new_num_set):
        demands = []
        probs = []

        for node_id in scen_tree.leaves:
            node = scen_tree.nodes[node_id]
            demand = float(node['obs'][j])
            demand = max(0, round(demand))
            demands.append(demand)
            probs.append(node['path_prob'])

        demands_agg, probs_agg = scen_tree.aggregate_discrete_demands(demands, probs)

        start = time.perf_counter()
        demands_reduced, probs_reduced, sse_kj = scen_tree.reduce_scenarios_kmeans_1D(demands_agg, probs_agg, k=k)
        end = time.perf_counter()
        times_k.append(end - start)
        sse[k-1, j] = sse_kj # estraggo il valore dell'SSE per la clusterizzazione a k scenari, del j-esimo campione 
        
        # Stampa scenari ridotti 
        # print(f"\n📉 Scenari ridotti via Clustering KMeans (set {j+1}, k={k}):")
        # for d, p in zip(demands_reduced, probs_reduced):
        #     print(f"d = {d}, π = {p:.2f}")
        # print(f"🔎 Somma delle probabilità: {sum(probs_reduced):.2f}")

        start = time.perf_counter()
        result = solve_newsvendor(demands_reduced, probs_reduced, verbose=False)
        end = time.perf_counter()
        times_k_solve.append(end - start)
        profits_k.append(result['objective'])

    profit_mean = np.mean(profits_k)
    profit_std = np.std(profits_k)    
    red_time_mean = np.mean(times_k)
    solve_time_mean = np.mean(times_k_solve)
    all_means.append(profit_mean)
    all_stds.append(profit_std)
    all_times_red.append(red_time_mean)
    all_times_solve.append(solve_time_mean)
    print(f"k={k:2d} | profitto atteso = {profit_mean:8.2f}€, std = {profit_std:6.2f}€, "
          f"tempo riduzione = {red_time_mean:.4f}s, tempo soluzione = {solve_time_mean:.4f}s")
    
    timing_results[f'KMeans_Reduction_k{k}'] = red_time_mean
    timing_results[f'KMeans_Solution_k{k}'] = solve_time_mean


# Traccio il grafico dell'SSE per ciascun campione
k_values = np.array(range(1,16))
for i in range(n_sets):
	plt.plot(k_values,sse[:,i])
plt.show()
# Traccio il grafico dei profitti attesi medi e deviazioni standard
plt.errorbar(range(k_min, k_max+1), all_means, yerr=all_stds, fmt='-o')
plt.xlabel('Numero di cluster k')
plt.ylabel('Profitto atteso medio (€)')
plt.show()

###################################################################################################

# --- Riduzione degli scenari via Wasserstein ---
k_min = 1
k_max = 15
all_means, all_stds, all_times_red, all_times_solve = [], [], [], []
print("\n===========================")
print(f"Riduzione degli scenari via Wasserstein (k={k_min}-{k_max})")

for k in range(k_min, k_max+1):
    profits_k = []
    times_k = []
    times_k_solve = []
    
    for j in range(new_num_set):
        demands = []
        probs = []

        for node_id in scen_tree.leaves:
            node = scen_tree.nodes[node_id]
            demand = float(node['obs'][j])
            demand = max(0, round(demand))
            demands.append(demand)
            probs.append(node['path_prob'])

        demands_agg, probs_agg = scen_tree.aggregate_discrete_demands(demands, probs)

        start = time.perf_counter()
        demands_reduced, probs_reduced = scen_tree.reduce_scenarios_wasserstein_1D(demands_agg, probs_agg, k=k)
        end = time.perf_counter()
        times_k.append(end - start)
        
        # Stampa scenari ridotti 
        # print(f"\n📉 Scenari ridotti via Clustering KMeans (set {j+1}, k={k}):")
        # for d, p in zip(demands_reduced, probs_reduced):
        #     print(f"d = {d}, π = {p:.2f}")
        # print(f"🔎 Somma delle probabilità: {sum(probs_reduced):.2f}")

        start = time.perf_counter()
        result = solve_newsvendor(demands_reduced, probs_reduced, verbose=False)
        end = time.perf_counter()
        times_k_solve.append(end - start)
        profits_k.append(result['objective'])

    profit_mean = np.mean(profits_k)
    profit_std = np.std(profits_k)    
    red_time_mean = np.mean(times_k)
    solve_time_mean = np.mean(times_k_solve)
    all_means.append(profit_mean)
    all_stds.append(profit_std)
    all_times_red.append(red_time_mean)
    all_times_solve.append(solve_time_mean)
    print(f"k={k:2d} | profitto atteso = {profit_mean:8.2f}€, std = {profit_std:6.2f}€, "
          f"tempo riduzione = {red_time_mean:.4f}s, tempo soluzione = {solve_time_mean:.4f}s")
    
    timing_results[f'Wass_Reduction_k{k}'] = red_time_mean
    timing_results[f'Wass_Solution_k{k}'] = solve_time_mean


# Traccio il grafico dei profitti attesi medi e deviazioni standard
plt.errorbar(range(k_min, k_max+1), all_means, yerr=all_stds, fmt='-o')
plt.xlabel('Numero di cluster k')
plt.ylabel('Profitto atteso medio (€)')
plt.show()


###################################################################################################

# Stampa i tempi di esecuzione
labels = list(timing_results.keys())
values = list(timing_results.values())

plt.figure(figsize=(10, 5))
plt.bar(labels, values, color='skyblue')
plt.ylabel("Tempo [s]")
plt.title("Confronto tempistiche riduzione e soluzione Newsvendor")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

df_time = pd.DataFrame(list(timing_results.items()), columns=['Operazione', 'Tempo [s]'])
print(df_time.to_string(index=False))
	