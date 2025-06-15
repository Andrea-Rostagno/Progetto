import numpy as np
import pandas as pd
import scipy.stats as stats
import time
import matplotlib.pyplot as plt
from scenario_tree import *
from models.ato_model import solve_ato

n_scenarios = 40      # numero di scenari per ogni set
n_sets = 20           # numero di set/scenari paralleli

class EasyStochasticModel(StochModel):
    def __init__(self, sim_setting):
        self.averages = sim_setting['averages']
        self.dim_obs = len(sim_setting['averages'])
        self.cov_matrix = np.diag(sim_setting.get("variances", [100, 225]))

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
    'averages': [10, 16] * n_sets,
    'variances': [70, 90] * n_sets,
    'seed': 123
}
easy_model = EasyStochasticModel(sim_setting)
scen_tree = ScenarioTree(
    name="std_MC_ato_tree",
    branching_factors=[n_scenarios],
    len_vector=40,
    initial_value=[0, 0],
    stoch_model=easy_model,
)

scen_tree.plot() 

#################################################################################################

# Simulazione dello scenario
confidence_level = 0.95
width = 1.0

results = []
timing_results = {}

# Parametri ATO (slide pizzaiolo)
C = [3, 2, 2]         # costi componenti (aumentali)
P = [7, 10]            # prezzi prodotti (margine più basso)
T = [0.5, 0.25, 0.25]
L = 8                # ore disponibili (più rilassato)
G = [
    [1, 1],
    [1, 1],
    [0, 1]
]

for j in range(n_sets):
    demands = []
    probs = []
    for node_id in scen_tree.leaves:
        node = scen_tree.nodes[node_id]
        d1 = max(0, round(node['obs'][j]))        # Margherita (j-esimo set)
        d2 = max(0, round(node['obs'][(j+1)%n_sets])) # 4 Stagioni (shift per esempio; personalizza secondo come hai generato i dati)
        demands.append([d1, d2])
        probs.append(node['path_prob'])

    # Aggregazione dei vettori domanda/probabilità
    demands_agg, probs_agg = scen_tree.aggregate_vectorial_demands(demands, probs)

    # print(f"\n--- SET {j+1} ---")
    # print("Domande distinte (ordinate) e probabilità:")
    # for d, p in zip(demands_agg, probs_agg):
    #     print(f"d = {d}, π = {p}")
    # print(f"Somma totale delle probabilità: {sum(probs_agg):.2f}")

    # Risoluzione ATO e timing
    result = solve_ato(
        demands_agg,
        probs_agg,
        C=C,
        P=P,
        T=T,
        L=L,
        G=G,
        verbose=False
    )
    results.append(result['objective'])

    # print("\n🔧 Quantità ottimali di componenti da produrre:")
    # for i, q in enumerate(result['x']):
    #     print(f"  Componente {i}: {q:.2f}")

    # print(f"\n📦 Obiettivo massimo (ricavo atteso - costo): {result['objective']:.2f}€")

# Statistiche finali su tutti i set
results = np.array(results)
print("\n===========================")
print(f"Statistiche sui 20 set:")
print(f"Media ricavo atteso: {np.mean(results):.2f}€")
print(f"Deviazione standard: {np.std(results):.2f}€")
print("===========================")

z = stats.norm.ppf((1 + confidence_level) / 2)  # Z-score for 95% confidence interval
lower_bound = np.mean(results) - z * np.std(results) / np.sqrt(n_sets)
upper_bound = np.mean(results) + z * np.std(results) / np.sqrt(n_sets)

# Display the results
print(f"Estimated profit: {np.mean(results):.2f}€")
print(f"95% confidence interval: ({lower_bound:.2f}, {upper_bound:.2f})")
actual_width = upper_bound - lower_bound
print(f"actual_width: {actual_width:.2f}")

###################################################################################################

# --- Riduzione degli scenari per avere un intervallo di confidenza di 10€ ---
new_num_set =  int((np.std(results) * 2 * z/ width)**2)
print(f"new_num_set: {new_num_set}")

sim_setting = {
    'averages': [10, 16] * new_num_set,
    'variances': [70, 90] * new_num_set,
    'seed': 123
}

easy_model = EasyStochasticModel(sim_setting)

scen_tree = ScenarioTree(
    name="std_MC_ato_tree",
    branching_factors=[n_scenarios], #massimo 50 se no scoppia
    len_vector=new_num_set,
    initial_value=[0, 0],
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
        d1 = max(0, round(node['obs'][j]))        # Margherita (j-esimo set)
        d2 = max(0, round(node['obs'][(j+1)%new_num_set])) # 4 Stagioni (shift per esempio; personalizza secondo come hai generato i dati)
        demands.append([d1, d2])
        probs.append(node['path_prob'])

    demands_agg, probs_agg = scen_tree.aggregate_vectorial_demands(demands, probs) 

    # print(f"\n--- SET {j+1} ---")
    # print("Domande distinte (ordinate) e probabilità:")
    # for d, p in zip(demands_agg, probs_agg):
    #     print(f"d = {d}, π = {p}")
    # print(f"Somma totale delle probabilità: {sum(probs_agg):.2f}") 

    start = time.perf_counter()
    result = solve_ato(
        demands_agg,
        probs_agg,
        C=C,
        P=P,
        T=T,
        L=L,
        G=G,
        verbose=False
    )
    end = time.perf_counter()
    times.append(end - start)
    
    # salva risultati
    results.append(result['objective'])

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

# Riduci gli scenari con KMeans

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
            d1 = max(0, round(node['obs'][j]))        # Margherita (j-esimo set)
            d2 = max(0, round(node['obs'][(j+1)%new_num_set])) # 4 Stagioni (shift per esempio; )
            demands.append([d1, d2])
            probs.append(node['path_prob'])

        # Aggregazione dei vettori domanda/probabilità
        demands_agg, probs_agg = scen_tree.aggregate_vectorial_demands(demands, probs)

        start = time.perf_counter()
        demands_reduced, probs_reduced, sse_kj = scen_tree.reduce_scenarios_kmeans_multiD(demands_agg, probs_agg, k=k)
        end = time.perf_counter()
        times_k.append(end - start)
        sse[k-1, j] = sse_kj # estraggo il valore dell'SSE per la clusterizzazione a k scenari, del j-esimo campione 

        start = time.perf_counter()
        result = solve_ato(
            demands=demands_reduced,
            probabilities=probs_reduced,
            C=C,
            P=P,
            T=T,
            L=L,
            G=G,
            verbose=False
        )
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
    
    timing_results[f'Kmeans_Reduction_k{k}'] = red_time_mean
    timing_results[f'Kmeans_Solution_k{k}'] = solve_time_mean


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
            d1 = max(0, round(node['obs'][j]))          # domanda Margherita, set j
            d2 = max(0, round(node['obs'][(j+1)%new_num_set]))  # domanda 4 Stagioni, set j+1 (personalizza pairing se necessario)
            demands.append([d1, d2])
            probs.append(node['path_prob'])

        demands_agg, probs_agg = scen_tree.aggregate_vectorial_demands(demands, probs)

        # print(f"\n--- SET {j+1} ---")
        # print("Domande distinte (ordinate) e probabilità was:")
        # for d, p in zip(demands_agg, probs_agg):
        #     print(f"d = {d}, π = {p}")
        # print(f"Somma totale delle probabilità: {sum(probs_agg):.2f}")
        # print(f"Numero di scenari aggregati: {len(demands_agg)}")

        start = time.perf_counter()
        demands_reduced, probs_reduced = scen_tree.reduce_scenarios_wasserstein_multiD(
            X  = np.array(demands_agg),
            mu = np.array(probs_agg),
            k  = k
        )
        end = time.perf_counter()
        times_k.append(end - start)
        
        start = time.perf_counter()
        result = solve_ato(
            demands=demands_reduced,
            probabilities=probs_reduced,
            C=C,
            P=P,
            T=T,
            L=L,
            G=G,
            verbose=False
        )
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
plt.title("Confronto tempistiche riduzione e soluzione ATO")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

df_time = pd.DataFrame(list(timing_results.items()), columns=['Operazione', 'Tempo [s]'])
print(df_time.to_string(index=False))
