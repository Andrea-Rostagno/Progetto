import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scenario_tree import *
from models.ato_model import solve_ato


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
    'averages': [90, 200] * 20,
    'variances': [100, 2000] * 20,
    'seed': 123
}
easy_model = EasyStochasticModel(sim_setting)
scen_tree = ScenarioTree(
    name="std_MC_ato_tree",
    branching_factors=[50],
    len_vector=40,
    initial_value=[0, 0],
    stoch_model=easy_model,
)

scen_tree.plot() 

#################################################################################################

# Simulazione dello scenario
n_sets = 20           # numero di set/scenari paralleli
n_scenarios = 50      # numero di scenari per ogni set
timing_results = {}

# Parametri ATO (slide pizzaiolo)
C = [1, 1, 3]           # costi componenti: impasto, pomodoro, verdure
P = [6, 8.5]            # prezzi prodotti: Margherita, 4 Stagioni
T = [0.5, 0.25, 0.25]   # tempi produzione per macchina (in ore)
L = 6.0                 # ore disponibili
G = [
    [1, 1],  # impasto per entrambi
    [1, 1],  # pomodoro per entrambi
    [0, 1]   # verdure solo per 4 Stagioni
]

results = []
times = []

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

    print(f"\n--- SET {j+1} ---")
    print("Domande distinte (ordinate) e probabilità:")
    for d, p in zip(demands_agg, probs_agg):
        print(f"d = {d}, π = {p}")
    print(f"Somma totale delle probabilità: {sum(probs_agg):.2f}")

    # Risoluzione ATO e timing
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
    results.append(result['objective'])

    print("\n🔧 Quantità ottimali di componenti da produrre:")
    for i, q in enumerate(result['x']):
        print(f"  Componente {i}: {q:.2f}")

    print(f"\n📦 Obiettivo massimo (ricavo atteso - costo): {result['objective']:.2f}€")
    print(f"⏱️ Tempo ottimizzazione: {end-start:.4f} s")

# Statistiche finali su tutti i set
results = np.array(results)
mean_time = np.mean(times)
timing_results['Full_Solution'] = mean_time
print("\n===========================")
print(f"Statistiche sui 20 set:")
print(f"Media ricavo atteso: {np.mean(results):.2f}€")
print(f"Deviazione standard: {np.std(results):.2f}€")
print(f"Tempo medio ottimizzazione: {mean_time:.4f} s")
print("===========================")

###################################################################################################

# Riduci gli scenari con KMeans
n_sets = 20
k_min = 1
k_max = 15
set_mean_profits = []
set_mean_times = []
set_mean_times_solve = []
sse = np.zeros((k_max, n_sets))

# Parametri ATO (slide pizzaiolo)
C = [1, 1, 3]
P = [6, 8.5]
T = [0.5, 0.25, 0.25]
L = 6.0
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
        d1 = max(0, round(node['obs'][j]))          # domanda Margherita, set j
        d2 = max(0, round(node['obs'][(j+1)%n_sets]))  # domanda 4 Stagioni, set j+1 (personalizza se vuoi cambiare pairing!)
        demands.append([d1, d2])
        probs.append(node['path_prob'])

    demands_agg, probs_agg = scen_tree.aggregate_vectorial_demands(demands, probs)

    profits_k = []
    times_k = []
    times_k_solve = []
    for k in range(k_min, k_max+1):
        start = time.perf_counter()
        demands_reduced, probs_reduced, sse_kj = scen_tree.reduce_scenarios_kmeans_multiD(demands_agg, probs_agg, k=k)
        end = time.perf_counter()
        times_k.append(end - start)
        sse[k-1, j] = sse_kj # estraggo il valore dell'SSE per la clusterizzazione a k scenari, del j-esimo campione

        print(f"\n📉 Scenari ATO ridotti via Clustering KMeans (set {j+1}, k={k}):")
        for d, p in zip(demands_reduced, probs_reduced):
            print(f"d = {d}, π = {p:.2f}")
        print(f"🔎 Somma delle probabilità: {sum(probs_reduced):.2f}")

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

    set_mean = np.mean(profits_k)
    set_mean_time = np.mean(times_k)
    set_mean_time_solve = np.mean(times_k_solve)
    set_mean_profits.append(set_mean)
    set_mean_times.append(set_mean_time)
    set_mean_times_solve.append(set_mean_time_solve)
    print(f"\nSet {j+1:2d}: ricavo atteso medio (K={k_min}-{k_max}) = {set_mean:.2f}€, tempo medio riduzione = {set_mean_time:.4f}s")
    print(f"Tempo medio soluzione (K={k_min}-{k_max}): {set_mean_time_solve:.4f} s")

# Traccio il grafico dell'SSE per ciascun campione
k_values = np.array(range(1,16))
for i in range(n_sets):
	plt.plot(k_values,sse[:,i])
plt.show()
# Statistiche finali sulle 20 medie
set_mean_profits = np.array(set_mean_profits)
set_mean_times = np.array(set_mean_times)
set_mean_times_solve = np.array(set_mean_times_solve)
mean_overall = np.mean(set_mean_profits)
std_overall = np.std(set_mean_profits)
mean_time_overall = np.mean(set_mean_times)
mean_time_solve_overall = np.mean(set_mean_times_solve)

print("\n============================")
print(f"Media dei ricavi medi sui 20 set: {mean_overall:.2f}€")
print(f"Deviazione standard dei ricavi medi: {std_overall:.2f}€")
print(f"Tempo medio di riduzione scenari (sui 20 set): {mean_time_overall:.4f} s")
print(f"Tempo medio di soluzione (sui 20 set): {mean_time_solve_overall:.4f} s")
print("============================")

timing_results['KMeans_Reduction'] = mean_time_overall
timing_results['KMeans_Solution'] = mean_time_solve_overall

###################################################################################################

# Riduzione via Wasserstein
n_sets = 20
k_min = 5
k_max = 15
set_mean_profits = []
set_mean_times = []
set_mean_times_solve = []

# Parametri ATO (slide pizzaiolo)
C = [1, 1, 3]
P = [6, 8.5]
T = [0.5, 0.25, 0.25]
L = 6.0
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
        d1 = max(0, round(node['obs'][j]))          # domanda Margherita, set j
        d2 = max(0, round(node['obs'][(j+1)%n_sets]))  # domanda 4 Stagioni, set j+1 (personalizza pairing se necessario)
        demands.append([d1, d2])
        probs.append(node['path_prob'])

    demands_agg, probs_agg = scen_tree.aggregate_vectorial_demands(demands, probs)

    profits_k = []
    times_k = []
    times_k_solve = []
    for k in range(k_min, k_max+1):
        start = time.perf_counter()
        demands_reduced, probs_reduced = scen_tree.reduce_scenarios_wasserstein_multiD(
            X  = np.array(demands_agg),
            mu = np.array(probs_agg),
            k  = k
        )
        end = time.perf_counter()
        times_k.append(end - start)

        print(f"\n📉 Scenari ATO ridotti via Wasserstein (set {j+1}, k={k}):")
        for d, p in zip(demands_reduced, probs_reduced):
            print(f"d = {d}, π = {p:.2f}")
        print(f"🔎 Somma delle probabilità: {sum(probs_reduced):.2f}")

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

    set_mean = np.mean(profits_k)
    set_mean_time = np.mean(times_k)
    set_mean_time_solve = np.mean(times_k_solve)
    set_mean_profits.append(set_mean)
    set_mean_times.append(set_mean_time)
    set_mean_times_solve.append(set_mean_time_solve)
    print(f"\nSet {j+1:2d}: ricavo atteso medio (K={k_min}-{k_max}) = {set_mean:.2f}€, tempo medio riduzione = {set_mean_time:.4f}s")
    print(f"Tempo medio soluzione (K={k_min}-{k_max}): {set_mean_time_solve:.4f} s")

# Statistiche finali sulle 20 medie
set_mean_profits = np.array(set_mean_profits)
set_mean_times = np.array(set_mean_times)
set_mean_times_solve = np.array(set_mean_times_solve)
mean_overall = np.mean(set_mean_profits)
std_overall = np.std(set_mean_profits)
mean_time_overall = np.mean(set_mean_times)
mean_time_solve_overall = np.mean(set_mean_times_solve)

print("\n============================")
print(f"Media dei ricavi medi sui 20 set: {mean_overall:.2f}€")
print(f"Deviazione standard dei ricavi medi: {std_overall:.2f}€")
print(f"Tempo medio di riduzione scenari (sui 20 set): {mean_time_overall:.4f} s")
print(f"Tempo medio di soluzione (sui 20 set): {mean_time_solve_overall:.4f} s")
print("============================")

timing_results['Wasserstein_Reduction'] = mean_time_overall
timing_results['Wasserstein_Solution'] = mean_time_solve_overall

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
