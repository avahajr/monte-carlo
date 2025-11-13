import numpy as np
import matplotlib.pyplot as plt

def get_historical_data() -> dict[str, int]:
    # It's better to get this data from Jira, but I hand-counted
    return {
        '09/24/2025':3,
        '09/25/2025':0,
        '09/26/2025':0,
        '09/29/2025':2,
        '10/01/2025':3,
        '10/02/2025':2,
        '10/03/2025':8,
        '10/06/2025':16,
        '10/07/2025':6,
    }

def plot_results(results: np.ndarray, save_path="velocity_estimation_results.png"):
    plt.hist(results, bins=len(results)//2500, color="#4C72B0", edgecolor="black")
    plt.title("Velocity Estimation MCS results")
    plt.xlabel("Points completed")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_with_risk(results: np.ndarray, save_path="velocity_estimation_results_with_risk.png"):
    bins = max(10, len(results) // 2500)
    fig, ax = plt.subplots()
    counts, bin_edges, _ = ax.hist(results, bins=bins, color="#4C72B0", edgecolor="black")
    risk_levels = {5: "Low", 50: "Medium", 95: "High"}
    vals = np.percentile(results, list(risk_levels.keys()))

    ymax = counts.max() if counts.size else 1
    for ((pct, risk), val) in zip(risk_levels.items(), vals):
        ax.axvline(val, color="lawngreen", linestyle="--", linewidth=2, alpha=0.9)
        ax.text(val, ymax * 0.92, f"{pct}% ≤ {val:.0f}\n({risk} risk)", color="darkgreen",
                ha="center", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    ax.set_title("Velocity Estimation MCS results (with risk percentiles)")
    ax.set_xlabel("Points completed")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    num_simulations = 100000
    history = get_historical_data()
    print(sum(history.values()))
    simulation = np.random.choice(list(history.values()), size=(num_simulations, 9))
    results = np.sum(simulation, axis=1)

    print('first simulation', simulation[0])
    print(results[0])
    plot_results(results)
    plot_with_risk(results)



