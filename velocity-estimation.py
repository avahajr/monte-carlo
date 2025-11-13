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


if __name__ == "__main__":
    num_simulations = 100000
    history = get_historical_data()
    print(sum(history.values()))
    simulation = np.random.choice(list(history.values()), size=(num_simulations, 9))
    results = np.sum(simulation, axis=1)

    print('first simulation', simulation[0])
    print(results[0])
    plot_results(results)



