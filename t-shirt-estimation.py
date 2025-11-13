import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_story_durations(path: str = 'jira-data.csv'):
    df = pd.read_csv(path)
    df.drop("Issue id", axis='columns', inplace=True)
    df["Actual Start Date"] = pd.to_datetime(df["Actual Start Date"])
    df["Resolved"] = pd.to_datetime(df["Resolved"])

    df["Days taken"] = (df["Resolved"] - df["Actual Start Date"]).dt.days
    return df


def plot_historical_data(df: pd.DataFrame, save_path="story_durations.png"):
    plt.figure(figsize=(12, 6))
    plt.hist(df["Days taken"], bins=range(0, df["Days taken"].max() + 2), color='darkred', edgecolor="black", align='left')
    plt.title("Story Durations over 90 days")
    plt.xlabel("Days taken")
    plt.ylabel("Number of Stories")
    plt.xticks(range(0, df["Days taken"].max() + 2))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_results(results: np.ndarray, stories_in_epic: int, save_path="t-shirt-estimation_results.png"):
    plt.hist(results, color='darkred', bins=len(results) // 2500, edgecolor="black")
    plt.title(f"T-shirt sizing MCS results ({stories_in_epic} stories in epic)")
    plt.xlabel("Days needed")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_with_risk(results: np.ndarray, save_path="t_shirt_estimation_results_with_risk.png"):
    bins = max(10, len(results) // 2500)
    fig, ax = plt.subplots()
    counts, bin_edges, _ = ax.hist(results, bins=bins, color="darkred", edgecolor="black")
    risk_levels = {5: "Low", 50: "Medium", 95: "High"}
    vals = np.percentile(results, list(risk_levels.keys()))

    ymax = counts.max() if counts.size else 1
    for ((pct, risk), val) in zip(risk_levels.items(), vals):
        ax.axvline(val, color="lawngreen", linestyle="--", linewidth=2, alpha=0.9)
        ax.text(val, ymax * 0.92, f"{pct}% ≤ {val:.0f}\n({risk} risk)", color="darkgreen",
                ha="center", va="top", fontsize=9, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    ax.set_title("T-shirt Estimation MCS results (with risk percentiles)")
    ax.set_xlabel("Days needed")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    num_simulations = 100000
    stories_in_epic = 30
    historical_data = get_story_durations()

    historical_data[["Issue key", "Days taken"]].to_csv('story-durations.csv', index=False)
    plot_historical_data(historical_data)

    simulations = np.random.choice(historical_data["Days taken"], size=(num_simulations, stories_in_epic), replace=True)
    results = np.sum(simulations, axis=1)
    print('Days needed in the first simulation:', simulations[0])

    plot_results(results, stories_in_epic)
    plot_with_risk(results)
