import matplotlib.pyplot as plt

def plot_anomaly_scores(scores, threshold=None):
    plt.hist(scores, bins=50, alpha=0.75, color='darkorange', edgecolor='black')
    if threshold is not None:
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.2f}')
        plt.legend()
    plt.title("Distribution of Anomaly Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
