import matplotlib.pyplot as plt
import seaborn as sns

def plot_anomaly_scores(scores, title="Anomaly Score Distribution"):
    sns.histplot(scores, bins=50, kde=True)
    plt.title(title)
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.show()
