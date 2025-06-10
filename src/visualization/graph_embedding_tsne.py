from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_tsne(embeddings, labels=None, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.8)
        plt.legend(*scatter.legend_elements())
    else:
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.8)
    plt.title("t-SNE of Transaction Embeddings")
    plt.grid(True)
    plt.show()
