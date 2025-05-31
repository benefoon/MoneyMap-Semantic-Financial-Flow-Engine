from sklearn.metrics import silhouette_score, davies_bouldin_score

def evaluate_clustering(X, labels):
    silhouette = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    return {
        "Silhouette": round(silhouette, 3),
        "Davies-Bouldin": round(db_score, 3)
    }
