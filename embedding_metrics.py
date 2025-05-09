import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import calinski_harabasz_score, silhouette_score, normalized_mutual_info_score, davies_bouldin_score
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import itertools

from dataclasses import dataclass, asdict

@dataclass
class EmbeddingMetrics:
    dataset: str
    split: str
    model: str
    calinski_harabasz_seen: float
    calinski_harabasz_unseen: float
    silhouette_seen: float
    silhouette_unseen: float
    centroid_distance_mean: float
    nearest_seen_centroid_dist: float
    knn_consistency: float
    angular_separation_mean: float
    nmi: float

def compute_metrics(features, labels, unseen_classes, k=10):
    centroids = np.array([np.mean(features[labels == c], axis=0) for c in np.unique(labels)])
    centroid_distances = squareform(pdist(centroids, metric='euclidean'))
    knn = NearestNeighbors(n_neighbors=k + 1).fit(features)
    neighbors = knn.kneighbors(features, return_distance=False)
    
    seen_mask = np.isin(labels, unseen_classes, invert=True)
    unseen_mask = np.isin(labels, unseen_classes)
    
    seen_features, unseen_features = features[seen_mask], features[unseen_mask]
    seen_labels, unseen_labels = labels[seen_mask], labels[unseen_mask]

    # Compute distances from unseen samples to seen class centroids
    seen_centroids = np.array([np.mean(seen_features[seen_labels == c], axis=0) for c in np.unique(seen_labels)])

    unseen_to_seen_dist = np.mean(np.min(cdist(unseen_features, seen_centroids), axis=1))

    return {
        "davies_bouldin": davies_bouldin_score(features, labels) if len(np.unique(labels)) > 1 else np.nan,
        "calinski_harabasz": calinski_harabasz_score(features, labels) if len(np.unique(labels)) > 1 else np.nan,
        "silhouette": silhouette_score(features, labels) if len(np.unique(labels)) > 1 else np.nan,
        "centroid_distance_mean": np.mean(centroid_distances),
        "nearest_seen_centroid_dist": unseen_to_seen_dist,
        "knn_consistency": sum(np.sum(labels[nbrs[1:]] == labels[i]) for i, nbrs in enumerate(neighbors)) / (len(features) * k),
        "angular_separation_mean": np.mean(np.abs((centroids / np.linalg.norm(centroids, axis=1, keepdims=True)) @ (centroids / np.linalg.norm(centroids, axis=1, keepdims=True)).T - np.eye(len(centroids)))),
        "nmi": normalized_mutual_info_score(labels, KMeans(n_clusters=len(np.unique(labels)), random_state=42).fit(features).labels_)
    }


def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    dataset = path.split("/")[1]
    split = path.split("/")[-1].split("_")[2].split(".")[0]

    # Get unseen classes
    if dataset == "esc50":
        if split == "fold0":
            classes = [27, 46, 38, 3, 29, 48, 40, 31, 2, 35]
        elif split == "fold1":
            classes = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
        elif split == "fold2":
            classes = [23, 41, 14, 24, 33, 30, 4, 17, 10, 45]
        elif split == "fold3":
            classes = [47, 34, 20, 44, 25, 6, 7, 1, 28, 18]
        else:
            classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
    elif dataset == "fsc22":
        classes = [5, 7, 15, 17, 21, 23, 26]
        if split != "test":
            classes = [6, 8, 9, 12, 13, 18, 22]

    labels = np.array(data["labels"])
    features = np.array([list(d.to("cpu")[0]) for d in data["features"]])

    # Normalise features
    features = features / np.linalg.norm(features, axis=1, keepdims=True)

    return compute_metrics(features, labels, classes)

def plot_metrics(table):
    # Table is a 3D numpy array with shape (models, folds, metrics)
    # All metrics will be plotted on separate subplots stacked vertically for each model

    fold_order = {
        "YAMNet": ["fold1", "fold2", "test", "fold0", "fold3", "val", "cat1", "fold4", "cat0", "cat3", "cat4", "cat2"],
        "VGGish": ["test", "val", "fold1", "fold0", "fold2", "fold3", "cat3", "cat1", "cat0", "cat4", "cat2", "fold4"],
        "Inception": ["val", "test", "fold0", "fold1", "fold3", "cat1", "fold4", "fold2", "cat4", "cat2", "cat0", "cat3"]
    }

    metrics = ["Davies-Bouldin index", "Centroid distance mean", "Nearest seen centroid distance", "K-nn consistency", "Angular separation mean", "Normalised mutual information score", "Silhouette score"]

    y_max = 2.0  # Cap for the y-axis

    # Create a figure with subplots stacked vertically
    fig, axes = plt.subplots(len(fold_order), 1, figsize=(12, 4 * len(fold_order)), sharex=False)

    for i, (model, ax) in enumerate(zip(["YAMNet", "VGGish", "Inception"], axes)):
        fold_ordered = fold_order[model]

        # Reorder the table data to match the fold_ordered
        fold_indices = [fold_order[model].index(fold) for fold in fold_ordered]
        reordered_table = table[i, fold_indices, :]

        for j, metric in enumerate(metrics):
            y_values = reordered_table[:, j]
            capped_y_values = np.clip(y_values, None, y_max)  # Cap values at y_max

            # Plot the capped values
            ax.plot(range(len(fold_ordered)), capped_y_values, marker='o', label=metric)

            # Add arrows and labels for outliers
            for x, y in enumerate(y_values):
                if y > y_max:
                    ax.annotate(
                        '↑',
                        (x, y_max),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha='center',
                        fontsize=12,
                        color='red'
                    )
                    ax.annotate(
                        f"{y:.2f}",
                        (x, y_max),
                        textcoords="offset points",
                        xytext=(0, 15),
                        ha='center',
                        fontsize=10,
                        color='red'
                    )

        # Set the title to just the model name
        ax.set_title(model)
        ax.set_ylabel("Score")
        ax.set_ylim(0, y_max + 0.1)  # Set y-axis limit slightly above the cap
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))  # Adjust legend position

        # Set x-axis labels specific to the model
        ax.set_xticks(range(len(fold_ordered)))
        ax.set_xticklabels(fold_ordered, rotation=45)

    # Adjust layout with more vertical margins
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(hspace=0.5)  # Increase space between subplots
    plt.savefig("metrics_stacked.png")
    plt.show()

import matplotlib.patches as mpatches

def plot_box(table, models, metrics, datasets, folds):
    """
    Visualize the differences between models using subplots for each dataset and metric group.
    Models are on the x-axis, and box plots are color-coded by metrics.

    Args:
        table (np.ndarray): A 3D numpy array with shape (models, folds, metrics).
        models (list): List of model names.
        metrics (list): List of metric names.
        datasets (list): List of dataset names corresponding to each fold.
        folds (list): List of fold names.
    """
    dataset_model_orders = {
        "esc50": ["YAMNet", "VGGish", "Inception"],
        "fsc22": ["VGGish", "YAMNet", "Inception"]
    }

    # Separate metrics into two groups
    metric_groups = {
        "group1": ["Davies-Bouldin index", "Centroid distance mean", "Normalised mutual information score"],
        "group2": ["k-NN consistency", "Nearest seen centroid distance", "Angular separation mean", "Silhouette score"]
    }
    metric_names = {
        "Davies-Bouldin index": "davies_bouldin",
        "Centroid distance mean": "centroid_distance_mean",
        "Normalised mutual information score": "nmi",
        "k-NN consistency": "knn_consistency",
        "Nearest seen centroid distance": "nearest_seen_centroid_dist",
        "Angular separation mean": "angular_separation_mean",
        "Silhouette score": "silhouette"
    }

    y_ranges = {
        "esc50": {"group1": (0.75, 2.0), "group2": (0, 1)},
        "fsc22": {"group1": (0.5, 4.0), "group2": (0, 0.8)}
    }

    # Define a unique color for each metric
    all_metrics = metric_groups["group1"] + metric_groups["group2"]
    colors = {metric: plt.cm.tab20(i / len(all_metrics)) for i, metric in enumerate(all_metrics)}

    for dataset in ["esc50", "fsc22"]:
        # Filter folds for the current dataset
        dataset_folds = [fold for fold, ds in zip(folds, datasets) if ds == dataset]
        dataset_indices = [folds.index(fold) for fold in dataset_folds]

        # Filter the table for the current dataset
        dataset_table = table[:, dataset_indices, :]

        # Get the model order for the current dataset
        model_order = dataset_model_orders[dataset]
        model_indices = [models.index(model) for model in model_order]
        dataset_table = dataset_table[model_indices, :, :]

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)  # Reduced height

        for ax, (group_name, group_metrics) in zip(axes, metric_groups.items()):
            # Prepare data for the box plots
            box_data = {metric: [dataset_table[model_idx, :, metrics.index(metric_names[metric])] for model_idx in range(len(model_order))]
                        for metric in group_metrics}

            for metric_idx, metric in enumerate(group_metrics):
                color = colors[metric]
                # Offset the positions for each metric to avoid overlap
                positions = np.arange(len(model_order)) + metric_idx * 0.2
                boxplot = ax.boxplot(
                    box_data[metric],
                    positions=positions,
                    widths=0.15,
                    patch_artist=True,
                    showfliers=True,  # Ensure outliers are displayed
                    boxprops=dict(facecolor=color, color=color),
                    medianprops=dict(color="black"),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    flierprops=dict(marker='o', color=color, alpha=0.3)  # Customize outlier appearance
                )

                # Add translucent lines connecting the medians
                medians = [item.get_ydata()[0] for item in boxplot['medians']]
                ax.plot(positions, medians, color=color, alpha=0.5, linestyle='--', linewidth=1)

            # Add labels and legend
            ax.set_xticks(np.arange(len(model_order)) + (len(group_metrics) - 1) * 0.1 / 2)
            ax.set_xticklabels(model_order, rotation=45)
            ax.set_xlabel("Models")
            ax.set_ylabel("Score")
            ax.set_ylim(*y_ranges[dataset][group_name])  # Set y-axis range for the group

            # Create a custom legend for the metrics
            legend_patches = [mpatches.Patch(color=colors[metric], label=metric) for metric in group_metrics]
            ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0.0, 1.2))  # Place legend outside the plot

        # Adjust layout and save the figure
        # fig.suptitle(dataset.upper(), fontsize=16)
        plt.tight_layout(rect=[0, 0, 1.0, 1.0])  # Add space for the title
        plt.savefig(f"model_comparison_{dataset}_subplots.png")
        plt.show()

if __name__ == "__main__":
    # Get data for each fold & dataset
    models = ["VGGish", "YAMNet", "Inception"]
    datasets = ["esc50", "esc50", "esc50", "esc50", "esc50", "esc50", "esc50", "esc50", "esc50", "esc50", "fsc22", "fsc22"]
    folds = ["cat0", "cat1", "cat2", "cat3", "cat4", "fold0", "fold1", "fold2", "fold3", "fold4", "val", "test"]
    metrics = ["davies_bouldin", "centroid_distance_mean", "nearest_seen_centroid_dist", "knn_consistency", "angular_separation_mean", "nmi", "silhouette"]
    
    # Make a 3-d table with models as rows, folds as columns, and metrics as depth
    # For each model-fold pair get the metrics and add them to the table in the depth
    table = np.zeros((len(models), len(folds), len(metrics)))
    
    # open table
    table = np.load("metrics_table.pickle", allow_pickle=True)

    # for model, fold in itertools.product(models, folds):
    #     print("Metrics for", model, fold, datasets[folds.index(fold)])
    #     dataset = datasets[folds.index(fold)]
        
    #     file = f"pickles/{dataset}/{model}_synonyms_{fold}.pickle"
    #     if not os.path.exists(file):
    #         print(f"File {file} does not exist.")
    #         continue

    #     results = load_data(file)
    #     for key, value in results.items():
    #         if key in metrics:
    #             table[models.index(model), folds.index(fold), metrics.index(key)] = value

    # # Save table to a pickle file
    # with open("metrics_table.pickle", "wb") as f:
    #     pickle.dump(table, f)
    #     print("Saved metrics table to metrics_table.pickle")

    plot_box(table, models, metrics, datasets, folds)