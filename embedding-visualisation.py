import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import argparse

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    type=str,
    help="Which dataset to use.",
)
parser.add_argument(
    "split",
    type=str,
    help="Which split to use.",
)
parser.add_argument(
    "pickle",
    type=str,
    help="Path of the pickle file to load.",
)
args = parser.parse_args()

if args.dataset == "ESC-50":
    class_names = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects flying", "sheep", "crow", "rain", "sea waves", "crackling fire", "crickets", "chirping birds", "water drops", "wind", "pouring water", "toilet flush", "thunderstorm", "crying baby", "sneezing", "clapping", "breathing", "coughing", "footsteps", "laughing", "brushing teeth", "snoring", "sipping", "door knock", "mouse click", "keyboard typing", "door wood creaks", "can opening", "washing machine", "vacuum cleaner", "clock alarm", "clock tick", "glass breaking", "helicoper", "chainsaw", "siren", "car horn", "engine", "train", "church bells", "airplane", "fireworks", "hand saw"]
else:
    class_names = ["fire", "rain", "thunderstorm", "water_drops", "wind", "silence", "tree_falling", "helicoper", "vehicle_engine", "ax", "chainsaw", "generator", "handsaw", "firework", "gunshot", "woodchop", "whistling", "speaking", "footsteps", "clapping", "insect", "frog", "bird_chirping", "wing_flapping", "lion", "wolf_howl", "squirrel"]

# Open the pickle and get the embeddings
with open(args.pickle, "rb") as f:
    data = pickle.load(f)

# Get val classes
if args.dataset == "ESC-50":
    if args.split == "fold0":
        classes = [27, 46, 38, 3, 29, 48, 40, 31, 2, 35]
    elif args.split == "fold1":
        classes = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
    elif args.split == "fold2":
        classes = [23, 41, 14, 24, 33, 30, 4, 17, 10, 45]
    elif args.split == "fold3":
        classes = [47, 34, 20, 44, 25, 6, 7, 1, 28, 18]
    elif args.split == "cat0":
        classes = [0,1,2,3,4,5,6,7,8,9]
    elif args.split == "cat1":
        classes = [10,11,12,13,14,15,16,17,18,19]
    elif args.split == "cat2":
        classes = [20,21,22,23,24,25,26,27,28,29]
    elif args.split == "cat3":
        classes = [30,31,32,33,34,35,36,37,38,39]
    elif args.split == "cat4":
        classes = [40,41,42,43,44,45,46,47,48,49]
    else:
        classes = [43, 5, 37, 12, 9, 0, 11, 8, 15, 16]
elif args.dataset == "FSC22":
    classes = [5, 7, 15, 17, 21, 23, 26]
    if args.split != "test":
        classes = [6, 8, 9, 12, 13, 18, 22]

# Get the embeddings
all_labels = np.array(data["labels"])
all_features = np.array([list(d.to("cpu")[0]) for d in data["features"]])
all_auxiliary = np.array(data["auxiliary"])

# Get only the embeddings for the val classes
val_indices = np.where(np.isin(all_labels, classes))[0]
all_labels = all_labels[val_indices]
all_features = all_features[val_indices]

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=9, n_iter=1000)
all_features = tsne.fit_transform(all_features)

# Plot the embeddings
plt.figure(figsize=(10, 10))
colors = plt.cm.tab20.colors
for i, class_ in enumerate(classes):
    indices = np.where(all_labels == class_)[0]
    plt.scatter(all_features[indices, 0], all_features[indices, 1], label=class_, color=colors[i])

# Get class names for legend
classes = [str(class_) + " " + class_names[class_] for class_ in classes]

# Add legend
custom_lines = [Line2D([0], [0], color=colors[i], lw=4) for i in range(len(classes))]
# plt.legend(custom_lines, classes, title="Classes", fontsize=22, title_fontsize=22, bbox_to_anchor=(1, 1), loc="upper left")
# plt.title(f"Feature Embedding t-SNE for {dataset} {split}", fontsize=26)
# plt.title(f"t-SNE for ESC-50 {args.split}", fontsize=16)

# Remove axis labels
plt.xticks([])
plt.yticks([])

# Fit so the legend is not cut off
# plt.tight_layout()

# Set aspect ratio
# x_range = all_features[:, 0].max() - all_features[:, 0].min()
# y_range = all_features[:, 1].max() - all_features[:, 1].min()
# aspect_ratio = x_range / y_range
# plt.gca().set_aspect(aspect_ratio)

# plt.xlabel("Dimension 1", fontsize=14)
# plt.ylabel("Dimension 2", fontsize=14)
# remove axis labels
# plt.xticks([])
# plt.yticks([])

# add padding on the axes to center the plot and ensure points are not too close to the edges
plt.xlim(all_features[:, 0].min() - 10, all_features[:, 0].max() + 10)
plt.ylim(all_features[:, 1].min() - 10, all_features[:, 1].max() + 10)

# add grid
plt.grid(color='gray', linestyle='--', linewidth=1.0, alpha=0.3)

# remove border
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Save the plot
plt.savefig(f"{args.dataset}-{args.split}-tSNE.png", dpi=400)
# plt.savefig("vggish_embeddings.svg", format='svg', bbox_inches='tight')

# Show the plot
# plt.show()
