import os
import pickle
import argparse
import torch

# Word2Vec - text semantic vector
import gensim

# from audio_to_vggish import audio_to_vggish
from word2vec import word2vec

from models.YAMNet import YAMNet
from models.VGGish import VGGNet
from models.Inception import InceptionV4
from inference import audio_to_embedding

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "model_path",
    type=str,
    help="Where the YAMNet model is located.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="../../../datasets/GTZAN/",
    help="Where the UrbanSound9k wav files are located.",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run inference on.",
)
parser.add_argument(
    "--classes",
    type=str,
    default="synonyms",
    help="Add synonyms or not.",
)
parser.add_argument(
    "--split",
    type=str,
    default="fold0",
    help="Which split to use.",
)
parser.add_argument(
    "--model",
    type=str,
    default="YAMNet",
    help="Which model to use.",
)
args = parser.parse_args()

if args.classes == "synonyms":
    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
elif args.classes == "normal":
    classes = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
    "./word2vec.txt", binary=True
)

data = {
    "labels": [],
    "features": [],
    "auxiliary": [],
}

# Load the YAMNet model
state_dict = torch.load(args.model_path, map_location=torch.device("cuda"))
# Get the number of classes from the last layer
if args.model == "YAMNet":
    num_classes = state_dict['classifier.weight'].shape[0]
    model = YAMNet(channels=1, num_classes=num_classes)
elif args.model == "VGGish":
    model = VGGNet(num_classes=7)
elif args.model == "Inception":
    model = InceptionV4(num_classes=7)

model.load_state_dict(state_dict)
model = model.to(args.device)
model.eval()

path = os.path.join(args.data_path, "Data/genres_original")
for i, c in enumerate(classes):
    # Get the class from the file name
    target = i
    
    for file in os.listdir(os.path.join(path, c)):
        full_path = f"{path}/{c}/{file}"
        channels = 3 if args.model == "Inception" else 1
        audio_embedding = audio_to_embedding(full_path, model, args.device, channels)
        
        # Use Word2Vec to get text embedding
        word_embedding = word2vec(w2v_model, classes[target], double_first=True)

        data["labels"].append(target)
        data["features"].append(audio_embedding)
        data["auxiliary"].append(word_embedding)

try:
    os.mkdir("../pickles/gtzan/")
except:
    pass

# Data is a dictionary of seen/unseen classes, labels, auxiliary data (word2vec) and audio features
filename = "../pickles/gtzan/{}_{}_{}.pickle".format(args.model, args.classes, args.split)

with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
