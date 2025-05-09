import os
import re
import pickle
import argparse
import torch
import csv

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
    default="../../../datasets/urbansound8k/",
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
    classes = ["air_conditioner_appliance", "car_horn_vehicle_noise_blast_loud_honk", "children_playing_child_laughter_squeal_chatter_kid_play", "dog_canine_bark_woof_yap_call_animal_puppy", "drilling_tool_power_drill_machine_loud", "engine_idling_rumble_vehicle_chug_revving_car_drive", "gunshot_gun_firearm_weapon_shot", "jackhammer_construction_tool_roadwork_loud", "siren_alarm_alert_bell_horn_noise_loud", "street_music_melody_instruments_outdoors"]  # 10 classes
elif args.classes == "normal":
    classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gunshot", "jackhammer", "siren", "street_music"]  # 10 classes

#   `{FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav`

#   - `{FOLD}` - index of the cross-validation fold,
#   - `{CLIP_ID}` - ID of the original Freesound clip,
#   - `{TAKE}` - letter disambiguating between different fragments from the same Freesound clip,
#   - `{TARGET}` - class in numeric format [0, 49].

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

# Read the CSV file to get the data
csv_file = os.path.join(args.data_path, "metadata/UrbanSound8K.csv")
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Get the class from the file name
        target = int(row["classID"])
        
        audio_file = f"{args.data_path}/audio/fold{row['fold']}/{row['slice_file_name']}"

        channels = 3 if args.model == "Inception" else 1
        audio_embedding = audio_to_embedding(audio_file, model, args.device, channels)
        
        # Use Word2Vec to get text embedding
        word_embedding = word2vec(w2v_model, classes[target], double_first=True)

        data["labels"].append(target)
        data["features"].append(audio_embedding)
        data["auxiliary"].append(word_embedding)

try:
    os.mkdir("../pickles/urbansound8k/")
except:
    pass

# Data is a dictionary of seen/unseen classes, labels, auxiliary data (word2vec) and audio features
filename = "../pickles/urbansound8k/{}_{}_{}.pickle".format(args.model, args.classes, args.split)

with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
