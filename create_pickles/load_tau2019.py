import os
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
    default="../../../datasets/TAU Urban Acoustics/",
    help="Where the TAU wav files are located.",
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
    default="train",
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
    classes = [
        "airport_terminal_airplane_flight_travel_aviation_lounge_gate",
        "shopping_mall_store_commerce_retail_center_shops_indoor_crowd",
        "metro_station_subway_platform_underground_train_transit_railway",
        "street_pedestrian_walk_footpath_sidewalk_people_urban_outdoor",
        "public_square_plaza_open_space_urban_gathering_people_outdoor",
        "street_traffic_road_cars_buses_honking_vehicle_junction_city",
        "tram_streetcar_light_rail_transit_tracks_passenger_vehicle",
        "bus_vehicle_transit_transportation_commute_stop_passengers",
        "metro_subway_train_tunnel_underground_commute_passenger",
        "park_nature_trees_grass_birds_outdoor_recreation"
    ]
elif args.classes == "normal":
    classes = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", "street_traffic", "tram", "bus", "metro", "park"]

classes_index = ["airport", "shopping_mall", "metro_station", "street_pedestrian", "public_square", "street_traffic", "tram", "bus", "metro", "park"]

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

# Read the CSV file to get the class labels for each audio file
csv_file = os.path.join(args.data_path, "meta.csv")
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # Get the class from the file name
        target = classes_index.index(row["scene_label"])

        audio_file = f"{args.data_path}/{row['filename']}"
        channels = 3 if args.model == "Inception" else 1
        audio_embedding = audio_to_embedding(audio_file, model, args.device, channels)
        
        # Use Word2Vec to get text embedding
        word_embedding = word2vec(w2v_model, classes[target], double_first=True)

        data["labels"].append(target)
        data["features"].append(audio_embedding)
        data["auxiliary"].append(word_embedding)

try:
    os.mkdir("../pickles/tau2019/")
except:
    pass

# Data is a dictionary of seen/unseen classes, labels, auxiliary data (word2vec) and audio features
filename = "../pickles/tau2019/{}_{}_{}.pickle".format(args.model, args.classes, args.split)

with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
