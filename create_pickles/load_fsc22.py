import os
import re
import sys
import pickle
import argparse
import torch

# Word2Vec - text semantic vector
import gensim

# from audio_to_vggish import audio_to_vggish
from word2vec import word2vec

sys.path.append("../Audio-Embeddings-ZSL/")
from models.YAMNet import YAMNet
from models.VGGish import VGGNet
from models.Inception import InceptionV4
from inference import audio_to_embedding

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "save_name",
    type=str,
    help="What name to give the saved data.",
)
parser.add_argument(
    "model_path",
    type=str,
    help="Where the YAMNet model is located.",
)
parser.add_argument(
    "--data_path",
    type=str,
    default="../../datasets/FSC22/audio/",
    help="Where the FSC22 wav files are located.",
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
    default="normal",
    help="Add synonyms or not.",
)
args = parser.parse_args()

if args.classes == "synonyms":
    classes = [
        "fire_crackling_hissing_sizzling_flame_bonfire_campfire_nature",  # fire
        "rain_drizzle_wet_sprinkle_shower_water_nature",
        "thunderstorm_thunder_storm_nature_lightning",
        "water_drops_splash_droplet_drip",
        "wind_nature_gust_gale_blow_breeze_howl",
        "silence_quiet_silent_soft_nature",
        "tree_falling_crackling_wood_nature_crash",
        "helicoper_chopping_engine_blades_whirring_swish_chopper_electrical_noise_vehicle_loud",
        "vehicle_engine_rumble_chug_revving_car_drive",
        "ax_chop_cutting_wood_tool",
        "chainsaw_saw_electrical_noise_tool_loud",
        "generator_hum_electrical_machine",
        "hand_saw_squeak_sawing_cut_hack_tool",
        "fireworks_burst_bang_firecracker",
        "gunshot_gun_firearm_weapon_shot",
        "wood_chop_breaking_splintering_crack",
        "whistling_whistle_high_pitch",
        "speaking_talking_speech_conversation",
        "footsteps_walking_walk_pace_step_gait_march",
        "clapping_clap_applause_applaud_praise",
        "insect_flying_buzz_hum_bug",
        "frog_toad_croak_call_animal",
        "bird_chirping_animal_call_song_tweet_chirp_twitter_trill_warble_chatter_cheep",  # chirping
        "wing_flapping_flap_bird_animal",
        "lion_roar_growl_call_animal",
        "wolf_howl_canine_call_animal",
        "squirrel_call_animal_chatter_chirp_bark_whistle",
    ]
elif args.classes == "normal":
    classes = [
        "fire",
        "rain",
        "thunderstorm",
        "water_drops",
        "wind",
        "silence",
        "tree_falling",
        "helicoper",
        "vehicle_engine",
        "ax", # 9
        "chainsaw", # 10
        "generator", # 11
        "handsaw",
        "firework",
        "gunshot",
        "woodchop", # 15
        "whistling",
        "speaking", # 17
        "footsteps",
        "clapping", # 19
        "insect", # 20
        "frog", # 21
        "bird_chirping", # 22
        "wing_flapping", # 23
        "lion", # 24
        "wolf_howl", # 25
        "squirrel", # 26
    ]

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
num_classes = state_dict['classifier.weight'].shape[0]
model = YAMNet(channels=1, num_classes=num_classes)
# model = InceptionV4(num_classes=13)
model.load_state_dict(state_dict)
model = model.to(args.device)
model.eval()

# Iterate on all of the wav files and create the dataset
for file in os.listdir(args.data_path):
    # Get the class index from the first number in the file name
    target = int(file.split("_")[0])

    audio_embedding = audio_to_embedding(os.path.join(args.data_path, file), model, args.device, 3)

    # Use Word2Vec to get text embedding
    word_embedding = word2vec(w2v_model, classes[target-1], double_first=True)

    # Get all embeddings individually
    data["labels"].append((target-1))
    data["features"].append(audio_embedding)
    data["auxiliary"].append(word_embedding)

try:
    os.mkdir("./pickles/FSC22")
except:
    pass

# Data is a dictionary of seen/unseen classes, labels, auxiliary data (word2vec) and audio features
filename = "./pickles/FSC22/{}.pickle".format(args.save_name)

with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
