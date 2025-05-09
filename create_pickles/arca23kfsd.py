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
    default="/media/ysi/secondary/FSD50K",
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
    print("This dataset is too big for manual synonyms.")
    import sys
    sys.exit(1)
elif args.classes == "normal":            
    classes = ['Crash_cymbal', 'Run', 'Zipper_(clothing)', 'Acoustic_guitar', 'Gong', 'Knock', 'Train', 'Crack', 'Cough', 'Cricket']
    classes += ['Electric_guitar', 'Chewing_and_mastication', 'Keys_jangling', 'Female_speech_and_woman_speaking', 'Crumpling_and_crinkling', 'Skateboard', 'Computer_keyboard', 'Bass_guitar', 'Stream', 'Toilet_flush']
    classes += ['Tap', 'Water_tap_and_faucet', 'Squeak', 'Snare_drum', 'Finger_snapping', 'Walk_and_footsteps', 'Meow', 'Rattle_(instrument)', 'Bowed_string_instrument', 'Sawing']
    classes += ['Rattle', 'Slam', 'Whoosh_and_swoosh_and_swish', 'Hammer', 'Fart', 'Harp', 'Coin_(dropping)', 'Printer', 'Boom', 'Giggle']
    classes += ['Clapping', 'Crushing', 'Livestock_and_farm_animals_and_working_animals', 'Scissors', 'Writing', 'Wind', 'Crackle', 'Tearing', 'Piano', 'Microwave_oven']
    classes += ['Trumpet', 'Wind_instrument_and_woodwind_instrument', 'Child_speech_and_kid_speaking', 'Drill', 'Thump_and_thud', 'Drawer_open_or_close', 'Male_speech_and_man_speaking', 'Gunshot_and_gunfire', 'Burping_and_eructation', 'Splash_and_splatter']
    classes += ['Female_singing', 'Wind_chime', 'Dishes_and_pots_and_pans', 'Scratching_(performance_technique)', 'Crying_and_sobbing', 'Waves_and_surf', 'Screaming', 'Bark', 'Camera', 'Organ']

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
audio_paths = [os.path.join(args.data_path, "FSD50K.dev_audio"), 
                os.path.join(args.data_path, "FSD50K.eval_audio")]
csv_files = ["ARCA23K-FSD.ground_truth/train.csv", "ARCA23K-FSD.ground_truth/val.csv","ARCA23K-FSD.ground_truth/test.csv"]

for csv_file in csv_files:
    with open(os.path.join(args.data_path, csv_file), "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get the class from the file name
            target = classes.index(row["label"])

            # Find the audio file in any of the audio paths
            audio_file = None
            for path in audio_paths:
                if os.path.exists(os.path.join(path, row["fname"] + ".wav")):
                    audio_file = os.path.join(path, row["fname"] + ".wav")
                    break

            channels = 3 if args.model == "Inception" else 1
            audio_embedding = audio_to_embedding(audio_file, model, args.device, channels)
            
            # Use Word2Vec to get text embedding
            word_embedding = word2vec(w2v_model, classes[target], double_first=True)

            data["labels"].append(target)
            data["features"].append(audio_embedding)
            data["auxiliary"].append(word_embedding)

try:
    os.mkdir("../pickles/arca23kfsd/")
except:
    pass

# Data is a dictionary of seen/unseen classes, labels, auxiliary data (word2vec) and audio features
filename = "../pickles/arca23kfsd/{}_{}_{}.pickle".format(args.model, args.classes, args.split)

with open(filename, "wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saved {}".format(filename))
