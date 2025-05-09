import sounddevice as sd
import time
import torch
import argparse
import numpy as np
import soundfile as sf
import resampy
import pickle

from models.YAMNet import YAMNet
from models.classifier import Compatibility
from utils import waveform_to_examples

# Define constants
FORMAT = 'int16'  # Data format
CHANNELS = 1  # Mono audio
M_BINS = 64  # Number of mel bins
SAMPLE_RATE = 16000  # Sample rate
TARGET_DURATION = 5.0  # Duration of audio in seconds

# List available devices
print("Available devices:")
devices = sd.query_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['name']}")

DEVICE_INDEX = int(input("Enter the index of the microphone to use: "))

# Get sample rate
sample_rate = int(devices[DEVICE_INDEX]["default_samplerate"])
print(f"Using sample rate: {sample_rate}")
chunk = int(sample_rate * TARGET_DURATION)

# Load in arguments for the model
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_path", type=str, default="YAMNet-ESC-50-fold14.pt", help="Dataset to train on."
)
parser.add_argument(
    "--compat_path", type=str, default="classifier.pt", help="Dataset to train on."
)
parser.add_argument("--device", type=str, default="auto", help="Dataset to train on.")
args = parser.parse_args()
if args.device == "auto":
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

# Create the embedding model
audio_embedding_model = YAMNet(channels=1, num_classes=30)
audio_embedding_model.load_state_dict(
    torch.load(args.model_path, map_location=torch.device(args.device))
)
audio_embedding_model = audio_embedding_model.to(args.device)
audio_embedding_model.eval()

# Create the model and load the weights in
compatibility_model = Compatibility(128, 300).to(args.device)

# Load the state dictionary if the file contains it
state_dict = torch.load(args.compat_path, map_location=torch.device(args.device))
if isinstance(state_dict, dict):
    compatibility_model.load_state_dict(state_dict)
else:
    # If the file contains a full model object, use it directly
    compatibility_model = state_dict.to(args.device)

compatibility_model.eval()

# Load the word embeddings from a pickle file
with open('word_embeddings.pkl', 'rb') as f:
    word_embeddings = pickle.load(f)
word_embeddings = torch.tensor(word_embeddings).float().to(args.device)

# Define the classes
# fmt: off
classes = [
    "dog_canine_bark_woof_yap_call_animal_puppy", "rooster_cockerel_call_animal", "pig_hog_sow_swine_squeal_oink_grunt_call_animal", "cow_moo_call_bull_oxen_animal", "frog_toad_croak_call_animal",
    "cat_meow_mew_purr_hiss_chirp_kitten_feline_call_animal", "hen_cluck_chicken_animal_call", "insects_flying_buzz_hum_bug", "sheep_bleat_animal_call_lamb", "crow_squawk_screech_caw_bird_call_cry_animal", "rain_drizzle_wet_sprinkle_shower_water_nature", "sea_waves_water_swell_tide_ocean_surf_nature",  # sea waves
    "crackling_fire_hissing_sizzling_flame_bonfire_campfire_nature", "crickets_insects_insect_bug_cicada_call", "chirping_birds_animal_call_song_tweet_chirp_twitter_trill_warble_chatter_cheep", 
    "water_drops_splash_droplet_drip", "wind_nature_gust_gale_blow_breeze_howl", "pouring_water_slosh_gargle_splash_splosh", "toilet_flush_water_flow_wash", "thunderstorm_thunder_storm_nature_lightning", "crying_baby_cry_human_whine_infant_child_wail_bawl_sob_scream_call",
    "sneezing_sneeze", "clapping_clap_applause_applaud_praise", "breathing_breath_breathe_gasp_exhale", "coughing_cough_hack", "footsteps_walking_walk_pace_step_gait_march", "laughing_cackle_laugh_chuckle_giggle_funny",
    "brushing_teeth_scrape_rub_brush", "snoring_snore_sleep_snore_snort_wheeze_breath", "drinking_sipping_gulp_gargle_drink_sip_breath", "door_wood_knock_tap_bang_thump", "mouse_click_computer_tap",
    "keyboard_typing_tap_mechanical_computer", "door_wood_creaks_squeak_creak_screech_scrape", "can_opening_hiss_fizz_air", "washing_machine_electrical_hum_thump_noise_loud",
    "vacuum_cleaner_electrical_noise_loud", "clock_alarm_signal_buzzer_alert_ring_beep", "clock_tick_tock_click_clack_beat_tap_ticking", "glass_breaking_crunch_crack_smash_clink_break_noise", "helicoper_chopping_engine_blades_whirring_swish_chopper_electrical_noise_vehicle_loud",
    "chainsaw_saw_electrical_noise_tool_loud", "siren_alarm_alert_bell_horn_noise_loud", "car_horn_vehicle_noise_blast_loud_honk", "engine_rumble_vehicle_chug_revving_car_drive",
    "train_clack_horn_clatter_vehicle_squeal_rattle", "church_bells_tintinnabulation_ring_chime_bell", "airplane_plane_motor_engine_hum_loud_noise", "fireworks_burst_bang_firecracker", "hand_saw_squeak_sawing_cut_hack_tool",
]
# fmt: on

fold1 = [22, 13, 39, 49, 32, 26, 42, 21, 19, 36]
fold1_classes = [classes[i] for i in fold1]

# Define a callback function to process the audio data
def callback(indata, frames, time, status):
    if status:
        print(status)
    
    audio_data = indata[:, 0].astype(np.float32)

    assert audio_data.dtype == np.float32, "Bad sample type: %r" % audio_data.dtype
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize to [-1.0, +1.0]

    # Calculate the current duration of the audio in seconds
    current_duration = len(audio_data) / sample_rate

    if current_duration > TARGET_DURATION:
        # Chop the audio to the target duration
        excess_samples = int((current_duration - TARGET_DURATION) * sample_rate)
        start_idx = excess_samples // 2
        end_idx = len(audio_data) - (excess_samples - start_idx)
        audio_data = audio_data[start_idx:end_idx]
    else:
        # Pad the audio with zeros to reach the target duration
        samples_to_pad = int((TARGET_DURATION - current_duration) * sample_rate)
        audio_data = np.pad(
            audio_data, (samples_to_pad // 2, samples_to_pad // 2), mode="constant"
        )

    # Convert to mono.
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    # Resample to the rate assumed by the model
    if sample_rate != SAMPLE_RATE:
        audio_data = resampy.resample(audio_data, sample_rate, SAMPLE_RATE)

    # Convert to log mel spectrogram and get embedding from YAMNet
    sp_data = waveform_to_examples(audio_data, SAMPLE_RATE, 64)

    audio_input = np.array([[sp_data[0]]])
    audio_input = torch.from_numpy(audio_input).float().to(args.device)
    embedding = audio_embedding_model.inference(audio_input)

    # Calculate the compatibility score of the embedding with each class
    compatibility_scores = compatibility_model(embedding, word_embeddings)

    # Get the class with the highest compatibility score
    _, predicted_class = torch.max(compatibility_scores, dim=1)

    # Print the predicted class
    print("Predicted to be:", fold1_classes[predicted_class.item()])

# Open a streaming interface with the callback
with sd.InputStream(
    samplerate=sample_rate,
    channels=CHANNELS,
    device=DEVICE_INDEX,
    callback=callback,
    blocksize=chunk,
    dtype=FORMAT
):
    # Keep the stream open until a KeyboardInterrupt is received
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass