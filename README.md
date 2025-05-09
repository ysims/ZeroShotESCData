# Audio-ZSL

## Set Up

### VGGish

Make sure pip is updated and wheel is installed.

    sudo python -m pip install --upgrade pip wheel

Install required stuff. ((Todo: move this to a requirements.txt file))

    sudo pip install numpy resampy tensorflow tf_slim six soundfile

VGGish is a folder within the Tensorflow models repository on GitHub. Clone the repo and grab the VGGish code out of it and then delete the rest of it.

    cd data
    git clone https://github.com/tensorflow/models.git
    mv models/research/audioset/vggish .
    rm -rf models

Download the VGGish pretrained model.

    cd vggish
    curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
    curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
    cd ..

### Word2Vec

Download from [LexVec](https://github.com/alexandres/lexvec#pre-trained-vectors). Use LexVec -> Common Crawl -> Word Vectors. Extract and rename the file to `vectors.txt` and put it in the `data` directory of the repository. 

### ESC-50 Dataset

Clone the repository

    cd data
    git clone https://github.com/karolpiczak/ESC-50

### Setting up the Dataset

1. Install the dependencies

    cd data
    pip install librosa gensim
