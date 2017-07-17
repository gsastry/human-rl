#!/usr/bin/env bash
set -ex

# conda create --name humanrl python=3.5
source activate humanrl

if [ "$(uname)" == "Darwin" ]; then
    brew install tmux htop cmake golang libjpeg-turbo
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    sudo apt-get install -y tmux htop cmake golang libjpeg-dev
    pwd > $HOME/anaconda3/envs/humanrl/lib/python3.5/site-packages/humanrl.pth
fi

pip install "gym[atari]==0.7.3"
pip install universe
pip install six
pip install tensorflow-gpu
conda install --channel loopbio --channel conda-forge --channel pkgw-forge gtk2 ffmpeg ffmpeg-feature gtk2-feature opencv openblas
conda install -y numpy
conda install -y scipy
conda install -y -c conda-forge matplotlib
conda install -c conda-forge jupyter_contrib_nbextensions
pip install 'atari-py==0.0.21' --force-reinstall
pip install pydotplus sklearn pandas tqdm
conda install -y jupyter
ipython kernel install --prefix /home/william/anaconda3/envs/humanrl
