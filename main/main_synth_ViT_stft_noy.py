import os
import sys
import hydra
import argparse

sys.path.append('/home/moshela/work/moshe/pycharm/Projects/')
sys.path.append('/home/a/home/cc/students/cs/moshelaufer/Projects/Projects_torch/')
from Projects_torch import task
from omegaconf import DictConfig

"""
path2data: path2dataset/{train, test, valid}/{data, labels, .csv}
path2data: path2dataset/{train, test, valid}/{data, .csv}
path2data: path2dataset/{data, labels, .csv}
path2data: path2dataset/{data, .csv}

path2save: path to empty folder which all the outputs will be saved 

"""


@hydra.main(config_path=os.path.join('../config', 'synth_ViT_STFT_noy'), config_name='config')
def main(cfg: DictConfig) -> None:
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--path2data', required=False,
                        # default='/home/a/home/cc/students/cs/moshelaufer/dataset/fm',
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\noy',
                        help='path2data')
    parser.add_argument('--path2save', required=False,
                        # default='/home/a/home/cc/students/cs/moshelaufer/outputs',
                        default=r'C:\Users\moshe\PycharmProjects\commercial_synth_dataset\outputs_main',
                        help='path2save')

    args = parser.parse_args()

    task.TrainTaskSupervised(cfg, args)


if __name__ == "__main__":
    main()
