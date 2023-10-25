import os
import sys
import hydra
import argparse
from Projects_torch import task
from omegaconf import DictConfig


@hydra.main(config_path=os.path.join('../config', 'data2vec_cfg'), config_name='config')
def main(cfg: DictConfig) -> None:

    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--path2data', required=False,
                        default=r'C:\Users\moshe\PycharmProjects',
                        help='path2data')
    parser.add_argument('--path2save', required=False,
                        default=r'C:\Users\moshe\PycharmProjects\outputs_main',
                        help='path2save')

    args = parser.parse_args()

    task.TrainTaskSupervised(cfg, args)


if __name__ == "__main__":
    main()
