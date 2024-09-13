import argparse
from preprocess import image_resize
from environs import Env

from train import train

env = Env()
env.read_env()
ROOT_DIR = env.str("ROOT_DIR")


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--resize", action="store_true")
parser.add_argument("-t", "--train", action="store_true")

args = parser.parse_args()

if args.resize:
    image_resize("data", "data_resized", 640, 640)

if args.train:
    train(root_dir=ROOT_DIR)
