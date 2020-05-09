"""Conv Nets testing script."""

import click
import numpy as np

import data
import util
from nn import create_net, test_model


@click.command()
@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
              help='Path or name of configuration module.')
@click.option('--weights_from', default='weights/detector.h5', show_default=True,
              help='Path to initial weights file.')
@click.option('--input_dir', default='', show_default=True,
              help='Path to input image dir.')
def main(cnf, weights_from, input_dir):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    # load every data found in train_dir
    files = data.get_image_files(input_dir)
    names = data.get_names(files)

    model = create_net(config)
    model.summary()

    try:
        model.load_weights(weights_from)
        print(("loaded weights from {}".format(weights_from)))
    except IOError:
        print("couldn't load weights, exit")
        exit()

    print("Testing ...")
    test_model(model, config, files, np.zeros(files.shape))


if __name__ == '__main__':
    main()

