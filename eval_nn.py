"""Conv Nets training script."""

import click
import numpy as np

import data
import util
from nn import create_net, test_model


@click.command()
@click.option('--cnf', default='configs/c_512_4x4_32.py', show_default=True,
              help='Path or name of configuration module.')
@click.option('--weights_from', default=None, show_default=True,
              help='Path to initial weights file.')
def main(cnf, weights_from):

    config = util.load_module(cnf).config

    if weights_from is None:
        weights_from = config.weights_file
    else:
        weights_from = str(weights_from)

    # load every data found in train_dir
    files = data.get_image_files(config.get('train_dir'))
    names = data.get_names(files)
    labels = data.get_labels(names).astype(np.float32)

    model = create_net(config)
    model.summary()

    try:
        model.load_weights(weights_from)
        print(("loaded weights from {}".format(weights_from)))
    except IOError:
        print("couldn't load weights, exit")
        exit()

    print("Testing ...")
    test_model(model, config, files, labels)


if __name__ == '__main__':
    main()

