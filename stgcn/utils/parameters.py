import argparse


def get_parser(description):
    parser = argparse.ArgumentParser(
        description=description)
    parser.add_argument(
        '--config',
        default='./configs/baseline.yaml',
        help='path to the configuration file')
    return parser
    