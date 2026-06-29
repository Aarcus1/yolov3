import argparse


def parse_program_options(args):
    parser = argparse.ArgumentParser(
        prog='YOLO V3',
        description='Implements the YOLO V3 neural network model, based on the original paper.')
    parser.add_argument('-c', '--config_folder', help='Location of the configuration folder', required=True)
    return parser.parse_args(args)

