import json
import argparse
import importlib
import torch
from lib.trainer import NetTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help='JSON File Train configuration')

    args = parser.parse_args()
    with open(args.config) as json_file:
        config = json.load(json_file)
        file, class_name = config['model']['name'].split('.')
        model = getattr(importlib.import_module('models.') + file, class_name)(**config['model']['params'])
        if 'state_dict' in config['model']:
            model.load_state_dict(torch.load(config['model']['state_dict']))
        del config['model']
        config['net'] = model
        trainer = NetTrainer(**config)
        trainer.train()


if __name__ == '__main__':
    main()


