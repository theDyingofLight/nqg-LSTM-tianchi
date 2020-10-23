from trainer import Trainer
from infenrence import BeamSearcher
import config
import argparse
import logging,json,logging.config,os
import warnings
warnings.filterwarnings("ignore")

def main(args):
    logger = logging.getLogger('paragraph-level')
    if args.train:
        trainer = Trainer(args)
        trainer.train()
    else:
        beamsearcher = BeamSearcher(args.model_path, args.output_dir)
        beamsearcher.decode()

def setup_logging(default_path = "logging.json",default_level = logging.INFO,env_key = "LOG_CFG"):
    path = default_path
    value = os.getenv(env_key,None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path,"r") as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level = default_level)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", type=str, default="",
                        help="path to the saved checkpoint")
    parser.add_argument("--output_dir", type=str, default="race/high/result/pointer_maxout_ans")
    args = parser.parse_args()
    setup_logging(default_path='logger.json')
    main(args)
