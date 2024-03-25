import os
from datetime import datetime
import time
import logging
import sys
from typing import Any, Dict, List
import json

from ..models._ot import _OT


class Experiment:
    def __init__(
        self,
        model: Dict[int, _OT],
        exp_name: str,
        log_dir: str = "logs",
    ):
        # set up loggers .........................
        self.exp_name = exp_name
        self.log_dir = os.path.join(log_dir, exp_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.cur_time = datetime.now().strftime("%y%m%d_%H%M%S")
        self.logger = logging.getLogger(f"experiment/{exp_name.upper()}")
        self.logger.setLevel(logging.INFO)
        
        s_handler = logging.StreamHandler(stream=sys.stdout)
        s_handler.setLevel(logging.INFO)
        self.logger.addHandler(s_handler)

        f_handler = logging.FileHandler(os.path.join(self.log_dir, f"{self.cur_time}.log"), mode='w')
        f_handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", 
                                                 datefmt="%m/%d %H:%M:%S"))
        f_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(f_handler)
        self.logger.propagate = False
        
        # set up OT models .........................
        self.model = model
        self.record_: Dict[str, Dict] = {}


    def run(self, **kwargs) -> Any:
        pass

    def plot(self, **kwargs) -> Any:
        pass

    def checkpoint(self):
        log_path = os.path.join(self.log_dir, self.cur_time + ".json")
        with open(log_path, 'w') as f:
            json.dump(self.record_, f)
        self.logger.debug("Checkpoint at {log_path}")

    def load(self, log_path: str) -> Dict:
        with open(log_path, "r") as f:
            self.record_ = json.load(f)
        return self.record_


    