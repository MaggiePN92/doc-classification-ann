from pathlib import Path
import os
import json
from typing import Dict, Any, List
import random
import string


class Logger:
    def __init__(
        self,
        model_name : str,
        training_time : float,
        f1_score : float, 
        accuracy : float,
        clf_report : Dict[str, Any],
        accum_train_loss : List[float],
        accum_val_loss : List[float]
    ) -> None:
        """Logger is used to log results from model evaluation.

        Args:
            model_name (str): descriptive name of model,
            training_time (float): time it took to train model.
            f1_score (float): f1 score. 
            accuracy (float): accuracy. 
            clf_report (Dict[str, Any]): dict with metrics, made
            with scikit-learns classification_report. 
            accum_train_loss (List[float]): accumulated training
            loss.
            accum_val_loss (List[float]): accumulated validation
            loss.
        """
        self.model_name = model_name 
        self.training_time = training_time 
        self.f1_score = f1_score
        self.accuracy = accuracy 
        self.clf_report = clf_report
        self.accum_train_loss = accum_train_loss
        self.accum_val_loss = accum_val_loss

    def log(self) -> None:
        """Writes results of model evaluation to file. Makes
        a new directory and writes file to this dir. 
        """
        dir2make = self._count_dirs()
        path2dir = Path(f"modellogs/{dir2make}")
        
        # don't want to overwrite log files, so if directory already
        # exits make new one
        if path2dir.exists():
            # generate random string. Code retrieved from:
            # https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
            rand_dirname = ''.join(
                random.choice(
                    string.ascii_uppercase + string.digits) for _ in range(5))
            path2dir = Path(f"modellogs/{rand_dirname}")
        
        path2dir.mkdir(exist_ok=False)

        print(f"Writing model log to: {path2dir}")
        with open(f"{path2dir}/{self.model_name}.log", "w") as f:

            json2log = json.dumps(
                {
                    "model_name":self.model_name,
                    "train_time":self.training_time,
                    "f1_score":self.f1_score,
                    "accuracy":self.accuracy,
                    "clf_rep":self.clf_report,
                    "accum_train_loss":self.accum_train_loss,
                    "accum_val_loss":self.accum_val_loss
                }
            )
            f.write(json2log)

    def _count_dirs(self, logdir = "modellogs"):
        """Counts number of directories in modellogs directory. 

        Args:
            logdir (str, optional): path to log dir. Defaults to "modellogs".

        Returns:
            int: number of dirs in folder. 
        """
        return len(next(os.walk(logdir))[1])


if __name__ == "__main__":
    # for testing 
    test_log = Logger("a", 1.1, 1.1, 1.1, dict(), [], [])
    test_log.log()
