import typing
import argparse
import logging
import sklearn.metrics
import sklearn.model_selection
import xgboost
import xgboost.callback
import pandas as pd
import sklearn
import numpy as np


class CustomCallback(xgboost.callback.TrainingCallback):
    def __init__(self, iters_per_log: int, x_test_df, y_test_df) -> None:
        super().__init__()
        self._iters_per_log = iters_per_log
        self._x_test_df = x_test_df
        self._y_test_df = y_test_df

    def after_iteration(self, model: typing.Any, epoch: int, evals_log: xgboost.callback.Dict[str, xgboost.callback.Dict[str, xgboost.callback.List[float] | xgboost.callback.List[xgboost.callback.Tuple[float]]]]) -> bool:
        if epoch % self._iters_per_log == 0:
            x_test_matrix = xgboost.DMatrix(data=self._x_test_df.values)
            y_pred = model.predict(x_test_matrix)
            logging.info(y_pred)


def main():
    '''
    Mount the datasets folder to /mnt/datasets.
    These files should exist.
    + /mnt/datasets/x_train.csv
    + /mnt/datasets/y_train.csv
    + /mnt/datasets/x_test.csv
    + /mnt/datasets/y_test.csv
    '''
    # Training settings
    parser = argparse.ArgumentParser(description="XGBoost")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--ne", type=int, default=1000, metavar="NE", 
                        help="n estimators (default:1000)")
    parser.add_argument("--booster", type=str, choices=["gbtree", "gblinear", "dart"], default="gbtree", 
                        help="Choose the booster", metavar="B")
    parser.add_argument("--rs", type=int, default=1, metavar="RS",
                        help="random state (default: 1)")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "cuda"], default="cuda", 
                        help="Choose the device", metavar="DEV")

    args = parser.parse_args()
    print(args)

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        level=logging.DEBUG)
    
    x_train_df = pd.read_csv("/mnt/datasets/x_train.csv")
    y_train_df = pd.read_csv("/mnt/datasets/y_train.csv")
    x_test_df = pd.read_csv("/mnt/datasets/x_test.csv")
    y_test_df = pd.read_csv("/mnt/datasets/y_test.csv")

    model = xgboost.XGBClassifier(
        n_estimators=args.ne, 
        learning_rate=args.lr, 
        booster=args.booster, 
        device=args.device, 
        callbacks=[CustomCallback(iters_per_log=100, x_test_df=x_test_df, y_test_df=y_test_df)]
    )
    model.fit(x_train_df.values, y_train_df.values)

    y_pred = model.predict(x_test_df.values)
    accuracy = sklearn.metrics.accuracy_score(y_test_df.values, y_pred)
    msg = f"accuracy={accuracy}\n"
    logging.info(msg)

if __name__ == '__main__':
    main()