#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import glob
import logging
import os
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="train_val_test_split")
    run.config.update(args)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    logger.info(f"Fetching artifact {args.input}")
    artifact_dir = run.use_artifact(args.input).download(
        root=os.path.join(".", "artifacts", "input_split")
    )
    csv_files = glob.glob(os.path.join(artifact_dir, "*.csv"))
    artifact_local_path = csv_files[0]

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        # NamedTemporaryFile with delete=False is required on Windows — the default
        # exclusive lock prevents pandas from reopening the file by name.
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fp:
            tmp_name = fp.name

        try:
            df.to_csv(tmp_name, index=False)
            log_artifact(
                f"{k}_data.csv",
                f"{k}_data",
                f"{k} split of dataset",
                tmp_name,
                run,
            )
        finally:
            os.remove(tmp_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
