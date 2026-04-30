#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import argparse
import logging
import os
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact.
    # Use download(root=...) instead of .file() to avoid Windows path issues
    # caused by the colon in artifact names like "sample.csv:latest".
    logger.info("Downloading artifact %s", args.input_artifact)
    artifact = run.use_artifact(args.input_artifact)
    artifact_dir = artifact.download(root=os.path.join(".", "artifacts", "input"))
    # Use the artifact manifest to get exactly the file that belongs to this
    # artifact version, instead of glob which would pick up stale files from
    # previous runs stored in the same directory.
    artifact_local_path = os.path.join(artifact_dir, list(artifact.manifest.entries)[0])

    df = pd.read_csv(artifact_local_path)

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(subset=["price"])

    # Drop price outliers
    logger.info(
        "Filtering price between %.2f and %.2f", args.min_price, args.max_price
    )
    df = df[df["price"].between(args.min_price, args.max_price)]

    # Filter rows outside NYC geographic boundaries
    # idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    # df = df[idx].copy()

    # Convert last_review to datetime
    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Cleaned data has %s rows and %s columns", *df.shape)

    logger.info("Saving cleaned data to clean_sample.csv")
    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)

    run.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name of the input artifact in W&B",
        required=True,
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the cleaned output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True,
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum price to keep when filtering outliers",
        required=True,
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum price to keep when filtering outliers",
        required=True,
    )

    args = parser.parse_args()

    go(args)
