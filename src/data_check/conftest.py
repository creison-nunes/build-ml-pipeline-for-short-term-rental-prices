import glob
import os
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")


@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Use download() instead of file() to avoid Windows path issues caused
    # by colons in artifact names (e.g. "clean_sample.csv:latest").
    artifact_dir = run.use_artifact(request.config.option.csv).download(
        root=os.path.join(".", "artifacts", "csv_data")
    )
    csv_files = glob.glob(os.path.join(artifact_dir, "*.csv"))

    if not csv_files:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(csv_files[0])

    return df


@pytest.fixture(scope='session')
def ref_data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Use download() instead of file() to avoid Windows path issues caused
    # by colons in artifact names (e.g. "clean_sample.csv:reference").
    artifact_dir = run.use_artifact(request.config.option.ref).download(
        root=os.path.join(".", "artifacts", "ref_data")
    )
    csv_files = glob.glob(os.path.join(artifact_dir, "*.csv"))

    if not csv_files:
        pytest.fail("You must provide the --ref option on the command line")

    df = pd.read_csv(csv_files[0])

    return df


@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)
