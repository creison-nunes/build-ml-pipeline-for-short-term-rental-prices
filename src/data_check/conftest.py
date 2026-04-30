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
def wandb_run(request):
    """Single W&B run shared across the entire test session. Created fresh each time (no resume)."""
    run = wandb.init(job_type="data_tests")
    yield run
    # Propagate pytest exit code so wandb marks the run as Failed when tests fail
    exit_code = 1 if request.session.testsfailed else 0
    run.finish(exit_code=exit_code)


@pytest.fixture(scope='session')
def data(request, wandb_run):
    # Use download() instead of file() to avoid Windows path issues caused
    # by colons in artifact names (e.g. "clean_sample.csv:latest").
    artifact_dir = wandb_run.use_artifact(request.config.option.csv).download(
        root=os.path.join(".", "artifacts", "csv_data")
    )
    csv_files = glob.glob(os.path.join(artifact_dir, "*.csv"))

    if not csv_files:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(csv_files[0])

    return df


@pytest.fixture(scope='session')
def ref_data(request, wandb_run):
    # Use download() instead of file() to avoid Windows path issues caused
    # by colons in artifact names (e.g. "clean_sample.csv:reference").
    artifact_dir = wandb_run.use_artifact(request.config.option.ref).download(
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
