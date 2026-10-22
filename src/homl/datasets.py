"""Download-and-cache loaders for every dataset used in Part 1."""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("HOML_DATA_DIR", REPO_ROOT / "datasets"))

BOOK_DATA_URL = "https://raw.githubusercontent.com/ageron/data/main/"
SPAM_URL = "https://spamassassin.apache.org/old/publiccorpus/"
BIKESHARE_URL = (
    "https://raw.githubusercontent.com/intro-stat-learning/ISLP/main/ISLP/data/Bikeshare.csv"
)


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def _download_and_extract(url: str, archive: Path, extracted: Path) -> Path:
    if not extracted.exists():
        _download(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(path=archive.parent, filter="data")
    return extracted


def load_housing() -> pd.DataFrame:
    """California housing districts (Chapter 2)."""
    csv = _download_and_extract(
        BOOK_DATA_URL + "housing.tgz", DATA_DIR / "housing.tgz", DATA_DIR / "housing"
    ) / "housing.csv"
    return pd.read_csv(csv)


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """MNIST: 70,000 images of 28x28 pixels, flattened to 784 features.

    Returns ``X`` (uint8 pixel intensities 0-255) and ``y`` (int64 digits).
    The first 60,000 rows are the standard training set, the last 10,000 the test set.
    """
    cache = DATA_DIR / "mnist_784.npz"
    if not cache.exists():
        from sklearn.datasets import fetch_openml

        mnist = fetch_openml("mnist_784", as_frame=False, parser="liac-arff")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache, X=mnist.data.astype(np.uint8), y=mnist.target.astype(np.int64)
        )
    data = np.load(cache)
    return data["X"], data["y"]


def load_titanic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Titanic passengers (Chapter 3, Ex 3). Returns (train, test); test has no labels."""
    folder = _download_and_extract(
        BOOK_DATA_URL + "titanic.tgz", DATA_DIR / "titanic.tgz", DATA_DIR / "titanic"
    )
    return pd.read_csv(folder / "train.csv"), pd.read_csv(folder / "test.csv")


def load_spam() -> tuple[list[bytes], list[bytes]]:
    """Apache SpamAssassin public corpus (Chapter 3, Ex 4): easy ham + spam (2003-02-28).

    Returns (ham_emails, spam_emails) as raw bytes, one item per email.
    """
    folder = DATA_DIR / "spam"
    for name, sub in [("20030228_easy_ham.tar.bz2", "easy_ham"), ("20030228_spam.tar.bz2", "spam")]:
        _download_and_extract(SPAM_URL + name, folder / name, folder / sub)

    def read_all(sub: str) -> list[bytes]:
        files = sorted(p for p in (folder / sub).iterdir() if p.name != "cmds")
        return [p.read_bytes() for p in files]

    return read_all("easy_ham"), read_all("spam")


def load_bikeshare() -> pd.DataFrame:
    """Hourly Capital Bikeshare rentals for 2011 (Chapter 2, Ex 7).

    This is the UCI Bike Sharing dataset as packaged for the ISLP book.
    """
    return pd.read_csv(_download(BIKESHARE_URL, DATA_DIR / "Bikeshare.csv"))


def load_olivetti():
    """Olivetti faces (Chapter 8): 400 images, 64x64, 40 people. Uses scikit-learn's fetcher."""
    from sklearn.datasets import fetch_olivetti_faces

    return fetch_olivetti_faces(data_home=DATA_DIR / "sklearn")
