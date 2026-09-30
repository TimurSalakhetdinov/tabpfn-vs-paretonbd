"""Fetch raw public datasets, verify hashes, and convert them to one purchase-occasion table.

A purchase occasion is one customer on one calendar day with at least one valid purchase.
Several invoices or line items on the same day collapse into one occasion. This matches the
convention used for CDNOW in the BTYD literature and is applied identically to every dataset.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

# Repository root: the working directory unless TABPFN_VS_PARETONBD_ROOT is set (the package may be installed non-editable).
ROOT = Path(os.environ.get("TABPFN_VS_PARETONBD_ROOT", Path.cwd())).resolve()
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str  # hash of the downloaded file (zip or rds)
    license: str
    attribution: str


SOURCES: dict[str, Source] = {
    "online_retail_ii": Source(
        "online_retail_ii",
        "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb",
        "CC BY 4.0",
        "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D",
    ),
    "cdnow_master": Source(
        "cdnow_master",
        "https://www.brucehardie.com/datasets/CDNOW_master.zip",
        "94081977a983e80dff8edfa58c83a97db33ec2bdd553151987f0dd2f63aaff49",
        "No license stated; publicly posted by B. Hardie for research use. Not redistributed here.",
        "Fader, P. S. and Hardie, B. G. S. (2001). Forecasting Repeat Sales at CDNOW: A Case Study. Interfaces 31(3), S94-S107.",
    ),
    "cdnow_sample": Source(
        "cdnow_sample",
        "https://www.brucehardie.com/datasets/CDNOW_sample.zip",
        "7c487a00ae6c66a49b8edcd287f81f585785dcdfdb5c9c2c8606414803aa282f",
        "No license stated; publicly posted by B. Hardie for research use. Not redistributed here.",
        "1/10th systematic sample of the CDNOW cohort (Fader and Hardie 2001); used by Abe (2009).",
    ),
    "completejourney": Source(
        "completejourney",
        "https://github.com/bradleyboehmke/completejourney/blob/master/data/transactions.rds?raw=true",
        "1fa0700033f1e5d9bb6b09e2be063d8d68474d346e95c50f2833e09d083e0007",
        "R package CC0; data originated from 84.51 Complete Journey study",
        "84.51, The Complete Journey; completejourney R package by B. Boehmke",
    ),
    "donations_1995": Source(
        "donations_1995",
        "https://www.brucehardie.com/datasets/1995_cohort_binary.zip",
        "12e5b70655471f6fa07ae43ba1b75673e95d094ecd8ee2752a6f5c68310320b5",
        "No license stated; publicly posted by B. Hardie for research use.",
        "Fader, Hardie and Shang (2010). Customer-Base Analysis in a Discrete-Time Noncontractual Setting. Marketing Science 29(6).",
    ),
}

_FILES = {
    "online_retail_ii": "online_retail_ii.zip",
    "cdnow_master": "CDNOW_master.zip",
    "cdnow_sample": "CDNOW_sample.zip",
    "completejourney": "completejourney_transactions.rds",
    "donations_1995": "1995_cohort_binary.zip",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, force: bool = False) -> Path:
    """Download a source into data/raw and verify its SHA-256. Returns the local path."""
    src = SOURCES[name]
    path = RAW / _FILES[name]
    if force or not path.exists():
        RAW.mkdir(parents=True, exist_ok=True)
        # brucehardie.com rejects the default python-requests User-Agent (HTTP 465)
        r = requests.get(src.url, timeout=300, headers={"User-Agent": "Mozilla/5.0 (tabpfn-vs-paretonbd data fetch)"})
        r.raise_for_status()
        path.write_bytes(r.content)
    got = sha256(path)
    if got != src.sha256:
        raise ValueError(f"{name}: sha256 mismatch, expected {src.sha256}, got {got}")
    return path


# --- raw loaders -----------------------------------------------------------------------


def load_online_retail_raw() -> pd.DataFrame:
    """Both workbook sheets, with the 1-9 Dec 2010 overlap (identical rows in both sheets) removed."""
    cache = INTERIM / "online_retail_ii_lines.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    z = zipfile.ZipFile(fetch("online_retail_ii"))
    xlsx = io.BytesIO(z.read("online_retail_II.xlsx"))
    sheets = pd.read_excel(xlsx, sheet_name=None, dtype={"Invoice": str, "StockCode": str, "Description": str})
    a, b = sheets["Year 2009-2010"], sheets["Year 2010-2011"]
    overlap_start = b["InvoiceDate"].min().normalize()
    a = a[a["InvoiceDate"] < overlap_start]
    df = pd.concat([a, b], ignore_index=True)
    df = df.rename(columns={"Customer ID": "CustomerID"})
    INTERIM.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def _read_fixed_width_zip(name: str, member: str, cols: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(fetch(name))
    text = z.read(member).decode("latin-1")
    return pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=cols, dtype=str)


def load_cdnow_raw(variant: str = "master") -> pd.DataFrame:
    if variant == "master":
        df = _read_fixed_width_zip("cdnow_master", "CDNOW_master.txt", ["cust", "date", "cds", "sales"])
    else:
        df = _read_fixed_width_zip("cdnow_sample", "CDNOW_sample.txt", ["cust", "sampleid", "date", "cds", "sales"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["cds"] = df["cds"].astype(int)
    df["sales"] = df["sales"].astype(float)
    return df


# --- purchase occasions ----------------------------------------------------------------

OCCASION_COLS = ["customer_id", "date", "n_tx", "units", "amount"]


def clean_online_retail_lines(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply the line-level rules and aggregate to customer-day occasions. Returns (occasions, clean lines, log).

    Rules: drop lines without a customer ID; drop cancellation invoices (prefix 'C') and lines with
    quantity <= 0 or price <= 0 (returns, adjustments, zero-price samples); drop exact duplicate lines.
    Cancellations are dropped, not netted against earlier purchases, because a cancellation does not
    remove the fact that a purchase occasion happened.
    """
    log = {"lines_after_sheet_overlap_removed": len(df)}
    log["lines_missing_customer"] = int(df["CustomerID"].isna().sum())
    df = df[df["CustomerID"].notna()]
    is_cancel = df["Invoice"].astype(str).str.startswith("C")
    log["lines_cancellation"] = int(is_cancel.sum())
    df = df[~is_cancel]
    bad = (df["Quantity"] <= 0) | (df["Price"] <= 0)
    log["lines_nonpositive_qty_or_price"] = int(bad.sum())
    df = df[~bad]
    dup = df.duplicated()
    log["lines_exact_duplicate"] = int(dup.sum())
    df = df[~dup]
    log["lines_used"] = len(df)
    df = df.assign(customer_id=df["CustomerID"].astype(int).astype(str), date=df["InvoiceDate"].dt.normalize(),
                   amount_line=df["Quantity"] * df["Price"])
    occ = (df.groupby(["customer_id", "date"], as_index=False)
             .agg(n_tx=("Invoice", "nunique"), units=("Quantity", "sum"), amount=("amount_line", "sum")))
    log["occasions"] = len(occ)
    log["customers"] = int(occ["customer_id"].nunique())
    return occ, df, log


def online_retail_occasions(return_log: bool = False):
    """Customer-day purchase occasions for Online Retail II (rules in `clean_online_retail_lines`)."""
    occ, df, log = clean_online_retail_lines(load_online_retail_raw())
    country = df.groupby("customer_id")["Country"].agg(lambda s: s.mode().iat[0])
    occ.attrs["country"] = country
    return (occ, log) if return_log else occ


def cdnow_occasions(variant: str = "master", return_log: bool = False):
    df = load_cdnow_raw(variant)
    log = {"records": len(df)}
    df = df.assign(customer_id=df["cust"].str.lstrip("0"))
    occ = (df.groupby(["customer_id", "date"], as_index=False)
             .agg(n_tx=("cds", "size"), units=("cds", "sum"), amount=("sales", "sum")))
    log["occasions"] = len(occ)
    log["same_day_records_merged"] = len(df) - len(occ)
    log["customers"] = int(occ["customer_id"].nunique())
    return (occ, log) if return_log else occ


def load_occasions(dataset: str) -> pd.DataFrame:
    if dataset == "online_retail_ii":
        return online_retail_occasions()
    if dataset in ("cdnow_master", "cdnow_sample"):
        return cdnow_occasions(dataset.split("_")[1])
    raise KeyError(dataset)
