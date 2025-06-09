import bz2
import gzip
import io
import json
import math
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from puddy.readers import (
    read_csv,
    read_data,
    read_dataframe,
    read_json_array,
    read_jsonl,
    read_parquet,
)


def normalize_nans(obj):
    """Recursively convert all float('nan') values to None in dicts/lists."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: normalize_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_nans(x) for x in obj]
    return obj


# --- JSONL TESTS ---


def create_jsonl_file(tmp_path: Path, lines: list[dict], suffix: str = "") -> Path:
    filename = "test.jsonl" + suffix
    path = tmp_path / filename
    content = "\n".join(json.dumps(line) for line in lines) + "\n"
    if suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    elif suffix == ".bz2":
        with bz2.open(path, "wb") as f:
            f.write(content.encode("utf-8"))
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_read_jsonl_plain(tmp_path: Path):
    lines = [{"a": 1}, {"b": 2}]
    path = create_jsonl_file(tmp_path, lines)
    result = list(read_jsonl(str(path)))
    assert result == lines


def test_read_jsonl_gzip(tmp_path: Path):
    lines = [{"x": 9}]
    path = create_jsonl_file(tmp_path, lines, suffix=".gz")
    result = list(read_jsonl(str(path)))
    assert result == lines


def test_read_jsonl_bz2(tmp_path: Path):
    lines = [{"y": 8}]
    path = create_jsonl_file(tmp_path, lines, suffix=".bz2")
    result = list(read_jsonl(str(path)))
    assert result == lines


def test_read_jsonl_file_like():
    content = '{"foo": 1}\n{"bar": 2}\n'
    f = io.StringIO(content)
    result = list(read_jsonl(f))
    assert result == [{"foo": 1}, {"bar": 2}]


def test_read_jsonl_invalid(tmp_path: Path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("not a json line\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_jsonl(str(path)))


# --- JSON ARRAY TESTS ---


def create_json_array_file(tmp_path: Path, arr: list[dict], suffix: str = "") -> Path:
    filename = "test.json" + suffix
    path = tmp_path / filename
    content = json.dumps(arr)
    if suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    elif suffix == ".bz2":
        with bz2.open(path, "wb") as f:
            f.write(content.encode("utf-8"))
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_read_json_array_plain(tmp_path: Path):
    arr = [{"a": 1}, {"b": 2}]
    path = create_json_array_file(tmp_path, arr)
    result = list(read_json_array(str(path)))
    assert result == arr


def test_read_json_array_gzip(tmp_path: Path):
    arr = [{"foo": 2}]
    path = create_json_array_file(tmp_path, arr, suffix=".gz")
    result = list(read_json_array(str(path)))
    assert result == arr


def test_read_json_array_bz2(tmp_path: Path):
    arr = [{"baz": 7}]
    path = create_json_array_file(tmp_path, arr, suffix=".bz2")
    result = list(read_json_array(str(path)))
    assert result == arr


def test_read_json_array_file_like():
    content = '[{"x": 1}, {"y": 2}]'
    f = io.StringIO(content)
    result = list(read_json_array(f))
    assert result == [{"x": 1}, {"y": 2}]


def test_read_json_array_invalid(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("[not a json array]", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(read_json_array(str(path)))


# --- CSV TESTS ---


def create_csv_file(
    tmp_path: Path, rows: list[dict], headers: list[str], suffix: str = ""
) -> Path:
    filename = "test.csv" + suffix
    path = tmp_path / filename
    content = ",".join(headers) + "\n"
    for row in rows:
        content += ",".join(str(row.get(h, "")) for h in headers) + "\n"
    if suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(content)
    elif suffix == ".bz2":
        with bz2.open(path, "wb") as f:
            f.write(content.encode("utf-8"))
    else:
        path.write_text(content, encoding="utf-8")
    return path


def test_read_csv_plain(tmp_path: Path):
    headers = ["a", "b"]
    rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    path = create_csv_file(tmp_path, rows, headers)
    assert list(read_csv(str(path))) == rows


def test_read_csv_gzip(tmp_path: Path):
    headers = ["x", "y"]
    rows = [{"x": "foo", "y": "bar"}]
    path = create_csv_file(tmp_path, rows, headers, suffix=".gz")
    assert list(read_csv(str(path))) == rows


def test_read_csv_bz2(tmp_path: Path):
    headers = ["n", "m"]
    rows = [{"n": "val1", "m": "val2"}]
    path = create_csv_file(tmp_path, rows, headers, suffix=".bz2")
    assert list(read_csv(str(path))) == rows


def test_read_csv_file_like():
    content = "c,d\nx,y\nz,w\n"
    fake = io.StringIO(content)
    assert list(read_csv(fake)) == [{"c": "x", "d": "y"}, {"c": "z", "d": "w"}]


def test_read_csv_chunk_size(tmp_path: Path):
    headers = ["id", "value"]
    rows = [{"id": "1", "value": "a"}, {"id": "2", "value": "b"}]
    path = create_csv_file(tmp_path, rows, headers)
    assert list(read_csv(str(path), chunk_size=1)) == rows


def test_read_csv_invalid(tmp_path: Path):
    path = tmp_path / "invalid.csv"
    path.write_text("col1,col2\nvalue1,value2\nbadrowwithoutcomma", encoding="utf-8")
    with pytest.raises(RuntimeError):
        list(read_csv(str(path)))


# --- PARQUET TESTS ---


def test_read_parquet_valid(tmp_path: Path):
    df = pd.DataFrame([{"x": 1, "y": "foo"}, {"x": 2, "y": "bar"}])
    path = tmp_path / "test.parquet"
    df.to_parquet(path)
    result = list(read_parquet(str(path)))
    assert result == df.to_dict(orient="records")


def test_read_parquet_with_batch_size(tmp_path: Path):
    df = pd.DataFrame([{"a": 1}, {"a": 2}, {"a": 3}])
    path = tmp_path / "batch.parquet"
    df.to_parquet(path)
    result = list(read_parquet(str(path), batch_size=2))
    assert result == df.to_dict(orient="records")


def test_read_parquet_invalid(tmp_path: Path):
    path = tmp_path / "bad.parquet"
    path.write_text("not a parquet file", encoding="utf-8")
    with pytest.raises(RuntimeError):
        list(read_parquet(str(path)))


# --- DATAFRAME TESTS ---


def test_read_dataframe_pandas():
    df = pd.DataFrame([{"x": 1}, {"y": 2}])
    result = list(read_dataframe(df))
    expected = df.to_dict(orient="records")
    assert normalize_nans(result) == normalize_nans(expected)


def test_read_dataframe_pyarrow():
    df = pd.DataFrame([{"x": 3}, {"y": 4}])
    table = pa.Table.from_pandas(df)
    result = list(read_dataframe(table))
    expected = df.to_dict(orient="records")
    assert normalize_nans(result) == normalize_nans(expected)


def test_read_dataframe_invalid():
    with pytest.raises(RuntimeError):
        list(read_dataframe("not a dataframe or table"))


# --- read_data() / entry point tests ---


def test_read_data_jsonl(tmp_path: Path):
    lines = [{"a": 10}, {"b": 20}]
    path = create_jsonl_file(tmp_path, lines)
    assert read_data(str(path)) == lines


def test_read_data_json_array(tmp_path: Path):
    arr = [{"a": 1}, {"a": 2}]
    path = create_json_array_file(tmp_path, arr)
    assert read_data(str(path)) == arr


def test_read_data_csv(tmp_path: Path):
    headers = ["a", "b"]
    rows = [{"a": "11", "b": "12"}, {"a": "13", "b": "14"}]
    path = create_csv_file(tmp_path, rows, headers)
    assert read_data(str(path)) == rows


def test_read_data_parquet(tmp_path: Path):
    df = pd.DataFrame([{"foo": 1, "bar": 2}])
    path = tmp_path / "foo.parquet"
    df.to_parquet(path)
    assert read_data(str(path)) == df.to_dict(orient="records")


def test_read_data_sample_size(tmp_path: Path):
    rows = [{"a": i} for i in range(5)]
    path = create_jsonl_file(tmp_path, rows)
    assert read_data(str(path), sample_size=2) == rows[:2]


def test_read_data_invalid_extension(tmp_path: Path):
    path = tmp_path / "badfile.weird"
    path.write_text("nonsense", encoding="utf-8")
    with pytest.raises(RuntimeError):
        read_data(str(path))