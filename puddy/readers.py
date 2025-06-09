import bz2
import csv
import gzip
import json
from io import BytesIO, TextIOBase
from typing import Any, Dict, Iterator, List, Optional, TextIO, Union

import ijson
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(source: Union[str, TextIO]) -> Iterator[Dict[str, Any]]:
    """
    Read a JSON Lines (JSONL) file or file-like object and yield one record (dict) at a time.
    Supports plain text, .gz, and .bz2 files. Raises JSONDecodeError on invalid JSON.
    """
    if isinstance(source, str):
        file: Any
        if source.endswith(".gz"):
            file = gzip.open(source, "rt", encoding="utf-8")
        elif source.endswith(".bz2"):
            file = bz2.open(source, "rt", encoding="utf-8")
        else:
            file = open(source, "rt", encoding="utf-8")
        should_close = True
    else:
        if isinstance(source, TextIOBase):
            raw = source.read()
            file = BytesIO(raw.encode("utf-8"))
            should_close = True
        else:
            file = source  # type: ignore
            should_close = False

    try:
        for idx, raw_line in enumerate(file, start=1):
            line_data = raw_line
            if isinstance(line_data, memoryview):
                line_data = line_data.tobytes()
            if isinstance(line_data, (bytes, bytearray)):
                line_data = line_data.decode("utf-8")
            if not isinstance(line_data, str):
                continue
            text = line_data.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as err:
                raise json.JSONDecodeError(
                    f"Error parsing JSON on line {idx}: {err.msg}", err.doc, err.pos
                )
    finally:
        if should_close:
            file.close()


def read_json_array(source: Union[str, TextIO]) -> Iterator[Dict[str, Any]]:
    """
    Read a JSON array from a file or file-like object and yield each element as a dict.
    Supports plain JSON, .gz, and .bz2 files via streaming parsing. Raises JSONDecodeError on errors.
    """
    if isinstance(source, str):
        file: Any
        if source.endswith(".gz"):
            file = gzip.open(source, "rb")
        elif source.endswith(".bz2"):
            file = bz2.open(source, "rb")
        else:
            file = open(source, "rb")
        should_close = True
    else:
        if isinstance(source, TextIOBase):
            raw = source.read()
            file = BytesIO(raw.encode("utf-8"))
            should_close = True
        else:
            file = source  # type: ignore
            should_close = False

    try:
        for item in ijson.items(file, "item"):
            yield item
    except Exception as err:
        raise json.JSONDecodeError(f"Error parsing JSON array: {err}", "", 0)
    finally:
        if should_close:
            file.close()


def read_csv(
    source: Union[str, TextIO],
    delimiter: str = ",",
    encoding: str = "utf-8",
    chunk_size: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Read a CSV file and yield each row as a dict.
    Always converts all values to str.
    Supports plain text, .gz, .bz2 files, and file-like objects.
    If chunk_size is None, streams via csv.DictReader;
    else uses pandas.read_csv with chunksize.
    """
    if isinstance(source, str):
        file: Any
        if source.endswith(".gz"):
            file = gzip.open(source, "rt", encoding=encoding)
        elif source.endswith(".bz2"):
            file = bz2.open(source, "rt", encoding=encoding)
        else:
            file = open(source, "rt", encoding=encoding)
        should_close = True
    else:
        file = source
        should_close = False

    try:
        if chunk_size is None:
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                if any(v is None for v in row.values()):
                    raise RuntimeError(f"Malformed CSV row at line {reader.line_num}")
                yield {k: v for k, v in row.items()}
        else:
            df_iter = pd.read_csv(
                source if isinstance(source, str) else file,
                delimiter=delimiter,
                encoding=encoding,
                chunksize=chunk_size,
                dtype=str,
            )
            for df_chunk in df_iter:
                for record in df_chunk.to_dict(orient="records"):
                    yield {k: v for k, v in record.items()}
    except Exception as err:
        raise RuntimeError(f"Error reading CSV: {err}") from err
    finally:
        if should_close:
            file.close()


def read_parquet(
    source: Union[str, TextIO], batch_size: Optional[int] = None
) -> Iterator[Dict[str, Any]]:
    """
    Read a Parquet file and yield each row as a dict.
    Uses PyArrow for streaming row batches.
    """
    if isinstance(source, str):
        try:
            pq_file = pq.ParquetFile(source)
        except Exception as err:
            raise RuntimeError(f"Error opening Parquet: {err}") from err
        iterator = (
            pq_file.iter_batches(batch_size=batch_size)
            if batch_size is not None
            else pq_file.iter_batches()
        )
        should_close = False
    else:
        raw = source.read() if hasattr(source, "read") else b""
        if isinstance(raw, bytes):
            buffer_data = raw
        elif isinstance(raw, str):
            buffer_data = raw.encode("utf-8")
        else:
            buffer_data = str(raw).encode("utf-8")
        file_obj = BytesIO(buffer_data)
        try:
            pq_file = pq.ParquetFile(file_obj)
        except Exception as err:
            raise RuntimeError(f"Error opening Parquet: {err}") from err
        iterator = (
            pq_file.iter_batches(batch_size=batch_size)
            if batch_size is not None
            else pq_file.iter_batches()
        )
        should_close = True

    try:
        for batch in iterator:
            for rec in batch.to_pylist():
                yield rec
    except Exception as err:
        raise RuntimeError(f"Error reading Parquet: {err}") from err
    finally:
        if should_close:
            file_obj.close()  # type: ignore


def read_dataframe(source: Union[pd.DataFrame, pa.Table]) -> Iterator[Dict[str, Any]]:
    """
    Read rows from a pandas DataFrame or PyArrow Table, yielding dicts. Raises RuntimeError on unsupported type.
    """
    try:
        if isinstance(source, pd.DataFrame):
            for rec in source.to_dict(orient="records"):
                yield rec  # type: ignore
        elif isinstance(source, pa.Table):
            for batch in source.to_batches():
                for rec in batch.to_pylist():
                    yield rec  # type: ignore
        else:
            raise RuntimeError(f"Unsupported type: {type(source)}")
    except Exception as err:
        raise RuntimeError(f"Error reading DataFrame: {err}") from err


def read_data(
    source: Union[str, TextIO, pd.DataFrame, pa.Table],
    *,
    sample_size: Optional[int] = None,
) -> list[Dict[str, Any]]:
    """
    Entry point: Read records from various sources into a list of dicts.

    Parameters:
        source: path, file-like, DataFrame, or Table
        sample_size: max records to sample (None for all records)

    Returns:
        List of records as dicts, with all NaN values replaced by None.
    """
    # Select reader based on type or extension
    if isinstance(source, (pd.DataFrame, pa.Table)):
        records = read_dataframe(source)
    else:
        if isinstance(source, str) and source.endswith(".jsonl"):
            records = read_jsonl(source)
        elif isinstance(source, str) and source.endswith(".json"):
            records = read_json_array(source)
        elif isinstance(source, str) and source.endswith(".csv"):
            records = read_csv(source)
        elif isinstance(source, str) and source.endswith(".parquet"):
            records = read_parquet(source)
        else:
            raise RuntimeError(f"Unsupported source type or extension: {source}")

    # Apply sampling
    if sample_size is not None and sample_size >= 0:
        result = []
        for idx, rec in enumerate(records):
            if idx >= sample_size:
                break
            result.append(rec)
    else:
        result = list(records)

    # Normalize NaN to None for all results
    def _replace_nan_with_none(
        obj: Union[Dict[str, Any], List[Any], float, int, str, None],
    ) -> Any:
        """
        Recursively replace all float NaN values with None in lists and dicts.
        """
        if isinstance(obj, float) and pd.isna(obj):
            return None
        elif isinstance(obj, dict):
            return {k: _replace_nan_with_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_replace_nan_with_none(item) for item in obj]
        else:
            return obj

    return [_replace_nan_with_none(rec) for rec in result]
