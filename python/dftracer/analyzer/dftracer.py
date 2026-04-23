import dask
import dask.bag as db
import dask.dataframe as dd
import glob
import json
import math
import numpy as np
import os
import pandas as pd
import portion as I
import structlog
import pyarrow as pa
import pyarrow.compute as pc
from dftracer.utils import Indexer, AggregationConfig
from dftracer.utils.dask import (
    DFTracerUtilsDaskWorkerPlugin,
    _assign_files_by_pid,
    distributed_aggregate,
    distributed_aggregate_all,
    distributed_index as _distributed_index,
)
from dask.distributed import Client, get_client, wait
from typing import Callable, Dict, List, Optional, Tuple

from .analyzer import Analyzer, HLM_AGG, HLM_EXTRA_COLS
from .analysis_utils import (
    build_view_rename_map,
    derive_call_stats,
    fix_dtypes,
    fix_std_cols,
    set_unique_counts,
)
from .utils.dask_agg import unique_set, unique_set_flatten
from .constants import (
    COL_ACC_PAT,
    COL_COUNT,
    COL_FILE_NAME,
    COL_FUNC_NAME,
    COL_HOST_NAME,
    COL_IO_CAT,
    COL_PROC_NAME,
    COL_SIZE,
    COL_TIME,
    COL_TIME_END,
    COL_TIME_RANGE,
    COL_TIME_START,
    POSIX_IO_CAT_MAPPING,
    POSIX_METADATA_FUNCTIONS,
    IOCategory,
)
from .types import ReadTraceResult, ViewType
from .utils.log_utils import log_block

logger = structlog.get_logger()

CAT_POSIX = "POSIX"
CAT_STDIO = "STDIO"
IGNORED_FILE_PATTERNS = [
    "/dev/",
    "/etc/",
    "/gapps/python",
    "/lib/python",
    "/proc/",
    "/software/",
    "/sys/",
    "/usr/lib",
    "/usr/tce/backend",
    "/usr/tce/packages",
    "/venv",
    "__pycache__",
]
IGNORED_FUNC_NAMES = [
    "DLIOBenchmark.__init__",
    # 'DLIOBenchmark._train',
    "DLIOBenchmark.initialize",
    # 'DLIOBenchmark.run',
    "FileStorage.__init__",
    "TorchDataset.__init__",
    # "TorchDataset.worker_init",
]
IGNORED_FUNC_PATTERNS = [
    "Checkpointing.__init__",
    "Checkpointing.finalize",
    "Checkpointing.get_tensor",
    "DataLoader.__init__",
    "DataLoader.finalize",
    "DataLoader.get_tensor",
    "DataLoader.next",
    "Framework.get_loader",
    "Framework.init_loader",
    "Framework.is_nativeio_available",
    "Framework.trace_object",
    "Reader.__init__",
    "Reader.load_index",
    "Reader.next",
    "Reader.read_index",
    ".save_state",
    "checkpoint_end_",
    "checkpoint_start_",
]
TRACE_COL_MAPPING = {
    "dur": COL_TIME,
    "name": COL_FUNC_NAME,
    "te": COL_TIME_END,
    "trange": COL_TIME_RANGE,
    "ts": COL_TIME_START,
}
TYPE_EVENT = 0
TYPE_FILE_HASH = 1
TYPE_HOST_HASH = 2
TYPE_STRING_HASH = 3
TYPE_METADATA = 4
TYPE_PROC_METADATA = 5
TYPE_PROFILE = 6
TYPE_SYSTEM = 7
PROFILE_COLUMN_MAPPING = {
    "count": "Int64",
    "count_max": "Int64",
    "count_min": "Int64",
    "count_sum": "Int64",
    "dft_cnt": "Int64",
    "dur": "Int64",
    "dur_max": "Int64",
    "dur_min": "Int64",
    "dur_sum": "Int64",
    "epoch": "Int64",
    "flags": "Int64",
    "offset": "Int64",
    "ret": "Int64",
    "offset_max": "Int64",
    "offset_min": "Int64",
    "offset_sum": "Int64",
    "ret_max": "Int64",
    "ret_min": "Int64",
    "ret_sum": "Int64",
    "whence": "Int64",
    "whence_max": "Int64",
    "whence_min": "Int64",
    "whence_sum": "Int64",
}
PROFILE_OUTPUT_COLUMNS = {
    "cat": "string",
    COL_FUNC_NAME: "string",
    "pid": "Int64",
    "tid": "Int64",
    "epoch": "Int64",
    "step": "Int64",
    "file_hash": "string",
    "host_hash": "string",
    COL_FILE_NAME: "string",
    COL_HOST_NAME: "string",
    COL_PROC_NAME: "string",
    COL_IO_CAT: "Int8",
    COL_ACC_PAT: "Int8",
    COL_COUNT: "Int64",
    COL_TIME: "float64",
    COL_SIZE: "Int64",
    "time_min": "float64",
    "time_max": "float64",
    "size_min": "Int64",
    "size_max": "Int64",
    "offset_min": "Int64",
    "offset_max": "Int64",
    COL_TIME_RANGE: "Int64",
    COL_TIME_START: "Int64",
    COL_TIME_END: "Int64",
}
PROFILE_MEASURE_COLUMNS = [COL_COUNT, COL_TIME, COL_SIZE]
PROFILE_STAT_COLUMNS = ["time_min", "time_max", "size_min", "size_max", "offset_min", "offset_max"]
PROFILE_IDENTITY_COLUMNS = [
    col for col in PROFILE_OUTPUT_COLUMNS
    if col not in PROFILE_MEASURE_COLUMNS and col not in PROFILE_STAT_COLUMNS
]

# System metric columns extracted from cat="sys" ph="C" events
SYSTEM_CPU_METRICS = ["user_pct", "system_pct", "iowait_pct", "idle_pct", "irq_pct", "softirq_pct"]
SYSTEM_MEMORY_METRICS = ["MemAvailable", "MemFree", "Cached", "Dirty", "Active"]
SYSTEM_COLUMN_MAPPING = {
    **{m: "float64" for m in SYSTEM_CPU_METRICS},
    **{m: "float64" for m in SYSTEM_MEMORY_METRICS},
}
SYSTEM_OUTPUT_COLUMNS = {
    "host_hash": "string",
    COL_TIME_RANGE: "Int64",
    "sys_cpu_iowait_pct": "float64",
    "sys_cpu_user_pct": "float64",
    "sys_cpu_system_pct": "float64",
    "sys_cpu_idle_pct": "float64",
    "sys_core_iowait_pct_max": "float64",
    "sys_core_iowait_pct_p95": "float64",
    "sys_mem_dirty": "float64",
    "sys_mem_cached": "float64",
    "sys_mem_available": "float64",
}


def _make_empty_hlm(hlm_groupby, hlm_agg, bin_cols):
    """Return an empty DataFrame matching the HLM meta schema."""
    int_groupby = {"pid", "tid", "io_cat", "acc_pat", "time_range"}
    time_metric_cols = {
        "time", "time_sq", "time_min", "time_max",
        "time_call_min", "time_call_max", "time_start", "time_end",
    }
    bin_set = set(bin_cols)
    data_cols = {}
    for col in hlm_agg:
        if col in hlm_groupby or col in bin_set:
            continue
        if col in time_metric_cols:
            data_cols[col] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        else:
            data_cols[col] = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
    meta = pd.DataFrame(data_cols)
    idx_arrays = []
    for col in hlm_groupby:
        if col in int_groupby:
            idx_arrays.append(pd.array([], dtype=pd.ArrowDtype(pa.int64())))
        else:
            idx_arrays.append(pd.array([], dtype=pd.ArrowDtype(pa.string())))
    if idx_arrays:
        meta.index = pd.MultiIndex.from_arrays(idx_arrays, names=list(hlm_groupby))
    return meta


def _worker_hlm_partial(ipc_result, data_type, hlm_groupby, hlm_agg, bin_cols):
    """Per-worker partial HLM from already-resident IPC bytes.

    Workers own disjoint PID sets and proc_name is always in hlm_groupby, so
    per-worker partials have disjoint keys and need no cross-worker merge.
    """
    import time as _time

    empty = lambda: _make_empty_hlm(hlm_groupby, hlm_agg, bin_cols)

    t0 = _time.time()
    ipc_bytes = ipc_result[data_type] if isinstance(ipc_result, dict) else None
    if ipc_bytes is None:
        return empty()
    reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
    table = reader.read_all()
    if table.num_rows == 0:
        return empty()

    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))

    time_col = table.column("time")
    size_col = table.column("size")
    table = table.append_column("time_sq", pc.multiply(time_col, time_col))
    size_filled = pc.if_else(pc.is_null(size_col), pa.scalar(0, pa.int64()), size_col)
    table = table.append_column("size_sq", pc.multiply(size_filled, size_filled))
    table = table.append_column("time_call_min", time_col)
    table = table.append_column("time_call_max", time_col)
    table = table.append_column("size_call_min", size_col)
    table = table.append_column("size_call_max", size_col)

    available_groupby = [c for c in hlm_groupby if c in table.column_names]
    if not available_groupby:
        return empty()

    agg_specs = []
    for col, agg_fn in hlm_agg.items():
        if col in table.column_names and agg_fn in ("sum", "min", "max"):
            agg_specs.append((col, agg_fn))

    t1 = _time.time()
    result = table.group_by(available_groupby).aggregate(agg_specs)
    t2 = _time.time()

    rename = {f"{col}_{agg_fn}": col for col, agg_fn in agg_specs}
    result = result.rename_columns([rename.get(c, c) for c in result.column_names])

    # Do lowercase + zero-to-null in Arrow (vectorized) before to_pandas.
    cat_idx = result.schema.get_field_index("cat")
    if cat_idx >= 0:
        cat_col = result.column(cat_idx)
        if pa.types.is_string(cat_col.type) or pa.types.is_large_string(cat_col.type):
            result = result.set_column(cat_idx, "cat", pc.utf8_lower(cat_col))

    groupby_set = set(available_groupby)
    for i, field in enumerate(result.schema):
        if field.name in groupby_set:
            continue
        t_ = field.type
        if pa.types.is_integer(t_) or pa.types.is_floating(t_):
            col = result.column(i)
            zero = pa.scalar(0 if pa.types.is_integer(t_) else 0.0, t_)
            null = pa.scalar(None, t_)
            result = result.set_column(
                i, field.name, pc.if_else(pc.equal(col, zero), null, col)
            )

    # Zero-copy to pandas; set MultiIndex (names are required by downstream),
    # skip sort since Dask can't preserve global order across partitions.
    pdf = result.to_pandas(types_mapper=pd.ArrowDtype)
    pdf = pdf.set_index(available_groupby)
    t3 = _time.time()
    print(
        f"[_worker_hlm_partial] type={data_type} in_rows={table.num_rows} "
        f"decode={t1-t0:.2f}s groupby={t2-t1:.2f}s finalize={t3-t2:.2f}s "
        f"out_rows={len(pdf)} total={t3-t0:.2f}s",
        flush=True,
    )
    return pdf


def _ipc_to_pandas(ipc_bytes):
    """Decode Arrow IPC bytes to pandas with Arrow-backed string columns."""
    reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
    table = reader.read_all()
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))
    return table.to_pandas()


def _partial_arrow_view_groupby(
    df,
    view_type,
    full_cols,
    sum_cols,
    min_cols,
    max_cols,
    set_cols_items,
):
    """Per-partition Arrow groupby emitting mergeable partial aggregates.

    full_cols => five partial cols per input: sum, count, sumsq, min, max.
    sum/min/max_cols emit a single col each.
    set_cols_items is a list of (col_name, dd.Aggregation) whose chunk fn
    produces the per-group BetterSet (handled via pandas; Arrow can't express
    a BetterSet union).
    """
    from betterset import BetterSet as S

    view_type_in_index = (
        isinstance(df.index, pd.MultiIndex) and view_type in df.index.names
    ) or (df.index.name == view_type)
    work = df.reset_index() if view_type_in_index else df
    if work.empty:
        empty_cols = {}
        for c in full_cols:
            empty_cols[f"{c}_sum"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            empty_cols[f"{c}_count"] = pd.Series(dtype=pd.ArrowDtype(pa.int64()))
            empty_cols[f"{c}_sumsq"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            empty_cols[f"{c}_min"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
            empty_cols[f"{c}_max"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        for c in sum_cols:
            empty_cols[f"{c}_sum"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        for c in min_cols:
            empty_cols[f"{c}_min"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        for c in max_cols:
            empty_cols[f"{c}_max"] = pd.Series(dtype=pd.ArrowDtype(pa.float64()))
        for c, _ in set_cols_items:
            empty_cols[f"{c}_unique"] = pd.Series(dtype="object")
        out = pd.DataFrame(empty_cols)
        out.index = pd.Index([], name=view_type)
        return out

    arrow_keep = [view_type]
    for lst in (full_cols, sum_cols, min_cols, max_cols):
        for c in lst:
            if c in work.columns and c not in arrow_keep:
                arrow_keep.append(c)
    tbl = pa.Table.from_pandas(work[arrow_keep], preserve_index=False)

    agg_specs = []
    for c in full_cols:
        if c not in tbl.schema.names:
            continue
        col_arr = pc.cast(tbl.column(c), pa.float64())
        tbl = tbl.append_column(f"{c}__sq", pc.multiply(col_arr, col_arr))
        agg_specs += [
            (c, "sum"), (c, "count"), (c, "min"), (c, "max"),
            (f"{c}__sq", "sum"),
        ]
    for c in sum_cols:
        if c in tbl.schema.names:
            agg_specs.append((c, "sum"))
    for c in min_cols:
        if c in tbl.schema.names:
            agg_specs.append((c, "min"))
    for c in max_cols:
        if c in tbl.schema.names:
            agg_specs.append((c, "max"))

    if agg_specs:
        result = tbl.group_by([view_type]).aggregate(agg_specs)
        out = result.to_pandas(types_mapper=pd.ArrowDtype)
        rename = {f"{c}__sq_sum": f"{c}_sumsq" for c in full_cols}
        if rename:
            out = out.rename(columns=rename)
        out = out.set_index(view_type)
    else:
        uniq = work[view_type].drop_duplicates().reset_index(drop=True)
        out = pd.DataFrame(index=pd.Index(uniq, name=view_type))

    for col, agg in set_cols_items:
        if col not in work.columns:
            continue
        sgb = work.groupby(view_type)[col]
        chunk_fn = getattr(agg, "chunk", None)
        partial = chunk_fn(sgb) if chunk_fn is not None else sgb.apply(S.flatten)
        partial.name = f"{col}_unique"
        out = out.join(partial, how="left")
    return out


def _finalize_view_partials(df, full_cols):
    """Compute mean/std per view_type row from merged partials; drop helper cols."""
    if df.empty:
        return df
    out = df.copy()
    drop = []
    for c in full_cols:
        sum_c = f"{c}_sum"
        count_c = f"{c}_count"
        sq_c = f"{c}_sumsq"
        if sum_c not in out.columns or count_c not in out.columns:
            continue
        s = out[sum_c].astype("float64")
        n = out[count_c].astype("float64")
        mean_v = s / n
        out[f"{c}_mean"] = mean_v.astype(pd.ArrowDtype(pa.float64()))
        if sq_c in out.columns:
            sq = out[sq_c].astype("float64")
            var_v = (sq - (s * s) / n) / (n - 1)
            var_v = var_v.where(var_v >= 0, 0)
            out[f"{c}_std"] = np.sqrt(var_v).astype(pd.ArrowDtype(pa.float64()))
            drop.append(sq_c)
        drop.append(count_c)
    if drop:
        out = out.drop(columns=drop)
    return out


def _worker_scan_to_ipc(files, index_path, time_granularity, time_resolution, query):
    """Dask worker task: scan aggregation and return Arrow IPC bytes per type.

    Returns compact IPC bytes, not pandas DataFrames, to minimize worker memory.
    """
    indexer = Indexer(
        files=files,
        index_dir=os.path.dirname(index_path) if index_path else "",
        require_checkpoint=False,
        require_bloom=False,
        require_manifest=False,
        require_aggregation=False,
        force_rebuild=False,
    )
    all_batches = indexer.iter_arrow_dfanalyzer_all(
        time_granularity=time_granularity,
        time_resolution=time_resolution,
        query=query,
    )
    result = {}
    for data_type in ("events", "profiles", "system"):
        batches = [pa.record_batch(b) for b in all_batches.get(data_type, [])]
        if batches:
            sink = pa.BufferOutputStream()
            writer = pa.ipc.new_stream(sink, batches[0].schema)
            for batch in batches:
                writer.write_batch(batch)
            writer.close()
            result[data_type] = sink.getvalue().to_pybytes()
        else:
            result[data_type] = None
    return result




def create_index(filename):
    index_file = f"{filename}.idx"
    if not os.path.exists(index_file):
        indexer = Indexer(filename, index_file, checkpoint_size=32 * 1024 * 1024)
        indexer.build()
        logger.debug("Creating index", filename=filename)
    return filename


def generate_batches(filename, max_bytes):
    batch_size = 4 * 1024 * 1024  # 4 MB
    for start in range(0, max_bytes, batch_size):
        # this range is intended since DFTracerJsonLinesBytesReader do
        # line boundary algorithm internally to chop incomplete line
        end = min(start + batch_size, max_bytes)
        logger.debug("Created batch", filename=filename, start=start, end=end)
        yield filename, start, end


def get_size(filename):
    size = 0
    if filename.endswith(".pfw"):
        size = os.stat(filename).st_size
    elif filename.endswith(".pfw.gz"):
        index_file = f"{filename}.idx"
        indexer = Indexer(filename, index_file)
        size = indexer.get_max_bytes()
    logger.debug("File has size", filename=filename, size=size / 1024**3)
    return filename, int(size)


def get_io_cat(func_name: str):
    if func_name in POSIX_METADATA_FUNCTIONS:
        return IOCategory.METADATA.value
    if func_name in POSIX_IO_CAT_MAPPING:
        return POSIX_IO_CAT_MAPPING[func_name].value
    return IOCategory.OTHER.value


def io_columns():
    columns = {
        "file_hash": "string",
        "host_hash": "string",
        "image_id": "Int64",
        "io_cat": "Int8",
        "size": "Int64",
        "offset": "Int64",
    }
    return columns


def io_function(json_dict: dict):
    d = {}
    d[COL_IO_CAT] = IOCategory.OTHER.value
    if "args" in json_dict:
        if "fhash" in json_dict["args"]:
            d["file_hash"] = str(json_dict["args"]["fhash"])
        if "size_sum" in json_dict["args"]:
            d["size"] = int(json_dict["args"]["size_sum"])
        elif json_dict["cat"] in [CAT_POSIX, CAT_STDIO]:
            name = json_dict["name"]
            io_cat = get_io_cat(name)
            if "ret" in json_dict["args"]:
                size = int(json_dict["args"]["ret"])
                if size > 0:
                    if io_cat in [IOCategory.READ.value, IOCategory.WRITE.value]:
                        d["size"] = size
            if "offset" in json_dict["args"]:
                offset = int(json_dict["args"]["offset"])
                if offset >= 0:
                    d["offset"] = offset
            d[COL_IO_CAT] = io_cat
        else:
            if "image_idx" in json_dict["args"]:
                image_id = int(json_dict["args"]["image_idx"])
                if image_id > 0:
                    d["image_id"] = image_id
            if "image_size" in json_dict["args"]:
                name = json_dict["name"].lower()
                # e.g. NPZReader.open image_size is not correct
                if "open" not in name:
                    size = int(json_dict["args"]["image_size"])
                    if size > 0:
                        d["size"] = size
    return d


def profile_function(json_dict: dict):
    args = json_dict.get("args", {})
    d = {}
    d[COL_IO_CAT] = IOCategory.OTHER.value
    if "fhash" in args:
        d["file_hash"] = str(args["fhash"])
    if "hhash" in args:
        d["host_hash"] = str(args["hhash"])
    if json_dict.get("cat") in [CAT_POSIX, CAT_STDIO]:
        d[COL_IO_CAT] = get_io_cat(json_dict["name"])
    for key in PROFILE_COLUMN_MAPPING:
        if key in args:
            d[key] = int(args[key])
    return d


def system_function(json_dict: dict):
    """Extract CPU/memory metric args from a cat='sys' ph='C' event."""
    args = json_dict.get("args", {})
    d = {}
    if "hhash" in args:
        d["host_hash"] = str(args["hhash"])
    for key in SYSTEM_COLUMN_MAPPING:
        if key in args:
            d[key] = float(args[key])
    return d


def load_indexed_gzip_files(filename, start, end):
    index_file = f"{filename}.idx"
    reader = Reader(filename, index_file)
    json_lines = reader.read_line_bytes_json(start, end)
    logger.debug("Read json lines", filename=filename, start=start, end=end, num_lines=len(json_lines))
    return json_lines


def load_objects_dict(
    json_dict: dict,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    final_dict = {}
    logger.debug("Loading dict", json_dict=json_dict)
    if json_dict is not None:
        try:
            ph = json_dict.get("ph")
            if "name" in json_dict:
                final_dict["name"] = json_dict["name"]
            if "cat" in json_dict:
                final_dict["cat"] = json_dict["cat"].lower()
            if "pid" in json_dict:
                final_dict["pid"] = json_dict["pid"]
            if "tid" in json_dict:
                final_dict["tid"] = json_dict["tid"]
            if "args" in json_dict:
                if "hhash" in json_dict["args"]:
                    final_dict["host_hash"] = str(json_dict["args"]["hhash"])
                if (
                    "epoch" in json_dict["args"]
                    and json_dict["args"]["epoch"] != "train"
                    and json_dict["args"]["epoch"] != "valid"
                ):
                    epoch = int(json_dict["args"]["epoch"])
                    if epoch >= 0:
                        final_dict["epoch"] = epoch
                if "step" in json_dict["args"]:
                    step = int(json_dict["args"]["step"])
                    if step >= 0:
                        final_dict["step"] = step
            if "M" == ph:
                if final_dict["name"] == "FH":
                    final_dict["type"] = TYPE_FILE_HASH
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                elif final_dict["name"] == "HH":
                    final_dict["type"] = TYPE_HOST_HASH
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                elif final_dict["name"] == "SH":
                    final_dict["type"] = TYPE_STRING_HASH
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                elif final_dict["name"] == "PR":
                    final_dict["type"] = TYPE_PROC_METADATA
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["hash"] = str(json_dict["args"]["value"])
                else:
                    final_dict["type"] = TYPE_METADATA
                    if "args" in json_dict and "name" in json_dict["args"] and "value" in json_dict["args"]:
                        final_dict["name"] = json_dict["args"]["name"]
                        final_dict["value"] = str(json_dict["args"]["value"])
            elif "C" == ph:
                is_system = json_dict.get("cat", "").lower() == "sys"
                final_dict["type"] = TYPE_SYSTEM if is_system else TYPE_PROFILE
                if "ts" in json_dict:
                    if type(json_dict["ts"]) is not int:
                        json_dict["ts"] = int(json_dict["ts"])
                    final_dict["ts"] = json_dict["ts"]
                if is_system:
                    final_dict.update(system_function(json_dict))
                else:
                    final_dict.update(profile_function(json_dict))
                    final_dict.update(extra_columns_fn(json_dict) if extra_columns_fn else {})
            else:
                final_dict["type"] = TYPE_EVENT
                if "dur" in json_dict:
                    if type(json_dict["dur"]) is not int:
                        json_dict["dur"] = int(json_dict["dur"])
                    if type(json_dict["ts"]) is not int:
                        json_dict["ts"] = int(json_dict["ts"])
                    final_dict["ts"] = json_dict["ts"]
                    final_dict["dur"] = json_dict["dur"]
                    final_dict["te"] = final_dict["ts"] + final_dict["dur"]
                    if not time_approximate:
                        final_dict["tinterval"] = I.to_string(
                            I.closed(json_dict["ts"], json_dict["ts"] + json_dict["dur"])
                        )
                final_dict.update(io_function(json_dict))
                final_dict.update(extra_columns_fn(json_dict) if extra_columns_fn else {})
            # check if all extra columns are present
            if extra_columns and not all(col in final_dict for col in extra_columns):
                missing_cols = [col for col in extra_columns if col not in final_dict]
                raise ValueError(f"Missing extra columns: {missing_cols}")
            logger.debug("Built a dictionary for dict", final_dict=final_dict)
            yield final_dict
        except ValueError as error:
            logger.error("Processing dict failed", dict=json_dict, error=error)
    return {}


def load_objects_str(
    line: str,
    time_approximate: bool,
    extra_columns: Optional[Dict[str, str]],
    extra_columns_fn: Optional[Callable[[dict], dict]],
):
    if line is not None and line != "" and len(line) > 0 and "[" != line[0] and "]" != line[0] and line != "\n":
        try:
            unicode_line = "".join([i if ord(i) < 128 else "#" for i in line])
            json_dict = json.loads(unicode_line, strict=False)
            yield from load_objects_dict(json_dict, time_approximate, extra_columns, extra_columns_fn)
        except ValueError as error:
            logger.error("Processing line failed", line=line, error=error)
    return {}


def _resolve_trace_inputs(
    trace_path: str,
    trace_groups: Optional[List[str]],
) -> Tuple[str, Optional[List[str]]]:
    """Resolve a trace path into (directory, files) for the Indexer.

    If trace_path is a directory containing manifest.json (dftracer_organize
    output) AND trace_groups is set, glob only the subdirs for the requested
    groups. Otherwise, preserve the legacy behavior (directory, or glob list).
    """
    if not os.path.isdir(trace_path):
        matched = glob.glob(trace_path) if "*" in trace_path else [trace_path]
        files = [f for f in matched if f.endswith(".pfw") or f.endswith(".pfw.gz")]
        return "", files

    manifest_path = os.path.join(trace_path, "manifest.json")
    has_manifest = os.path.isfile(manifest_path)

    if not has_manifest:
        if trace_groups:
            raise FileNotFoundError(
                f"trace_groups={trace_groups} requested but no manifest.json at "
                f"{manifest_path}. Run dftracer_organize to produce it, or unset "
                "trace_groups."
            )
        return trace_path, None

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    group_map = manifest.get("groups") or {}

    selected = trace_groups if trace_groups else sorted(group_map.keys())
    missing = [g for g in selected if g not in group_map]
    if missing:
        raise KeyError(
            f"trace_groups {missing} not found in manifest at {manifest_path}; "
            f"available groups: {sorted(group_map.keys())}"
        )

    files: List[str] = []
    for g in selected:
        subdir = os.path.join(trace_path, group_map[g])
        files.extend(glob.glob(os.path.join(subdir, "*.pfw.gz")))
        files.extend(glob.glob(os.path.join(subdir, "*.pfw")))
    return "", files


class DFTracerAnalyzer(Analyzer):
    def __init__(self, preset, assign_epochs=False, **kwargs):
        super().__init__(preset, **kwargs)
        self.assign_epochs = assign_epochs

    @staticmethod
    def build_index_distributed(
        directory: str = "",
        files: Optional[List[str]] = None,
        index_path: str = "",
        local_staging: str = "",
        lustre_staging: str = "",
        client: Optional["Client"] = None,
        checkpoint_size: int = 32 * 1024 * 1024,
        bloom_dimensions: Optional[List[str]] = None,
        build_manifest: bool = True,
        force_rebuild: bool = False,
        partition: str = "lpt",
        rebuild_root_summaries: bool = True,
        parallelism_per_worker: int = 0,
        flush_every_files: int = 0,
        aggregation: Optional["AggregationConfig"] = None,
        auto_register_plugin: bool = True,
    ) -> dict:
        """Build the dftracer index across a Dask cluster.

        DFTracer-specific: the whole SST-based pipeline assumes .pfw/.pfw.gz
        inputs. Other analyzers (Darshan, Recorder) use their own tooling.

        Steps:
            1. Parallel directory scan + LPT bin-pack files across workers.
            2. Coordinator pre-registers files and assigns file_id ranges.
            3. Each Dask worker builds per-CF SSTs under `local_staging`,
               then moves them to `lustre_staging` for coordinator ingest.
            4. If `aggregation` is given, each worker also attaches an
               SST-backed AggregationVisitor per file. Per-file aggregation
               SSTs (mixed Put+Merge operands) are produced bounded by
               AggregationVisitor::FLUSH_THRESHOLD. Cross-worker overlapping
               `(pid, time_bucket, ...)` keys are combined by the rocksdb
               merge_operator at read/compaction time.
            5. Coordinator runs bulk_ingest (one-at-a-time for content-
               addressed CFs + AGGREGATION + SYSTEM_METRICS) and
               rebuild_root_summaries.

        After this returns, `DFTracerAnalyzer.read_trace()` will find the
        index already built (including aggregation when requested) and
        skip the serial ensure_indexed phase entirely.

        Args:
            directory: Trace directory to scan. Mutually exclusive with
                `files`.
            files: Explicit file list.
            index_path: Target .dftindex on shared FS.
            local_staging: Per-worker SST build dir (prefer node-local,
                e.g. /l/ssd/dftracer_sst). Required.
            lustre_staging: Shared-FS dir the coordinator reads SSTs from.
                Defaults to local_staging when unset (single-FS mode).
            client: Dask Client. If None, a cluster-local Client is looked
                up; if that fails too, tasks run inline serially.
            auto_register_plugin: If True, install the DFTracer worker
                plugin so per-worker C++ Runtime threads match
                hw_concurrency / n_workers_on_node.
            aggregation: If given, workers fill AGGREGATION +
                SYSTEM_METRICS CFs in parallel via SST-backed
                AggregationVisitors.

        Returns:
            dict with `total_files`, `per_worker` (sizes), `index_path`,
            `artifact_batches`, and (if aggregation) `aggregation_files`.
        """
        if client is None:
            try:
                client = get_client()
            except ValueError:
                client = None

        if auto_register_plugin and client is not None:
            DFTracerAnalyzer._register_dask_plugin()

        result = _distributed_index(
            directory=directory,
            files=files,
            index_path=index_path,
            local_staging=local_staging,
            lustre_staging=lustre_staging,
            client=client,
            checkpoint_size=checkpoint_size,
            bloom_dimensions=bloom_dimensions,
            build_manifest=build_manifest,
            force_rebuild=force_rebuild,
            partition=partition,
            rebuild_root_summaries=rebuild_root_summaries,
            parallelism_per_worker=parallelism_per_worker,
            flush_every_files=flush_every_files,
            aggregation_config=aggregation,
        )

        return result

    @staticmethod
    def _register_dask_plugin():
        """Register the DFTracer Dask worker plugin if a distributed client is active.

        Computes C++ Runtime threads as hardware_concurrency / n_workers_on_node
        so the Runtime uses all available cores without oversubscription.
        """
        if DFTracerUtilsDaskWorkerPlugin is None:
            return
        try:
            client = get_client()
            scheduler_info = client.scheduler_info()
            workers = scheduler_info.get("workers", {})

            from collections import Counter

            host_counts = Counter(w["host"] for w in workers.values())

            class _AutoThreadPlugin(DFTracerUtilsDaskWorkerPlugin):
                def __init__(self, host_worker_counts):
                    super().__init__(threads=0)
                    self._host_worker_counts = host_worker_counts

                def setup(self, worker):
                    total_cpus = (
                        len(os.sched_getaffinity(0))
                        if hasattr(os, "sched_getaffinity")
                        else os.cpu_count() or 1
                    )
                    my_host = worker.address.split("://")[-1].rsplit(":", 1)[0]
                    n_local = self._host_worker_counts.get(my_host, 1)
                    self.threads = max(1, total_cpus // n_local)
                    super().setup(worker)

            client.register_plugin(_AutoThreadPlugin(dict(host_counts)))
            logger.info(
                "Registered DFTracerUtilsDaskWorkerPlugin",
                host_worker_counts=dict(host_counts),
            )
        except (ValueError, ImportError):
            pass

    def old_read_trace(self, trace_path, extra_columns, extra_columns_fn):
        with log_block("glob_files"):
            pfw_pattern, pfw_gz_pattern = [], []
            if os.path.isdir(trace_path):
                pfw_pattern = glob.glob(os.path.join(trace_path, "*.pfw"))
                pfw_gz_pattern = glob.glob(os.path.join(trace_path, "*.pfw.gz"))
            elif trace_path.endswith(".pfw.gz"):
                pfw_gz_pattern = glob.glob(trace_path) if "*" in trace_path else [trace_path]
            elif trace_path.endswith(".pfw"):
                pfw_pattern = glob.glob(trace_path) if "*" in trace_path else [trace_path]
            all_files = pfw_pattern + pfw_gz_pattern
            if not all_files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")
        logger.debug("Processing files", files=all_files)
        if len(pfw_gz_pattern) > 0:
            with log_block("create_index"):
                db.from_sequence(pfw_gz_pattern).map(create_index).compute()
                logger.info("Created index for files", num_files=len(pfw_gz_pattern))
        with log_block("sum_total_size"):
            sizes = db.from_sequence(all_files).map(get_size).compute()
            total_size = sum(size for _, size in sizes)
            logger.info("Total size of all files", total_size=total_size)
        gz_bag = None
        pfw_bag = None
        if len(pfw_gz_pattern) > 0:
            with log_block("gzip_index_and_batches"):
                logger.debug("Max bytes per file", sizes=sizes)
                json_line_delayed = []
                total_lines = 0
                for filename, max_bytes in sizes:
                    total_lines += max_bytes
                    for _, start, end in generate_batches(filename, max_bytes):
                        json_line_delayed.append((filename, start, end))

                logger.info(
                    "Loading batches",
                    num_batches=len(json_line_delayed),
                    num_files=len(pfw_gz_pattern),
                    total_lines=total_lines,
                )
                json_line_bags = []
                for filename, start, end in json_line_delayed:
                    json_line_bags.append(dask.delayed(load_indexed_gzip_files)(filename, start, end))
                json_lines = db.concat(json_line_bags)
            with log_block("parse_gzip_json_lines"):
                gz_bag = (
                    json_lines.map(
                        load_objects_dict,
                        time_approximate=self.time_approximate,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    .flatten()
                    .filter(lambda x: "name" in x)
                )
        main_bag = None
        if len(pfw_pattern) > 0:
            with log_block("parse_json_lines"):
                pfw_bag = (
                    db.read_text(pfw_pattern)
                    .map(
                        load_objects_str,
                        time_approximate=self.time_approximate,
                        extra_columns=extra_columns,
                        extra_columns_fn=extra_columns_fn,
                    )
                    .flatten()
                    .filter(lambda x: "name" in x)
                )
        if len(pfw_gz_pattern) > 0 and len(pfw_pattern) > 0:
            main_bag = db.concat([pfw_bag, gz_bag])
        elif len(pfw_gz_pattern) > 0:
            main_bag = gz_bag
        elif len(pfw_pattern) > 0:
            main_bag = pfw_bag
        if main_bag:
            self._columns = self._get_columns(extra_columns)
            with log_block("to_dataframe"):
                raw_traces = main_bag.to_dataframe(meta=self._columns)
            with log_block("_handle_metadata"):
                traces, profiles, system_events = self._handle_metadata(raw_traces)
            with log_block("compute_time_origin"):
                trace_min, profile_min, system_min = dask.compute(
                    traces["ts"].min(), profiles["ts"].min(), system_events["ts"].min()
                )
                time_origin_candidates = [
                    ts for ts in [trace_min, profile_min, system_min] if pd.notna(ts)
                ]
                time_origin = min(time_origin_candidates) if time_origin_candidates else 0
                has_profiles = pd.notna(profile_min)
                has_system = pd.notna(system_min)
                if has_profiles:
                    # DFTracer counter buckets are emitted on absolute 5s boundaries,
                    # while trace_min is arbitrary. Snap the shared origin down to the
                    # 5s profile grid so a single 5s profile bucket cannot straddle two
                    # analyzer bins and get assigned to only one time_range.
                    profile_grid_width = int(self.profile_time_granularity * self.time_resolution)
                    time_origin = (time_origin // profile_grid_width) * profile_grid_width
            self._npartitions = math.ceil(total_size / (128 * 1024**2))
            logger.debug(f"Number of partitions used are {self._npartitions}")
            with log_block("repartition+persist"):
                traces = traces.repartition(npartitions=self._npartitions).persist()
                if has_profiles:
                    profiles = profiles.repartition(npartitions=self._npartitions).persist()
                else:
                    profiles = None
            with log_block("normalize_records+persist"):
                traces = self._fix_time(traces, time_origin=time_origin).persist()
                if profiles is not None:
                    profiles = self._standardize_profiles(profiles, time_origin=time_origin).persist()
                if has_system:
                    system_metrics = self._standardize_system(system_events, time_origin=time_origin).persist()
                else:
                    system_metrics = None
            with log_block("wait_all"):
                wait_list = [traces, self._file_hashes, self._host_hashes, self._string_hashes, self._metadata]
                if profiles is not None:
                    wait_list.append(profiles)
                if system_metrics is not None:
                    wait_list.append(system_metrics)
                wait(wait_list)
        else:
            logger.error("Unable to load traces")
            exit(1)
        return ReadTraceResult(
            traces=self._rename_columns(traces),
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def read_trace_local(self, trace_path, extra_columns=None, extra_columns_fn=None):
        """Read trace using C++ aggregation pipeline.

        This is a faster alternative to read_trace() that uses the C++ Indexer
        with fused aggregation to produce pre-aggregated data directly.
        All transformation (hash resolution, time normalization, io_cat, proc_name)
        is done in C++ for performance.

        Args:
            trace_path: Directory containing trace files or glob pattern.
            extra_columns: Not used (kept for API compatibility).
            extra_columns_fn: Not used (kept for API compatibility).

        Returns:
            ReadTraceResult with traces, profiles, and system_metrics.
        """
        with log_block("cpp_indexer_setup"):
            # Configure aggregation to match analyzer time granularity
            time_interval_ms = self.time_granularity * 1000.0  # seconds to ms

            directory, files = _resolve_trace_inputs(trace_path, self.trace_groups)

            if not directory and not files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")

            indexer = Indexer(
                directory=directory,
                files=files if files else None,
                require_checkpoint=True,
                require_bloom=True,
                require_manifest=True,
                require_aggregation=AggregationConfig(
                    time_interval_ms=time_interval_ms,
                    compute_percentiles=False,
                ),
                force_rebuild=False,
            )

        with log_block("cpp_ensure_indexed"):
            status = indexer.ensure_indexed()
            logger.info(
                "C++ indexing complete",
                total_files=status.total_files,
                ready=len(status.ready),
                needs_work=len(status.needs_work),
            )

        with log_block("cpp_load_hash_tables"):
            # Store hash tables for compatibility with other methods
            file_hashes = indexer.get_hash_table("file")
            host_hashes = indexer.get_hash_table("host")
            self._file_hashes = pd.DataFrame(
                {"name": list(file_hashes.values())},
                index=pd.Index(list(file_hashes.keys()), name="hash", dtype="string"),
            )
            self._host_hashes = pd.DataFrame(
                {"name": list(host_hashes.values())},
                index=pd.Index(list(host_hashes.keys()), name="hash", dtype="string"),
            )
            self._string_hashes = pd.DataFrame(columns=["name"])
            self._metadata = pd.DataFrame(columns=["name", "value"])

        with log_block("cpp_iter_arrow_dfanalyzer"):
            # Use fused C++ API that scans all types in one pass (~3x faster)
            all_batches = indexer.iter_arrow_dfanalyzer_all(
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
            )
            event_batches = [pa.record_batch(b) for b in all_batches.get("events", [])]
            profile_batches = [pa.record_batch(b) for b in all_batches.get("profiles", [])]
            system_batches = [pa.record_batch(b) for b in all_batches.get("system", [])]

        with log_block("cpp_to_dask"):
            # Convert Arrow batches to Dask DataFrames
            if event_batches:
                events_table = pa.Table.from_batches(event_batches)
                traces = dd.from_pandas(
                    events_table.to_pandas(),
                    npartitions=max(1, len(event_batches) // 10),
                )
            else:
                traces = dd.from_pandas(
                    pd.DataFrame(columns=list(PROFILE_OUTPUT_COLUMNS.keys())),
                    npartitions=1,
                )

            if profile_batches:
                profiles_table = pa.Table.from_batches(profile_batches)
                profiles = dd.from_pandas(
                    profiles_table.to_pandas(),
                    npartitions=max(1, len(profile_batches) // 10),
                )
            else:
                profiles = None

            if system_batches:
                system_table = pa.Table.from_batches(system_batches)
                system_metrics = dd.from_pandas(
                    system_table.to_pandas(),
                    npartitions=max(1, len(system_batches) // 10),
                )
            else:
                system_metrics = None

        return ReadTraceResult(
            traces=traces,
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def read_trace(
        self,
        trace_path,
        extra_columns=None,
        extra_columns_fn=None,
        client: Optional[Client] = None,
    ):
        """Read trace using C++ aggregation pipeline.

        Uses the active Dask client for distributed execution across workers.
        Data stays on workers as Dask DataFrame partitions (no coordinator
        materialization).

        Args:
            trace_path: Directory containing trace files or glob pattern.
            extra_columns: Not used (kept for API compatibility).
            extra_columns_fn: Not used (kept for API compatibility).
            client: Dask distributed Client. If None, uses the active client.

        Returns:
            ReadTraceResult with traces, profiles, and system_metrics.
        """
        with log_block("distributed_setup"):
            time_interval_ms = self.time_granularity * 1000.0
            self._register_dask_plugin()

            directory, files = _resolve_trace_inputs(trace_path, self.trace_groups)

            if not directory and not files:
                raise FileNotFoundError("No matching .pfw or .pfw.gz files found.")

        with log_block("cpp_indexer"):
            indexer = Indexer(
                directory=directory,
                files=files,
                require_checkpoint=True,
                require_bloom=True,
                require_manifest=True,
                require_aggregation=AggregationConfig(
                    time_interval_ms=time_interval_ms,
                    compute_percentiles=False,
                ),
                force_rebuild=False,
            )
            status = indexer.ensure_indexed()

            if status.total_files == 0:
                self._file_hashes = pd.DataFrame(columns=["name"])
                self._host_hashes = pd.DataFrame(columns=["name"])
                self._string_hashes = pd.DataFrame(columns=["name"])
                self._metadata = pd.DataFrame(columns=["name", "value"])
                return ReadTraceResult(
                    traces=dd.from_pandas(
                        pd.DataFrame(columns=list(PROFILE_OUTPUT_COLUMNS.keys())),
                        npartitions=1,
                    ),
                    profiles=None,
                    profile_time_granularity=None,
                    system_metrics=None,
                )

        with log_block("distribute_scan"):
            file_id_to_path, file_pids = indexer.query_file_info()
            index_path = os.path.abspath(status.index_path)
            indexer.close()

            try:
                dask_client = client or get_client()
            except ValueError:
                dask_client = None

            if dask_client is None:
                return self.read_trace_local(trace_path)

            n_workers = len(dask_client.scheduler_info().get("workers", {})) or 1
            all_file_ids = set(file_id_to_path.keys())
            full_file_pids = {fid: file_pids.get(fid, set()) for fid in all_file_ids}
            worker_file_ids = _assign_files_by_pid(full_file_pids, n_workers)
            worker_list = list(dask_client.scheduler_info().get("workers", {}).keys())

            event_futures = []
            worker_scan_args = []
            for worker_id, fids in worker_file_ids.items():
                wfiles = [file_id_to_path[fid] for fid in fids if fid in file_id_to_path]
                if not wfiles:
                    continue
                pids = set()
                for fid in fids:
                    if fid in file_pids:
                        pids.update(file_pids[fid])
                query = None
                if pids:
                    pid_conditions = " or ".join(f"pid == {pid}" for pid in sorted(pids))
                    query = f"({pid_conditions})"
                worker_addr = worker_list[worker_id % len(worker_list)] if worker_list else None
                future = dask_client.submit(
                    _worker_scan_to_ipc,
                    wfiles,
                    index_path,
                    self.time_granularity,
                    self.time_resolution,
                    query,
                    workers=[worker_addr] if worker_addr else None,
                    pure=False,
                )
                event_futures.append(future)
                worker_scan_args.append((worker_addr, wfiles, query))

            self._worker_ipc_futures = event_futures
            self._worker_scan_args = worker_scan_args
            self._index_path = index_path
            self._dask_client = dask_client

        with log_block("build_dask_dataframe"):
            events_meta = pd.DataFrame({
                "cat": pd.Series(dtype="object"),
                COL_FUNC_NAME: pd.Series(dtype="object"),
                "pid": pd.Series(dtype="int64"),
                "tid": pd.Series(dtype="int64"),
                "file_hash": pd.Series(dtype="object"),
                "host_hash": pd.Series(dtype="object"),
                COL_FILE_NAME: pd.Series(dtype="object"),
                COL_HOST_NAME: pd.Series(dtype="object"),
                COL_PROC_NAME: pd.Series(dtype="object"),
                COL_IO_CAT: pd.Series(dtype="int64"),
                COL_ACC_PAT: pd.Series(dtype="int64"),
                COL_COUNT: pd.Series(dtype="int64"),
                COL_TIME: pd.Series(dtype="float64"),
                COL_SIZE: pd.Series(dtype="int64"),
                "time_min": pd.Series(dtype="float64"),
                "time_max": pd.Series(dtype="float64"),
                "size_min": pd.Series(dtype="int64"),
                "size_max": pd.Series(dtype="int64"),
                COL_TIME_RANGE: pd.Series(dtype="int64"),
                COL_TIME_START: pd.Series(dtype="int64"),
                COL_TIME_END: pd.Series(dtype="int64"),
            })

            def _extract_and_decode(ipc_future, key, meta):
                ipc_bytes = ipc_future[key]
                if ipc_bytes is None:
                    return meta.iloc[:0].copy()
                return _ipc_to_pandas(ipc_bytes)

            event_delayed = [
                dask.delayed(_extract_and_decode)(dask.delayed(f), "events", events_meta)
                for f in event_futures
            ]
            traces = (
                dd.from_delayed(event_delayed, meta=events_meta)
                if event_delayed
                else dd.from_pandas(events_meta, npartitions=1)
            )

            def _has_data(result_dict, key):
                return result_dict[key] is not None

            has_profiles_futures = [
                dask_client.submit(_has_data, f, "profiles", pure=False)
                for f in event_futures
            ]
            has_profiles = any(dask_client.gather(has_profiles_futures))

            if has_profiles:
                profile_delayed = [
                    dask.delayed(_extract_and_decode)(dask.delayed(f), "profiles", events_meta)
                    for f in event_futures
                ]
                profiles = dd.from_delayed(profile_delayed, meta=events_meta)
            else:
                profiles = None
            system_metrics = None

        self._file_hashes = pd.DataFrame(columns=["name"])
        self._host_hashes = pd.DataFrame(columns=["name"])
        self._string_hashes = pd.DataFrame(columns=["name"])
        self._metadata = pd.DataFrame(columns=["name", "value"])

        return ReadTraceResult(
            traces=traces,
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

    def _distributed_hlm(self, data_type, view_types, traces):
        if not hasattr(self, "_worker_ipc_futures") or not self._worker_ipc_futures:
            return None

        hlm_groupby = list(dict.fromkeys(view_types + HLM_EXTRA_COLS))
        bin_cols = [col for col in traces.columns if "_bin_" in col]

        hlm_agg = dict(HLM_AGG)
        hlm_agg.update({col: "sum" for col in bin_cols})
        hlm_agg["time_sq"] = "sum"
        hlm_agg["size_sq"] = "sum"
        hlm_agg["time_call_min"] = "min"
        hlm_agg["time_call_max"] = "max"
        hlm_agg["size_call_min"] = "min"
        hlm_agg["size_call_max"] = "max"

        # Pin HLM tasks to the worker that already holds the IPC bytes.
        worker_addrs = [a for (a, _, _) in getattr(self, "_worker_scan_args", [])]
        if len(worker_addrs) < len(self._worker_ipc_futures):
            worker_addrs += [None] * (len(self._worker_ipc_futures) - len(worker_addrs))

        partial_futures = []
        for addr, ipc_future in zip(worker_addrs, self._worker_ipc_futures):
            fut = self._dask_client.submit(
                _worker_hlm_partial,
                ipc_future,
                data_type,
                list(hlm_groupby),
                dict(hlm_agg),
                list(bin_cols),
                workers=[addr] if addr else None,
                pure=False,
            )
            partial_futures.append(fut)

        # Partitions stay on their worker; persist() ships no big pandas.
        partial_delayed = [dask.delayed(f) for f in partial_futures]
        meta = self._build_hlm_meta(hlm_groupby, hlm_agg, bin_cols)
        ddf = dd.from_delayed(partial_delayed, meta=meta)
        return ddf

    @staticmethod
    def _build_hlm_meta(hlm_groupby, hlm_agg, bin_cols):
        """Meta for the Dask DataFrame from _worker_hlm_partial."""
        return _make_empty_hlm(hlm_groupby, hlm_agg, bin_cols)

    def _compute_high_level_metrics(self, traces, view_types, partition_size):
        result = self._distributed_hlm("events", view_types, traces)
        if result is not None:
            return result
        return super()._compute_high_level_metrics(traces, view_types, partition_size)

    def _compute_profile_hlm(self, profiles, view_types, partition_size):
        result = self._distributed_hlm("profiles", view_types, profiles)
        if result is not None:
            return result
        if profiles is None:
            return None
        return super()._compute_profile_hlm(profiles, view_types, partition_size)

    def _compute_view(self, layer, records, view_key, view_type, view_types):
        from .constants import VIEW_TYPES
        import itertools as it

        keep_object_cols = (
            set(VIEW_TYPES)
            | set(view_types)
            | set(it.chain.from_iterable(self.logical_views.values()))
        )
        drop_cols = []
        for col in records.columns:
            dtype_str = str(records[col].dtype)
            if dtype_str in ("string", "category"):
                records[col] = records[col].astype("object")
                dtype_str = "object"
            if dtype_str == "object" and col not in keep_object_cols and not any(
                col.endswith(vt) for vt in keep_object_cols
            ):
                drop_cols.append(col)
        if drop_cols:
            records = records.drop(columns=drop_cols, errors="ignore")

        # Arrow-based view computation
        view_types_diff = set(VIEW_TYPES).difference(view_types)
        local_view_types = records.index._meta.names
        local_view_types_diff = set(local_view_types).difference([view_type])

        view_agg = {}
        for col in records.columns:
            if "_bin_" in col:
                view_agg[col] = ["sum"]
            elif any(map(col.endswith, view_types_diff)):
                view_agg[col] = [unique_set_flatten()]
            elif col in it.chain.from_iterable(self.logical_views.values()):
                view_agg[col] = [unique_set_flatten()]
            elif col.endswith("_sq"):
                view_agg[col] = ["sum"]
            elif col.endswith("_call_min"):
                view_agg[col] = ["min"]
            elif col.endswith("_call_max"):
                view_agg[col] = ["max"]
            elif pd.api.types.is_numeric_dtype(records[col].dtype):
                view_agg[col] = ["sum", "min", "max", "mean", "std"]
            else:
                raise TypeError(
                    f"Unsupported dtype '{records[col].dtype}' for column '{col}'"
                )
        view_agg.update({col: [unique_set()] for col in local_view_types_diff})

        # Decompose view_agg into tree-reducible partials. Each input column's
        # aggregation list is split into per-partition chunks that later merge
        # via Dask's tree-reduce (sum/min/max/set-union are all associative).
        full_cols, sum_cols, min_cols, max_cols = [], [], [], []
        set_cols_items = []
        for col, aggs in view_agg.items():
            if col not in records.columns:
                continue
            if all(isinstance(a, str) for a in aggs):
                s = set(aggs)
                if s == {"sum", "min", "max", "mean", "std"}:
                    full_cols.append(col)
                elif s == {"sum"}:
                    sum_cols.append(col)
                elif s == {"min"}:
                    min_cols.append(col)
                elif s == {"max"}:
                    max_cols.append(col)
                else:
                    raise ValueError(f"unsupported agg combo for {col}: {aggs}")
            else:
                set_cols_items.append((col, aggs[0]))

        std_cols = list(full_cols)
        records = records.map_partitions(fix_std_cols, std_cols=std_cols)

        partial_meta = self._build_partial_meta(
            records, view_type, full_cols, sum_cols, min_cols, max_cols, set_cols_items
        )
        partials = records.map_partitions(
            _partial_arrow_view_groupby,
            view_type,
            full_cols,
            sum_cols,
            min_cols,
            max_cols,
            set_cols_items,
            meta=partial_meta,
        )

        merge_aggs = {}
        for c in full_cols:
            merge_aggs[f"{c}_sum"] = "sum"
            merge_aggs[f"{c}_count"] = "sum"
            merge_aggs[f"{c}_sumsq"] = "sum"
            merge_aggs[f"{c}_min"] = "min"
            merge_aggs[f"{c}_max"] = "max"
        for c in sum_cols:
            merge_aggs[f"{c}_sum"] = "sum"
        for c in min_cols:
            merge_aggs[f"{c}_min"] = "min"
        for c in max_cols:
            merge_aggs[f"{c}_max"] = "max"
        for c, _ in set_cols_items:
            merge_aggs[f"{c}_unique"] = unique_set_flatten()

        merged = partials.groupby(view_type).agg(merge_aggs)

        final_meta = self._build_final_meta(merged, full_cols)
        final = merged.map_partitions(
            _finalize_view_partials, full_cols, meta=final_meta
        )
        final = final.rename(columns=build_view_rename_map(final.columns))
        final = final.replace(0, pd.NA)
        final = (
            final.map_partitions(derive_call_stats)
            .map_partitions(set_unique_counts, layer=layer)
            .map_partitions(fix_dtypes, time_sliced=self.time_sliced)
            .persist()
        )
        return final

    @staticmethod
    def _build_partial_meta(
        records, view_type, full_cols, sum_cols, min_cols, max_cols, set_cols_items
    ):
        in_meta = records._meta
        def _dtype_of(col, default=pd.ArrowDtype(pa.float64())):
            if col in in_meta.columns:
                return in_meta[col].dtype
            if isinstance(in_meta.index, pd.MultiIndex) and col in in_meta.index.names:
                return in_meta.index.get_level_values(col).dtype
            return default

        # Column order must exactly match what Arrow's group_by+aggregate
        # emits. For full_cols agg_specs were appended as
        # [(c, sum), (c, count), (c, min), (c, max), (c__sq, sum)] and
        # c__sq_sum is renamed to c_sumsq after to_pandas.
        cols = {}
        for c in full_cols:
            cols[f"{c}_sum"] = _dtype_of(c)
            cols[f"{c}_count"] = pd.ArrowDtype(pa.int64())
            cols[f"{c}_min"] = _dtype_of(c)
            cols[f"{c}_max"] = _dtype_of(c)
            cols[f"{c}_sumsq"] = pd.ArrowDtype(pa.float64())
        for c in sum_cols:
            cols[f"{c}_sum"] = _dtype_of(c)
        for c in min_cols:
            cols[f"{c}_min"] = _dtype_of(c)
        for c in max_cols:
            cols[f"{c}_max"] = _dtype_of(c)
        for c, _ in set_cols_items:
            cols[f"{c}_unique"] = "object"

        meta = pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in cols.items()})
        idx_dtype = _dtype_of(view_type, default=pd.ArrowDtype(pa.int64()))
        meta.index = pd.Index([], name=view_type, dtype=idx_dtype)
        return meta

    @staticmethod
    def _build_final_meta(merged, full_cols):
        # The merge step drops count/sumsq and adds mean/std per full_col.
        cols = {}
        for c in merged.columns:
            if c.endswith("_count") and c[: -len("_count")] in full_cols:
                continue
            if c.endswith("_sumsq") and c[: -len("_sumsq")] in full_cols:
                continue
            cols[c] = merged._meta[c].dtype
        for c in full_cols:
            cols[f"{c}_mean"] = pd.ArrowDtype(pa.float64())
            cols[f"{c}_std"] = pd.ArrowDtype(pa.float64())
        meta = pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in cols.items()})
        meta.index = pd.Index(
            [], name=merged._meta.index.name, dtype=merged._meta.index.dtype
        )
        return meta

    @staticmethod
    def _arrow_view_groupby(pdf: pd.DataFrame, view_type: str, view_agg: dict) -> pd.DataFrame:
        """Groupby+aggregate pandas DataFrame using pyarrow for standard aggs.

        Falls back to pandas only for ``unique_set`` / ``unique_set_flatten``
        (custom Python aggregations Arrow can't express). Output column names
        match the base class pipeline after ``flatten_column_names``:
        ``col_sum``, ``col_min``, ``col_max``, ``col_mean``, ``col_std``.
        """
        from betterset import BetterSet as S

        arrow_aggs = []
        set_cols = {}
        for col, aggs in view_agg.items():
            if col not in pdf.columns:
                continue
            for a in aggs:
                if isinstance(a, str):
                    arrow_fn = "stddev" if a == "std" else a
                    arrow_aggs.append((col, arrow_fn))
                else:
                    set_cols[col] = a

        keep_cols = [view_type] + [c for c, _ in arrow_aggs]
        keep_cols = list(dict.fromkeys(keep_cols))
        arrow_pdf = pdf[keep_cols]

        tbl = pa.Table.from_pandas(arrow_pdf, preserve_index=False)
        result = tbl.group_by([view_type]).aggregate(arrow_aggs)
        out = result.to_pandas(types_mapper=pd.ArrowDtype)

        rename = {}
        for c in out.columns:
            if c.endswith("_stddev"):
                rename[c] = c[: -len("_stddev")] + "_std"
        if rename:
            out = out.rename(columns=rename)

        if set_cols:
            # Apply the dd.Aggregation's chunk fn to the SeriesGroupBy as
            # Dask itself would. Yields a Series indexed by group key.
            for col, agg in set_cols.items():
                if col not in pdf.columns:
                    continue
                sgb = pdf.groupby(view_type)[col]
                chunk = getattr(agg, "chunk", None)
                series = chunk(sgb) if chunk is not None else sgb.apply(S.flatten)
                series = series.reset_index().rename(columns={col: f"{col}_unique"})
                out = out.merge(series, on=view_type, how="left")

        out = out.set_index(view_type)
        return out

    @staticmethod
    def _normalize_arrow_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype("object")
        if "cat" in df.columns:
            df["cat"] = df["cat"].str.lower()
        return df

    def postread_trace(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
        traces = traces.map_partitions(self._normalize_arrow_dtypes)
        with log_block("filter_files"):
            traces = traces[
                traces[COL_FILE_NAME].isna() | ~traces[COL_FILE_NAME].str.contains("|".join(IGNORED_FILE_PATTERNS))
            ]

        # Set epochs
        with log_block("assign_epochs"):
            if self.assign_epochs:
                if "epoch" not in self.preset.layer_defs:
                    raise ValueError("Epoch layer definition is missing")
                epochs = traces.query(self.preset.layer_defs["epoch"]).compute()
                epochs_with_index = epochs.sort_values(["pid", "time_start"]).reset_index(drop=True)
                epochs_with_index["epoch"] = epochs_with_index.groupby("pid").cumcount() + 1
                epoch_boundaries = epochs_with_index[["pid", "time_start", "time_end", "epoch"]]
                traces = traces.map_partitions(self._set_epochs, epoch_boundaries=epoch_boundaries)

        # Ignore redundant function calls
        with log_block("filter_functions"):
            traces = traces[~traces[COL_FUNC_NAME].isin(IGNORED_FUNC_NAMES)]
            traces = traces[~traces[COL_FUNC_NAME].str.contains("|".join(IGNORED_FUNC_PATTERNS))]

        with log_block("wait"):
            _ = wait(traces)

        with log_block("set_basic_columns"):
            traces[COL_ACC_PAT] = 0
            traces[COL_COUNT] = 1

        # drop columns that are not needed
        # if COL_FILE_NAME not in view_types:
        #     traces = traces.drop(columns=[COL_FILE_NAME], errors='ignore')
        # if COL_HOST_NAME not in view_types:
        #     traces = traces.drop(columns=[COL_HOST_NAME], errors='ignore')

        # Set batches
        # traces['batch'] = traces.groupby(['func_name', 'step']).cumcount() + 1
        # batch_counts = traces['batch'].value_counts()
        # last_valid_batch = batch_counts[batch_counts > 1].index.max()
        # traces['batch'] = traces['batch'].mask(
        #     traces['batch'] > last_valid_batch, pd.NA
        # )

        # pytorch reads images instead of batches
        # e.g. 4 workers = 0..4 images = who starts/finishes first

        # epoch and step make sense in dlio layer

        # to put step back, target variable = previous compute + my io

        # Set steps depending on time ranges
        # step_time_ranges = traces.groupby(['pid', 'epoch', 'step']).agg({'ts': min, 'te': max})
        # traces = traces.map_partitions(
        #     self._set_steps, step_time_ranges=step_time_ranges.reset_index()
        # )

        return (
            traces.map_partitions(self._set_proc_names)
            .map_partitions(self._fix_file_posix_category)
            .map_partitions(self._sanitize_size_offset)
        )

    def get_job_time(self, traces):
        return super().get_job_time(traces) / self.time_resolution

    def get_time_boundary_layer(self):
        if self.assign_epochs:
            return "epoch"
        return super().get_time_boundary_layer()

    def get_unique_file_count(self, traces: dd.DataFrame):
        return traces["file_hash"].nunique()

    def get_unique_host_count(self, traces: dd.DataFrame):
        return traces["host_hash"].nunique()

    def get_unique_process_count(self, traces: dd.DataFrame):
        return traces["pid"].nunique()

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epoch_boundaries: pd.DataFrame):
        df["epoch"] = pd.NA

        # Iterate over each epoch boundary to find matching events
        for _, epoch_boundary in epoch_boundaries.iterrows():
            pid = epoch_boundary["pid"]
            start = epoch_boundary["time_start"]
            end = epoch_boundary["time_end"]

            # Find rows in the partition that match the pid and fall within the time interval
            mask = (df["pid"] == pid) & (df["time_start"] >= start) & (df["time_start"] < end)

            # Assign the epoch number to the matching rows
            df.loc[mask, "epoch"] = epoch_boundary["epoch"]

        return df

    @staticmethod
    def _fix_file_posix_category(df: pd.DataFrame):
        base_condition = (df["cat"].str.contains("posix|stdio")) & (~df["file_name"].isna())

        # Step 1: Map file purpose suffixes first
        purpose_updates = {"/data": "_reader", "/checkpoint": "_checkpoint"}

        for path, suffix in purpose_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        # Step 2: Map filesystem suffixes
        filesystem_updates = {"/lustre": "_lustre", "/ssd": "_ssd"}

        for path, suffix in filesystem_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        return df

    @staticmethod
    def _sanitize_size_offset(df: pd.DataFrame):
        df["size"] = df["size"].replace(0, np.nan)
        if "offset" in df.columns:
            df["offset"] = df["offset"].replace(0, np.nan)
        return df

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epoch_boundaries: pd.DataFrame):
        df["epoch"] = pd.NA

        # Iterate over each epoch boundary to find matching events
        for _, epoch_boundary in epoch_boundaries.iterrows():
            pid = epoch_boundary["pid"]
            start = epoch_boundary["time_start"]
            end = epoch_boundary["time_end"]

            # Find rows in the partition that match the pid and fall within the time interval
            mask = (df["pid"] == pid) & (df["time_start"] >= start) & (df["time_start"] < end)

            # Assign the epoch number to the matching rows
            df.loc[mask, "epoch"] = epoch_boundary["epoch"]

        return df

    @staticmethod
    def _fix_file_posix_category(df: pd.DataFrame):
        base_condition = df["cat"].str.contains("posix|stdio") & ~df["file_name"].isna()

        # Step 1: Map file purpose suffixes first
        purpose_updates = {"/data": "_reader", "/checkpoint": "_checkpoint"}

        for path, suffix in purpose_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        # Step 2: Map filesystem suffixes
        filesystem_updates = {"/lustre": "_lustre", "/ssd": "_ssd"}

        for path, suffix in filesystem_updates.items():
            mask = base_condition & df["file_name"].str.contains(path)
            df.loc[mask, "cat"] = df.loc[mask, "cat"] + suffix

        return df

    def _fix_time(self, traces: dd.DataFrame, time_origin: Optional[int] = None) -> dd.DataFrame:
        time_origin = traces["ts"].min() if time_origin is None else time_origin
        traces["ts"] = traces["ts"] - time_origin
        traces["te"] = traces["ts"] + traces["dur"]
        traces["trange"] = traces["ts"] // (self.time_granularity * self.time_resolution)
        traces["ts"] = traces["ts"].astype("Int64")
        traces["te"] = traces["te"].astype("Int64")
        traces["trange"] = traces["trange"].astype("Int16")
        traces["dur"] = traces["dur"] / self.time_resolution
        return traces

    def _standardize_profiles(self, profiles: dd.DataFrame, time_origin: int) -> dd.DataFrame:
        profiles = profiles.map_partitions(self._set_proc_names)
        profiles = profiles.map_partitions(
            self._standardize_profile_partition,
            profile_time_granularity=self.profile_time_granularity,
            time_granularity=self.time_granularity,
            time_origin=time_origin,
            time_resolution=self.time_resolution,
            meta=PROFILE_OUTPUT_COLUMNS,
        )
        profiles = profiles.map_partitions(self._fix_file_posix_category).map_partitions(self._sanitize_size_offset)
        return self._coalesce_profiles(profiles)

    def _coalesce_profiles(self, profiles: dd.DataFrame) -> dd.DataFrame:
        # dft-agg-full can emit multiple counter rows for the same canonical
        # profile bucket. Collapse them here so `read_trace()` returns a stable
        # analyzer-native profile table.
        split_out = max(1, math.ceil(math.sqrt(profiles.npartitions)))
        coalesced = (
            profiles.groupby(PROFILE_IDENTITY_COLUMNS, dropna=False)
            .agg(
                {
                    COL_COUNT: "sum",
                    COL_TIME: "sum",
                    COL_SIZE: "sum",
                    "time_min": "min",
                    "time_max": "max",
                    "size_min": "min",
                    "size_max": "max",
                    "offset_min": "min",
                    "offset_max": "max",
                },
                split_out=split_out,
            )
            .reset_index()
        )
        coalesced[COL_COUNT] = coalesced[COL_COUNT].astype("Int64")
        coalesced[COL_TIME] = coalesced[COL_TIME].astype("float64")
        coalesced[COL_SIZE] = coalesced[COL_SIZE].replace(0, pd.NA).astype("Int64")
        return coalesced[list(PROFILE_OUTPUT_COLUMNS)]

    @staticmethod
    def _standardize_profile_partition(
        df: pd.DataFrame,
        profile_time_granularity: float,
        time_origin: int,
        time_granularity: float,
        time_resolution: float,
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in PROFILE_OUTPUT_COLUMNS.items()})

        df = df.copy()
        duration = df["dur_sum"].where(df["dur_sum"].notna(), df["dur"]).fillna(0)
        size = df["ret_sum"].where(df["ret_sum"].notna(), df["ret"])
        is_sized_io = df[COL_IO_CAT].isin([IOCategory.READ.value, IOCategory.WRITE.value]) & size.notna() & (size > 0)

        profile_df = pd.DataFrame(index=df.index)
        profile_df["cat"] = df["cat"].astype("string")
        profile_df[COL_FUNC_NAME] = df["name"].astype("string")
        profile_df["pid"] = df["pid"].astype("Int64")
        profile_df["tid"] = df["tid"].astype("Int64")
        profile_df["epoch"] = df["epoch"].astype("Int64")
        profile_df["step"] = df["step"].astype("Int64")
        profile_df["file_hash"] = df["file_hash"].astype("string")
        profile_df["host_hash"] = df["host_hash"].astype("string")
        profile_df[COL_FILE_NAME] = df[COL_FILE_NAME].astype("string")
        profile_df[COL_HOST_NAME] = df[COL_HOST_NAME].astype("string")
        profile_df[COL_PROC_NAME] = df[COL_PROC_NAME].astype("string")
        profile_df[COL_IO_CAT] = df[COL_IO_CAT].fillna(IOCategory.OTHER.value).astype("Int8")
        profile_df[COL_ACC_PAT] = pd.Series(0, index=df.index, dtype="Int8")
        profile_df[COL_COUNT] = df["dft_cnt"].fillna(0).astype("Int64")
        profile_df[COL_TIME] = duration.astype("float64") / time_resolution
        profile_df[COL_SIZE] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df.loc[is_sized_io, COL_SIZE] = size.loc[is_sized_io].astype("Int64")
        dur_min = df["dur_min"].where(df["dur_min"].notna(), df["dur"])
        profile_df["time_min"] = dur_min.astype("float64") / time_resolution
        dur_max = df["dur_max"].where(df["dur_max"].notna(), df["dur"])
        profile_df["time_max"] = dur_max.astype("float64") / time_resolution
        profile_df["size_min"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df["size_max"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        profile_df.loc[is_sized_io, "size_min"] = (
            df["ret_min"].where(df["ret_min"].notna(), df["ret"]).loc[is_sized_io].astype("Int64")
        )
        profile_df.loc[is_sized_io, "size_max"] = (
            df["ret_max"].where(df["ret_max"].notna(), df["ret"]).loc[is_sized_io].astype("Int64")
        )
        profile_df["offset_min"] = df["offset_min"].where(df["offset_min"].notna(), df["offset"]).astype("Int64")
        profile_df["offset_max"] = df["offset_max"].where(df["offset_max"].notna(), df["offset"]).astype("Int64")
        profile_df[COL_TIME_START] = (df["ts"] - time_origin).astype("Int64")
        profile_df[COL_TIME_END] = profile_df[COL_TIME_START] + int(profile_time_granularity * time_resolution)
        profile_df[COL_TIME_RANGE] = (
            profile_df[COL_TIME_START] // int(time_granularity * time_resolution)
        ).astype("Int64")
        return profile_df[list(PROFILE_OUTPUT_COLUMNS)]

    @staticmethod
    def _standardize_system_partition(
        df: pd.DataFrame,
        time_origin: int,
        time_granularity: float,
        time_resolution: float,
    ) -> pd.DataFrame:
        """Aggregate raw system events into per-time_range system metric rows."""
        empty = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
        if df.empty:
            return empty

        df = df.copy()
        bucket_width_us = int(time_granularity * time_resolution)
        df[COL_TIME_RANGE] = ((df["ts"] - time_origin) // bucket_width_us).astype("Int64")

        group_keys = ["host_hash", COL_TIME_RANGE]

        # Aggregate CPU (name == "cpu"): mean of samples per bucket
        agg_cpu = df[df["name"] == "cpu"]
        cpu_agg = pd.DataFrame()
        if not agg_cpu.empty:
            agg_dict = {}
            for m, out in [("iowait_pct", "sys_cpu_iowait_pct"),
                           ("user_pct", "sys_cpu_user_pct"),
                           ("system_pct", "sys_cpu_system_pct"),
                           ("idle_pct", "sys_cpu_idle_pct")]:
                if m in agg_cpu.columns:
                    agg_dict[out] = (m, "mean")
            if agg_dict:
                cpu_agg = agg_cpu.groupby(group_keys).agg(**agg_dict).reset_index()

        # Per-core cross-core stats (name starts with "cpu-")
        per_core = df[df["name"].str.startswith("cpu-")]
        core_agg = pd.DataFrame()
        if not per_core.empty and "iowait_pct" in per_core.columns:
            core_agg = per_core.groupby(group_keys).agg(
                sys_core_iowait_pct_max=("iowait_pct", "max"),
                sys_core_iowait_pct_p95=("iowait_pct", lambda x: x.quantile(0.95)),
            ).reset_index()

        # Memory (name == "memory"): mean of samples per bucket
        mem = df[df["name"] == "memory"]
        mem_agg = pd.DataFrame()
        if not mem.empty:
            mem_dict = {}
            for m, out in [("Dirty", "sys_mem_dirty"),
                           ("Cached", "sys_mem_cached"),
                           ("MemAvailable", "sys_mem_available")]:
                if m in mem.columns:
                    mem_dict[out] = (m, "mean")
            if mem_dict:
                mem_agg = mem.groupby(group_keys).agg(**mem_dict).reset_index()

        # Merge all on (host_hash, time_range)
        dfs = [d for d in [cpu_agg, core_agg, mem_agg] if not d.empty]
        if not dfs:
            return empty

        result = dfs[0]
        for d in dfs[1:]:
            result = result.merge(d, on=group_keys, how="outer")

        for col, dtype in SYSTEM_OUTPUT_COLUMNS.items():
            if col not in result.columns:
                result[col] = pd.Series(dtype=dtype)
            result[col] = result[col].astype(dtype)
        return result[list(SYSTEM_OUTPUT_COLUMNS)]

    def _standardize_system(self, system_events: dd.DataFrame, time_origin: int) -> dd.DataFrame:
        """Standardize raw system events into per-time_range metrics."""
        meta = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in SYSTEM_OUTPUT_COLUMNS.items()})
        return system_events.map_partitions(
            self._standardize_system_partition,
            time_origin=time_origin,
            time_granularity=self.time_granularity,
            time_resolution=self.time_resolution,
            meta=meta,
        )

    def _get_columns(self, extra_columns: Optional[Dict[str, str]]):
        columns = {
            "name": "string",
            "cat": "string",
            "type": "Int8",
            "pid": "Int64",
            "tid": "Int64",
            "ts": "Int64",
            "te": "Int64",
            "dur": "Int64",
            "epoch": "Int64",
            "step": "Int64",
            "tinterval": "Int64" if self.time_approximate else "string",
            "trange": "Int64",
            "level": "Int8",
        }
        metadata_columns = {
            "hash": "string",
            "host_hash": "string",
            "value": "string",
        }
        columns.update(io_columns())
        columns.update(PROFILE_COLUMN_MAPPING)
        columns.update(SYSTEM_COLUMN_MAPPING)
        columns.update(metadata_columns)
        columns.update(extra_columns or {})
        logger.debug("get_columns", columns=columns)
        return columns

    def _handle_metadata(self, raw_traces: dd.DataFrame) -> Tuple[dd.DataFrame, dd.DataFrame, dd.DataFrame]:
        is_dask = isinstance(raw_traces, dd.DataFrame)
        traces = raw_traces.query(f"type == {TYPE_EVENT}")
        profiles = raw_traces.query(f"type == {TYPE_PROFILE}")
        system_events = raw_traces.query(f"type == {TYPE_SYSTEM}")
        file_hashes = raw_traces.query(f"type == {TYPE_FILE_HASH}")[["name", "hash"]].groupby("hash").first()
        host_hashes = raw_traces.query(f"type == {TYPE_HOST_HASH}")[["name", "hash"]].groupby("hash").first()
        string_hashes = raw_traces.query(f"type == {TYPE_STRING_HASH}")[["name", "hash"]].groupby("hash").first()
        metadata = raw_traces.query(f"type == {TYPE_METADATA}")[["name", "value"]]
        file_hashes.index = file_hashes.index.astype(str)
        host_hashes.index = host_hashes.index.astype(str)
        string_hashes.index = string_hashes.index.astype(str)
        if is_dask:
            file_hashes = file_hashes.persist()
            host_hashes = host_hashes.persist()
            string_hashes = string_hashes.persist()
            metadata = metadata.persist()
        traces = self._attach_metadata(traces, file_hashes=file_hashes, host_hashes=host_hashes)
        profiles = self._attach_metadata(profiles, file_hashes=file_hashes, host_hashes=host_hashes)
        self._file_hashes = file_hashes
        self._host_hashes = host_hashes
        self._string_hashes = string_hashes
        self._metadata = metadata
        return traces, profiles, system_events

    @staticmethod
    def _attach_metadata(records: dd.DataFrame, file_hashes: dd.DataFrame, host_hashes: dd.DataFrame):
        records = records.merge(
            file_hashes.rename(columns={"name": COL_FILE_NAME}),
            how="left",
            left_on="file_hash",
            right_index=True,
        )
        records = records.merge(
            host_hashes.rename(columns={"name": COL_HOST_NAME}),
            how="left",
            left_on="host_hash",
            right_index=True,
        )
        return records

    @staticmethod
    def _rename_columns(traces: dd.DataFrame) -> dd.DataFrame:
        return traces.rename(columns=TRACE_COL_MAPPING)

    @staticmethod
    def _sanitize_size_offset(df: pd.DataFrame):
        df["size"] = df["size"].replace(0, pd.NA)
        if "offset" in df.columns:
            df["offset"] = df["offset"].replace(0, pd.NA)
        return df

    @staticmethod
    def _set_epochs(df: pd.DataFrame, epoch_boundaries: pd.DataFrame):
        df["epoch"] = pd.NA

        # Iterate over each epoch boundary to find matching events
        for _, epoch_boundary in epoch_boundaries.iterrows():
            pid = epoch_boundary["pid"]
            start = epoch_boundary["time_start"]
            end = epoch_boundary["time_end"]

            # Find rows in the partition that match the pid and fall within the time interval
            mask = (df["pid"] == pid) & (df["time_start"] >= start) & (df["time_start"] < end)

            # Assign the epoch number to the matching rows
            df.loc[mask, "epoch"] = epoch_boundary["epoch"]

        return df

    @staticmethod
    def _set_proc_names(df: pd.DataFrame):
        if COL_PROC_NAME in df.columns and df[COL_PROC_NAME].notna().any():
            return df
        host_component = df[COL_HOST_NAME].astype(str) if COL_HOST_NAME in df.columns else pd.Series(pd.NA, index=df.index)
        if "host_hash" in df.columns:
            host_component = host_component.fillna(df["host_hash"].astype(str))
        df[COL_PROC_NAME] = (
            "app#"
            + host_component.fillna("unknown").astype(str)
            + "#"
            + df["pid"].astype(str)
            + "#"
            + df["tid"].astype(str)
        )
        return df

    @staticmethod
    def _set_steps(df: pd.DataFrame, step_time_ranges: pd.DataFrame):
        mapped_traces = df.copy()

        for pid in df["pid"].unique():
            pid_trace_cond = mapped_traces["pid"] == pid
            pid_traces = mapped_traces[pid_trace_cond]
            pid_step_ranges = step_time_ranges[step_time_ranges["pid"] == pid]

            # Sort step ranges by start timestamp
            pid_step_ranges_sorted = pid_step_ranges.sort_values("ts")

            # Create bins and labels
            bins = pid_step_ranges_sorted["ts"].tolist()
            if len(bins) > 0:
                bins.append(pid_step_ranges_sorted["te"].max())
            # print(pid, bins)
            steps = pid_step_ranges_sorted["step"].tolist()

            # Use np.digitize to find bin indices
            bin_indices = np.digitize(pid_traces["ts"], bins=bins) - 1

            # Map indices to steps, leaving as None for out-of-range timestamps
            mapped_traces.loc[pid_trace_cond, "step"] = [
                steps[idx] if 0 <= idx < len(steps) else pd.NA for idx in bin_indices
            ]

        return mapped_traces
