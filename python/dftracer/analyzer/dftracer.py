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
from dftracer.utils import Indexer, AggregationConfig
from dftracer.utils.dask import (
    DFTracerUtilsDaskWorkerPlugin,
    _assign_files_by_pid,
    distributed_aggregate,
    distributed_aggregate_all,
)
from dask.distributed import Client, get_client, wait
from typing import Callable, Dict, List, Optional, Tuple

from .analyzer import Analyzer, HLM_AGG, HLM_EXTRA_COLS
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


def _worker_hlm(ipc_result, data_type, hlm_groupby, hlm_agg):
    """Compute partial HLM on a worker from IPC bytes using Arrow compute."""
    import pyarrow.compute as pc
    import time as _time

    t0 = _time.time()
    ipc_bytes = ipc_result[data_type]
    if ipc_bytes is None:
        return pd.DataFrame()
    reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
    table = reader.read_all()
    if table.num_rows == 0:
        return pd.DataFrame()

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
        return pd.DataFrame()

    agg_map = {"sum": "sum", "min": "min", "max": "max"}
    agg_specs = []
    for col, agg_fn in hlm_agg.items():
        if col in table.column_names and agg_fn in agg_map:
            agg_specs.append((col, agg_map[agg_fn]))

    t1 = _time.time()
    result = table.group_by(available_groupby).aggregate(agg_specs)
    t2 = _time.time()

    rename = {}
    for col, agg_fn in agg_specs:
        rename[f"{col}_{agg_fn}"] = col

    out = result.rename_columns(
        [rename.get(c, c) for c in result.column_names]
    ).to_pandas()
    t3 = _time.time()
    print(f"[_worker_hlm] type={data_type} rows={table.num_rows} "
          f"decode={t1-t0:.2f}s groupby={t2-t1:.2f}s to_pandas={t3-t2:.2f}s "
          f"result_rows={len(out)} total={t3-t0:.2f}s", flush=True)
    return out


def _ipc_to_pandas(ipc_bytes):
    """Decode Arrow IPC bytes to pandas with Arrow-backed string columns."""
    reader = pa.ipc.open_stream(pa.BufferReader(ipc_bytes))
    table = reader.read_all()
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))
    return table.to_pandas(types_mapper=pd.ArrowDtype)


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


class DFTracerAnalyzer(Analyzer):
    def __init__(self, preset, assign_epochs=False, **kwargs):
        super().__init__(preset, **kwargs)
        self.assign_epochs = assign_epochs

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

            # Build file list
            files = []
            directory = ""
            if os.path.isdir(trace_path):
                directory = trace_path
            else:
                # Glob pattern or single file
                matched = glob.glob(trace_path) if "*" in trace_path else [trace_path]
                files = [f for f in matched if f.endswith(".pfw") or f.endswith(".pfw.gz")]

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

            files = None
            directory = ""
            if os.path.isdir(trace_path):
                directory = trace_path
            else:
                matched = glob.glob(trace_path) if "*" in trace_path else [trace_path]
                files = [f for f in matched if f.endswith(".pfw") or f.endswith(".pfw.gz")]

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
            all_files = status.ready + status.needs_work
            file_pids = indexer.query_all_file_pids()
            index_path = status.index_path
            indexer.close()

            try:
                dask_client = client or get_client()
            except ValueError:
                dask_client = None

            if dask_client is None:
                return self.read_trace_local(trace_path)

            n_workers = len(dask_client.scheduler_info().get("workers", {})) or 1
            worker_file_ids = _assign_files_by_pid(file_pids, n_workers)

            file_id_to_path = {i: path for i, path in enumerate(all_files)}
            worker_list = list(dask_client.scheduler_info().get("workers", {}).keys())

            event_futures = []
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

            self._worker_ipc_futures = event_futures
            self._dask_client = dask_client

        with log_block("build_dask_dataframe"):
            _str = pd.ArrowDtype(pa.string())
            events_meta = pd.DataFrame({
                "cat": pd.Series(dtype=_str),
                COL_FUNC_NAME: pd.Series(dtype=_str),
                "pid": pd.Series(dtype="int64"),
                "tid": pd.Series(dtype="int64"),
                "file_hash": pd.Series(dtype=_str),
                "host_hash": pd.Series(dtype=_str),
                COL_FILE_NAME: pd.Series(dtype=_str),
                COL_HOST_NAME: pd.Series(dtype=_str),
                COL_PROC_NAME: pd.Series(dtype=_str),
                COL_IO_CAT: pd.Series(dtype="int64"),
                COL_ACC_PAT: pd.Series(dtype="int64"),
                COL_COUNT: pd.Series(dtype="int64"),
                COL_TIME: pd.Series(dtype="double[pyarrow]"),
                COL_SIZE: pd.Series(dtype="int64"),
                "time_min": pd.Series(dtype="double[pyarrow]"),
                "time_max": pd.Series(dtype="double[pyarrow]"),
                "size_min": pd.Series(dtype="int64"),
                "size_max": pd.Series(dtype="int64"),
                COL_TIME_RANGE: pd.Series(dtype="int64"),
                COL_TIME_START: pd.Series(dtype="int64"),
                COL_TIME_END: pd.Series(dtype="int64"),
            })

            def _extract_and_decode(ipc_future, key):
                ipc_bytes = ipc_future[key]
                if ipc_bytes is None:
                    return pd.DataFrame()
                return _ipc_to_pandas(ipc_bytes)

            event_delayed = [
                dask.delayed(_extract_and_decode)(dask.delayed(f), "events")
                for f in event_futures
            ]
            traces = (
                dd.from_delayed(event_delayed, meta=events_meta)
                if event_delayed
                else dd.from_pandas(events_meta, npartitions=1)
            )

            profile_delayed = [
                dask.delayed(_extract_and_decode)(dask.delayed(f), "profiles")
                for f in event_futures
            ]
            profiles = dd.from_delayed(profile_delayed, meta=events_meta) if profile_delayed else None
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

        partial_futures = [
            self._dask_client.submit(_worker_hlm, f, data_type, hlm_groupby, hlm_agg, pure=False)
            for f in self._worker_ipc_futures
        ]
        partial_results = self._dask_client.gather(partial_futures)
        non_empty = [r for r in partial_results if not r.empty]
        if not non_empty:
            return dd.from_pandas(pd.DataFrame(), npartitions=1)
        combined = pd.concat(non_empty, ignore_index=True)

        if combined.empty:
            return dd.from_pandas(combined, npartitions=1)

        available_groupby = [c for c in hlm_groupby if c in combined.columns]
        merge_agg = {col: agg for col, agg in hlm_agg.items() if col in combined.columns}
        hlm = combined.groupby(available_groupby, as_index=False).agg(merge_agg)
        hlm = hlm.replace(0, pd.NA)
        for col in bin_cols:
            if col in hlm.columns:
                hlm[col] = hlm[col].astype("Int32")
        return dd.from_pandas(hlm, npartitions=max(1, len(hlm) // 100000))

    def _compute_high_level_metrics(self, traces, view_types, partition_size):
        result = self._distributed_hlm("events", view_types, traces)
        if result is not None:
            return result
        return super()._compute_high_level_metrics(traces, view_types, partition_size)

    def _compute_profile_hlm(self, profiles, view_types, partition_size):
        result = self._distributed_hlm("profiles", view_types, profiles)
        if result is not None:
            return result
        return super()._compute_profile_hlm(profiles, view_types, partition_size)

    def postread_trace(
        self,
        traces: dd.DataFrame,
        view_types: List[ViewType],
    ) -> dd.DataFrame:
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
        host_component = df[COL_HOST_NAME] if COL_HOST_NAME in df.columns else pd.Series(pd.NA, index=df.index)
        if "host_hash" in df.columns:
            host_component = host_component.fillna(df["host_hash"])
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
