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
from dftracer.utils import Indexer, TraceReader
from dftracer.utils.utilities import AggregatorUtility
from dask.distributed import wait

try:
    from dftracer.utils.dask import DFTracerUtilsDaskWorkerPlugin
except ImportError:
    DFTracerUtilsDaskWorkerPlugin = None
from typing import Callable, Dict, List, Optional, Tuple

from .analyzer import Analyzer
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
from .utils.log_utils import console_block, log_block

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
    col for col in PROFILE_OUTPUT_COLUMNS if col not in PROFILE_MEASURE_COLUMNS and col not in PROFILE_STAT_COLUMNS
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


def _parse_args(val):
    """Parse a single args value (JSON string or dict) into a dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str) and val:
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _process_arrow_table(table, time_approximate, extra_columns, extra_columns_fn, meta):
    """Convert a C++-normalized Arrow Table to pandas with correct dtypes.

    When normalize=True is used in iter_arrow, the C++ side already produces the
    semantic output schema (type, cat, name, ts, dur, te, io_cat, size, etc.).
    This function just converts to pandas and enforces the meta dtypes.
    """
    import pyarrow as pa

    empty = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in meta.items()})
    if table.num_rows == 0:
        return empty

    # tinterval: computed in Python only when time_approximate=False
    if not time_approximate and 'ts' in table.column_names and 'te' in table.column_names:
        import pyarrow.compute as pc
        ts_col = table.column('ts')
        te_col = table.column('te')
        both_valid = pc.and_(pc.is_valid(ts_col), pc.is_valid(te_col))
        if pc.any(both_valid).as_py():
            tinterval_list = [None] * table.num_rows
            ts_arr = ts_col.to_pylist()
            te_arr = te_col.to_pylist()
            for i in range(table.num_rows):
                if ts_arr[i] is not None and te_arr[i] is not None:
                    tinterval_list[i] = I.to_string(I.closed(ts_arr[i], te_arr[i]))
            table = table.append_column('tinterval', pa.array(tinterval_list, type=pa.string()))

    # Extra columns callback (rare per-row fallback)
    if extra_columns_fn:
        result = table.to_pandas()
        ev_or_prof = result['type'].isin([TYPE_EVENT, TYPE_PROFILE])
        if ev_or_prof.any():
            for idx in result.index[ev_or_prof]:
                row_dict = {col: result.at[idx, col] for col in result.columns if pd.notna(result.at[idx, col])}
                extra = extra_columns_fn(row_dict)
                for k, v in extra.items():
                    if k not in result.columns:
                        result[k] = pd.NA
                    result.at[idx, k] = v
        for col, dtype in meta.items():
            if col not in result.columns:
                fill = np.nan if dtype == "float64" else pd.NA
                result[col] = pd.Series(fill, index=result.index, dtype=dtype)
            else:
                try:
                    result[col] = result[col].astype(dtype)
                except (ValueError, TypeError):
                    pass
        return result[list(meta.keys())]

    result = table.to_pandas(types_mapper={
        pa.int8(): pd.Int8Dtype(),
        pa.int64(): pd.Int64Dtype(),
        pa.string(): pd.StringDtype(),
    }.get)

    for col, dtype in meta.items():
        if col not in result.columns:
            fill = np.nan if dtype == "float64" else pd.NA
            result[col] = pd.Series(fill, index=result.index, dtype=dtype)
        else:
            try:
                result[col] = result[col].astype(dtype)
            except (ValueError, TypeError):
                pass

    return result[list(meta.keys())]


def load_partition_arrow(filename, start_byte, end_byte, time_approximate, extra_columns, extra_columns_fn, meta):
    """Load a byte-range partition via Arrow zero-copy and process into the semantic schema."""
    import pyarrow as pa
    from dftracer.utils.arrow import ArrowBatch

    reader = TraceReader(filename)
    empty = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in meta.items()})

    batches = []
    for capsule in reader.iter_arrow(batch_size=10000, start_byte=start_byte, end_byte=end_byte, normalize=True):
        batches.append(ArrowBatch(capsule)._to_pa_batch())

    if not batches:
        return empty

    table = pa.concat_tables([pa.Table.from_batches([b]) for b in batches], promote_options='default')
    return _process_arrow_table(table, time_approximate, extra_columns, extra_columns_fn, meta)


class DFTracerAnalyzer(Analyzer):
    def __init__(self, preset, assign_epochs=False, **kwargs):
        super().__init__(preset, **kwargs)
        self.assign_epochs = assign_epochs

    def analyze_trace(
        self,
        trace_path,
        view_types=None,
        accuracy="optimistic",
        exclude_characteristics=None,
        extra_columns=None,
        extra_columns_fn=None,
        logical_view_types=False,
        metric_boundaries=None,
        unoverlapped_posix_only=False,
    ):
        """Override: bypass Dask entirely using PyArrow/pandas for all computation."""
        import pyarrow as pa
        import pyarrow.compute as pc
        from .analysis_utils import (
            fix_dtypes, set_size_bins, derive_call_stats,
            set_unique_counts, build_view_rename_map,
        )
        from .utils.dask_utils import flatten_column_names
        from .metrics import set_main_metrics, set_view_metrics, set_cross_layer_metrics
        from .types import AnalysisResult
        from .constants import VIEW_TYPES as _VIEW_TYPES

        if view_types is None:
            view_types = ["proc_name", "time_range"]
        if exclude_characteristics is None:
            exclude_characteristics = []
        if metric_boundaries is None:
            metric_boundaries = {}

        proc_view_types = self.ensure_proc_view_type(view_types=view_types)

        # --- Read trace & stats (C++ aggregator) ---
        with console_block("Read trace & stats"):
            read_result = self.read_trace(trace_path, extra_columns, extra_columns_fn)

            # Validate profile granularity (same check as base class)
            if read_result.profiles is not None:
                ptg = read_result.profile_time_granularity or self.profile_time_granularity
                profiles = self._validate_and_expand_profiles(
                    profiles=read_result.profiles,
                    profile_time_granularity=ptg,
                )
                read_result.profiles = profiles

            traces_pd = read_result.traces.compute()
            traces_pd = set_size_bins(traces_pd)

            traces_pd[COL_ACC_PAT] = 0

            from .types import RawStats
            profiles_pd = read_result.profiles.compute() if read_result.profiles is not None else None
            profile_count = int(profiles_pd[COL_COUNT].sum()) if profiles_pd is not None and COL_COUNT in profiles_pd.columns else 0
            trace_count = int(traces_pd[COL_COUNT].sum())
            raw_stats = RawStats(
                job_time=(traces_pd[COL_TIME_END].max() - traces_pd[COL_TIME_START].min()) / self.time_resolution if COL_TIME_START in traces_pd.columns else 0,
                time_granularity=self.time_granularity,
                time_resolution=self.time_resolution,
                trace_event_count=trace_count,
                profile_event_count=profile_count,
                total_event_count=trace_count + profile_count,
                unique_file_count=traces_pd['file_hash'].nunique() if 'file_hash' in traces_pd.columns else 0,
                unique_host_count=traces_pd['host_hash'].nunique() if 'host_hash' in traces_pd.columns else 0,
                unique_process_count=traces_pd['pid'].nunique() if 'pid' in traces_pd.columns else 0,
            )

        # --- Compute HLM (pandas groupby) ---
        with console_block("Compute high-level metrics"):
            hlm_groupby = list(dict.fromkeys(proc_view_types + ["cat", COL_IO_CAT, COL_ACC_PAT, COL_FUNC_NAME]))
            view_types_diff = list(set(_VIEW_TYPES).difference(proc_view_types))

            def _compute_hlm_pandas(df):
                bin_cols = [col for col in df.columns if "_bin_" in col]
                df = df.assign(
                    time_sq=df[COL_TIME] ** 2,
                    size_sq=df[COL_SIZE] ** 2,
                    time_call_min=df[COL_TIME],
                    time_call_max=df[COL_TIME],
                    size_call_min=df[COL_SIZE],
                    size_call_max=df[COL_SIZE],
                )
                agg = {COL_TIME: "sum", COL_COUNT: "sum", COL_SIZE: "sum"}
                agg.update({col: "sum" for col in bin_cols})
                for col in view_types_diff:
                    if col in df.columns:
                        agg[col] = lambda x: frozenset(x.dropna())
                agg["time_sq"] = "sum"
                agg["size_sq"] = "sum"
                agg["time_call_min"] = "min"
                agg["time_call_max"] = "max"
                agg["size_call_min"] = "min"
                agg["size_call_max"] = "max"
                result = df.groupby(hlm_groupby).agg(agg)
                result = result.replace(0, np.nan)
                if bin_cols:
                    result[bin_cols] = result[bin_cols].astype("Int32")
                return result

            trace_hlm = _compute_hlm_pandas(traces_pd)

            profiles_pd = read_result.profiles.compute() if read_result.profiles is not None else None
            profile_hlm = None
            if profiles_pd is not None and not profiles_pd.empty:
                profiles_pd = set_size_bins(profiles_pd)
                profiles_pd[COL_ACC_PAT] = 0
                profile_hlm = _compute_hlm_pandas(profiles_pd)

        # --- Compute views (pandas groupby) ---
        with console_block("Compute views"):
            hlms = {}
            main_views = {}
            main_indexes = {}
            views = {}
            view_keys = set()

            def _reconcile_per_layer(t_hlm, p_hlm, condition):
                """Reconcile trace + profile HLM for a single layer."""
                lt = t_hlm.copy()
                if condition:
                    lt = lt.query(condition)
                if p_hlm is None:
                    return lt
                lp = p_hlm.copy()
                if condition:
                    lp = lp.query(condition)
                if lp.empty:
                    return lt
                # Merge: keep profile rows that don't overlap with trace rows
                lt_reset = lt.reset_index()
                lp_reset = lp.reset_index()
                trace_keys = lt_reset[hlm_groupby].drop_duplicates()
                merged = lp_reset.merge(trace_keys, on=hlm_groupby, how="left", indicator=True)
                profile_only = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
                if profile_only.empty:
                    return lt
                combined = pd.concat([lt_reset, profile_only], ignore_index=True)
                result = combined.groupby(hlm_groupby).agg(
                    {col: "sum" for col in combined.columns if col not in hlm_groupby}
                ).replace(0, np.nan)
                return result

            for layer, layer_condition in self.preset.layer_defs.items():
                layer_hlm = _reconcile_per_layer(trace_hlm, profile_hlm, layer_condition)

                # _compute_main_view equivalent
                size_layers = {cl.lower() for cl in (self.preset.size_layers or [])}
                if layer.lower() not in size_layers:
                    size_cols = [c for c in layer_hlm.columns if c.startswith("size")]
                    layer_hlm = layer_hlm.drop(columns=size_cols, errors='ignore')
                    if "file_name" in layer_hlm.columns:
                        layer_hlm = layer_hlm.drop(columns=["file_name"])

                layer_hlm = self.set_layer_metrics(
                    layer_hlm.reset_index(),
                    derived_metrics=self.preset.derived_metrics[layer],
                    size_derived_metrics=(self.preset.size_derived_metrics or {}).get(layer.lower(), []),
                )

                main_agg = {}
                for col in layer_hlm.columns:
                    if col in proc_view_types or col in ["cat", COL_IO_CAT, COL_ACC_PAT, COL_FUNC_NAME]:
                        continue
                    if any(map(col.endswith, set(_VIEW_TYPES).difference(proc_view_types))):
                        main_agg[col] = lambda x: frozenset(x.dropna())
                    elif col.endswith("_call_min"):
                        main_agg[col] = "min"
                    elif col.endswith("_call_max"):
                        main_agg[col] = "max"
                    else:
                        main_agg[col] = "sum"

                layer_main_view = layer_hlm.groupby(list(proc_view_types)).agg(main_agg)
                layer_main_view = set_main_metrics(layer_main_view)
                layer_main_view = layer_main_view.replace(0, np.nan)
                layer_main_view = fix_dtypes(layer_main_view, time_sliced=self.time_sliced)

                layer_main_index = layer_main_view.index.to_frame().reset_index(drop=True)

                # _compute_view for each view permutation
                layer_views = {}
                for view_key in self.view_permutations(view_types=proc_view_types):
                    view_type = view_key[-1]
                    parent_records = layer_main_view
                    for parent_vt in view_key[:-1]:
                        parent_records = parent_records.query(
                            f"{parent_vt} in @indices",
                            local_dict={"indices": layer_views[(parent_vt,)].index},
                        )

                    records = parent_records.reset_index()
                    # Pre-grouping
                    if view_type != COL_PROC_NAME:
                        pre_agg = {}
                        for col in records.columns:
                            if col in (view_type, COL_PROC_NAME):
                                continue
                            if col.endswith("_call_min"):
                                pre_agg[col] = "min"
                            elif col.endswith("_call_max"):
                                pre_agg[col] = "max"
                            else:
                                pre_agg[col] = "sum"
                        records = records.groupby([view_type, COL_PROC_NAME]).agg(pre_agg).reset_index()

                    # Build view agg dict
                    view_agg = {}
                    local_view_types = [c for c in layer_main_view.index.names if c != view_type]
                    view_types_diff_v = set(_VIEW_TYPES).difference(proc_view_types)
                    for col in records.columns:
                        if col == view_type or col == COL_PROC_NAME:
                            continue
                        if "_bin_" in col:
                            view_agg[col] = ["sum"]
                        elif any(map(col.endswith, view_types_diff_v)):
                            view_agg[col] = [lambda x: frozenset(x.dropna())]
                        elif col.endswith("_sq"):
                            view_agg[col] = ["sum"]
                        elif col.endswith("_call_min"):
                            view_agg[col] = ["min"]
                        elif col.endswith("_call_max"):
                            view_agg[col] = ["max"]
                        elif pd.api.types.is_numeric_dtype(records[col].dtype):
                            view_agg[col] = ["sum", "min", "max", "mean", "std"]
                        elif col in local_view_types:
                            view_agg[col] = [lambda x: frozenset(x.dropna())]

                    view_agg.update({col: [lambda x: frozenset(x.dropna())] for col in local_view_types if col not in view_agg})

                    view = records.groupby([view_type]).agg(view_agg).replace(0, np.nan)
                    view = flatten_column_names(view)
                    view = view.rename(columns=build_view_rename_map(view.columns))
                    view = derive_call_stats(view)
                    view = set_unique_counts(view, layer=layer)
                    view = fix_dtypes(view, time_sliced=self.time_sliced)

                    layer_views[view_key] = view

                hlms[layer] = layer_hlm
                main_views[layer] = layer_main_view
                main_indexes[layer] = layer_main_index
                views[layer] = layer_views
                view_keys.update(layer_views.keys())

        # --- Process views ---
        with console_block("Process views"):
            flat_views = {}
            for layer in views:
                for view_key in views[layer]:
                    view = views[layer][view_key].copy()
                    view.columns = view.columns.map(lambda col: layer.lower() + "_" + col)
                    if view_key in flat_views:
                        flat_views[view_key] = flat_views[view_key].merge(view, how="outer", left_index=True, right_index=True)
                    else:
                        flat_views[view_key] = view

            for view_key in flat_views:
                view_type = view_key[-1]
                top_layer = list(self.preset.layer_defs)[0]
                time_proc_suffix = "time_sum" if self.is_view_process_based(view_key) else "time_proc_max"
                time_boundary = flat_views[view_key][f"{top_layer}_{time_proc_suffix}"].sum()
                metric_boundaries.setdefault(view_type, {})
                for layer in self.preset.layer_defs:
                    metric_boundaries[view_type][f"{layer}_{time_proc_suffix}"] = time_boundary
                flat_views[view_key] = self._process_flat_view(
                    flat_view=flat_views[view_key],
                    view_key=view_key,
                    metric_boundaries=metric_boundaries,
                )

            if self.checkpoint:
                for view_key, fv in flat_views.items():
                    name = self.get_checkpoint_name("flat_view", *list(view_key))
                    path = self.get_checkpoint_path(name)
                    fv_out = fv.copy()
                    for col in fv_out.select_dtypes(include=['object']).columns:
                        fv_out[col] = fv_out[col].apply(lambda x: str(x) if isinstance(x, frozenset) else x)
                    fv_out.to_parquet(f"{path}.parquet")

        # Wrap pandas DataFrames so they support .compute() for compatibility
        class _PandasCompat:
            """Wraps a pandas DataFrame to add .compute() → returns self."""
            def __init__(self, df):
                self._df = df
            def compute(self, **kwargs):
                return self._df
            def __getattr__(self, name):
                return getattr(self._df, name)
            def __getitem__(self, key):
                return self._df[key]
            def __len__(self):
                return len(self._df)

        def _wrap(v):
            if isinstance(v, pd.DataFrame):
                return _PandasCompat(v)
            return v

        dask_hlms = {k: _wrap(v) for k, v in hlms.items()}
        dask_main_views = {k: _wrap(v) for k, v in main_views.items()}
        dask_views = {}
        for layer in views:
            dask_views[layer] = {k: _wrap(v) for k, v in views[layer].items()}

        result = AnalysisResult(
            _hlms=dask_hlms,
            _main_views=dask_main_views,
            _metric_boundaries=metric_boundaries,
            additional_metrics={
                view_type: list(metrics.keys())
                for view_type, metrics in (self.preset.additional_metrics or {}).items()
            },
            checkpoint_dir=self.checkpoint_dir,
            flat_views=flat_views,
            layers=self.layers,
            raw_stats=raw_stats,
            view_types=proc_view_types,
            views=dask_views,
        )
        result._read_result = read_result
        result.view_types = view_types
        return result

    def _register_dask_plugin(self):
        """Register the DFTracer Dask worker plugin if a distributed client is active.

        Computes C++ Runtime threads as hardware_concurrency / n_workers_on_node
        so the Runtime uses all available cores without oversubscription.
        """
        if DFTracerUtilsDaskWorkerPlugin is None:
            return
        try:
            from dask.distributed import get_client

            client = get_client()
            scheduler_info = client.scheduler_info()
            workers = scheduler_info.get("workers", {})

            # Count workers per host
            from collections import Counter
            host_counts = Counter(w["host"] for w in workers.values())

            class _AutoThreadPlugin(DFTracerUtilsDaskWorkerPlugin):
                def __init__(self, host_worker_counts):
                    super().__init__(threads=0)
                    self._host_worker_counts = host_worker_counts

                def setup(self, worker):
                    import os
                    total_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
                    # Extract host from worker's own address (e.g. "tcp://127.0.0.1:1234" -> "127.0.0.1")
                    my_host = worker.address.split("://")[-1].rsplit(":", 1)[0]
                    n_local = self._host_worker_counts.get(my_host, 1)
                    self.threads = max(1, total_cpus // n_local)
                    super().setup(worker)

            client.register_plugin(_AutoThreadPlugin(dict(host_counts)))
            logger.info("Registered DFTracerUtilsDaskWorkerPlugin", host_worker_counts=dict(host_counts))
        except (ValueError, ImportError):
            pass

    def read_trace_legacy(self, trace_path, extra_columns, extra_columns_fn):
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

        with log_block("register_dask_plugin"):
            self._register_dask_plugin()

        if pfw_gz_pattern:
            with log_block("create_index"):
                db.from_sequence(pfw_gz_pattern).map(create_index).compute()
                logger.info("Created index for files", num_files=len(pfw_gz_pattern))

        with log_block("sum_total_size"):
            sizes = db.from_sequence(all_files).map(get_size).compute()
            total_size = sum(size for _, size in sizes)
            logger.info("Total size of all files", total_size=total_size)

        self._columns = self._get_columns(extra_columns)
        meta = self._columns
        meta_df = pd.DataFrame({col: pd.Series(dtype=dtype) for col, dtype in meta.items()})

        with log_block("create_arrow_partitions"):
            delayed_parts = []
            for filename, max_bytes in sizes:
                for _, start, end in generate_batches(filename, max_bytes):
                    delayed_parts.append(
                        dask.delayed(load_partition_arrow)(
                            filename,
                            start,
                            end,
                            self.time_approximate,
                            extra_columns,
                            extra_columns_fn,
                            meta,
                        )
                    )
            logger.info("Created Arrow partitions", num_partitions=len(delayed_parts))

        if delayed_parts:
            with log_block("to_dataframe"):
                raw_traces = dd.from_delayed(delayed_parts, meta=meta_df)
            with log_block("_handle_metadata"):
                traces, profiles, system_events = self._handle_metadata(raw_traces)
            with log_block("compute_time_origin"):
                trace_min, profile_min, system_min = dask.compute(
                    traces["ts"].min(), profiles["ts"].min(), system_events["ts"].min()
                )
                time_origin_candidates = [ts for ts in [trace_min, profile_min, system_min] if pd.notna(ts)]
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

    def _read_metadata(self, trace_path):
        """Quick pass to extract hash tables (file/host/string hashes) from trace metadata."""
        import pyarrow as pa
        from dftracer.utils.arrow import ArrowBatch

        pfw_files = []
        if os.path.isdir(trace_path):
            pfw_files = glob.glob(os.path.join(trace_path, "*.pfw")) + glob.glob(os.path.join(trace_path, "*.pfw.gz"))
        elif trace_path.endswith((".pfw", ".pfw.gz")):
            pfw_files = glob.glob(trace_path) if "*" in trace_path else [trace_path]

        file_hashes = {}
        host_hashes = {}
        string_hashes = {}
        proc_metadata = {}

        for f in pfw_files:
            reader = TraceReader(f)
            for capsule in reader.iter_arrow(batch_size=10000, normalize=True):
                batch = ArrowBatch(capsule)._to_pa_batch()
                if 'type' not in batch.column_names:
                    continue
                type_col = batch.column('type')
                for i in range(batch.num_rows):
                    t = type_col[i].as_py()
                    if t is None:
                        continue
                    if t in (TYPE_FILE_HASH, TYPE_HOST_HASH, TYPE_STRING_HASH):
                        h = batch.column('hash')[i].as_py()
                        n = batch.column('name')[i].as_py()
                        if h and n:
                            if t == TYPE_FILE_HASH:
                                file_hashes[h] = n
                            elif t == TYPE_HOST_HASH:
                                host_hashes[h] = n
                            else:
                                string_hashes[h] = n
                    elif t == TYPE_PROC_METADATA:
                        h = batch.column('hash')[i].as_py()
                        n = batch.column('name')[i].as_py()
                        if h and n:
                            proc_metadata[h] = n

        self._file_hashes_dict = file_hashes
        self._host_hashes_dict = host_hashes
        self._string_hashes_dict = string_hashes
        self._proc_metadata_dict = proc_metadata

    def read_trace(self, trace_path, extra_columns=None, extra_columns_fn=None):
        """Read and aggregate traces using C++ AggregatorUtility.

        Returns aggregated HLM-like pandas DataFrame directly, bypassing
        Dask for the heavy computation. Metadata hash tables are read
        in a separate quick pass.
        """
        import pyarrow as pa
        from dftracer.utils import Runtime

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

        if pfw_gz_pattern:
            with log_block("create_index"):
                db.from_sequence(pfw_gz_pattern).map(create_index).compute()

        with log_block("read_metadata"):
            self._read_metadata(trace_path)

        with log_block("aggregate"):
            total_cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
            rt = Runtime(threads=total_cpus)
            agg = AggregatorUtility(runtime=rt)
            time_interval_ms = self.time_granularity * self.time_resolution / 1000.0
            trace_dir = trace_path if os.path.isdir(trace_path) else os.path.dirname(trace_path)
            arrow_result = agg.process(
                trace_dir,
                time_interval_ms=time_interval_ms,
                custom_metric_fields=['offset'],
            )
            rt.shutdown()

        with log_block("to_pandas"):
            # Batches may have different schemas (e.g. events lack offset columns
            # that profiles have). Use concat_tables with promote to unify.
            from dftracer.utils.arrow import ArrowBatch
            pa_batches = [pa.record_batch(b) for b in arrow_result._batches]
            if pa_batches:
                tables = [pa.Table.from_batches([b]) for b in pa_batches]
                pa_table = pa.concat_tables(tables, promote_options='default')
            else:
                pa_table = pa.table({})
            df = pa_table.to_pandas()
            logger.info("Aggregated traces", rows=len(df), cols=len(df.columns))

        with log_block("normalize"):
            # Clamp uninitialized min values (uint64 max → NaN)
            uint64_max = np.iinfo(np.uint64).max
            for col in ['dur_min', 'size_min']:
                if col in df.columns:
                    df.loc[df[col] == uint64_max, col] = np.nan

            # Rename hash columns before resolving so _set_proc_names can find them
            if 'fhash' in df.columns:
                df = df.rename(columns={'fhash': 'file_hash'})
            if 'hhash' in df.columns:
                df = df.rename(columns={'hhash': 'host_hash'})

            # Resolve hash → name
            if 'file_hash' in df.columns:
                df[COL_FILE_NAME] = df['file_hash'].map(self._file_hashes_dict)
            if 'host_hash' in df.columns:
                df[COL_HOST_NAME] = df['host_hash'].map(self._host_hashes_dict)
            df = self._set_proc_names(df)

            # Lowercase cat to match layer conditions
            if 'cat' in df.columns:
                df['cat'] = df['cat'].str.lower()

            # IO category from function name
            if 'name' in df.columns:
                df[COL_IO_CAT] = df['name'].map(get_io_cat).astype('int8')

            # Rename to HLM schema
            rename = {
                'name': COL_FUNC_NAME,
                'dur_total': COL_TIME,
                'count': COL_COUNT,
                'size_total': COL_SIZE,
                'dur_min': 'time_call_min',
                'dur_max': 'time_call_max',
                'dur_std': 'time_std',
                'dur_mean': 'time_mean',
                'size_min': 'size_call_min',
                'size_max': 'size_call_max',
                'size_std': 'size_std',
                'size_mean': 'size_mean',
                'ts': COL_TIME_START,
                'te': COL_TIME_END,
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

            # Time normalization: make time_bucket relative
            if 'time_bucket' in df.columns:
                time_origin = df['time_bucket'].min()
                bucket_size = int(time_interval_ms * 1000)  # to microseconds
                df[COL_TIME_RANGE] = ((df['time_bucket'] - time_origin) // bucket_size).astype('int64')
                df = df.drop(columns=['time_bucket'])
            if COL_TIME_START in df.columns:
                if 'time_origin' not in dir():
                    time_origin = df[COL_TIME_START].min()
                df[COL_TIME_START] = df[COL_TIME_START] - time_origin
                df[COL_TIME_END] = df[COL_TIME_END] - time_origin

            # Convert duration from microseconds to time_resolution units
            if COL_TIME in df.columns:
                df[COL_TIME] = df[COL_TIME].astype('float64') / self.time_resolution
            for col in ['time_call_min', 'time_call_max']:
                if col in df.columns:
                    df[col] = df[col].astype('float64') / self.time_resolution

            df[COL_ACC_PAT] = 0

            # Replace zero size/offset with NaN
            if COL_SIZE in df.columns:
                df[COL_SIZE] = df[COL_SIZE].replace(0, np.nan)

        # Filtering happens in postread_trace (called by analyze_trace),
        # not here — so read_trace output matches legacy behavior.

        # Split by batch_type: 0=EVENT, 1=PROFILE, 2=SYSTEM
        events_df = df[df['batch_type'] == 0].drop(columns=['batch_type']) if 'batch_type' in df.columns else df
        profiles_df = df[df['batch_type'] == 1].drop(columns=['batch_type']) if 'batch_type' in df.columns else pd.DataFrame()
        system_df = df[df['batch_type'] == 2].drop(columns=['batch_type']) if 'batch_type' in df.columns else pd.DataFrame()

        traces = dd.from_pandas(events_df, npartitions=1)

        if not profiles_df.empty:
            # Fix profile time_end: C++ used time_granularity but profiles use profile_time_granularity
            ptg_us = int(self.profile_time_granularity * self.time_resolution)
            if COL_TIME_START in profiles_df.columns and COL_TIME_END in profiles_df.columns:
                profiles_df[COL_TIME_END] = profiles_df[COL_TIME_START] + ptg_us

            # Add time_min/time_max/size_min/size_max aliases expected by _expand_profile_buckets
            if 'time_call_min' in profiles_df.columns:
                profiles_df['time_min'] = profiles_df['time_call_min']
            if 'time_call_max' in profiles_df.columns:
                profiles_df['time_max'] = profiles_df['time_call_max']
            if 'size_call_min' in profiles_df.columns:
                profiles_df['size_min'] = profiles_df['size_call_min']
            if 'size_call_max' in profiles_df.columns:
                profiles_df['size_max'] = profiles_df['size_call_max']

            # offset: custom metrics may provide offset_min/max, or fall back to NA
            for ofs_col in ['offset_min', 'offset_max']:
                if ofs_col not in profiles_df.columns:
                    profiles_df[ofs_col] = pd.NA
                else:
                    # Clamp uint64 max to NA
                    uint64_max = np.iinfo(np.uint64).max
                    profiles_df.loc[profiles_df[ofs_col] == uint64_max, ofs_col] = pd.NA

            # Match legacy _standardize_profiles nullable dtypes
            str_cols = [COL_FUNC_NAME, COL_FILE_NAME, COL_HOST_NAME, COL_PROC_NAME,
                        'cat', 'file_hash', 'host_hash']
            for col in str_cols:
                if col in profiles_df.columns:
                    profiles_df[col] = profiles_df[col].astype('string')
            int_cols = [COL_COUNT, COL_SIZE, 'size_call_min', 'size_call_max',
                        'size_min', 'size_max', COL_TIME_RANGE, 'pid', 'tid']
            for col in int_cols:
                if col in profiles_df.columns:
                    profiles_df[col] = pd.to_numeric(profiles_df[col], errors='coerce').astype('Int64')
            profiles = dd.from_pandas(profiles_df, npartitions=1)
        else:
            profiles = None

        # System events need raw per-metric fields (CPU %, memory values)
        # that the aggregator doesn't preserve. Read them via TraceReader.
        has_system = not system_df.empty
        system_metrics = None
        if has_system:
            from dftracer.utils.arrow import ArrowBatch
            sys_batches = []
            trace_dir = trace_path if os.path.isdir(trace_path) else os.path.dirname(trace_path)
            all_files = glob.glob(os.path.join(trace_dir, "*.pfw")) + glob.glob(os.path.join(trace_dir, "*.pfw.gz"))
            for f in all_files:
                reader = TraceReader(f)
                for capsule in reader.iter_arrow(batch_size=10000, normalize=True):
                    batch = ArrowBatch(capsule)._to_pa_batch()
                    if 'type' in batch.column_names:
                        import pyarrow.compute as pc
                        mask = pc.equal(batch.column('type'), TYPE_SYSTEM)
                        filtered = batch.filter(mask)
                        if filtered.num_rows > 0:
                            sys_batches.append(filtered)
            if sys_batches:
                sys_table = pa.concat_tables(
                    [pa.Table.from_batches([b]) for b in sys_batches],
                    promote_options='default',
                )
                sys_df = sys_table.to_pandas()
                # Resolve hashes + set proc_names
                if 'host_hash' in sys_df.columns:
                    sys_df[COL_HOST_NAME] = sys_df['host_hash'].map(self._host_hashes_dict)
                sys_df = self._set_proc_names(sys_df)
                # Apply time normalization
                if 'ts' in sys_df.columns:
                    sys_df['ts'] = sys_df['ts'] - time_origin
                    sys_df['te'] = sys_df.get('te', sys_df['ts'])
                    if 'te' in sys_df.columns:
                        sys_df['te'] = sys_df['te'] - time_origin
                system_metrics = dd.from_pandas(sys_df, npartitions=1)
                system_metrics = self._standardize_system(system_metrics, time_origin=0)

        return ReadTraceResult(
            traces=traces,
            profiles=profiles,
            profile_time_granularity=self.profile_time_granularity if profiles is not None else None,
            system_metrics=system_metrics,
        )

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
        profile_df[COL_TIME_RANGE] = (profile_df[COL_TIME_START] // int(time_granularity * time_resolution)).astype(
            "Int64"
        )
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
            for m, out in [
                ("iowait_pct", "sys_cpu_iowait_pct"),
                ("user_pct", "sys_cpu_user_pct"),
                ("system_pct", "sys_cpu_system_pct"),
                ("idle_pct", "sys_cpu_idle_pct"),
            ]:
                if m in agg_cpu.columns:
                    agg_dict[out] = (m, "mean")
            if agg_dict:
                cpu_agg = agg_cpu.groupby(group_keys).agg(**agg_dict).reset_index()

        # Per-core cross-core stats (name starts with "cpu-")
        per_core = df[df["name"].str.startswith("cpu-")]
        core_agg = pd.DataFrame()
        if not per_core.empty and "iowait_pct" in per_core.columns:
            core_agg = (
                per_core.groupby(group_keys)
                .agg(
                    sys_core_iowait_pct_max=("iowait_pct", "max"),
                    sys_core_iowait_pct_p95=("iowait_pct", lambda x: x.quantile(0.95)),
                )
                .reset_index()
            )

        # Memory (name == "memory"): mean of samples per bucket
        mem = df[df["name"] == "memory"]
        mem_agg = pd.DataFrame()
        if not mem.empty:
            mem_dict = {}
            for m, out in [
                ("Dirty", "sys_mem_dirty"),
                ("Cached", "sys_mem_cached"),
                ("MemAvailable", "sys_mem_available"),
            ]:
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
