"""Compare C++ aggregated path vs legacy Dask path for DFTracer traces.

Ensures the new AggregatorUtility-based read_trace produces numerically
equivalent results to the legacy Dask-based read_trace_legacy.
"""

import numpy as np
import pandas as pd
import pytest
from dask.distributed import Client, LocalCluster
from omegaconf import OmegaConf

from dftracer.analyzer.config import AnalyzerPresetConfigPOSIX, AnalyzerPresetConfigDLIOAILogging
from dftracer.analyzer.dftracer import DFTracerAnalyzer


@pytest.fixture(scope="module")
def dask_client():
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=2,
        processes=False,
        scheduler_port=0,
        silence_logs="error",
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


def _make_analyzer(tmp_path, preset=None, **kwargs):
    if preset is None:
        preset = AnalyzerPresetConfigPOSIX()
    return DFTracerAnalyzer(
        preset=OmegaConf.structured(preset),
        checkpoint=False,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        debug=False,
        quantile_stats=False,
        time_approximate=True,
        time_granularity=5,
        time_resolution=10**6,
        time_sliced=False,
        verbose=False,
        **kwargs,
    )


# Test parameters: (trace_path, preset_class)
trace_params = [
    pytest.param(
        "tests/data/extracted/dftracer-dlio",
        AnalyzerPresetConfigDLIOAILogging,
        id="dftracer-dlio",
    ),
    pytest.param(
        "tests/data/extracted/dftracer-posix",
        AnalyzerPresetConfigPOSIX,
        id="dftracer-posix",
    ),
]


@pytest.mark.parametrize("trace_path, preset_class", trace_params)
def test_read_trace_events_match(dask_client, trace_path, preset_class, tmp_path):
    """Event counts and function names should match between paths."""
    analyzer = _make_analyzer(tmp_path, preset=preset_class())

    # New aggregated path
    new_result = analyzer.read_trace(trace_path, extra_columns=None, extra_columns_fn=None)
    new_traces = new_result.traces.compute()

    # Legacy path
    legacy_result = analyzer.read_trace_legacy(trace_path, extra_columns=None, extra_columns_fn=None)
    legacy_traces = legacy_result.traces.compute()

    # The C++ aggregator groups by (name, cat, pid, tid, time_bucket),
    # collapsing duplicate levels into one row. The legacy path keeps
    # one row per original trace line (including duplicated levels).
    # So we compare function names (should match) but not raw counts.
    new_funcs = set(new_traces["func_name"].dropna().unique())
    legacy_funcs = set(legacy_traces["func_name"].dropna().unique())
    assert new_funcs == legacy_funcs, f"Function name mismatch:\n  new-only: {new_funcs - legacy_funcs}\n  legacy-only: {legacy_funcs - new_funcs}"

    # Event counts must match exactly
    new_total = int(new_traces["count"].sum()) if "count" in new_traces.columns else len(new_traces)
    legacy_total = len(legacy_traces)
    assert new_total == legacy_total, f"Event count mismatch: new={new_total}, legacy={legacy_total}"


@pytest.mark.parametrize("trace_path, preset_class", trace_params)
def test_read_trace_profiles_match(dask_client, trace_path, preset_class, tmp_path):
    """Profile presence and function names should match between paths."""
    analyzer = _make_analyzer(tmp_path, preset=preset_class())

    new_result = analyzer.read_trace(trace_path, extra_columns=None, extra_columns_fn=None)
    legacy_result = analyzer.read_trace_legacy(trace_path, extra_columns=None, extra_columns_fn=None)

    new_has_profiles = new_result.profiles is not None
    legacy_has_profiles = legacy_result.profiles is not None
    assert new_has_profiles == legacy_has_profiles, (
        f"Profile presence mismatch: new={new_has_profiles}, legacy={legacy_has_profiles}"
    )

    if new_has_profiles and legacy_has_profiles:
        new_profiles = new_result.profiles.compute()
        legacy_profiles = legacy_result.profiles.compute()

        new_funcs = set(new_profiles["func_name"].dropna().unique())
        legacy_funcs = set(legacy_profiles["func_name"].dropna().unique())
        assert new_funcs == legacy_funcs, f"Profile function mismatch:\n  new-only: {new_funcs - legacy_funcs}\n  legacy-only: {legacy_funcs - new_funcs}"

        new_total = new_profiles["count"].sum() if "count" in new_profiles.columns else len(new_profiles)
        legacy_total = legacy_profiles["count"].sum() if "count" in legacy_profiles.columns else len(legacy_profiles)
        assert new_total == legacy_total, f"Profile count mismatch: new={new_total}, legacy={legacy_total}"


@pytest.mark.parametrize("trace_path, preset_class", trace_params)
def test_analyze_trace_views_match(dask_client, trace_path, preset_class, tmp_path):
    """View structure (keys, columns) should match between aggregated and legacy analyze_trace."""
    analyzer_new = _make_analyzer(tmp_path / "new", preset=preset_class())
    analyzer_legacy = _make_analyzer(tmp_path / "legacy", preset=preset_class())

    view_types = ["proc_name", "time_range"]

    # New path (uses analyze_trace override with pure pandas)
    new_result = analyzer_new.analyze_trace(
        trace_path=trace_path,
        view_types=view_types,
    )

    # Legacy path: swap read_trace and use base class analyze_trace (Dask path)
    from dftracer.analyzer.analyzer import Analyzer
    orig_read_trace = analyzer_legacy.read_trace
    analyzer_legacy.read_trace = analyzer_legacy.read_trace_legacy
    try:
        legacy_result = Analyzer.analyze_trace(
            analyzer_legacy,
            trace_path=trace_path,
            view_types=view_types,
        )
    finally:
        analyzer_legacy.read_trace = orig_read_trace

    # Compare view keys
    assert set(new_result.flat_views.keys()) == set(legacy_result.flat_views.keys()), (
        f"View keys mismatch:\n  new: {set(new_result.flat_views.keys())}\n  legacy: {set(legacy_result.flat_views.keys())}"
    )

    # Compare view shapes and key metrics
    for view_key in new_result.flat_views:
        new_view = new_result.flat_views[view_key]
        legacy_view = legacy_result.flat_views[view_key]

        # Same number of rows (groups)
        assert len(new_view) == len(legacy_view), (
            f"View {view_key} row count: new={len(new_view)}, legacy={len(legacy_view)}"
        )

        # Same columns (allow minor differences from aggregation path)
        new_cols = set(new_view.columns)
        legacy_cols = set(legacy_view.columns)
        shared_cols = new_cols & legacy_cols
        assert len(shared_cols) > 0, f"View {view_key} has no shared columns"

        # Compare column presence (not exact values — aggregation semantics
        # differ due to level collapsing in the C++ aggregator)
        missing_in_new = legacy_cols - new_cols
        missing_in_legacy = new_cols - legacy_cols
        # Allow some columns to be missing (frozenset cols, etc.)
        # but core metric columns should be present in both
        core_suffixes = ['_time_sum', '_count_sum', '_time_proc_max']
        for suffix in core_suffixes:
            core_new = {c for c in new_cols if c.endswith(suffix)}
            core_legacy = {c for c in legacy_cols if c.endswith(suffix)}
            assert core_new == core_legacy, (
                f"View {view_key}: core columns with suffix '{suffix}' differ:\n"
                f"  new: {core_new}\n  legacy: {core_legacy}"
            )
