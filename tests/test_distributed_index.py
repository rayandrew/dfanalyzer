"""End-to-end test for DFTracerAnalyzer.build_index_distributed.

Runs the full SST-based distributed indexer on a Dask LocalCluster with
synthetic .pfw.gz traces and compares the resulting index against a
serial `Indexer(...).ensure_indexed()` baseline on identical inputs.
"""

import gzip
import os

import pytest

pytest.importorskip("dask.distributed")
pytest.importorskip("dftracer.analyzer.dftracer")
pytest.importorskip("pyarrow")

import pyarrow as pa  # noqa: E402

from dftracer.analyzer.dftracer import DFTracerAnalyzer  # noqa: E402
from dftracer.utils import AggregationConfig, Indexer  # noqa: E402


def _write_trace(path: str, pid: int, n_events: int) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hhash = f"h{pid}"
    fhash = f"f{pid}"
    io_names = ["read", "write", "open", "close", "pread", "pwrite", "fread", "fwrite"]
    cats = ["POSIX"] * 6 + ["STDIO"] * 2
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(
            f'{{"name":"HH","ph":"M","pid":{pid},"tid":1,'
            f'"args":{{"name":"host{pid}","value":"{hhash}"}}}}\n'
        )
        f.write(
            f'{{"name":"FH","ph":"M","pid":{pid},"tid":1,'
            f'"args":{{"name":"/data/file{pid}.dat","value":"{fhash}"}}}}\n'
        )
        for i in range(n_events):
            name = io_names[i % len(io_names)]
            cat = cats[i % len(cats)]
            f.write(
                f'{{"name":"{name}","cat":"{cat}","pid":{pid},"tid":{1 + i % 3},'
                f'"ts":{1_000_000 + i * 1000},"dur":{100 + i * 10},'
                f'"ph":"X","args":{{"ret":{1024 * (i + 1)},'
                f'"hhash":"{hhash}","fhash":"{fhash}"}}}}\n'
            )


def _make_workload(root: str, n_files: int = 4, n_events: int = 200) -> list:
    files = []
    for i in range(n_files):
        p = os.path.join(root, f"trace_p{100 + i}.pfw.gz")
        _write_trace(p, pid=100 + i, n_events=n_events)
        files.append(p)
    return files


def _read_agg_tables(indexer: Indexer) -> dict:
    result = indexer.iter_arrow_dfanalyzer_all(time_granularity=1.0, time_resolution=1e6)
    out = {}
    for key in ("events", "profiles", "system"):
        batches = [pa.record_batch(cap) for cap in result.get(key, [])]
        out[key] = pa.Table.from_batches(batches) if batches else pa.table({})
    return out


def test_distributed_index_localcluster_matches_serial(tmp_path):
    from dask.distributed import Client, LocalCluster

    serial_dir = tmp_path / "serial"
    dist_dir = tmp_path / "dist"
    stage_dir = tmp_path / "stage"
    serial_dir.mkdir()
    dist_dir.mkdir()
    stage_dir.mkdir()

    serial_files = _make_workload(str(serial_dir))
    dist_files = _make_workload(str(dist_dir))

    agg = AggregationConfig(time_interval_ms=5000)

    serial_idx = Indexer(
        files=serial_files,
        require_aggregation=agg,
        force_rebuild=True,
    )
    serial_status = serial_idx.ensure_indexed()
    assert len(serial_status.ready) == len(serial_files)
    serial_tables = _read_agg_tables(serial_idx)
    serial_idx.close()

    index_path = str(dist_dir / ".dftindex")
    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        client.wait_for_workers(2, timeout=60)
        result = DFTracerAnalyzer.build_index_distributed(
            files=dist_files,
            index_path=index_path,
            local_staging=str(stage_dir),
            lustre_staging=str(stage_dir),
            client=client,
            aggregation=agg,
            auto_register_plugin=False,
        )
    finally:
        client.close()
        cluster.close()

    assert result["total_files"] == len(dist_files)
    assert result["artifact_batches"] > 0
    assert os.path.exists(index_path)
    assert sum(1 for n in result["per_worker"] if n > 0) >= 1

    # Distributed builds keep aggregation data in per-worker SSTs addressed
    # by agg_manifest.json; the unified DB's AGGREGATION CF only carries the
    # config + per-file completion markers. Consolidate to fold the manifest
    # SSTs into the unified DB so `iter_arrow_dfanalyzer_all` (legacy scan
    # path) reads real data. This is the portability escape hatch documented
    # in dftracer.utils.dask.consolidate_index.
    from dftracer.utils.dask import consolidate_index

    consolidate_result = consolidate_index(index_path)
    assert consolidate_result.get("consolidated") is True, consolidate_result
    assert consolidate_result.get("agg_ssts_ingested", 0) >= 1

    dist_idx = Indexer(
        files=dist_files,
        index_dir=str(dist_dir),
        require_aggregation=agg,
        force_rebuild=False,
    )
    dist_status = dist_idx.ensure_indexed()
    assert len(dist_status.ready) == len(dist_files), (
        f"distributed index missed files: {dist_status.needs_work}"
    )
    dist_tables = _read_agg_tables(dist_idx)
    dist_idx.close()

    for key in ("events", "profiles", "system"):
        assert dist_tables[key].num_rows == serial_tables[key].num_rows, (
            f"{key}: distributed={dist_tables[key].num_rows} "
            f"serial={serial_tables[key].num_rows}"
        )

    for key in ("events", "profiles"):
        if "count" in serial_tables[key].column_names:
            s_total = pa.compute.sum(serial_tables[key].column("count")).as_py()
            d_total = pa.compute.sum(dist_tables[key].column("count")).as_py()
            assert s_total == d_total, f"{key}: count sum differs {d_total} vs {s_total}"


def test_distributed_index_then_analyze_trace(tmp_path):
    """Mirrors bench_pipeline_dist.py: phase 1 = build_index_distributed,
    phase 2 = DFTracerAnalyzer().analyze_trace() on the same cluster+index.

    Exercises the full scaling pipeline flow on a LocalCluster so we can
    validate locally before submitting to Flux."""
    from dask.distributed import Client, LocalCluster
    from omegaconf import OmegaConf

    from dftracer.analyzer.config import AnalyzerPresetConfigPOSIX

    dist_dir = tmp_path / "dist"
    stage_dir = tmp_path / "stage"
    dist_dir.mkdir()
    stage_dir.mkdir()

    files = _make_workload(str(dist_dir), n_files=4, n_events=200)
    agg = AggregationConfig(time_interval_ms=5000)

    cluster = LocalCluster(n_workers=2, threads_per_worker=1, processes=True)
    client = Client(cluster)
    try:
        client.wait_for_workers(2, timeout=60)

        # Phase 1: distributed index build (also registers _AutoThreadPlugin).
        DFTracerAnalyzer.build_index_distributed(
            files=files,
            index_path=str(dist_dir / ".dftindex"),
            local_staging=str(stage_dir),
            lustre_staging=str(stage_dir),
            client=client,
            aggregation=agg,
            auto_register_plugin=True,
        )

        # Phase 2: instantiate analyzer directly (no init_with_hydra) and
        # call analyze_trace. read_trace will call _register_dask_plugin a
        # SECOND time; the idempotency guard (scheduler-address set) must
        # make that a no-op, otherwise Dask's teardown+re-setup broadcast
        # hangs on moodycamel ~ImplicitProducer of the previous Runtime.
        analyzer = DFTracerAnalyzer(
            preset=OmegaConf.structured(AnalyzerPresetConfigPOSIX()),
            checkpoint=False,
            index_dir=str(dist_dir),
            time_granularity=1.0,
        )
        result = analyzer.analyze_trace(
            trace_path=str(dist_dir),
            view_types=["time_range"],
        )
    finally:
        client.close()
        cluster.close()

    # analyze_trace returns an AnalysisResult; loosely assert shape.
    assert result is not None
    # Canonical fields on AnalysisResult across the current API.
    assert hasattr(result, "views") and hasattr(result, "flat_views")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
