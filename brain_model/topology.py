"""Realised cell layout, contact topology and canonical sparse transmission."""

import gc
import time

import torch

_TOPOLOGY_CACHE = {}


def region_sizes(cfg, n):
    regions = cfg["regions"]
    minimum = cfg["runtime"]["min_cells_per_region"]
    if n < minimum * len(regions):
        raise ValueError(f"The network needs at least {minimum * len(regions):,} cells.")
    total = sum(region["relative_size"] for region in regions)
    raw = [n * region["relative_size"] / total for region in regions]
    sizes = [max(minimum, int(value)) for value in raw]
    # Hand out (or take back) the rounding remainder by largest fractional part.
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    index = 0
    while sum(sizes) != n:
        i = order[index % len(order)]
        if sum(sizes) < n:
            sizes[i] += 1
        elif sizes[i] > minimum:
            sizes[i] -= 1
        index += 1
    return sizes


def layout(cfg, n):
    """Cells are stored region by region: projection cells first, then SST/PV interneurons, then VIP cells."""
    rows, start = [], 0
    for region, size in zip(cfg["regions"], region_sizes(cfg, n), strict=False):
        projection = max(1, min(size - 1, round(size * region["projection_fraction"])))
        remaining = size - projection
        # Split interneurons: SST/PV (sst_fraction) and VIP (vip_fraction)
        vip_frac = region.get("vip_fraction", 0.0)
        vip_count = max(1, round(remaining * vip_frac)) if vip_frac > 0 else 0
        sst_count = remaining - vip_count
        rows.append(
            {
                "id": region["id"],
                "start": start,
                "size": size,
                "projection": projection,
                "sst_count": sst_count,
                "vip_count": vip_count,
                "gaba_projection": region["transmitter"] == "gaba",
                "fast_output": region.get("fast_output", True),
                "cell_type": region["projection_cell_type"],
                "has_vip": vip_count > 0,
            }
        )
        start += size
    return rows


def _cell_pools(row):
    p = (row["start"], row["projection"])
    sst = (row["start"] + row["projection"], row["sst_count"])
    vip = (sst[0] + sst[1], row["vip_count"])
    return p, sst, vip


def _vip_parts(row, local, weights):
    (p_start, p_count), (sst_start, sst_count), (vip_start, vip_count) = _cell_pools(row)
    parts = []
    vip_to_sst_k = max(1, round(local["interneuron_from_interneuron"] * 0.6))
    parts.append(
        (
            "local_gaba",
            sst_start,
            sst_count,
            vip_start,
            vip_count,
            vip_to_sst_k,
            weights["gaba_onto_interneuron"] * 1.8,
        )
    )  # Strong VIP→SST
    # VIP to pyramidal (very weak direct influence)
    vip_to_p_k = max(1, round(local["projection_from_interneuron"] * 0.1))
    parts.append(
        (
            "local_gaba",
            p_start,
            p_count,
            vip_start,
            vip_count,
            vip_to_p_k,
            weights["gaba_onto_projection"] * 0.2,
        )
    )  # Weak direct
    # Pyramidal to VIP (excitatory drive to VIP)
    pyr_to_vip_k = max(1, round(local["interneuron_from_projection"] * 0.3))
    parts.append(
        (
            "local_glu",
            vip_start,
            vip_count,
            p_start,
            p_count,
            pyr_to_vip_k,
            weights["glutamate_onto_interneuron"] * 0.8,
        )
    )
    # VIP to VIP (sparse)
    if vip_count > 4:
        parts.append(
            (
                "local_gaba",
                vip_start,
                vip_count,
                vip_start,
                vip_count,
                2,
                weights["gaba_onto_interneuron"] * 0.3,
            )
        )

    return parts


def _local_parts(cfg, rows):
    syn = cfg["synapses"]
    weights = syn["weights"]
    collateral = weights.get("gaba_projection_collateral_scale", 1.0)
    parts: list[tuple] = []
    for region, row in zip(cfg["regions"], rows, strict=False):
        local = {**syn["local_in_degree"], **region.get("local_in_degree", {})}
        (p_start, p_count), (sst_start, sst_count), (_vip_start, _vip_count) = _cell_pools(row)
        if row["gaba_projection"]:
            parts.append(
                (
                    "local_gaba",
                    p_start,
                    p_count,
                    p_start,
                    p_count,
                    local["projection_from_projection"],
                    weights["gaba_onto_projection"] * collateral,
                )
            )
            parts.append(
                (
                    "local_gaba",
                    sst_start,
                    sst_count,
                    p_start,
                    p_count,
                    local["interneuron_from_projection"],
                    weights["gaba_onto_interneuron"] * collateral,
                )
            )
        else:
            parts.append(
                (
                    "local_glu",
                    p_start,
                    p_count,
                    p_start,
                    p_count,
                    local["projection_from_projection"],
                    weights["glutamate_onto_projection"],
                )
            )
            parts.append(
                (
                    "local_glu",
                    sst_start,
                    sst_count,
                    p_start,
                    p_count,
                    local["interneuron_from_projection"],
                    weights["glutamate_onto_interneuron"],
                )
            )
        # SST/PV to pyramidal (local inhibition)
        parts.append(
            (
                "local_gaba",
                p_start,
                p_count,
                sst_start,
                sst_count,
                local["projection_from_interneuron"],
                weights["gaba_onto_projection"],
            )
        )
        # SST/PV to SST/PV (sparse lateral inhibition)
        parts.append(
            (
                "local_gaba",
                sst_start,
                sst_count,
                sst_start,
                sst_count,
                local["interneuron_from_interneuron"],
                weights["gaba_onto_interneuron"],
            )
        )
        # VIP disinhibition circuit: VIP → SST/PV (inhibits the inhibitor)
        if row["has_vip"]:
            parts.extend(_vip_parts(row, local, weights))
    return [(*part, False) for part in parts]


def _projection_contacts(edge, source, target, syn):
    if not source["fast_output"]:
        raise ValueError(
            f"{edge['from']} acts only through its chemical; remove its synaptic projection."
        )
    weight = float(edge["weight"])
    gaba = source["gaba_projection"]
    if gaba and weight < 0:
        raise ValueError(
            f"GABAergic projection {edge['from']} -> {edge['to']} must have a positive weight."
        )
    to_interneurons = edge.get("target") == "interneurons" or weight < 0
    source_pool = _cell_pools(source)[0]
    target_pool = _cell_pools(target)[1 if to_interneurons else 0]
    group = "long_gaba" if gaba else "long_glu"
    transmitter = "gaba" if gaba else "glutamate"
    target_type = "interneuron" if to_interneurons else "projection"
    k = max(1, round(abs(weight) * syn["long_range_in_degree_per_weight"]))
    plastic = bool(edge.get("plastic_synapse", False)) and not gaba
    parts = [
        (
            group,
            *target_pool,
            *source_pool,
            k,
            syn["weights"][f"{transmitter}_onto_{target_type}"],
            plastic,
        )
    ]
    k_ff = round(k * syn["feedforward_interneuron_fraction"])
    if not gaba and not to_interneurons and k_ff:
        parts.append(
            (
                "long_glu",
                *_cell_pools(target)[1],
                *source_pool,
                k_ff,
                syn["weights"]["glutamate_onto_interneuron"],
                plastic,
            )
        )
    return parts


def _projection_parts(cfg, rows):
    by_id = {row["id"]: row for row in rows}
    parts = []
    for edge in cfg["projections"]:
        parts.extend(
            _projection_contacts(edge, by_id[edge["from"]], by_id[edge["to"]], cfg["synapses"])
        )
    return parts


def _synapse_parts(cfg, rows):
    """Declared contacts: group, target/source pools, in-degree, weight and plasticity."""
    parts = _local_parts(cfg, rows) + _projection_parts(cfg, rows)
    return [part for part in parts if part[2] > 0 and part[4] > 0 and part[5] > 0]


def planned_synapses(cfg, n):
    return sum(part[2] * part[5] for part in _synapse_parts(cfg, layout(cfg, n)))


def estimated_bytes(cfg, n):
    """Conservative peak including contact identities, CSR, sorting and learning copies."""
    rt = cfg["runtime"]
    delay = cfg["synapses"]["long_range_delay_ms"] / rt["time_step_ms"]
    steps = rt["duration_seconds"] * 1000 / rt["time_step_ms"]
    raster = steps * rt["raster_cells_per_region"] * len(cfg["regions"])
    return int(planned_synapses(cfg, n) * 112 + n * (4 * (60 + delay)) + raster + 300 * 2**20)


def contact_matrix(contacts: dict, n: int, weights=None):
    """Aggregate repeated contacts into sorted unique CSR slots, retaining identities."""
    source, target = contacts["source"], contacts["target"]
    keys, slots = torch.unique(target * n + source, sorted=True, return_inverse=True)
    values = torch.zeros(keys.numel(), device=source.device)
    values.index_add_(0, slots, contacts["weight"] if weights is None else weights)
    crow = torch.zeros(n + 1, dtype=torch.int64, device=source.device)
    crow[1:] = torch.bincount(keys // n, minlength=n).cumsum(0)
    matrix = torch.sparse_csr_tensor(crow, keys % n, values, (n, n), check_invariants=True)
    return matrix, slots


def dynamics_dimensions(cfg: dict) -> dict:
    dt = cfg["runtime"]["time_step_ms"]
    return {
        "dt_ms": dt,
        "delay": max(1, round(cfg["synapses"]["long_range_delay_ms"] / dt)),
        "modulators": len(cfg["neuromodulators"]),
        "channels": len(cfg["field_potential_channels"]),
    }


def _contacts(parts, n, device, generator, jitter):
    sources, targets, weights, plastic, blocks = [], [], [], [], []
    for block, (_, dst_start, dst_count, src_start, src_count, k, weight, marked) in enumerate(
        parts
    ):
        target = torch.arange(dst_start, dst_start + dst_count, device=device).repeat_interleave(k)
        source = (
            torch.randint(src_count, (dst_count * k,), device=device, generator=generator)
            + src_start
        )
        if src_count > 1:
            source = torch.where(
                source == target, src_start + (source - src_start + 1) % src_count, source
            )
        keep = source != target
        sources.append(source[keep])
        targets.append(target[keep])
        values = weight * (
            1 + jitter * (2 * torch.rand(dst_count * k, device=device, generator=generator) - 1)
        )
        weights.append(values[keep])
        plastic.append(torch.full_like(source[keep], marked, dtype=torch.bool))
        blocks.append(torch.full_like(source[keep], block))
    empty = torch.empty(0, dtype=torch.int64, device=device)
    contacts = {
        "source": torch.cat(sources) if sources else empty,
        "target": torch.cat(targets) if targets else empty.clone(),
        "weight": torch.cat(weights) if weights else empty.float(),
        "plastic": torch.cat(plastic) if plastic else empty.bool(),
        "block": torch.cat(blocks) if blocks else empty.clone(),
    }
    matrix, contacts["slot"] = contact_matrix(contacts, n)
    return contacts, matrix


def _cell_parameters(cfg, rows, n, device, generator):
    # Per-cell parameters from the cell types.
    names = (
        "v_rest_mV",
        "v_reset_mV",
        "v_threshold_mV",
        "delta_t_mV",
        "v_peak_mV",
        "tau_m_ms",
        "a",
        "b_mV",
        "tau_w_ms",
        "refractory_ms",
        "threshold_jitter_mV",
        "release_u",
        "recovery_ms",
        "facilitation_ms",
    )
    params = {name: torch.empty(n, device=device) for name in names}
    region_of_cell = torch.empty(n, dtype=torch.int64, device=device)
    interneuron = torch.zeros(n, dtype=torch.bool, device=device)  # SST/PV (standard interneurons)
    vip_cell = torch.zeros(n, dtype=torch.bool, device=device)  # VIP disinhibitory cells
    gaba_cell = torch.zeros(n, dtype=torch.bool, device=device)
    types = cfg["cell_types"]
    for index, row in enumerate(rows):
        p_slice = slice(row["start"], row["start"] + row["projection"])
        sst_slice = slice(
            row["start"] + row["projection"], row["start"] + row["projection"] + row["sst_count"]
        )
        vip_slice = slice(
            row["start"] + row["projection"] + row["sst_count"], row["start"] + row["size"]
        )
        region_of_cell[row["start"] : row["start"] + row["size"]] = index
        interneuron[sst_slice] = True
        vip_cell[vip_slice] = True
        gaba_cell[sst_slice] = True
        gaba_cell[vip_slice] = True
        if row["gaba_projection"]:
            gaba_cell[p_slice] = True
        # Projection cells
        for name in names:
            params[name][p_slice] = float(types[row["cell_type"]][name])
        # SST/PV interneurons (fast_spiking)
        for name in names:
            params[name][sst_slice] = float(types["fast_spiking"][name])
        # VIP disinhibitory cells
        if row["has_vip"]:
            for name in names:
                params[name][vip_slice] = float(types["vip_disinhibitory"][name])
    params["v_threshold_mV"] += params.pop("threshold_jitter_mV") * torch.randn(
        n, device=device, generator=generator
    )

    return {
        "region_of_cell": region_of_cell,
        "interneuron": interneuron,
        "vip_cell": vip_cell,
        "gaba_cell": gaba_cell,
        "bin_of_cell": region_of_cell * 3
        + vip_cell.to(torch.int64) * 2
        + interneuron.to(torch.int64),
        "params": params,
    }


def _assemblies(cfg, rows, device):
    assembly_generator = torch.Generator(device=device).manual_seed(
        cfg["runtime"]["topology_seed"] + 17
    )
    assemblies = {}
    by_id = {row["id"]: row for row in rows}
    for channel in cfg["inputs"]:
        for region_id in channel["targets"]:
            row = by_id[region_id]
            assemblies[channel["id"] + ":" + region_id] = (
                torch.rand(row["size"], device=device, generator=assembly_generator)
                < channel["assembly_fraction"]
            )
    return assemblies


def build_topology(cfg, n, device):
    """Create (and cache) the cell-level network for one size on one device."""
    from .config import configuration_hash, validate_network

    validate_network(cfg)
    key = (n, str(device), configuration_hash(cfg))
    if key in _TOPOLOGY_CACHE:
        return _TOPOLOGY_CACHE[key]
    _TOPOLOGY_CACHE.clear()  # keep a single graph resident
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    started = time.perf_counter()
    rows = layout(cfg, n)
    generator = torch.Generator(device=device).manual_seed(cfg["runtime"]["topology_seed"] + n)
    jitter = cfg["synapses"]["weights"]["jitter"]
    parts = _synapse_parts(cfg, rows)
    matrices, counts, contacts = {}, {}, {}
    for group in ("local_glu", "local_gaba", "long_glu", "long_gaba"):
        contacts[group], matrices[group] = _contacts(
            [p for p in parts if p[0] == group], n, device, generator, jitter
        )
        counts[group] = contacts[group]["weight"].numel()

    topology = {
        "n": n,
        "config_hash": key[2],
        "dynamics": dynamics_dimensions(cfg),
        "device": device,
        "contacts": contacts,
        "rows": rows,
        "region_count": len(rows),
        "matrices": matrices,
        "synapse_counts": counts,
        "synapse_count": sum(counts.values()),
        **_cell_parameters(cfg, rows, n, device, generator),
        "build_seconds": time.perf_counter() - started,
    }
    topology["assemblies"] = _assemblies(cfg, rows, device)
    _TOPOLOGY_CACHE[key] = topology
    return topology
