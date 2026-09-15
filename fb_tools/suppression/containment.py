"""
Containment rules for stepped ``pyretechnics`` runs.

A rule is fed once per step with what burned during the step, the engine's frontier
(``{"burned_cells": (ys, xs), "unburned_cells": (ys, xs)}``, cell pairs along the
perimeter) and a suppression-difficulty surface, and answers with the frontier cells
it held (fuel model 91, non-burnable) plus a log row. Two rules:

- ``"pyre_firelines"`` (:class:`PyreFirelines`) — **the engine's own rule**,
  ``pyretechnics.suppression.build_firelines`` with its shipped SDI priority: the
  frontier is ordered into contiguous perimeters, each burned/unburned cell pair is
  scored by priority (``max SDI − SDI`` of the unburned cell) and by construction time
  (side length / rate), and the one contiguous segment with the highest total score
  that fits ``max_construction_time`` is built. Nothing about the decision is ours;
  we supply the SDI surface, a constant construction rate and, optionally, ELMFIRE's
  direct-attack feasibility gate expressed through the rate function.
- ``"elmfire0"`` (:class:`SectorContainment`) — rule S1, a re-implementation of
  ELMFIRE's containment model 0, written here (not shipped by either engine).

Rule **S1** (:class:`SectorContainment`) is ELMFIRE's containment model 0, read from
``elmfire_suppression.f90`` (main, 2026-09-13): a target containment fraction that
grows at

    dC/dt = 0.01 · diurnal · MAX_CONTAINMENT_PER_DAY · (1 − (dA/dt) / AREA_NO_CONTAINMENT_CHANGE) · exp(−B_SDI · SDĪ)

per day, where dA/dt is the growth rate in acres/day and SDĪ the mean SDI of the
acres burned in the step (clipped to ``sdi_max``); the perimeter is split into
``n_sectors`` around the ignition and the slowest sectors are held first until the
held share of the perimeter reaches the target. A held sector stays held: fire in it
can only advance by spotting over the line. Every parameter is ELMFIRE's default and
none is calibrated for Colorado — treat them as reported sweep axes, like the crown
gate, never as truth.

The rule is blind to the arm. Treatments enter only through the physics: slower
growth (dA/dt), lower SDI on the burned acres, and by *being* the slow sectors.
"""

import numpy as np

ACRE_M2 = 4046.8564224

#: ELMFIRE model-0 defaults (``elmfire_suppression.f90``), plus how this package's SDI rasters are scaled.
ELMFIRE_MODEL0 = {
    "max_containment_per_day": 100.0,     # % of the perimeter per day at zero growth and zero SDI
    "area_no_containment_change": 10_000.0,  # acres/day of growth at which containment stalls
    "b_sdi": 1.0,                          # exp(-b_sdi * SDI_bar)
    "sdi_max": 3.0,                        # SDI_bar is clipped here (ELMFIRE clips 0-3)
    "sdi_scale": 100.0,                    # fb_tools SDI rasters are raw SDI x 100 (int16)
    "n_sectors": 36,                       # 10-degree sectors around the ignition
    "start_h": 0.0,                        # hours after ignition before any line is held
    "diurnal": None,                       # 24 hourly multipliers on dC/dt (None = 1.0 all day)
}

RULES = ("elmfire0", "pyre_firelines")

#: Defaults of the package rule. Rate = ELMFIRE's fireline capacity; gate off = pure ``build_firelines``.
PYRE_FIRELINES = {
    "construction_rate_m_h": 2000.0,   # ELMFIRE MAX_FIRELINE_CONSTRUCTION_RATE-style capacity, all crews together
    "priority": "sdi",                 # "sdi" (package make_sdi_suppression_priority_fn) | "flame_length" (package fn on the burned cells)
    "max_flame_length_m": None,        # ELMFIRE's direct-attack limit (8 ft = 2.44 m): pairs whose burned cell exceeds it get a ~0 rate
    "sdi_scale": 100.0,                # fb_tools SDI rasters are raw SDI x 100
    "start_h": 0.0,                    # hours after ignition before any line is built
}


def sector_index(rows, cols, centroid_rc, n_sectors):
    """Sector (0..n-1, counter-clockwise from east) of cells ``(rows, cols)`` around ``centroid_rc``."""
    cy, cx = centroid_rc
    ang = np.arctan2(-(np.asarray(rows, float) - cy), np.asarray(cols, float) - cx)  # row axis points down
    return (np.floor(ang % (2 * np.pi) / (2 * np.pi) * n_sectors).astype(int)) % n_sectors


class SectorContainment:
    """
    Rule S1 for one realization. Call :meth:`step` once per engine step, in order.

    Parameters
    ----------
    sdi : 2-D array
        Suppression difficulty on the run grid, ``NaN`` where unknown (filled with
        ``sdi_fill``, default the median of the known cells) — in the raster's units;
        divided by ``params["sdi_scale"]`` here.
    res_m : float
    centroid_rc : (float, float)
        Row/col of the ignition, the origin of the sectors.
    params : dict, optional
        Overrides of :data:`ELMFIRE_MODEL0`.
    """

    def __init__(self, sdi, res_m, centroid_rc, params=None, sdi_fill=None):
        p = {**ELMFIRE_MODEL0, **(params or {})}
        bad = set(p) - set(ELMFIRE_MODEL0)
        if bad:
            raise TypeError(f"unknown containment parameter(s) {sorted(bad)}; allowed: {sorted(ELMFIRE_MODEL0)}")
        if p["diurnal"] is not None and len(p["diurnal"]) != 24:
            raise ValueError("diurnal must hold 24 hourly multipliers")
        self.p = p
        sdi = np.asarray(sdi, dtype="float32") / float(p["sdi_scale"])
        known = np.isfinite(sdi)
        self.sdi_fill = float(np.nanmedian(sdi)) if sdi_fill is None and known.any() else float(sdi_fill or 0.0)
        self.sdi = np.where(known, sdi, self.sdi_fill).astype("float32")
        self.sdi_known_share = float(known.mean())
        self.res = float(res_m)
        self.centroid = tuple(float(v) for v in centroid_rc)
        self.target = 0.0
        self.frozen = np.zeros(int(p["n_sectors"]), dtype=bool)
        self.held_per_sector = np.zeros(int(p["n_sectors"]), dtype=np.int64)   # line already held: still perimeter
        self._held = set()
        self.log = []

    def step(self, hour, dt_h, new_burned, frontier, fuel=None, flame_length=None):
        """
        Advance the target and pick the frontier cells to hold.

        Parameters
        ----------
        hour : float
            Simulation hour at the end of the step.
        dt_h : float
            Length of the step in hours.
        new_burned : 2-D bool array
            Cells that burned during this step.
        frontier : dict or (rows, cols)
            The engine's ``frontier_cells`` (its ``"unburned_cells"`` are used), or the
            unburned index arrays directly.
        fuel, flame_length : unused here (the runner writes fuel 91 for the returned cells).

        Returns
        -------
        (rows, cols)
            Frontier cells to hold (a subset of the unburned frontier).
        """
        frontier_unburned = frontier["unburned_cells"] if isinstance(frontier, dict) else frontier
        p, n = self.p, int(self.p["n_sectors"])
        rr_b, cc_b = np.nonzero(new_burned)
        area_ac = rr_b.size * self.res ** 2 / ACRE_M2
        rate_ac_day = area_ac / dt_h * 24.0 if dt_h > 0 else 0.0
        sdi_bar = float(np.clip(self.sdi[rr_b, cc_b].mean(), 0.0, p["sdi_max"])) if rr_b.size else 0.0
        diurnal = 1.0 if p["diurnal"] is None else float(p["diurnal"][int(hour) % 24])
        d_target = 0.0
        if hour >= p["start_h"]:
            d_target = (0.01 * diurnal * p["max_containment_per_day"]
                        * (1.0 - rate_ac_day / p["area_no_containment_change"])
                        * np.exp(-p["b_sdi"] * sdi_bar) * dt_h / 24.0)
            self.target = float(np.clip(self.target + d_target, 0.0, 1.0))

        rr_f, cc_f = (np.asarray(a, dtype=np.intp) for a in frontier_unburned)
        sec_b = sector_index(rr_b, cc_b, self.centroid, n)
        sec_f = sector_index(rr_f, cc_f, self.centroid, n)
        growth = np.bincount(sec_b, minlength=n)          # cells burned this step, per sector
        perim = np.bincount(sec_f, minlength=n)           # free (burnable) frontier cells, per sector
        # Held cells leave the engine's frontier (they are non-burnable), but they are still perimeter:
        # the containment share is held line over free frontier plus held line.
        total = int(perim.sum() + self.held_per_sector.sum())
        if total:
            held_share = (perim[self.frozen].sum() + self.held_per_sector.sum()) / total
            for s in np.lexsort((perim, growth)):         # slowest sectors first
                if held_share >= self.target:
                    break
                if self.frozen[s] or perim[s] == 0:
                    continue
                self.frozen[s] = True
                held_share += perim[s] / total
        else:
            held_share = 0.0
        hold = self.frozen[sec_f]
        for r_, c_, s_ in zip(rr_f[hold].tolist(), cc_f[hold].tolist(), sec_f[hold].tolist()):
            if (r_, c_) not in self._held:
                self._held.add((r_, c_)); self.held_per_sector[s_] += 1
        self.log.append({"hour": round(float(hour), 3), "burned_ac_step": round(area_ac, 2),
                         "rate_ac_day": round(rate_ac_day, 1), "sdi_bar": round(sdi_bar, 3),
                         "d_target": round(float(d_target), 5), "target": round(self.target, 4),
                         "frontier_cells": total, "held_share": round(float(held_share), 4),
                         "held_cells": int(hold.sum()), "frozen_sectors": int(self.frozen.sum())})
        return rr_f[hold], cc_f[hold]


class PyreFirelines:
    """
    The engine's rule: ``pyretechnics.suppression.build_firelines`` once per step.

    Per step the crews get ``dt_h`` hours at ``construction_rate_m_h`` and build the
    single highest-priority contiguous segment that fits. Held cells are the unburned
    frontier cells the package set to fuel 91 (read back by diff). With
    ``max_flame_length_m`` set, pairs whose burned cell burned above it are made
    effectively unbuildable (rate 1e-6 m/h) — ELMFIRE's direct-attack limit, expressed
    through the package's own rate function.

    Parameters
    ----------
    sdi : 2-D array
        As for :class:`SectorContainment` (raster units; divided by ``sdi_scale``).
    res_m : float
    centroid_rc : unused (kept for the common signature).
    params : dict, optional
        Overrides of :data:`PYRE_FIRELINES`.
    """

    def __init__(self, sdi, res_m, centroid_rc=None, params=None, sdi_fill=None):
        from pyretechnics import suppression as ps
        p = {**PYRE_FIRELINES, **(params or {})}
        bad = set(p) - set(PYRE_FIRELINES)
        if bad:
            raise TypeError(f"unknown fireline parameter(s) {sorted(bad)}; allowed: {sorted(PYRE_FIRELINES)}")
        if p["priority"] not in ("sdi", "flame_length"):
            raise ValueError("priority must be 'sdi' or 'flame_length'")
        self.p, self._ps = p, ps
        sdi = np.asarray(sdi, dtype="float32") / float(p["sdi_scale"])
        known = np.isfinite(sdi)
        self.sdi_fill = float(np.nanmedian(sdi)) if sdi_fill is None and known.any() else float(sdi_fill or 0.0)
        self.sdi = np.where(known, sdi, self.sdi_fill).astype("float32")
        self.sdi_known_share = float(known.mean())
        self.res = float(res_m)
        self.rate_m_min = float(p["construction_rate_m_h"]) / 60.0
        self.target = np.nan                          # no containment target in this rule
        self.frozen = np.zeros(0, dtype=bool)
        self.log = []
        self._priority_sdi = ps.make_sdi_suppression_priority_fn(self.sdi)
        self._held = set()

    def _rate_fn(self, flame_length):
        gate, rate = self.p["max_flame_length_m"], self.rate_m_min
        if gate is None or flame_length is None:
            return self._ps.make_constant_value_fn(rate)

        def gated_rate(frontier):
            fl = flame_length[frontier["burned_cells"]]
            return np.where(np.isfinite(fl) & (fl <= gate), rate, 1e-6 / 60.0)
        return gated_rate

    def step(self, hour, dt_h, new_burned, frontier, fuel=None, flame_length=None):
        """Build this step's segment in ``fuel`` (the live 2-D fuel-model array); return the cells held."""
        if fuel is None:
            raise ValueError("pyre_firelines needs the live fuel-model array")
        ys, xs = (np.asarray(a, dtype=np.intp) for a in frontier["unburned_cells"])
        n_front = ys.size
        held = (np.zeros(0, np.intp), np.zeros(0, np.intp))
        if n_front and hour >= self.p["start_h"]:
            if self.p["priority"] == "flame_length":
                if flame_length is None:
                    raise ValueError("priority='flame_length' needs the flame-length matrix")
                priority = self._ps.make_flame_length_suppression_priority_fn(np.nan_to_num(flame_length, nan=0.0))
            else:
                priority = self._priority_sdi
            # build_firelines' own three steps, so an empty selection (nothing buildable within the
            # step) can be skipped instead of indexing the fuel matrix with an empty tuple.
            ordered = self._ps.order_frontier_cells(frontier)
            scores = self._ps.score_frontier_cells(ordered, self.res, self.res, priority, self._rate_fn(flame_length))
            sel = self._ps.select_frontier_cells_for_suppression(ordered, scores, dt_h * 60.0)
            if len(sel["unburned_cells"]) and len(sel["unburned_cells"][0]):
                sy, sx = (np.asarray(a, dtype=np.intp) for a in sel["unburned_cells"])
                new = fuel[sy, sx] != 91
                fuel[sy, sx] = 91.0
                held = (sy[new], sx[new])
                self._held.update(zip(held[0].tolist(), held[1].tolist()))
        self.log.append({"hour": round(float(hour), 3), "burned_ac_step": round(float(new_burned.sum()) * self.res ** 2 / ACRE_M2, 2),
                         "frontier_cells": int(n_front), "held_cells": int(held[0].size), "held_total": len(self._held),
                         "line_m": round(float(len(self._held)) * self.res, 0)})
        return held


def make_rule(spec, sdi, res_m, centroid_rc):
    """Build a rule from ``{"rule": name, **params}``."""
    spec = dict(spec)
    name = spec.pop("rule", "pyre_firelines")
    if name == "elmfire0":
        return SectorContainment(sdi, res_m, centroid_rc, params=spec)
    if name == "pyre_firelines":
        return PyreFirelines(sdi, res_m, centroid_rc, params=spec)
    raise ValueError(f"unknown containment rule {name!r}; known: {RULES}")
