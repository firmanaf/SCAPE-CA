# -*- coding: utf-8 -*-

from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingParameterMultipleLayers,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFolderDestination,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterEnum,
    QgsProcessingException,
    QgsProcessing,
)

import os
import re
import json
import math
import time
from collections import deque

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from scipy.ndimage import uniform_filter, gaussian_filter, maximum_filter, label as scipy_label
from sklearn.preprocessing import RobustScaler

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    _HAS_HGB = True
except Exception:
    _HAS_HGB = False

from sklearn.ensemble import RandomForestClassifier

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    )
    from reportlab.lib import colors
    _HAS_REPORTLAB = True
except Exception:
    _HAS_REPORTLAB = False

try:
    import shap as _shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False
# =========================================================================
# Stand-alone helpers (existing + v5 novelty additions)
# =========================================================================

def safe_log1p(a):
    return np.log1p(np.clip(a, 0.0, None))


def confusion_components(b0, b1_true, b1_pred, mask):
    """
    Pontius-style confusion components for binary land change.
    """
    domain = mask & np.isfinite(b0)
    obs_change = (b0 == 0) & (b1_true == 1) & domain
    pred_change = (b0 == 0) & (b1_pred == 1) & domain
    obs_persist = (b0 == 1) & (b1_true == 1) & domain
    pred_persist = (b0 == 1) & (b1_pred == 1) & domain

    hits = int(np.count_nonzero(obs_change & pred_change))
    misses = int(np.count_nonzero(obs_change & ~pred_change))
    false_alarms = int(np.count_nonzero(pred_change & ~obs_change))
    persistence_correct = int(np.count_nonzero(obs_persist & pred_persist))

    denom = hits + misses + false_alarms
    fom = hits / denom if denom > 0 else 0.0
    producer = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    user = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0

    return {
        "Hits": hits,
        "Misses": misses,
        "FalseAlarms": false_alarms,
        "PersistenceCorrect": persistence_correct,
        "FoM": fom,
        "ProducerAcc": producer,
        "UserAcc": user,
    }


def validation_diagnostics(b0, b1_true, b1_pred, mask):
    """Transition-only validation diagnostics (kept verbatim from user script)."""
    domain = mask & np.isfinite(b0) & np.isfinite(b1_true) & np.isfinite(b1_pred)
    candidate = domain & (b0 == 0)
    obs_change = candidate & (b1_true == 1)
    pred_change = candidate & (b1_pred == 1)

    hits = int(np.count_nonzero(obs_change & pred_change))
    misses = int(np.count_nonzero(obs_change & ~pred_change))
    false_alarms = int(np.count_nonzero(pred_change & ~obs_change))

    observed_change_cells = int(np.count_nonzero(obs_change))
    predicted_change_cells = int(np.count_nonzero(pred_change))
    candidate_cells = int(np.count_nonzero(candidate))
    valid_cells = int(np.count_nonzero(domain))

    denom = hits + misses + false_alarms
    fom = hits / denom if denom > 0 else 0.0
    producer = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    user = hits / (hits + false_alarms) if (hits + false_alarms) > 0 else 0.0

    false_alarm_ratio = false_alarms / predicted_change_cells if predicted_change_cells > 0 else 0.0
    net_growth_error = predicted_change_cells - observed_change_cells
    overprediction_factor = predicted_change_cells / observed_change_cells if observed_change_cells > 0 else 0.0

    return {
        "ValidCells": valid_cells,
        "CandidateCells": candidate_cells,
        "ObservedChangeCells": observed_change_cells,
        "PredictedChangeCells": predicted_change_cells,
        "Hits": hits,
        "Misses": misses,
        "FalseAlarms": false_alarms,
        "FoM": fom,
        "ProducerAcc": producer,
        "UserAcc": user,
        "FalseAlarmRatio": false_alarm_ratio,
        "NetGrowthError": net_growth_error,
        "OverpredictionFactor": overprediction_factor,
    }


def fom_at_resolution(b0, b1_true, b1_pred, mask, window):
    if window <= 1:
        return confusion_components(b0, b1_true, b1_pred, mask)["FoM"]

    obs_change = ((b0 == 0) & (b1_true == 1) & mask).astype(np.uint8)
    pred_change = ((b0 == 0) & (b1_pred == 1) & mask).astype(np.uint8)

    obs_dil = maximum_filter(obs_change, size=window) > 0
    pred_dil = maximum_filter(pred_change, size=window) > 0

    hits = int(np.count_nonzero(pred_change & obs_dil))
    misses = int(np.count_nonzero(obs_change & ~pred_dil))
    false_alarms = int(np.count_nonzero(pred_change & ~obs_dil))
    denom = hits + misses + false_alarms
    return hits / denom if denom > 0 else 0.0


def brier_score(prob, truth, mask):
    p = prob[mask]
    y = truth[mask].astype(np.float32)
    return float(np.mean((p - y) ** 2))


def reliability_curve(prob, truth, mask, n_bins=10):
    p = prob[mask]
    y = truth[mask].astype(np.float32)
    edges = np.linspace(0, 1, n_bins + 1)
    centres, fracs, counts = [], [], []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() > 0:
            centres.append(float(p[m].mean()))
            fracs.append(float(y[m].mean()))
            counts.append(int(m.sum()))
        else:
            centres.append(float(0.5 * (lo + hi)))
            fracs.append(np.nan)
            counts.append(0)

    return np.array(centres), np.array(fracs), np.array(counts)


# -------------------------------------------------------------------------
# v5 novelty: landscape metrics
# -------------------------------------------------------------------------

def landscape_metrics(binary_map, mask):
    """
    NumPatches, LPI (largest patch index, %), MPS (mean patch size, cells),
    EdgeDensity (cells per mask cell). 8-connectivity for patch labeling.
    """
    b = (binary_map == 1) & mask
    if not np.any(b):
        return {"NumPatches": 0, "LPI_pct": 0.0, "MPS_cells": 0.0, "EdgeDensity": 0.0}
    lab, n = scipy_label(b, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return {"NumPatches": 0, "LPI_pct": 0.0, "MPS_cells": 0.0, "EdgeDensity": 0.0}
    sizes = np.bincount(lab.ravel())[1:]
    map_cells = int(mask.sum())
    lpi = float(sizes.max() / map_cells * 100.0) if map_cells > 0 else 0.0
    mps = float(sizes.mean())
    diff = np.zeros_like(b, dtype=bool)
    diff[:-1, :] |= (b[:-1, :] != b[1:, :])
    diff[1:, :]  |= (b[:-1, :] != b[1:, :])
    diff[:, :-1] |= (b[:, :-1] != b[:, 1:])
    diff[:, 1:]  |= (b[:, :-1] != b[:, 1:])
    edges = int((diff & b).sum())
    return {
        "NumPatches": int(n),
        "LPI_pct": lpi,
        "MPS_cells": mps,
        "EdgeDensity": float(edges / map_cells) if map_cells > 0 else 0.0,
    }


# -------------------------------------------------------------------------
# v5 novelty: Bayesian demand posterior (bootstrap)
# -------------------------------------------------------------------------

def bootstrap_demand_posterior(years, counts, future_years,
                                n_samples=2000, seed=42, model="loglinear"):
    """
    Bootstrap-resampled posterior over future demand.
    'loglinear': fit log(count) ~ a + b*year; resample residuals.
    'linear':    fit count ~ a + b*year.
    Returns {fy: {median, lo95, hi95, samples (np.int64[n_samples])}}
    """
    rng = np.random.default_rng(seed)
    yrs = np.array(years, dtype=float)
    cs = np.array(counts, dtype=float)

    if model == "loglinear" and (cs > 0).all():
        y = np.log(cs)
        X = np.vstack([np.ones_like(yrs), yrs]).T
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        resid = y - yhat
        transform_back = np.exp
    else:
        X = np.vstack([np.ones_like(yrs), yrs]).T
        beta, *_ = np.linalg.lstsq(X, cs, rcond=None)
        yhat = X @ beta
        resid = cs - yhat
        transform_back = lambda v: v

    out = {}
    for fy in future_years:
        samples = np.empty(n_samples, dtype=np.float64)
        for k in range(n_samples):
            rs = rng.choice(resid, size=len(resid), replace=True)
            y_b = yhat + rs
            beta_b, *_ = np.linalg.lstsq(X, y_b, rcond=None)
            pred = beta_b[0] + beta_b[1] * fy + rng.choice(resid)
            samples[k] = transform_back(pred)
        samples = np.maximum(samples, 1.0)
        out[int(fy)] = {
            "median": int(np.median(samples)),
            "lo95": int(np.quantile(samples, 0.025)),
            "hi95": int(np.quantile(samples, 0.975)),
            "samples": samples.astype(np.int64),
        }
    return out


# -------------------------------------------------------------------------
# v5 novelty: driver correlation matrix + redundancy warnings
# -------------------------------------------------------------------------

def correlation_matrix(drv_arr, mask, names):
    """Pearson correlation among drivers; warns when |r| > 0.85."""
    F = drv_arr.shape[2]
    flat = drv_arr.reshape(-1, F)
    m = mask.ravel()
    idx = np.flatnonzero(m)
    if len(idx) > 100_000:
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, 100_000, replace=False)
    X = flat[idx]
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    corr = np.clip((Xs.T @ Xs) / Xs.shape[0], -1.0, 1.0)
    warnings = []
    for i in range(F):
        for j in range(i + 1, F):
            if abs(corr[i, j]) >= 0.85:
                warnings.append({
                    "a": names[i], "b": names[j], "r": float(corr[i, j])
                })
    return corr, warnings


# =========================================================================
# QGIS Processing Algorithm
# =========================================================================

class SCAPECAAlgorithm(QgsProcessingAlgorithm):

    # Existing parameter IDs (kept verbatim for backward compatibility)
    BUILT_STACK = "Built_Up"
    BUILT_CLASS = "BUILT_CLASS"
    DRIVERS = "DRIVERS"
    DRIVER_TRANSFORMS = "DRIVER_TRANSFORMS"
    INHIBIT = "INHIBIT"
    TEMPLATE = "TEMPLATE"

    FUTURE_LIST = "FUTURE_LIST"
    TARGET_SERIES = "Target_Series"

    # Demand engine
    DEMAND_MODE = "DEMAND_MODE"
    POP_BASE_RASTER = "POP_BASE_RASTER"
    POP_FUTURE_SERIES = "POP_FUTURE_SERIES"
    POP_PER_BUILT_CELL = "POP_PER_BUILT_CELL"
    PLAN_RASTER = "PLAN_RASTER"
    PLAN_BUILT_CLASSES = "PLAN_BUILT_CLASSES"
    PLAN_POLICY_MODE = "PLAN_POLICY_MODE"
    PLAN_BONUS = "PLAN_BONUS"
    PLAN_PENALTY = "PLAN_PENALTY"

    MODEL_KIND = "MODEL_KIND"
    MAX_TRAIN_SAMPLES = "MAX_TRAIN_SAMPLES"
    TRAINING_TARGET_MODE = "TRAINING_TARGET_MODE"
    AUTO_DIAGNOSTIC_RECOMMENDATION = "AUTO_DIAGNOSTIC_RECOMMENDATION"

    STEP_DEMAND = "STEP_DEMAND"
    ITER_ADVANTAGE = "ITER_ADVANTAGE"
    NB_SIZE = "NB_SIZE"

    SEED_THR = "SEED_THR"
    CAND_MULT = "CAND_MULT"

    PATCH_MEAN = "PATCH_MEAN"
    PATCH_SIGMA = "PATCH_SIGMA"

    TMC_SIGMA = "TMC_SIGMA"
    DO_VALIDATE = "DO_VALIDATE"
    DO_CALIBRATION = "DO_CALIBRATION"
    DO_PERMUTATION = "DO_PERMUTATION"
    DO_SPATIAL_CV = "DO_SPATIAL_CV"
    ENSEMBLE_N = "ENSEMBLE_N"
    DO_REPORT = "DO_REPORT"

    OUTPUT_PROB = "OUTPUT_PROB"
    OUTPUT_PDF = "OUTPUT_PDF"
    RANDOM_SEED = "RANDOM_SEED"
    OUTPUT = "OUTPUT"

    # ---------------------------------------------------------------------
    # NEW v6 / v5 NOVELTY PARAMETERS
    # ---------------------------------------------------------------------
    CONVERTIBLE_CLASSES = "CONVERTIBLE_CLASSES"   # multi-class mode
    DO_SHAP = "DO_SHAP"                            # SHAP attribution
    DO_CORRELATION = "DO_CORRELATION"              # driver correlation heatmap
    DO_LANDSCAPE = "DO_LANDSCAPE"                  # landscape metrics
    AUTO_NB = "AUTO_NB"                            # auto kernel-size tuning
    SMART_BEST = "SMART_BEST"                      # smart best-run selection
    BAYESIAN_DEMAND = "BAYESIAN_DEMAND"            # Bayesian demand posterior
    BAYESIAN_DEMAND_MODEL = "BAYESIAN_DEMAND_MODEL"

    # ---- meta ----------------------------------------------------------

    def tr(self, s):
        return QCoreApplication.translate("SCAPECA", s)

    def name(self):
        return "scape_ca"

    def displayName(self):
        return self.tr("SCAPE-CA : Spatial Cellular Automata with Patch-based Evolution")

    def createInstance(self):
        return SCAPECAAlgorithm()

    def shortHelpString(self):
        return self.tr(
            "<p><b>Created By: Firman Afrianto, Maya Safira</b></p>"
            "<p><b>Technical validation and debugging support:</b> Evan, Muhammad Fauzan Putra and Dimas Tri Rendra Graha</p>"

            "<p><b>SCAPE-CA: Spatial Cellular Automata with Patch-based Evolution</b> "
            "is a research-grade land-use and built-up growth simulation tool that combines "
            "<b>machine learning transition modelling</b>, <b>cellular automata</b>, "
            "<b>patch-based spatial allocation</b>, <b>multi-class land-cover transition learning</b>, "
            "<b>planning constraints</b>, <b>ensemble uncertainty</b>, and "
            "<b>automated diagnostic reporting</b>. It is designed to support spatial planning analysis, "
            "urban growth simulation, scenario-based planning, and spatial policy evaluation.</p>"

            "<p><b>Main modelling concept</b></p>"
            "<ul>"
            "<li><b>Transition probability model</b>: learns historical built-up transition patterns from multi-year raster inputs.</li>"
            "<li><b>Cellular automata engine</b>: allocates future growth using neighbourhood influence and suitability scores.</li>"
            "<li><b>Patch-based growth</b>: simulates clustered urban expansion instead of isolated pixel-by-pixel conversion.</li>"
            "<li><b>Demand control</b>: defines how many built-up cells should exist in each future year.</li>"
            "<li><b>Scenario logic</b>: allows baseline, population-driven, planned-growth, and constrained-growth simulations.</li>"
            "</ul>"

            "<p><b>Typical use cases</b></p>"
            "<ul>"
            "<li>Built-up growth simulation for future years such as 2030, 2035, 2040, and 2045.</li>"
            "<li>Urban expansion modelling for spatial planning, metropolitan planning, and strategic spatial planning.</li>"
            "<li>Testing planned built-up areas, zoning policy, and spatial development constraints.</li>"
            "<li>Comparing baseline growth, population-driven growth, and spatial-plan-constrained growth.</li>"
            "<li>Identifying likely growth corridors influenced by roads, POI, slope, accessibility, and other drivers.</li>"
            "<li>Producing professional PDF, PNG, Markdown, and JSON diagnostic reports.</li>"
            "</ul>"

            "<p><b>Required inputs</b></p>"
            "<ul>"
            "<li><b>Built_Up rasters</b>: time-series rasters representing historical built-up or land-cover maps. "
            "At least three layers are required. Years are automatically parsed from filenames.</li>"
            "<li><b>Built_Class</b>: class value representing built-up cells. Common examples: "
            "6 for Google Dynamic World built-up class, or 7 for several land-cover products.</li>"
            "<li><b>Template Raster</b>: reference grid used for raster alignment, extent, resolution, CRS, and nodata handling.</li>"
            "<li><b>Drivers</b>: continuous explanatory rasters such as proximity to roads, proximity to POI, slope, elevation, "
            "population density, nighttime light, land value, accessibility, or service density.</li>"
            "</ul>"

            "<p><b>Optional inputs</b></p>"
            "<ul>"
            "<li><b>Inhibiting Factors</b>: one or more raster constraints where 1 = forbidden and 0 = allowed. "
            "Multiple inhibiting layers are combined using logical OR.</li>"
            "<li><b>Population raster</b>: optional base population raster for population-driven demand estimation.</li>"
            "<li><b>Future population series</b>: comma-separated future population values for population-driven scenarios.</li>"
            "<li><b>Planned Built-Up raster</b>: optional spatial plan or zoning raster used to guide, restrict, or cap future growth.</li>"
            "<li><b>Planned Built-Up classes</b>: raster class values representing planned or allowed built-up areas.</li>"
            "</ul>"

            "<p><b>Multi-class transition learning</b></p>"
            "<ul>"
            "<li><b>Convertible Classes</b>: optional list of source land-cover classes eligible to become built-up.</li>"
            "<li><b>Auto-detect mode</b>: leave Convertible_Classes empty to automatically detect classes that historically converted into built-up.</li>"
            "<li><b>Excluded classes</b>: class 0, class 255, nodata, and the selected Built_Class are excluded automatically.</li>"
            "<li><b>Purpose</b>: helps the model distinguish which land-cover classes are more likely to transition into built-up areas.</li>"
            "<li><b>Example</b>: classes 1, 2, 4, 5, 10, and 11 may have different conversion tendencies depending on the dataset.</li>"
            "</ul>"

            "<p><b>Driver transform policy</b></p>"
            "<ul>"
            "<li><b>Auto-detect</b>: distance-like drivers are transformed using log1p; other drivers are robust-scaled.</li>"
            "<li><b>Force log1p</b>: all drivers are transformed using log1p.</li>"
            "<li><b>Force robust-scale</b>: all drivers are scaled without logarithmic transformation.</li>"
            "<li><b>Recommendation</b>: use auto-detect for mixed drivers such as proximity, slope, density, and accessibility.</li>"
            "</ul>"

            "<p><b>Demand engine</b></p>"
            "<ul>"
            "<li><b>Auto CAGR from historical Built_Up</b>: projects future demand using historical compound annual growth rate.</li>"
            "<li><b>Manual Target_Series</b>: uses user-defined target built-up cells for each future year.</li>"
            "<li><b>Population-driven target</b>: converts future population into built-up cell demand.</li>"
            "<li><b>Planned Built-Up raster target</b>: uses planned built-up capacity as the future development target.</li>"
            "<li><b>Hybrid population target capped by planned raster</b>: uses population demand but limits it using planned built-up capacity.</li>"
            "</ul>"

            "<p><b>Demand input examples</b></p>"
            "<pre>"
            "Future years: 2030,2035,2040,2045\n"
            "Manual Target_Series: 25000,30000,36000,43000\n"
            "Future population series: 350000,410000,480000,560000\n"
            "Planned built-up classes: 1,7,8,9"
            "</pre>"

            "<p><b>Spatial plan policy modes</b></p>"
            "<ul>"
            "<li><b>Target only</b>: plan raster is used only to estimate demand target.</li>"
            "<li><b>Soft preference</b>: cells inside planned areas receive a bonus and outside cells receive a penalty.</li>"
            "<li><b>Hard constraint</b>: growth is only allowed inside planned built-up areas.</li>"
            "<li><b>Hybrid target + soft preference</b>: combines demand target with soft spatial preference.</li>"
            "<li><b>Hybrid target + hard constraint</b>: combines demand target with strict spatial restriction.</li>"
            "</ul>"

            "<p><b>Core simulation parameters</b></p>"
            "<ul>"
            "<li><b>Classifier</b>: HistGradientBoosting or RandomForest. HistGradientBoosting is recommended for large raster datasets.</li>"
            "<li><b>Training target mode</b>: choose built-up status prediction or new built-up transition-only learning.</li>"
            "<li><b>Max training samples</b>: limits training samples to reduce memory and processing time.</li>"
            "<li><b>Neighborhood kernel size</b>: controls local built-up concentration influence.</li>"
            "<li><b>Auto-tune neighborhood kernel</b>: tests several kernel sizes and selects the best one based on validation FoM.</li>"
            "<li><b>Step demand per iteration</b>: controls the number of cells targeted for conversion in each iteration.</li>"
            "<li><b>Competitive advantage</b>: increases growth pressure during iterations to avoid stagnation.</li>"
            "<li><b>Seed threshold</b>: minimum suitability score required for a new growth seed.</li>"
            "<li><b>Seed candidate multiplier</b>: controls the number of candidate seed cells evaluated during allocation.</li>"
            "<li><b>Mean patch size</b>: average size of simulated new built-up patches.</li>"
            "<li><b>Patch sigma</b>: variability of patch size using a lognormal distribution.</li>"
            "<li><b>Time Monte Carlo sigma</b>: probability noise used to represent temporal uncertainty.</li>"
            "<li><b>Random seed</b>: ensures reproducible simulation results.</li>"
            "</ul>"

            "<p><b>Ensemble simulation outputs</b></p>"
            "<ul>"
            "<li><b>Ensemble size</b>: number of simulation members per future year.</li>"
            "<li><b>Agreement map</b>: frequency of cells becoming built-up across ensemble members, from 0 to 1.</li>"
            "<li><b>Uncertainty map</b>: calculated as 4*p*(1-p), highest when ensemble probability is close to 0.5.</li>"
            "<li><b>Probability map</b>: machine learning probability surface for future growth allocation.</li>"
            "<li><b>Smart best-run selection</b>: saves the ensemble member closest to the median ensemble pattern.</li>"
            "</ul>"

            "<p><b>Validation diagnostics</b></p>"
            "<ul>"
            "<li><b>Hits</b>: correctly predicted new built-up cells.</li>"
            "<li><b>Misses</b>: observed new built-up cells that were not predicted.</li>"
            "<li><b>False Alarms</b>: predicted new built-up cells that did not occur.</li>"
            "<li><b>FoM, Figure of Merit</b>: Hits divided by Hits + Misses + False Alarms.</li>"
            "<li><b>Producer Accuracy</b>: fraction of real change successfully captured.</li>"
            "<li><b>User Accuracy</b>: fraction of predicted change that is correct.</li>"
            "<li><b>False Alarm Ratio</b>: share of predicted change that is false alarm.</li>"
            "<li><b>Overprediction Factor</b>: ratio between predicted change and observed change.</li>"
            "</ul>"

            "<p><b>Advanced diagnostic modules</b></p>"
            "<ul>"
            "<li><b>Multi-resolution FoM</b>: evaluates allocation accuracy at several spatial windows.</li>"
            "<li><b>Probability calibration</b>: computes Brier score and reliability diagnostics.</li>"
            "<li><b>Permutation driver contribution</b>: ranks drivers based on DeltaFoM when each feature is shuffled.</li>"
            "<li><b>SHAP-style attribution</b>: explains feature influence using SHAP or a fallback attribution method.</li>"
            "<li><b>Driver correlation heatmap</b>: detects redundancy among explanatory drivers.</li>"
            "<li><b>Spatial block cross-validation</b>: evaluates spatial transferability using block-based validation.</li>"
            "<li><b>Bayesian demand posterior</b>: estimates median demand and 95 percent credible intervals.</li>"
            "<li><b>Landscape metrics</b>: measures projected morphology and fragmentation of built-up patterns.</li>"
            "</ul>"

            "<p><b>Landscape metrics</b></p>"
            "<ul>"
            "<li><b>NumPatches</b>: number of separated built-up patches.</li>"
            "<li><b>LPI</b>: largest patch index, expressed as percentage of valid area.</li>"
            "<li><b>MPS</b>: mean patch size in cells.</li>"
            "<li><b>EdgeDensity</b>: edge intensity of built-up patches per valid cell.</li>"
            "</ul>"

            "<p><b>Automated diagnostic interpretation</b></p>"
            "<ul>"
            "<li><b>GOOD</b>: model performance is acceptable for baseline scenario exploration.</li>"
            "<li><b>OVER_PREDICTIVE</b>: model produces too many false alarms.</li>"
            "<li><b>UNDER_PREDICTIVE</b>: model misses too many observed changes.</li>"
            "<li><b>WEAK_ALLOCATION</b>: spatial allocation accuracy is weak and driver quality should be reviewed.</li>"
            "<li><b>MODERATE</b>: model is usable for exploration but requires sensitivity analysis.</li>"
            "</ul>"

            "<p><b>What it produces</b></p>"

            "<p><b>A) Future simulation rasters</b></p>"
            "<ul>"
            "<li><code>landcover_&lt;year&gt;.tif</code>: simulated land-cover map for each future year. In multi-class mode, this preserves non-converted classes and updates simulated built-up cells.</li>"
            "<li><code>agreement_&lt;year&gt;.tif</code>: ensemble agreement raster.</li>"
            "<li><code>uncertainty_&lt;year&gt;.tif</code>: ensemble uncertainty raster.</li>"
            "<li><code>prob_&lt;year&gt;.tif</code>: model probability surface, if probability export is enabled.</li>"
            "</ul>"

            "<p><b>B) Validation and explanation rasters</b></p>"
            "<ul>"
            "<li><code>validation_pred_&lt;lastyear&gt;.tif</code>: hindcast validation prediction map.</li>"
            "<li><code>shap_top_driver.tif</code>: per-cell attribution map for the top explanatory feature.</li>"
            "<li><code>plan_allowed_mask.tif</code>: binary planned built-up mask when a plan raster is provided.</li>"
            "<li><code>inhibit_combined_mask.tif</code>: combined inhibiting factor mask.</li>"
            "</ul>"

            "<p><b>C) Report outputs</b></p>"
            "<ul>"
            "<li><code>metrics.json</code>: machine-readable diagnostic metrics and model configuration.</li>"
            "<li><code>report.md</code>: Markdown diagnostic narrative.</li>"
            "<li><code>report.png</code>: visual diagnostic summary.</li>"
            "<li><code>report.pdf</code>: professional PDF diagnostic report, if ReportLab is available.</li>"
            "</ul>"

            "<p><b>Professional PDF report</b></p>"
            "<ul>"
            "<li>Executive model summary and diagnostic status.</li>"
            "<li>Key performance indicator cards.</li>"
            "<li>Run configuration and model settings.</li>"
            "<li>Validation metrics and automated interpretation.</li>"
            "<li>Multi-resolution FoM, probability calibration, and spatial CV summary.</li>"
            "<li>Permutation importance and SHAP attribution ranking.</li>"
            "<li>Bayesian demand posterior table when enabled.</li>"
            "<li>Projected landscape metrics by future year.</li>"
            "<li>Output inventory and embedded visual report images.</li>"
            "</ul>"

            "<p><b>Recommended parameter strategy</b></p>"
            "<ul>"
            "<li><b>Baseline simulation</b>: use Auto CAGR, transition-only training, HistGradientBoosting, ensemble size 10.</li>"
            "<li><b>RDTR or zoning scenario</b>: use planned built-up raster with hybrid soft or hard plan policy.</li>"
            "<li><b>Population scenario</b>: use population-driven target or hybrid population plus plan cap.</li>"
            "<li><b>Conservative growth</b>: increase seed threshold, reduce patch size, reduce Monte Carlo sigma, and strengthen constraints.</li>"
            "<li><b>Exploratory growth</b>: reduce seed threshold, increase candidate multiplier, and use soft plan preference.</li>"
            "<li><b>Publication-grade diagnostics</b>: enable validation, calibration, permutation, SHAP, spatial CV, Bayesian demand, landscape metrics, and PDF report.</li>"
            "</ul>"

            "<p><b>Suggested output review workflow</b></p>"
            "<ol>"
            "<li>Open <code>report.pdf</code> for executive interpretation.</li>"
            "<li>Check <code>report.md</code> for technical details.</li>"
            "<li>Review <code>report.png</code> for quick visual diagnosis.</li>"
            "<li>Inspect <code>landcover_&lt;year&gt;.tif</code> for future land-cover or built-up spatial patterns.</li>"
            "<li>Compare <code>agreement_&lt;year&gt;.tif</code> and <code>uncertainty_&lt;year&gt;.tif</code> before drawing policy conclusions.</li>"
            "<li>Use <code>metrics.json</code> for dashboards, reproducibility, or automated reporting.</li>"
            "</ol>"

            "<p><b>Important notes</b></p>"
            "<ul>"
            "<li>SCAPE-CA is a scenario simulation tool, not a deterministic prediction of the future.</li>"
            "<li>High validation accuracy does not guarantee that future policy, infrastructure, or economic shocks are captured.</li>"
            "<li>Use a projected metric CRS for distance-based drivers and spatial modelling.</li>"
            "<li>Pre-aligned rasters are recommended even though the tool can align inputs to the template grid.</li>"
            "<li>Check Built_Class carefully before running the model.</li>"
            "<li>Use 1 = forbidden and 0 = allowed for inhibiting rasters.</li>"
            "<li>Class 0 and 255 should not be used as valid convertible classes.</li>"
            "<li>Driver contribution and SHAP results indicate statistical influence, not direct causal proof.</li>"
            "<li>Uncertainty and agreement maps should be reviewed before using simulated maps for policy decisions.</li>"
            "</ul>"

            "<p><b>Dependencies</b></p>"
            "<ul>"
            "<li><b>Required</b>: NumPy, Rasterio, SciPy, scikit-learn.</li>"
            "<li><b>Charts and PNG report</b>: Matplotlib.</li>"
            "<li><b>PDF report</b>: ReportLab.</li>"
            "<li><b>SHAP attribution</b>: SHAP package, optional. If unavailable, fallback attribution is used.</li>"
            "<li><b>QGIS Processing framework</b>.</li>"
            "</ul>"

            "<p><b>Install ReportLab for PDF output</b></p>"
            "<pre>"
            "\"C:\\Program Files\\QGIS 3.40.13\\apps\\Python312\\python.exe\" -m pip install reportlab"
            "</pre>"
        )

    # ---- init ---------------------------------------------------------

    def initAlgorithm(self, config=None):
        addP = self.addParameter

        addP(QgsProcessingParameterMultipleLayers(
            self.BUILT_STACK,
            "Built_Up rasters (years auto-parsed from filename, >=3 layers)",
            layerType=QgsProcessing.TypeRaster))

        addP(QgsProcessingParameterNumber(
            self.BUILT_CLASS,
            "Built_Class (Dynamic World default = 6, ESRI LC = 7)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=7))

        # NEW v6: multi-class transition mode
        addP(QgsProcessingParameterString(
            self.CONVERTIBLE_CLASSES,
            "Convertible_Classes (leave empty = auto-detect convertible classes from historical transition)",
            defaultValue="",
            optional=True))

        addP(QgsProcessingParameterMultipleLayers(
            self.DRIVERS,
            "Drivers (any number of continuous rasters: slope, distances, NDVI, density, ...)",
            layerType=QgsProcessing.TypeRaster))

        addP(QgsProcessingParameterEnum(
            self.DRIVER_TRANSFORMS,
            "Driver transform policy",
            options=[
                "Auto-detect: log1p for distance-like rasters, robust-scale otherwise",
                "Force log1p on every driver",
                "Force robust-scale only, no log",
            ],
            defaultValue=0))

        # v6.3: INHIBIT now accepts multiple raster layers.
        # Multiple constraint rasters are merged with logical OR:
        # a cell is forbidden if ANY of the input rasters marks it forbidden.
        # Each raster is interpreted as: value == 1 -> forbidden; 0 (and nodata) -> allowed.
        # The parameter remains optional; leave empty to disable inhibit constraints.
        addP(QgsProcessingParameterMultipleLayers(
            self.INHIBIT,
            "Inhibiting Factors (one or more rasters; 1 = forbidden, 0 = allowed; combined with logical OR)",
            layerType=QgsProcessing.TypeRaster,
            optional=True))

        addP(QgsProcessingParameterRasterLayer(
            self.TEMPLATE,
            "Template Raster (grid and nodata)"))

        addP(QgsProcessingParameterString(
            self.FUTURE_LIST,
            "Future years (comma separated)",
            defaultValue="2030,2035,2040,2045"))

        addP(QgsProcessingParameterString(
            self.TARGET_SERIES,
            "Target_Series (manual built-cell targets per future year; optional)",
            defaultValue="",
            optional=True))

        addP(QgsProcessingParameterEnum(
            self.DEMAND_MODE,
            "Demand mode",
            options=[
                "Auto CAGR from historical Built_Up",
                "Manual Target_Series",
                "Population-driven target",
                "Planned Built-Up raster target",
                "Hybrid population target capped by planned raster",
            ],
            defaultValue=0))

        addP(QgsProcessingParameterRasterLayer(
            self.POP_BASE_RASTER,
            "Base population raster, optional; used for population-driven target",
            optional=True))

        addP(QgsProcessingParameterString(
            self.POP_FUTURE_SERIES,
            "Future population values, optional, comma separated. Example: 300000,350000,420000,500000",
            defaultValue="",
            optional=True))

        addP(QgsProcessingParameterNumber(
            self.POP_PER_BUILT_CELL,
            "Population per built-up cell; 0 = auto from base population / latest built-up cells",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.0,
            minValue=0.0))

        addP(QgsProcessingParameterRasterLayer(
            self.PLAN_RASTER,
            "Planned Built-Up / spatial plan raster, optional",
            optional=True))

        addP(QgsProcessingParameterString(
            self.PLAN_BUILT_CLASSES,
            "Planned built-up classes, comma separated. Example: 1,7,8,9",
            defaultValue="1",
            optional=True))

        addP(QgsProcessingParameterEnum(
            self.PLAN_POLICY_MODE,
            "Spatial plan policy mode",
            options=[
                "Target only",
                "Soft preference",
                "Hard constraint",
                "Hybrid target + soft preference",
                "Hybrid target + hard constraint",
            ],
            defaultValue=3))

        addP(QgsProcessingParameterNumber(
            self.PLAN_BONUS,
            "Plan suitability bonus for planned built-up cells",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.20))

        addP(QgsProcessingParameterNumber(
            self.PLAN_PENALTY,
            "Plan suitability penalty outside planned built-up cells",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.40))

        # NEW v6: Bayesian demand posterior on top of demand engine
        addP(QgsProcessingParameterBoolean(
            self.BAYESIAN_DEMAND,
            "Add Bayesian demand uncertainty (95% CrI band on growth curve; ensemble samples demand from posterior)",
            defaultValue=False))

        addP(QgsProcessingParameterEnum(
            self.BAYESIAN_DEMAND_MODEL,
            "Bayesian demand model",
            options=["Loglinear (CAGR-like)", "Linear trend"],
            defaultValue=0))

        addP(QgsProcessingParameterEnum(
            self.MODEL_KIND,
            "Classifier",
            options=["HistGradientBoosting (recommended)", "RandomForest"],
            defaultValue=0))

        addP(QgsProcessingParameterNumber(
            self.MAX_TRAIN_SAMPLES,
            "Max training samples (stratified cap, 0 = no cap)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=500000,
            minValue=0))

        addP(QgsProcessingParameterEnum(
            self.TRAINING_TARGET_MODE,
            "Training target mode",
            options=[
                "Built-up status at next year",
                "New built-up transition only (recommended for growth simulation)",
            ],
            defaultValue=1))

        addP(QgsProcessingParameterBoolean(
            self.AUTO_DIAGNOSTIC_RECOMMENDATION,
            "Add automatic diagnostic interpretation and parameter recommendations",
            defaultValue=True))

        addP(QgsProcessingParameterNumber(
            self.STEP_DEMAND,
            "Step demand per iteration (cells)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=2000))

        addP(QgsProcessingParameterNumber(
            self.ITER_ADVANTAGE,
            "Competitive advantage per iteration",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.05))

        addP(QgsProcessingParameterNumber(
            self.NB_SIZE,
            "Neighborhood kernel size (odd integer)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=3))

        # NEW v6: auto-tune kernel
        addP(QgsProcessingParameterBoolean(
            self.AUTO_NB,
            "Auto-tune neighborhood kernel size on validation pair (tries k=3,5,7; slower)",
            defaultValue=False))

        addP(QgsProcessingParameterNumber(
            self.SEED_THR,
            "Seed threshold (0..1)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.6))

        addP(QgsProcessingParameterNumber(
            self.CAND_MULT,
            "Seed candidate multiplier",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=8))

        addP(QgsProcessingParameterNumber(
            self.PATCH_MEAN,
            "Mean patch size (cells)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=47.0))

        addP(QgsProcessingParameterNumber(
            self.PATCH_SIGMA,
            "Lognormal sigma (patch size variability)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.6))

        addP(QgsProcessingParameterNumber(
            self.TMC_SIGMA,
            "Time Monte Carlo sigma (probability noise)",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=0.10))

        addP(QgsProcessingParameterNumber(
            self.RANDOM_SEED,
            "Random seed",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=42))

        addP(QgsProcessingParameterBoolean(
            self.DO_VALIDATE,
            "Historical validation (FoM, Kappa, Pontius)",
            defaultValue=True))

        addP(QgsProcessingParameterBoolean(
            self.DO_CALIBRATION,
            "Probability calibration check (reliability curve + Brier)",
            defaultValue=True))

        addP(QgsProcessingParameterBoolean(
            self.DO_PERMUTATION,
            "Permutation-based driver contribution analysis",
            defaultValue=True))

        # NEW v6: SHAP
        addP(QgsProcessingParameterBoolean(
            self.DO_SHAP,
            "SHAP-style attribution (uses 'shap' if available, else partial-dependence fallback)",
            defaultValue=False))

        # NEW v6: correlation
        addP(QgsProcessingParameterBoolean(
            self.DO_CORRELATION,
            "Driver correlation heatmap + redundancy warnings (|r| > 0.85)",
            defaultValue=True))

        # NEW v6: landscape metrics
        addP(QgsProcessingParameterBoolean(
            self.DO_LANDSCAPE,
            "Landscape metrics per future year (NumPatches, LPI, MPS, EdgeDensity)",
            defaultValue=True))

        # NEW v6: smart best-run
        addP(QgsProcessingParameterBoolean(
            self.SMART_BEST,
            "Smart best-run selection (saves the ensemble member closest to median pattern)",
            defaultValue=True))

        addP(QgsProcessingParameterBoolean(
            self.DO_SPATIAL_CV,
            "Spatial block cross-validation (5x5 blocks, slower but honest)",
            defaultValue=False))

        addP(QgsProcessingParameterNumber(
            self.ENSEMBLE_N,
            "Ensemble size N per future year (1 = deterministic)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=10,
            minValue=1))

        addP(QgsProcessingParameterBoolean(
            self.DO_REPORT,
            "Generate auto PNG report + markdown narrative",
            defaultValue=True))

        addP(QgsProcessingParameterBoolean(
            self.OUTPUT_PROB,
            "Export probability maps",
            defaultValue=True))

        addP(QgsProcessingParameterBoolean(
            self.OUTPUT_PDF,
            "Generate PDF diagnostic report when reportlab is available",
            defaultValue=True))

        addP(QgsProcessingParameterFolderDestination(
            self.OUTPUT,
            "Output folder"))

    # ---- I/O helpers (kept from original SCAPE-CA) -------------------

    def _src_path(self, lyr):
        return lyr.source() if hasattr(lyr, "source") else lyr.dataProvider().dataSourceUri()

    def _read(self, lyr):
        with rasterio.open(self._src_path(lyr)) as s:
            return s.read(1), s.meta.copy(), s.nodata

    def _read_match_template(self, lyr, template_meta, resampling="nearest"):
        """
        Read a raster and force it to match the template grid.
        (Verbatim from user's SCAPE-CA: avoids dimension errors when
        Built_Up, Drivers, INHIBIT, and TEMPLATE differ in size, resolution,
        extent, transform, or CRS.)
        """
        src_path = self._src_path(lyr)

        with rasterio.open(src_path) as src:
            src_arr = src.read(1)
            src_meta = src.meta.copy()
            src_nodata = src.nodata

            dst_height = int(template_meta["height"])
            dst_width = int(template_meta["width"])
            dst_transform = template_meta["transform"]
            dst_crs = template_meta.get("crs", None)

            same_shape = src_arr.shape == (dst_height, dst_width)
            same_transform = src.transform == dst_transform
            same_crs = src.crs == dst_crs

            if same_shape and same_transform and same_crs:
                return src_arr, src_meta, src_nodata

            if src.crs is None or dst_crs is None:
                raise QgsProcessingException(
                    "Raster grid mismatch detected, but CRS is missing so automatic alignment cannot be performed. "
                    f"Problem raster: {os.path.basename(src_path)}. "
                    "Please assign CRS or pre-align all rasters to the same grid."
                )

            dst_nodata = src_nodata if src_nodata is not None else -9999.0
            dst_arr = np.full((dst_height, dst_width), dst_nodata, dtype=np.float32)

            rs = Resampling.bilinear if resampling == "bilinear" else Resampling.nearest

            reproject(
                source=src_arr,
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=dst_nodata,
                resampling=rs,
            )

            out_meta = template_meta.copy()
            out_meta.update({
                "height": dst_height,
                "width": dst_width,
                "transform": dst_transform,
                "crs": dst_crs,
                "dtype": "float32",
                "nodata": dst_nodata,
            })

            return dst_arr, out_meta, dst_nodata

    def _save(self, ref_meta, path, arr, dtype="uint8", nodata_value=None):
        """
        Save GeoTIFF safely.

        v6.1 class-safe fix:
        - Do not silently convert inherited negative nodata values to 255 for uint8 outputs.
          That behaviour can make class 255 appear in QGIS legends or maps.
        - Use nodata_value explicitly when a thematic class raster needs a nodata code,
          for example nodata_value=0 for multi-class land-cover outputs outside the boundary.
        """
        meta = dict(ref_meta)
        meta.update({
            "count": 1,
            "dtype": dtype,
            "compress": "LZW",
            "tiled": True,
            "predictor": 2 if dtype != "uint8" else 1,
            "BIGTIFF": "IF_SAFER",
        })

        if dtype in ("uint8", "uint16", "int16", "int32", "uint32"):
            if nodata_value is not None:
                meta["nodata"] = int(nodata_value)
            else:
                inherited_nodata = meta.get("nodata", None)
                if inherited_nodata is None:
                    meta.pop("nodata", None)
                else:
                    try:
                        nd = float(inherited_nodata)
                        if dtype == "uint8" and 0 <= nd <= 254:
                            meta["nodata"] = int(nd)
                        elif dtype != "uint8":
                            meta["nodata"] = int(nd)
                        else:
                            meta.pop("nodata", None)
                    except Exception:
                        meta.pop("nodata", None)
        elif dtype == "float32":
            if nodata_value is not None:
                meta["nodata"] = float(nodata_value)
            else:
                inherited_nodata = meta.get("nodata", None)
                if inherited_nodata is None:
                    meta["nodata"] = -9999.0
                else:
                    try:
                        meta["nodata"] = float(inherited_nodata)
                    except Exception:
                        meta["nodata"] = -9999.0

        with rasterio.open(path, "w", **meta) as dst:
            dst.write(arr.astype(dtype), 1)

    def _parse_year(self, layer):
        fname = os.path.basename(self._src_path(layer))
        years = re.findall(r"(?:19|20)\d{2}", fname)
        if not years:
            raise QgsProcessingException(f"Cannot detect year in: {fname}")
        return int(years[-1])

    def _driver_name(self, layer):
        name = os.path.splitext(os.path.basename(self._src_path(layer)))[0]
        name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        return name[:40]

    def _bin(self, arr, built_class):
        return (arr == built_class).astype(np.uint8)

    # ---- driver loading and transform (from original) ----------------

    def _load_drivers(self, driver_layers, mask, transform_policy, template_meta, fb):
        names, raws, transforms = [], [], []

        for layer in driver_layers:
            arr, _, nodata = self._read_match_template(
                layer, template_meta, resampling="bilinear"
            )
            arr = arr.astype(np.float32)

            if nodata is not None:
                arr = np.where(arr == nodata, np.nan, arr)

            inside = mask & np.isfinite(arr)
            median_value = float(np.nanmedian(arr[inside])) if inside.sum() > 0 else 0.0
            arr = np.where(np.isnan(arr), median_value, arr)
            arr[~mask] = 0.0

            names.append(self._driver_name(layer))
            raws.append(arr)

        for name, arr in zip(names, raws):
            lower_name = name.lower()
            if transform_policy == 1:
                transforms.append("log")
            elif transform_policy == 2:
                transforms.append("scale")
            else:
                if re.search(r"dist|distance|prox|proximity|jarak", lower_name):
                    transforms.append("log")
                elif mask.sum() > 0 and np.percentile(arr[mask], 99) > 50 and arr.min() >= 0:
                    transforms.append("log")
                else:
                    transforms.append("scale")

        features = []
        normalized = []

        for arr, transform in zip(raws, transforms):
            feat = safe_log1p(arr) if transform == "log" else arr.copy()
            features.append(feat.astype(np.float32))

            values = arr[mask]
            if values.size > 0:
                lo, hi = np.percentile(values, [2, 98])
            else:
                lo, hi = 0.0, 1.0

            if hi <= lo:
                norm = np.zeros_like(arr, dtype=np.float32)
            else:
                norm = np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)
                norm[~mask] = 0.0
            normalized.append(norm)

        drv_arr = np.dstack(features).astype(np.float32)
        drv_norm = np.dstack(normalized).astype(np.float32)

        fb.pushInfo(f"Loaded {len(names)} drivers and aligned them to the template grid:")
        for name, transform in zip(names, transforms):
            fb.pushInfo(f"  - {name} [{transform}]")

        return drv_arr, drv_norm, names, transforms

    # ---- neighborhood -------------------------------------------------

    def _neigh_uniform(self, built, kernel_size):
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        mean_val = uniform_filter(built.astype(np.float32), size=kernel_size, mode="nearest")
        cell_count = kernel_size * kernel_size
        return (mean_val * cell_count - built.astype(np.float32)) / (cell_count - 1)

    def _neigh_gauss(self, built, sigma):
        return gaussian_filter(built.astype(np.float32), sigma=sigma, mode="nearest")

    # ---- feature builder (extended with multi-class one-hots) --------

    def _build_features(self, built_state, drv_arr, drv_norm, kernel_size, mask,
                         source_class_map=None, convertible_classes=None):
        """
        Features:
          - built_state itself
          - neighborhood small/medium/large
          - urban_pressure
          - one-hot per convertible class (multi-class mode)
          - all driver features
        """
        built = built_state.astype(np.float32)
        nb_small = self._neigh_uniform(built, kernel_size)
        nb_medium = self._neigh_uniform(built, max(kernel_size * 2 + 1, 5))
        nb_large = self._neigh_gauss(built, sigma=max(2.0, kernel_size))

        comp = drv_norm.mean(axis=2)
        pressure = nb_medium * (1.0 - comp)

        feature_list = [built, nb_small, nb_medium, nb_large, pressure]

        # Multi-class one-hot indicators
        if convertible_classes and source_class_map is not None:
            for cls in convertible_classes:
                feature_list.append((source_class_map == cls).astype(np.float32))

        for i in range(drv_arr.shape[2]):
            feature_list.append(drv_arr[..., i])

        features = np.dstack(feature_list).astype(np.float32)
        features[~mask] = 0.0
        return features

    def _feature_names(self, driver_names, convertible_classes=None):
        base = [
            "built_state",
            "neigh_small",
            "neigh_medium",
            "neigh_large",
            "urban_pressure",
        ]
        if convertible_classes:
            for cls in convertible_classes:
                base.append(f"src_class_{int(cls)}")
        return base + list(driver_names)

    # ---- training (multi-class aware) --------------------------------

    def _train_model(self, Y_thw, drv_arr, drv_norm, kernel_size, mask,
                     max_samples, model_kind, seed, fb,
                     training_target_mode=1,
                     source_class_thw=None, convertible_classes=None):
        """
        Train transition model.

        training_target_mode:
          0 = predict built-up status at next year
          1 = predict new built-up transition only (recommended)

        Multi-class: when source_class_thw is supplied, one-hot indicators for
        each class in convertible_classes are added as features per training
        pair, using the source-time class map.
        """
        T = Y_thw.shape[0]
        all_features, all_labels, all_weights = [], [], []
        n_pairs = T - 1
        pair_weights = np.linspace(0.5, 1.0, n_pairs)
        rng = np.random.default_rng(seed)

        for t in range(n_pairs):
            scm_t = source_class_thw[t] if source_class_thw is not None else None
            X = self._build_features(
                Y_thw[t], drv_arr, drv_norm, kernel_size, mask,
                source_class_map=scm_t,
                convertible_classes=convertible_classes,
            )

            if int(training_target_mode) == 1:
                valid = mask & np.isfinite(X).all(2) & (Y_thw[t] == 0)
                Xv = X[valid]
                yv = ((Y_thw[t] == 0) & (Y_thw[t + 1] == 1))[valid].astype(np.uint8)
            else:
                valid = mask & np.isfinite(X).all(2)
                Xv = X[valid]
                yv = Y_thw[t + 1][valid].astype(np.uint8)

            if len(yv) == 0:
                continue

            if max_samples > 0:
                budget = max(1, max_samples // n_pairs)
                pos = np.flatnonzero(yv == 1)
                neg = np.flatnonzero(yv == 0)
                half = budget // 2
                n_pos = min(len(pos), half)
                n_neg = min(len(neg), budget - n_pos)

                pick_pos = rng.choice(pos, n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
                pick_neg = rng.choice(neg, n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
                pick = np.concatenate([pick_pos, pick_neg])
                rng.shuffle(pick)

                Xv = Xv[pick]
                yv = yv[pick]

            all_features.append(Xv)
            all_labels.append(yv)
            all_weights.append(np.full(len(yv), pair_weights[t], dtype=np.float32))

            fb.pushInfo(
                f"  pair {t}: kept {len(yv):,} samples "
                f"(positive {int(yv.sum()):,}, weight={pair_weights[t]:.2f})"
            )

        if not all_features:
            raise QgsProcessingException(
                "No training samples were generated. Check Built_Up classes, mask, and inhibiting layer."
            )

        Xall = np.vstack(all_features)
        yall = np.concatenate(all_labels).astype(np.uint8)
        wall = np.concatenate(all_weights).astype(np.float32)

        if int(yall.sum()) == 0:
            raise QgsProcessingException(
                "Training labels contain no positive transition. Check historical Built_Up change or use status mode."
            )

        if model_kind == 0 and _HAS_HGB:
            fb.pushInfo("Classifier: HistGradientBoosting")
            model = HistGradientBoostingClassifier(
                max_iter=400,
                learning_rate=0.06,
                max_leaf_nodes=63,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=seed,
            )
            model.fit(Xall, yall, sample_weight=wall)
        else:
            fb.pushInfo("Classifier: RandomForest")
            model = RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=seed,
            )
            model.fit(Xall, yall, sample_weight=wall)

        return model, Xall, yall

    def _predict_prob(self, model, X_hwf, mask, chunk_cells=1_000_000):
        """Chunked predict_proba; v6 memory guard exposes chunk_cells."""
        H, W, F = X_hwf.shape
        prob = np.zeros((H, W), dtype=np.float32)
        valid = mask & np.isfinite(X_hwf).all(2)
        if not np.any(valid):
            return prob

        Xv = X_hwf[valid]
        out = np.empty(len(Xv), dtype=np.float32)
        for i in range(0, len(Xv), chunk_cells):
            out[i:i + chunk_cells] = model.predict_proba(Xv[i:i + chunk_cells])[:, 1]
        prob[valid] = out
        return prob

    # ---- v6 NEW: auto kernel-size tuning -----------------------------

    def _auto_tune_nb(self, Y_thw, drv_arr, drv_norm, mask, inhibit,
                      max_samples, model_kind, seed, fb,
                      training_target_mode=1,
                      source_class_thw=None, convertible_classes=None,
                      candidate_kernels=(3, 5, 7)):
        """
        Quick grid search over candidate kernel sizes.
        For each k, train on pairs[0..n-2], evaluate FoM on the last pair
        with demand-based allocation.
        """
        results = []
        b0 = Y_thw[-2]
        b1 = Y_thw[-1]

        for k in candidate_kernels:
            scm_t = source_class_thw[-2] if source_class_thw is not None else None
            X = self._build_features(
                b0, drv_arr, drv_norm, k, mask,
                source_class_map=scm_t,
                convertible_classes=convertible_classes,
            )

            Y_train = Y_thw[:-1]
            scm_train = source_class_thw[:-1] if source_class_thw is not None else None

            try:
                m_, _, _ = self._train_model(
                    Y_train, drv_arr, drv_norm, k, mask,
                    max_samples=min(max_samples or 200_000, 200_000),
                    model_kind=model_kind, seed=seed, fb=fb,
                    training_target_mode=training_target_mode,
                    source_class_thw=scm_train,
                    convertible_classes=convertible_classes,
                )
            except QgsProcessingException as e:
                fb.pushInfo(f"  auto-NB k={k} skipped: {e}")
                continue

            prob = self._predict_prob(m_, X, mask)
            demand = int(np.count_nonzero((b0 == 0) & (b1 == 1) & mask & ~inhibit))
            cand = mask & ~inhibit & (b0 == 0)
            ci = np.flatnonzero(cand.ravel())
            pred = b0.copy()
            if demand > 0 and len(ci) > 0:
                cs = prob.ravel()[ci]
                k_take = min(demand, len(ci))
                top = np.argpartition(-cs, k_take - 1)[:k_take] if k_take < len(ci) else np.arange(len(ci))
                pred.ravel()[ci[top]] = 1
            comp = confusion_components(b0, b1, pred, mask & ~inhibit)
            results.append({"k": k, "FoM": comp["FoM"]})
            fb.pushInfo(f"  auto-NB k={k}: FoM={comp['FoM']:.4f}")

        if not results:
            return None, []
        best = max(results, key=lambda r: r["FoM"])
        fb.pushInfo(f"Auto-NB selected k={best['k']} (FoM={best['FoM']:.4f})")
        return best["k"], results

    # ---- v6 NEW: SHAP attribution ------------------------------------

    def _shap_attribution(self, model, X_hwf, mask, feat_names, fb,
                          sample_size=15000, seed=42):
        """
        SHAP attribution. Tries shap.TreeExplainer; falls back to a fast
        partial-dependence-style approximation if shap is unavailable or
        explainer construction fails.
        """
        rng = np.random.default_rng(seed)
        valid = mask & np.isfinite(X_hwf).all(2)
        idx = np.flatnonzero(valid.ravel())
        if len(idx) == 0:
            return None
        if len(idx) > sample_size:
            sub = rng.choice(idx, sample_size, replace=False)
        else:
            sub = idx

        Xs = X_hwf.reshape(-1, X_hwf.shape[2])[sub]

        shap_vals = None
        if _HAS_SHAP:
            try:
                expl = _shap.TreeExplainer(model)
                sv = expl.shap_values(Xs)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                shap_vals = np.asarray(sv)
                fb.pushInfo("SHAP: TreeExplainer used.")
            except Exception as e:
                fb.pushInfo(f"SHAP TreeExplainer failed ({e}); using PD fallback.")
                shap_vals = None

        if shap_vals is None:
            shap_vals = np.zeros_like(Xs, dtype=np.float32)
            base_p = model.predict_proba(Xs)[:, 1]
            for fi in range(Xs.shape[1]):
                Xp = Xs.copy()
                Xp[:, fi] = np.median(Xp[:, fi])
                p_no = model.predict_proba(Xp)[:, 1]
                shap_vals[:, fi] = (base_p - p_no).astype(np.float32)
            fb.pushInfo("SHAP: PD-based fallback computed.")

        mean_abs = np.mean(np.abs(shap_vals), axis=0)
        ranking = sorted(
            [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])}
             for i in range(len(feat_names))],
            key=lambda r: -r["mean_abs_shap"])

        plot_n = min(2500, shap_vals.shape[0])
        plot_idx = rng.choice(shap_vals.shape[0], plot_n, replace=False)
        beeswarm = {}
        for fi, name in enumerate(feat_names):
            beeswarm[name] = {
                "shap": shap_vals[plot_idx, fi].tolist(),
                "value": Xs[plot_idx, fi].tolist(),
            }

        top_name = ranking[0]["feature"]
        top_idx = feat_names.index(top_name)
        shap_grid = np.zeros(mask.shape, dtype=np.float32)
        rows = sub // mask.shape[1]
        cols = sub % mask.shape[1]
        shap_grid[rows, cols] = shap_vals[:, top_idx]

        return {
            "ranking": ranking,
            "top_feature": top_name,
            "shap_grid_top": shap_grid,
            "beeswarm": beeswarm,
            "n_samples": int(shap_vals.shape[0]),
        }

    # ---- demand and spatial-plan helpers (verbatim from original) ---

    def _parse_int_series(self, text, name, allow_empty=True):
        text = "" if text is None else str(text).strip()
        if not text:
            if allow_empty:
                return []
            raise QgsProcessingException(f"{name} is empty.")
        try:
            return [int(x.strip()) for x in text.split(",") if x.strip()]
        except Exception:
            raise QgsProcessingException(
                f"{name} must contain integer values separated by commas. Example: 3000,4500,6000"
            )

    def _parse_float_series(self, text, name, allow_empty=True):
        text = "" if text is None else str(text).strip()
        if not text:
            if allow_empty:
                return []
            raise QgsProcessingException(f"{name} is empty.")
        try:
            return [float(x.strip()) for x in text.split(",") if x.strip()]
        except Exception:
            raise QgsProcessingException(
                f"{name} must contain numeric values separated by commas. Example: 300000,350000,420000"
            )

    def _load_plan_allowed(self, plan_layer, template_meta, mask, built_classes, fb):
        if plan_layer is None:
            return None
        if not built_classes:
            raise QgsProcessingException(
                "PLAN_BUILT_CLASSES is empty. Provide classes such as 1 or 1,7,8,9."
            )

        arr, _, nodata = self._read_match_template(
            plan_layer, template_meta, resampling="nearest"
        )
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        allowed = np.zeros(arr.shape, dtype=bool)
        for cls in built_classes:
            allowed |= (arr == cls)
        allowed &= mask
        fb.pushInfo(
            f"Plan raster loaded: planned built-up classes={built_classes}; "
            f"planned cells={int(allowed.sum()):,}"
        )
        return allowed

    def _population_targets(self, pop_future, pop_base_layer, pop_per_cell,
                            template_meta, mask, latest_built, future_years, fb):
        if len(pop_future) != len(future_years):
            raise QgsProcessingException(
                "POP_FUTURE_SERIES length must match FUTURE_LIST. "
                f"Future years = {len(future_years)}, population values = {len(pop_future)}."
            )

        base_built_cells = max(int(latest_built.sum()), 1)
        if pop_per_cell <= 0:
            if pop_base_layer is None:
                raise QgsProcessingException(
                    "Population-driven demand requires either POP_PER_BUILT_CELL > 0 "
                    "or a Base population raster to estimate population per built-up cell."
                )
            pop_arr, _, pop_nodata = self._read_match_template(
                pop_base_layer, template_meta, resampling="bilinear"
            )
            pop_arr = pop_arr.astype(np.float32)
            if pop_nodata is not None:
                pop_arr = np.where(pop_arr == pop_nodata, np.nan, pop_arr)
            pop_arr[~mask] = np.nan
            total_pop_base = float(np.nansum(np.where(pop_arr > 0, pop_arr, 0)))
            if total_pop_base <= 0:
                raise QgsProcessingException(
                    "Base population raster has zero or invalid total population after alignment."
                )
            pop_per_cell = total_pop_base / base_built_cells
            fb.pushInfo(
                f"Auto POP_PER_BUILT_CELL = {pop_per_cell:.4f} from base population "
                f"{total_pop_base:,.2f} / latest built cells {base_built_cells:,}."
            )
        else:
            fb.pushInfo(f"Using manual POP_PER_BUILT_CELL = {pop_per_cell:.4f}.")

        targets = []
        for pop_value in pop_future:
            targets.append(max(base_built_cells, int(round(float(pop_value) / pop_per_cell))))

        return targets, float(pop_per_cell)

    def _planned_targets(self, plan_allowed, latest_built, years, future_years, fb):
        if plan_allowed is None:
            raise QgsProcessingException(
                "Planned Built-Up raster demand requires PLAN_RASTER."
            )
        base_year = years[-1]
        final_year = max(future_years)
        base_count = int(latest_built.sum())
        plan_capacity = int(plan_allowed.sum())
        final_capacity = max(base_count, plan_capacity)

        targets = []
        denom = max(final_year - base_year, 1)
        for fy in future_years:
            frac = np.clip((fy - base_year) / denom, 0.0, 1.0)
            target = int(round(base_count + frac * (final_capacity - base_count)))
            targets.append(max(base_count, target))

        fb.pushInfo(
            f"Planned-raster demand: existing={base_count:,}; "
            f"plan capacity={plan_capacity:,}; final target={final_capacity:,}."
        )
        return targets, plan_capacity

    def _resolve_targets(self, demand_mode, manual_target, pop_future, pop_base_layer,
                          pop_per_cell, plan_allowed, Y, years, future_years,
                          template_meta, mask, fb):
        base_year = years[-1]
        latest_built = Y[-1]
        base_count = int(latest_built.sum())

        info = {
            "demand_mode_index": int(demand_mode),
            "base_year": int(base_year),
            "base_built_cells": int(base_count),
        }

        if demand_mode == 0:
            first_count = max(int(Y[0].sum()), 1)
            span = max(base_year - years[0], 1)
            cagr = (base_count / first_count) ** (1.0 / span) - 1.0
            targets = []
            for fy in future_years:
                delta_year = fy - base_year
                if delta_year <= 0:
                    targets.append(base_count)
                else:
                    targets.append(max(base_count, int(round(base_count * (1 + cagr) ** delta_year))))
            info.update({"target_mode": "automatic_cagr", "cagr": float(cagr)})
            fb.pushInfo(f"Demand mode: Auto CAGR. CAGR={cagr * 100:.2f}%/year; base built cells={base_count:,}.")
            return targets, info

        if demand_mode == 1:
            if manual_target is None or len(manual_target) == 0:
                raise QgsProcessingException(
                    "Demand mode is Manual Target_Series, but Target_Series is empty. "
                    "Provide values such as 3000,4500,6000,7500."
                )
            if len(manual_target) != len(future_years):
                raise QgsProcessingException(
                    "Target_Series length must match FUTURE_LIST. "
                    f"Future years = {len(future_years)}, target values = {len(manual_target)}."
                )
            targets = [max(base_count, int(v)) for v in manual_target]
            info.update({"target_mode": "manual_Target_Series"})
            fb.pushInfo("Demand mode: Manual Target_Series.")
            return targets, info

        if demand_mode == 2:
            if not pop_future:
                raise QgsProcessingException(
                    "Demand mode is Population-driven target, but POP_FUTURE_SERIES is empty."
                )
            targets, resolved_ppc = self._population_targets(
                pop_future, pop_base_layer, pop_per_cell, template_meta, mask,
                latest_built, future_years, fb
            )
            info.update({
                "target_mode": "population_driven",
                "pop_per_built_cell": float(resolved_ppc),
                "future_population_series": [float(x) for x in pop_future],
            })
            fb.pushInfo("Demand mode: Population-driven target.")
            return targets, info

        if demand_mode == 3:
            targets, plan_capacity = self._planned_targets(
                plan_allowed, latest_built, years, future_years, fb
            )
            info.update({
                "target_mode": "planned_built_up_raster",
                "plan_capacity_cells": int(plan_capacity),
            })
            fb.pushInfo("Demand mode: Planned Built-Up raster target.")
            return targets, info

        if demand_mode == 4:
            if not pop_future:
                raise QgsProcessingException(
                    "Demand mode is Hybrid population + plan cap, but POP_FUTURE_SERIES is empty."
                )
            if plan_allowed is None:
                raise QgsProcessingException(
                    "Demand mode is Hybrid population + plan cap, but PLAN_RASTER is missing."
                )
            pop_targets, resolved_ppc = self._population_targets(
                pop_future, pop_base_layer, pop_per_cell, template_meta, mask,
                latest_built, future_years, fb
            )
            plan_targets, plan_capacity = self._planned_targets(
                plan_allowed, latest_built, years, future_years, fb
            )
            targets = [max(base_count, min(int(p), int(pl))) for p, pl in zip(pop_targets, plan_targets)]
            overshoot = [int(p) - int(pl) for p, pl in zip(pop_targets, plan_targets)]
            info.update({
                "target_mode": "hybrid_population_plan_cap",
                "pop_per_built_cell": float(resolved_ppc),
                "future_population_series": [float(x) for x in pop_future],
                "population_targets_uncapped": [int(x) for x in pop_targets],
                "planned_targets": [int(x) for x in plan_targets],
                "plan_capacity_cells": int(plan_capacity),
                "population_plan_overshoot_cells": [int(x) for x in overshoot],
            })
            fb.pushInfo("Demand mode: Hybrid population target capped by planned raster.")
            return targets, info

        raise QgsProcessingException(f"Unsupported DEMAND_MODE index: {demand_mode}")

    # ---- spatial block CV (multi-class aware) -----------------------

    def _spatial_block_cv(self, Y_thw, drv_arr, drv_norm, kernel_size, mask,
                          max_samples, model_kind, seed, n_blocks, fb,
                          source_class_thw=None, convertible_classes=None):
        H, W = mask.shape
        block_h = max(1, H // n_blocks)
        block_w = max(1, W // n_blocks)

        b0 = Y_thw[-2]
        b1 = Y_thw[-1]
        scm_t = source_class_thw[-2] if source_class_thw is not None else None
        X = self._build_features(
            b0, drv_arr, drv_norm, kernel_size, mask,
            source_class_map=scm_t, convertible_classes=convertible_classes,
        )

        rng = np.random.default_rng(seed)
        fold_foms = []

        for bi in range(n_blocks):
            for bj in range(n_blocks):
                hold_mask = np.zeros_like(mask, dtype=bool)
                r0 = bi * block_h
                r1 = (bi + 1) * block_h if bi < n_blocks - 1 else H
                c0 = bj * block_w
                c1 = (bj + 1) * block_w if bj < n_blocks - 1 else W
                hold_mask[r0:r1, c0:c1] = True

                train_mask = mask & ~hold_mask
                test_mask = mask & hold_mask

                if test_mask.sum() < 100 or train_mask.sum() < 1000:
                    continue

                valid_train = train_mask & np.isfinite(X).all(2)
                Xv = X[valid_train]
                yv = b1[valid_train].astype(np.uint8)

                pos = np.flatnonzero(yv == 1)
                neg = np.flatnonzero(yv == 0)
                half = min(20000, len(pos), len(neg))
                if half < 50:
                    continue

                pick = np.concatenate([
                    rng.choice(pos, half, replace=False),
                    rng.choice(neg, half, replace=False),
                ])
                Xv = Xv[pick]
                yv = yv[pick]

                if model_kind == 0 and _HAS_HGB:
                    model = HistGradientBoostingClassifier(
                        max_iter=200, learning_rate=0.08, max_leaf_nodes=31,
                        early_stopping=True, random_state=seed + bi * 7 + bj,
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=100, n_jobs=-1, min_samples_leaf=5,
                        random_state=seed + bi * 7 + bj,
                    )
                model.fit(Xv, yv)

                valid_test = test_mask & np.isfinite(X).all(2)
                if valid_test.sum() == 0:
                    continue
                prob_block = np.zeros_like(mask, dtype=np.float32)
                prob_block[valid_test] = model.predict_proba(X[valid_test])[:, 1]

                demand = int(((b1 == 1) & (b0 == 0) & test_mask).sum())
                if demand <= 0:
                    continue
                cand_mask = (b0 == 0) & test_mask
                cand_idx = np.flatnonzero(cand_mask.ravel())
                if len(cand_idx) <= demand:
                    chosen = cand_idx
                else:
                    scores = prob_block.ravel()[cand_idx]
                    chosen = cand_idx[np.argpartition(-scores, demand)[:demand]]

                pred = b0.copy()
                pred.ravel()[chosen] = 1
                comp = confusion_components(b0, b1, pred, test_mask)
                fold_foms.append(comp["FoM"])

        if fold_foms:
            arr = np.array(fold_foms)
            fb.pushInfo(
                f"Spatial CV: {len(arr)} folds, FoM mean={arr.mean():.4f}, "
                f"std={arr.std():.4f}, min={arr.min():.4f}, max={arr.max():.4f}"
            )
            return {
                "n_folds": int(len(arr)),
                "fom_mean": float(arr.mean()),
                "fom_std": float(arr.std()),
                "fom_min": float(arr.min()),
                "fom_max": float(arr.max()),
                "folds": [float(x) for x in arr],
            }
        return {"n_folds": 0}

    # ---- permutation importance (verbatim) --------------------------

    def _permutation_importance(self, model, X_hwf, y, mask, feat_names,
                                n_repeats, seed, fb):
        rng = np.random.default_rng(seed)
        valid = mask & np.isfinite(X_hwf).all(2)
        Xv = X_hwf[valid]
        yv = y[valid].astype(np.uint8)
        base_p = model.predict_proba(Xv)[:, 1]
        base_pred = (base_p >= 0.5).astype(np.uint8)
        denom = ((base_pred == 1) | (yv == 1)).sum()
        base_fom = float(((base_pred == 1) & (yv == 1)).sum() / denom) if denom > 0 else 0.0

        results = []
        for fi, name in enumerate(feat_names):
            drops = []
            for _ in range(n_repeats):
                Xp = Xv.copy()
                Xp[:, fi] = rng.permutation(Xp[:, fi])
                p = model.predict_proba(Xp)[:, 1]
                pred = (p >= 0.5).astype(np.uint8)
                d = ((pred == 1) | (yv == 1)).sum()
                fom = float(((pred == 1) & (yv == 1)).sum() / d) if d > 0 else 0.0
                drops.append(base_fom - fom)
            arr = np.array(drops)
            results.append({
                "feature": name,
                "delta_fom_mean": float(arr.mean()),
                "delta_fom_std": float(arr.std()),
            })
            fb.pushInfo(f"  perm {name}: deltaFoM={arr.mean():.4f} +/- {arr.std():.4f}")
        results.sort(key=lambda r: -r["delta_fom_mean"])
        return {"baseline_fom": base_fom, "ranking": results}

    # ---- patch grower (verbatim) -------------------------------------

    def _grow_patch(self, sim, score, allowed, seed_rc, target):
        H, W = sim.shape
        r0, c0 = int(seed_rc[0]), int(seed_rc[1])
        if sim[r0, c0] == 1 or not allowed[r0, c0]:
            return 0
        sim[r0, c0] = 1
        added = 1
        frontier = deque([(r0, c0)])
        seen = {r0 * W + c0}
        offsets = (
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        )
        while frontier and added < target:
            r, c = frontier.popleft()
            candidates = []
            for dr, dc in offsets:
                rr = r + dr
                cc = c + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W:
                    continue
                idx = rr * W + cc
                if idx in seen:
                    continue
                seen.add(idx)
                if sim[rr, cc] == 1 or not allowed[rr, cc]:
                    continue
                candidates.append((score[rr, cc], rr, cc))
            if not candidates:
                continue
            candidates.sort(reverse=True)
            for _, rr, cc in candidates:
                if added >= target:
                    break
                if sim[rr, cc] == 0:
                    sim[rr, cc] = 1
                    added += 1
                    frontier.append((rr, cc))
        return added

    # ---- one simulation run (verbatim, supports plan policy) --------

    def _simulate_one(self, prob, state, mask, inh, desired, params, run_seed,
                      plan_allowed=None, plan_policy_mode=0, plan_bonus=0.0, plan_penalty=0.0):
        rng = np.random.default_rng(run_seed)
        sim = state.copy().astype(np.uint8)
        advantage = 0.0

        step = params["step"]
        cand_mult = params["cand_mult"]
        seed_thr = params["seed_thr"]
        nb_k = params["nb_k"]
        sigma = max(1e-6, params["patch_sigma"])
        mean = max(1.0, params["patch_mean"])
        mu = math.log(mean) - 0.5 * sigma * sigma
        adv_step = params["adv_step"]

        use_soft_plan = plan_allowed is not None and plan_policy_mode in (1, 3)
        use_hard_plan = plan_allowed is not None and plan_policy_mode in (2, 4)

        def refresh_score(current_sim, current_advantage):
            nb = self._neigh_uniform(current_sim, nb_k)
            score = prob + current_advantage + 0.75 * nb
            if use_soft_plan:
                score = score.copy()
                score[plan_allowed] += float(plan_bonus)
                score[~plan_allowed] -= float(plan_penalty)
            score[~mask] = 0
            score[inh] = 0
            if use_hard_plan:
                score[~plan_allowed] = 0
            return score.astype(np.float32)

        score = refresh_score(sim, advantage)
        stagnation = 0

        for it in range(4000):
            current = int(sim.sum())
            if current >= desired:
                break

            allowed = mask & ~inh & (sim == 0)
            if use_hard_plan:
                allowed &= plan_allowed
            thr = seed_thr
            seeds = np.argwhere((score >= thr) & allowed)

            while thr > 0.2 and seeds.size == 0:
                thr -= 0.05
                seeds = np.argwhere((score >= thr) & allowed)

            if seeds.size == 0:
                break

            pool_n = min(seeds.shape[0], max(step * cand_mult, step))
            if pool_n < seeds.shape[0]:
                pick = rng.choice(seeds.shape[0], pool_n, replace=False)
                seeds = seeds[pick]

            order = np.argsort(-score[seeds[:, 0], seeds[:, 1]])
            seeds = seeds[order]

            remain = min(step, desired - current)
            added_total = 0

            for seed_cell in seeds:
                if remain <= 0:
                    break
                patch_size = max(1, min(int(round(rng.lognormal(mu, sigma))), remain))
                added = self._grow_patch(sim, score, allowed, seed_cell, patch_size)
                if added > 0:
                    added_total += added
                    remain -= added

            if added_total == 0:
                stagnation += 1
                if stagnation >= 3:
                    break
            else:
                stagnation = 0

            advantage += adv_step
            if it % 5 == 4:
                score = refresh_score(sim, advantage)

        sim[inh] = 0
        sim[~mask] = 0
        return sim

    # ---- processAlgorithm --------------------------------------------

    def processAlgorithm(self, p, c, fb):
        start_time = time.time()
        seed = int(self.parameterAsInt(p, self.RANDOM_SEED, c))

        # ---------- read parameters ----------
        built_layers = self.parameterAsLayerList(p, self.BUILT_STACK, c)
        if not built_layers or len(built_layers) < 3:
            raise QgsProcessingException("Provide at least 3 Built_Up rasters.")

        pairs = sorted(
            ((self._parse_year(layer), layer) for layer in built_layers),
            key=lambda x: x[0],
        )
        years = [year for year, _ in pairs]
        if len(set(years)) != len(years):
            raise QgsProcessingException(f"Duplicate years detected: {years}")
        built_layers = [layer for _, layer in pairs]
        fb.pushInfo(f"Years sorted from Built_Up input: {years}")

        built_class = self.parameterAsInt(p, self.BUILT_CLASS, c)

        # v6.2 AUTO-CONVERTIBLE MODE
        # ------------------------------------------------------------------
        # CONVERTIBLE_CLASSES is now optional:
        #   - empty  : auto-detect source classes from observed historical transitions
        #              into BUILT_CLASS; if no transition evidence is found, fallback
        #              to all non-built classes in the latest historical raster.
        #   - filled : manual override, e.g. 1,2,4,5,11.
        # Important: 0 and 255 are always treated as outside/nodata/artifact classes.
        conv_str = self.parameterAsString(p, self.CONVERTIBLE_CLASSES, c)
        conv_str = "" if conv_str is None else str(conv_str).strip()
        manual_convertible_classes = None
        if conv_str:
            try:
                manual_convertible_classes = [
                    int(x.strip()) for x in conv_str.split(",") if x.strip()
                ]
            except Exception:
                raise QgsProcessingException(
                    f"CONVERTIBLE_CLASSES must contain integer codes separated by commas. Got: {conv_str}"
                )

        convertible_classes = []
        multi_class = True  # keep full class stack available; final mode is resolved after raster loading
        if manual_convertible_classes is None:
            fb.pushInfo("CONVERTIBLE_CLASSES is empty: auto-detect mode will be used after raster loading.")
        else:
            fb.pushInfo(f"CONVERTIBLE_CLASSES manual input received: {manual_convertible_classes}")

        driver_layers = self.parameterAsLayerList(p, self.DRIVERS, c)
        if not driver_layers:
            raise QgsProcessingException("Provide at least 1 driver raster.")
        transform_policy = int(self.parameterAsEnum(p, self.DRIVER_TRANSFORMS, c))

        # v6.3: read inhibit as a list (may be empty if user skipped it)
        inhibit_layers = self.parameterAsLayerList(p, self.INHIBIT, c)
        if inhibit_layers is None:
            inhibit_layers = []
        template_layer = self.parameterAsRasterLayer(p, self.TEMPLATE, c)

        out_folder = self.parameterAsString(p, self.OUTPUT, c)
        if not out_folder:
            out_folder = p[self.OUTPUT]
        os.makedirs(out_folder, exist_ok=True)

        future_text = self.parameterAsString(p, self.FUTURE_LIST, c)
        try:
            future_years = [int(x.strip()) for x in str(future_text).split(",") if x.strip()]
        except Exception:
            raise QgsProcessingException(
                "FUTURE_LIST must contain future years separated by commas, "
                "for example: 2028,2032,2036,2040"
            )
        if not future_years:
            raise QgsProcessingException("FUTURE_LIST is empty.")

        target_text = self.parameterAsString(p, self.TARGET_SERIES, c)
        target_text = "" if target_text is None else str(target_text).strip()
        target_user = None
        if target_text:
            try:
                target_user = [int(x.strip()) for x in target_text.split(",") if x.strip()]
            except Exception:
                raise QgsProcessingException(
                    "Target_Series must contain integer built-cell targets separated by commas. "
                    "Leave it empty to use automatic projection."
                )
        if target_user is not None and len(target_user) != len(future_years):
            raise QgsProcessingException(
                "Target_Series length must match FUTURE_LIST. "
                f"Future years = {len(future_years)}, target values = {len(target_user)}."
            )

        demand_mode = int(self.parameterAsEnum(p, self.DEMAND_MODE, c))
        pop_base_layer = self.parameterAsRasterLayer(p, self.POP_BASE_RASTER, c)
        pop_future = self._parse_float_series(
            self.parameterAsString(p, self.POP_FUTURE_SERIES, c),
            "POP_FUTURE_SERIES", allow_empty=True,
        )
        pop_per_cell = float(self.parameterAsDouble(p, self.POP_PER_BUILT_CELL, c))
        plan_layer = self.parameterAsRasterLayer(p, self.PLAN_RASTER, c)
        plan_classes = self._parse_int_series(
            self.parameterAsString(p, self.PLAN_BUILT_CLASSES, c),
            "PLAN_BUILT_CLASSES", allow_empty=True,
        )
        plan_policy_mode = int(self.parameterAsEnum(p, self.PLAN_POLICY_MODE, c))
        plan_bonus = float(self.parameterAsDouble(p, self.PLAN_BONUS, c))
        plan_penalty = float(self.parameterAsDouble(p, self.PLAN_PENALTY, c))

        bayesian_demand = bool(self.parameterAsBool(p, self.BAYESIAN_DEMAND, c))
        bayesian_demand_model = ["loglinear", "linear"][int(self.parameterAsEnum(p, self.BAYESIAN_DEMAND_MODEL, c))]

        model_kind = int(self.parameterAsEnum(p, self.MODEL_KIND, c))
        max_samples = int(self.parameterAsInt(p, self.MAX_TRAIN_SAMPLES, c))
        training_target_mode = int(self.parameterAsEnum(p, self.TRAINING_TARGET_MODE, c))
        auto_diag = bool(self.parameterAsBool(p, self.AUTO_DIAGNOSTIC_RECOMMENDATION, c))

        sim_params = {
            "step": int(self.parameterAsInt(p, self.STEP_DEMAND, c)),
            "adv_step": float(self.parameterAsDouble(p, self.ITER_ADVANTAGE, c)),
            "nb_k": int(self.parameterAsInt(p, self.NB_SIZE, c)),
            "seed_thr": float(self.parameterAsDouble(p, self.SEED_THR, c)),
            "cand_mult": int(self.parameterAsInt(p, self.CAND_MULT, c)),
            "patch_mean": float(self.parameterAsDouble(p, self.PATCH_MEAN, c)),
            "patch_sigma": float(self.parameterAsDouble(p, self.PATCH_SIGMA, c)),
        }
        auto_nb = bool(self.parameterAsBool(p, self.AUTO_NB, c))

        tmc_sigma = float(self.parameterAsDouble(p, self.TMC_SIGMA, c))
        do_validate = bool(self.parameterAsBool(p, self.DO_VALIDATE, c))
        do_calibration = bool(self.parameterAsBool(p, self.DO_CALIBRATION, c))
        do_permutation = bool(self.parameterAsBool(p, self.DO_PERMUTATION, c))
        do_shap = bool(self.parameterAsBool(p, self.DO_SHAP, c))
        do_correlation = bool(self.parameterAsBool(p, self.DO_CORRELATION, c))
        do_landscape = bool(self.parameterAsBool(p, self.DO_LANDSCAPE, c))
        smart_best = bool(self.parameterAsBool(p, self.SMART_BEST, c))
        do_spatial_cv = bool(self.parameterAsBool(p, self.DO_SPATIAL_CV, c))
        ensemble_n = int(self.parameterAsInt(p, self.ENSEMBLE_N, c))
        do_report = bool(self.parameterAsBool(p, self.DO_REPORT, c))
        output_prob = bool(self.parameterAsBool(p, self.OUTPUT_PROB, c))
        output_pdf = bool(self.parameterAsBool(p, self.OUTPUT_PDF, c))

        # ---------- read template + masks ----------
        template_arr, template_meta, template_nodata = self._read(template_layer)
        mask = np.isfinite(template_arr)
        if template_nodata is not None:
            mask &= (template_arr != template_nodata)
        fb.pushInfo(
            f"Template grid: rows={template_meta['height']}, cols={template_meta['width']}, "
            f"crs={template_meta.get('crs')}"
        )

        # v6.3: Multi-layer inhibit fusion.
        # For each raster in inhibit_layers:
        #   - align to template grid (nearest-neighbour, categorical-style)
        #   - replace nodata with 0 (treated as "allowed")
        #   - mark forbidden where value == 1
        # Combine all layers with logical OR so that a cell is forbidden
        # if ANY input raster forbids it.
        # If no layers were supplied, inhibit is an all-False mask.
        inhibit = np.zeros(mask.shape, dtype=bool)
        inhibit_layer_names = []
        inhibit_per_layer_counts = []
        for idx, inh_lyr in enumerate(inhibit_layers):
            try:
                src_name = os.path.basename(self._src_path(inh_lyr))
            except Exception:
                src_name = f"inhibit_{idx + 1}"
            inhibit_layer_names.append(src_name)

            inh_arr, _, inh_nodata = self._read_match_template(
                inh_lyr, template_meta, resampling="nearest"
            )
            if inh_nodata is not None:
                inh_arr = np.where(inh_arr == inh_nodata, 0, inh_arr)
            this_layer_inhibit = (inh_arr == 1)

            if this_layer_inhibit.shape != mask.shape:
                raise QgsProcessingException(
                    f"Internal alignment failed for INHIBIT layer '{src_name}'. "
                    f"Layer shape={this_layer_inhibit.shape}, template shape={mask.shape}."
                )

            cells_forbidden = int(np.count_nonzero(this_layer_inhibit))
            inhibit_per_layer_counts.append(cells_forbidden)
            fb.pushInfo(
                f"  Inhibit layer {idx + 1}/{len(inhibit_layers)} '{src_name}': "
                f"{cells_forbidden:,} forbidden cells"
            )
            inhibit |= this_layer_inhibit

        if not inhibit_layers:
            fb.pushInfo("No inhibit raster supplied; running without spatial constraints.")
        else:
            cells_forbidden_union = int(np.count_nonzero(inhibit))
            fb.pushInfo(
                f"Combined inhibit mask (logical OR of {len(inhibit_layers)} raster"
                f"{'s' if len(inhibit_layers) != 1 else ''}): "
                f"{cells_forbidden_union:,} forbidden cells out of {int(mask.sum()):,} valid cells "
                f"({cells_forbidden_union / max(int(mask.sum()), 1) * 100:.2f}%)"
            )

        if inhibit.shape != mask.shape:
            raise QgsProcessingException(
                f"Internal alignment failed for INHIBIT union. "
                f"Inhibit shape={inhibit.shape}, template mask shape={mask.shape}."
            )

        # v6.3: save the combined inhibit mask for transparency / debugging
        # so the planner can verify the union of constraints in QGIS.
        if inhibit_layers:
            self._save(
                template_meta,
                os.path.join(out_folder, "inhibit_combined_mask.tif"),
                inhibit.astype(np.uint8), dtype="uint8",
            )

        # Read Built_Up: keep both binary built (Y) and full class map (SCM).
        # SCM is always loaded because CONVERTIBLE_CLASSES can be auto-detected
        # after all historical rasters are aligned.
        Y = []
        SCM = []
        built_nodata_values = set()
        for layer in built_layers:
            arr, _, nodata = self._read_match_template(
                layer, template_meta, resampling="nearest"
            )
            if nodata is not None:
                try:
                    built_nodata_values.add(int(nodata))
                except Exception:
                    pass
                arr = np.where(arr == nodata, 0, arr)

            cm = arr.astype(np.int32)
            # Preserve source classes inside the valid template boundary.
            # The inhibit raster only forbids conversion; it must not erase
            # source land-cover classes such as class 11.
            cm[~mask] = 0
            SCM.append(cm)

            built = self._bin(arr, built_class).astype(np.uint8)
            if built.shape != mask.shape:
                raise QgsProcessingException(
                    f"Internal alignment failed for Built_Up: {os.path.basename(self._src_path(layer))}"
                )
            built[~mask] = 0
            built[inhibit] = 0
            Y.append(built)
        Y = np.stack(Y, axis=0)
        SCM = np.stack(SCM, axis=0)
        fb.pushInfo(
            f"Built_Up stack aligned: {Y.shape}; valid mask cells={int(mask.sum()):,}"
        )

        # Resolve convertible classes after SCM is available.
        ignore_classes = {0, 255, int(built_class)}
        ignore_classes.update(int(x) for x in built_nodata_values if x is not None)

        if manual_convertible_classes is not None:
            convertible_classes = sorted({
                int(v) for v in manual_convertible_classes
                if int(v) not in ignore_classes and int(v) >= 0
            })
            fb.pushInfo(
                f"Convertible classes resolved from manual input, after excluding 0, 255, nodata, and BUILT_CLASS={built_class}: "
                f"{convertible_classes}"
            )
        else:
            # 1) Transition-aware auto detection: source classes that historically
            # became built-up. This is more selective and avoids forcing the user
            # to type class IDs manually.
            auto_classes = set()
            for t in range(Y.shape[0] - 1):
                new_built = (Y[t] == 0) & (Y[t + 1] == 1) & mask & (~inhibit)
                if np.any(new_built):
                    vals = np.unique(SCM[t][new_built])
                    for v in vals:
                        try:
                            iv = int(v)
                        except Exception:
                            continue
                        if iv not in ignore_classes and iv >= 0:
                            auto_classes.add(iv)

            if auto_classes:
                convertible_classes = sorted(auto_classes)
                fb.pushInfo(
                    "Auto-detected convertible classes from observed historical transitions into built-up: "
                    f"{convertible_classes}"
                )
            else:
                # 2) Fallback: all valid non-built classes in the latest historical raster.
                latest_candidate = mask & (~inhibit) & (Y[-1] == 0)
                vals = np.unique(SCM[-1][latest_candidate]) if np.any(latest_candidate) else np.array([], dtype=np.int32)
                fallback_classes = set()
                for v in vals:
                    try:
                        iv = int(v)
                    except Exception:
                        continue
                    if iv not in ignore_classes and iv >= 0:
                        fallback_classes.add(iv)
                convertible_classes = sorted(fallback_classes)
                fb.pushInfo(
                    "No historical source-class transition evidence found. Fallback auto-detected convertible classes from latest non-built cells: "
                    f"{convertible_classes}"
                )

        multi_class = len(convertible_classes) > 0
        if multi_class:
            fb.pushInfo(f"Multi-class feature mode ON. Final convertible classes: {convertible_classes}")
        else:
            fb.pushInfo(
                "Multi-class feature mode OFF because no eligible convertible classes were detected. "
                "The simulation will still export binary built-up outputs."
            )

        # Save detected source-class inventory for transparent debugging.
        try:
            class_inventory = []
            for t, yr in enumerate(years):
                vals, counts = np.unique(SCM[t][mask], return_counts=True)
                for val, cnt in zip(vals, counts):
                    iv = int(val)
                    if iv in (0, 255):
                        continue
                    class_inventory.append({"year": int(yr), "class": iv, "cells": int(cnt)})
            with open(os.path.join(out_folder, "class_inventory.json"), "w", encoding="utf-8") as f:
                json.dump(class_inventory, f, indent=2)
            with open(os.path.join(out_folder, "convertible_classes.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "built_class": int(built_class),
                    "manual_input": manual_convertible_classes,
                    "ignore_classes": sorted(int(x) for x in ignore_classes),
                    "convertible_classes": [int(x) for x in convertible_classes],
                    "mode": "manual" if manual_convertible_classes is not None else "auto_transition_aware_with_latest_fallback"
                }, f, indent=2)
        except Exception as e:
            fb.pushInfo(f"Class inventory JSON skipped: {e}")

        # plan raster
        plan_allowed = None
        if plan_layer is not None:
            plan_allowed = self._load_plan_allowed(
                plan_layer, template_meta, mask, plan_classes, fb
            )
            self._save(
                template_meta,
                os.path.join(out_folder, "plan_allowed_mask.tif"),
                plan_allowed.astype(np.uint8), dtype="uint8",
            )

        # drivers
        drv_arr, drv_norm, drv_names, tx_used = self._load_drivers(
            driver_layers, mask, transform_policy, template_meta, fb,
        )

        # NEW v6: driver correlation
        corr_data = None
        if do_correlation:
            fb.pushInfo("Driver correlation matrix ...")
            corr, warns = correlation_matrix(drv_arr, mask, drv_names)
            corr_data = {
                "matrix": corr.tolist(),
                "names": drv_names,
                "warnings": warns,
            }
            for w in warns:
                fb.pushInfo(f"  REDUNDANCY: {w['a']} ~ {w['b']} (r={w['r']:+.3f})")

        # NEW v6: auto kernel-size tuning
        auto_nb_results = None
        if auto_nb and len(years) >= 3:
            fb.pushInfo("Auto-tuning neighborhood kernel size ...")
            best_k, auto_nb_results = self._auto_tune_nb(
                Y, drv_arr, drv_norm, mask, inhibit,
                max_samples, model_kind, seed, fb,
                training_target_mode=training_target_mode,
                source_class_thw=SCM, convertible_classes=convertible_classes,
            )
            if best_k is not None:
                sim_params["nb_k"] = int(best_k)
                fb.pushInfo(f"Using auto-tuned neighborhood k={best_k}.")

        # ---------- training ----------
        fb.pushInfo("Training classifier ...")
        train_start = time.time()
        model, X_train, y_train = self._train_model(
            Y, drv_arr, drv_norm, sim_params["nb_k"], mask,
            max_samples, model_kind, seed, fb,
            training_target_mode=training_target_mode,
            source_class_thw=SCM, convertible_classes=convertible_classes,
        )
        fb.pushInfo(f"Training completed in {time.time() - train_start:.1f}s")

        feature_names = self._feature_names(drv_names, convertible_classes)

        metrics = {
            "version": "6.3.1-multi-inhibit-pdf-safe",
            "years_historical": years,
            "future_years": future_years,
            "drivers": [
                {"name": name, "transform": transform}
                for name, transform in zip(drv_names, tx_used)
            ],
            "feature_names": feature_names,
            "ensemble_n": ensemble_n,
            "demand_mode_index": int(demand_mode),
            "plan_policy_mode_index": int(plan_policy_mode),
            "plan_bonus": float(plan_bonus),
            "plan_penalty": float(plan_penalty),
            "training_target_mode": int(training_target_mode),
            "training_target_mode_label": "new_built_transition_only" if int(training_target_mode) == 1 else "built_status_next_year",
            "multi_class": multi_class,
            "convertible_classes": convertible_classes,
            "neighborhood_k": int(sim_params["nb_k"]),
            "auto_nb_results": auto_nb_results,
            "inhibit": {
                "n_layers": len(inhibit_layers),
                "layer_names": inhibit_layer_names,
                "per_layer_forbidden_cells": inhibit_per_layer_counts,
                "combined_forbidden_cells": int(np.count_nonzero(inhibit)),
                "valid_cells": int(np.count_nonzero(mask)),
            },
            "input_parameter_ids": {
                "built_up": self.BUILT_STACK,
                "target_series": self.TARGET_SERIES,
            },
        }
        if corr_data is not None:
            metrics["driver_correlation"] = {
                "warnings": corr_data["warnings"],
                "names": corr_data["names"],
            }

        # ---------- validation ----------
        validation = None
        prob_last = None
        pred_last = None
        X_last = None
        if do_validate and len(years) >= 3:
            fb.pushInfo("Historical validation: predicting last historical year ...")
            b0 = Y[-2]; b1_true = Y[-1]
            scm_t = SCM[-2] if multi_class else None
            X_last = self._build_features(
                b0, drv_arr, drv_norm, sim_params["nb_k"], mask,
                source_class_map=scm_t, convertible_classes=convertible_classes,
            )
            prob_last = self._predict_prob(model, X_last, mask)

            observed_change = (b0 == 0) & (b1_true == 1) & mask & ~inhibit
            demand = int(np.count_nonzero(observed_change))
            cand_mask = mask & ~inhibit & (b0 == 0)
            cand_idx = np.flatnonzero(cand_mask.ravel())
            pred_last = b0.copy()

            if demand > 0 and len(cand_idx) > 0:
                demand = min(demand, len(cand_idx))
                scores = prob_last.ravel()[cand_idx]
                take = min(demand, len(cand_idx))
                if take < len(cand_idx):
                    top = np.argpartition(-scores, take - 1)[:take]
                else:
                    top = np.arange(len(cand_idx))
                pred_last.ravel()[cand_idx[top]] = 1
            else:
                fb.pushInfo("No observed new built-up change in validation period.")

            comp = validation_diagnostics(b0, b1_true, pred_last, mask & ~inhibit)
            multires = {
                f"FoM_w{w}": fom_at_resolution(b0, b1_true, pred_last, mask & ~inhibit, w)
                for w in (1, 3, 5, 9)
            }
            validation = {"demand": demand, "components": comp, "multires_FoM": multires}
            metrics["validation"] = validation

            self._save(
                template_meta,
                os.path.join(out_folder, f"validation_pred_{years[-1]}.tif"),
                pred_last.astype(np.uint8), dtype="uint8",
            )

            if do_calibration:
                transition_truth = ((b0 == 0) & (b1_true == 1)).astype(np.uint8)
                calibration_mask = mask & ~inhibit & (b0 == 0)
                centres, fracs, counts = reliability_curve(
                    prob_last, transition_truth, calibration_mask, n_bins=10
                )
                bs = brier_score(prob_last, transition_truth, calibration_mask)
                metrics["calibration"] = {
                    "brier": bs,
                    "bin_centres": centres.tolist(),
                    "bin_observed_fraction": [None if np.isnan(x) else float(x) for x in fracs],
                    "bin_count": counts.tolist(),
                }
                fb.pushInfo(f"Brier score = {bs:.4f}")

        # ---------- permutation ----------
        if do_permutation and validation is not None and X_last is not None:
            fb.pushInfo("Permutation driver contribution analysis ...")
            if int(training_target_mode) == 1:
                perm_y = ((Y[-2] == 0) & (Y[-1] == 1)).astype(np.uint8)
                perm_mask = mask & ~inhibit & (Y[-2] == 0)
            else:
                perm_y = Y[-1]
                perm_mask = mask & ~inhibit
            metrics["permutation_importance"] = self._permutation_importance(
                model, X_last, perm_y, perm_mask, feature_names,
                n_repeats=3, seed=seed, fb=fb,
            )

        # ---------- v6 NEW: SHAP ----------
        shap_top_grid = None
        if do_shap and validation is not None and X_last is not None:
            fb.pushInfo("SHAP-style attribution ...")
            shap_res = self._shap_attribution(
                model, X_last, mask & ~inhibit, feature_names, fb,
                sample_size=15000, seed=seed,
            )
            if shap_res is not None:
                shap_top_grid = shap_res["shap_grid_top"]
                metrics["shap"] = {
                    "ranking": shap_res["ranking"],
                    "top_feature": shap_res["top_feature"],
                    "n_samples": shap_res["n_samples"],
                }
                metrics["shap_beeswarm"] = {
                    k: shap_res["beeswarm"][k]
                    for k in [r["feature"] for r in shap_res["ranking"][:5]]
                }
                self._save(
                    template_meta,
                    os.path.join(out_folder, "shap_top_driver.tif"),
                    shap_top_grid, dtype="float32",
                )

        # ---------- spatial CV ----------
        if do_spatial_cv:
            fb.pushInfo("Spatial block cross-validation 5x5 ...")
            metrics["spatial_cv"] = self._spatial_block_cv(
                Y, drv_arr, drv_norm, sim_params["nb_k"], mask,
                max_samples, model_kind, seed, n_blocks=5, fb=fb,
                source_class_thw=SCM, convertible_classes=convertible_classes,
            )

        # ---------- demand engine ----------
        target, demand_info = self._resolve_targets(
            demand_mode, target_user, pop_future, pop_base_layer,
            pop_per_cell, plan_allowed, Y, years, future_years,
            template_meta, mask, fb,
        )
        metrics.update(demand_info)
        metrics["target_per_year"] = dict(zip(future_years, target))

        # NEW v6: Bayesian demand posterior on top of demand engine
        posterior = None
        if bayesian_demand:
            fb.pushInfo(f"Bayesian demand posterior ({bayesian_demand_model}, 2000 bootstrap)...")
            hist_counts = [int(y.sum()) for y in Y]
            posterior = bootstrap_demand_posterior(
                years, hist_counts, future_years,
                n_samples=2000, seed=seed, model=bayesian_demand_model,
            )
            metrics["demand_posterior"] = {
                int(fy): {
                    "median": int(posterior[fy]["median"]),
                    "lo95": int(posterior[fy]["lo95"]),
                    "hi95": int(posterior[fy]["hi95"]),
                }
                for fy in future_years
            }
            for fy in future_years:
                pf = posterior[fy]
                fb.pushInfo(f"  {fy}: median={pf['median']:,}  95% CrI=[{pf['lo95']:,}..{pf['hi95']:,}]")

        # ---------- forward simulation ----------
        fb.pushInfo("Forward simulation with ensemble ...")
        state_best = Y[-1].copy().astype(np.uint8)
        ensemble_results = {}
        prob_per_year = {}
        landscape_per_year = {}
        future_class_maps = {}
        run_log = []

        # Running class-state for multi-class output and multi-class features.
        # Future years preserve non-converted classes and only change allocated
        # growth cells into BUILT_CLASS.
        projected_class_state = SCM[-1].astype(np.int32).copy() if (multi_class and SCM is not None) else None

        for year_index, year in enumerate(future_years):
            fb.pushInfo(f"  Simulating year {year} ...")
            year_start = time.time()

            scm_t = projected_class_state if multi_class else None
            X = self._build_features(
                state_best, drv_arr, drv_norm, sim_params["nb_k"], mask,
                source_class_map=scm_t, convertible_classes=convertible_classes,
            )
            prob_base = self._predict_prob(model, X, mask)
            prob_per_year[year] = prob_base.copy()

            # Determine demand for each ensemble member
            sims = []
            for ens in range(ensemble_n):
                run_seed = seed + 1000 * int(year) + ens
                rng_e = np.random.default_rng(run_seed)
                if posterior is not None:
                    samples_yr = posterior[year]["samples"]
                    dem_sample = int(samples_yr[ens % len(samples_yr)])
                    desired = max(int(dem_sample), int(state_best.sum()))
                else:
                    desired = max(int(target[year_index]), int(state_best.sum()))

                if tmc_sigma > 0:
                    noise = rng_e.normal(0.0, tmc_sigma, prob_base.shape).astype(np.float32)
                    prob_noise = np.clip(prob_base * (1.0 + noise), 0, 1)
                else:
                    prob_noise = prob_base

                sim = self._simulate_one(
                    prob_noise, state_best, mask, inhibit, desired,
                    sim_params, run_seed,
                    plan_allowed=plan_allowed,
                    plan_policy_mode=plan_policy_mode,
                    plan_bonus=plan_bonus, plan_penalty=plan_penalty,
                )
                sims.append(sim)

            ensemble_results[year] = sims

            # NEW v6: smart best-run selection
            stack_sims = np.stack(sims, axis=0).astype(np.float32)
            agreement = stack_sims.mean(axis=0).astype(np.float32)
            if smart_best and len(sims) > 1:
                median_pat = np.median(stack_sims, axis=0)
                dists = [float(np.mean(np.abs(s.astype(np.float32) - median_pat))) for s in sims]
                best_idx = int(np.argmin(dists))
            else:
                best_idx = 0
            sim_best = sims[best_idx]   # binary 0/1 built map

            # -----------------------------------------------------------
            # Save the BINARY built map (always produced; legacy output)
            # -----------------------------------------------------------
            self._save(
                template_meta,
                os.path.join(out_folder, f"scape_ca_built_{year}.tif"),
                sim_best.astype(np.uint8), dtype="uint8",
            )

            # -----------------------------------------------------------
            # FIX v6.1: True multi-class CA output without class 255.
            #
            # The CA engine remains binary for allocation, but the exported
            # multi-class land-cover raster preserves all non-converted classes.
            # Only cells selected as built-up by the simulation are changed to
            # BUILT_CLASS.
            #
            # Rules:
            #   - outside template boundary       -> 0 nodata/background
            #   - inhibited cells                 -> preserve original class
            #   - existing and simulated built-up -> built_class
            #   - all other cells                 -> previous projected class
            #
            # Extra diagnostic:
            #   - converted_from_class_<year>.tif records the source class of
            #     newly converted cells only; 0 elsewhere.
            # -----------------------------------------------------------
            if multi_class and projected_class_state is not None:
                previous_binary_state = state_best.copy().astype(np.uint8)
                previous_class_state = projected_class_state.copy().astype(np.int32)

                class_map = previous_class_state.copy()
                new_growth = (previous_binary_state == 0) & (sim_best == 1) & mask & (~inhibit)

                class_map[sim_best == 1] = int(built_class)
                class_map[~mask] = 0

                converted_from = np.zeros_like(class_map, dtype=np.int32)
                converted_from[new_growth] = previous_class_state[new_growth]
                converted_from[~mask] = 0

                max_class_value = int(np.nanmax(class_map)) if class_map.size else 0
                class_dtype = "uint8" if max_class_value <= 254 else "uint16"

                # Save only the definitive multi-class future land-cover map.
                # The older scape_ca_<year>.tif duplicate is intentionally removed
                # because it contains the same class_map as landcover_<year>.tif.
                self._save(
                    template_meta,
                    os.path.join(out_folder, f"landcover_{year}.tif"),
                    class_map.astype(class_dtype), dtype=class_dtype,
                    nodata_value=0,
                )
                self._save(
                    template_meta,
                    os.path.join(out_folder, f"converted_from_class_{year}.tif"),
                    converted_from.astype(class_dtype), dtype=class_dtype,
                    nodata_value=0,
                )

                # Propagate the multi-class state to the next future step.
                projected_class_state = class_map.copy()
                future_class_maps[int(year)] = class_map.copy()
            else:
                # Binary mode: save the selected future state as landcover_<year>.tif.
                # No scape_ca_<year>.tif is produced to avoid duplicate outputs.
                self._save(
                    template_meta,
                    os.path.join(out_folder, f"landcover_{year}.tif"),
                    sim_best.astype(np.uint8), dtype="uint8",
                    nodata_value=0,
                )

            uncertainty = (4.0 * agreement * (1.0 - agreement)).astype(np.float32)

            self._save(
                template_meta,
                os.path.join(out_folder, f"agreement_{year}.tif"),
                agreement, dtype="float32",
            )
            self._save(
                template_meta,
                os.path.join(out_folder, f"uncertainty_{year}.tif"),
                uncertainty, dtype="float32",
            )
            if output_prob:
                self._save(
                    template_meta,
                    os.path.join(out_folder, f"prob_{year}.tif"),
                    prob_base.astype(np.float32), dtype="float32",
                )

            # NEW v6: landscape metrics
            if do_landscape:
                lsm = landscape_metrics(sim_best, mask)
                landscape_per_year[year] = lsm
                fb.pushInfo(
                    f"    Landscape {year}: patches={lsm['NumPatches']}, "
                    f"LPI={lsm['LPI_pct']:.2f}%, MPS={lsm['MPS_cells']:.1f}"
                )

            elapsed = time.time() - year_start
            msg = (
                f"Year {year}: built={int(sim_best.sum()):,}, "
                f"target={target[year_index]:,}, ensemble_n={ensemble_n}, "
                f"best_run={best_idx}, time={elapsed:.1f}s"
            )
            fb.pushInfo(msg)
            run_log.append(msg)

            # Propagate the selected best run so the next simulation step is
            # consistent with the exported raster and the running class map.
            state_best = sim_best.copy().astype(np.uint8)
            state_best[inhibit] = 0
            state_best[~mask] = 0

        if landscape_per_year:
            metrics["landscape_per_year"] = {int(k): v for k, v in landscape_per_year.items()}

        metrics["runtime_total_s"] = time.time() - start_time
        metrics["sim_log"] = run_log

        # ---------- save metrics + report ----------
        with open(os.path.join(out_folder, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=float)

        if do_report and _HAS_PLT:
            try:
                self._build_report(
                    out_folder, Y, years, future_years,
                    ensemble_results, prob_per_year,
                    mask, inhibit, metrics, validation,
                    posterior=posterior,
                    shap_top_grid=shap_top_grid,
                    corr_data=corr_data,
                    landscape_per_year=landscape_per_year,
                    pred_last=pred_last,
                    SCM_final=projected_class_state if (multi_class and projected_class_state is not None) else None,
                    multi_class=multi_class,
                    built_class=built_class,
                    convertible_classes=convertible_classes,
                    SCM_hist=SCM if ('SCM' in locals() and SCM is not None) else None,
                    future_class_maps=future_class_maps,
                )
                fb.pushInfo(f"Report: {os.path.join(out_folder, 'report.png')}")
            except Exception as e:
                fb.pushInfo(f"Report generation skipped: {e}")
        elif do_report:
            fb.pushInfo("matplotlib not available; PNG report skipped.")

        if output_pdf:
            if _HAS_REPORTLAB:
                try:
                    self._build_pdf_report(out_folder, metrics, validation, years, future_years)
                    fb.pushInfo(f"PDF report: {os.path.join(out_folder, 'report.pdf')}")
                except Exception as e:
                    import traceback
                    fb.pushInfo(f"PDF report generation skipped: {e}")
                    fb.pushInfo(traceback.format_exc())
            else:
                fb.pushInfo("reportlab not available; PDF skipped. pip install reportlab to enable.")

        fb.pushInfo(f"All done in {metrics['runtime_total_s']:.1f}s.")
        return {self.OUTPUT: out_folder}

    # ---- visual report (modernised, multi-class aware) ---------------

    # ---------- color palettes ----------
    @staticmethod
    def _modern_class_palette(class_codes, built_class):
        """
        Return a dict {code: (r,g,b)} for known Dynamic World–like classes.
        Unknown codes get a perceptually-distinct fallback color.
        """
        # Material-design-ish palette tuned for land-cover legibility
        canonical = {
            0: (0.93, 0.93, 0.93),   # nodata / background
            1: (0.20, 0.45, 0.85),   # water — cool blue
            2: (0.10, 0.55, 0.20),   # trees — forest green
            3: (0.55, 0.78, 0.30),   # grass — light green
            4: (0.97, 0.82, 0.30),   # crops — warm yellow
            5: (0.78, 0.70, 0.40),   # shrub — olive khaki
            6: (0.85, 0.10, 0.10),   # built — strong red
            7: (0.65, 0.65, 0.65),   # bare — neutral grey
            8: (0.85, 0.85, 0.95),   # snow/ice — pale blue-white
        }
        # ensure built_class always shows red
        canonical[int(built_class)] = (0.85, 0.10, 0.10)
        # fallback distinct colours for unexpected codes
        fallbacks = [
            (0.45, 0.30, 0.65), (0.88, 0.55, 0.20), (0.30, 0.65, 0.65),
            (0.75, 0.30, 0.55), (0.45, 0.50, 0.20), (0.60, 0.40, 0.10),
        ]
        result = {}
        fb_i = 0
        for code in class_codes:
            code = int(code)
            if code in canonical:
                result[code] = canonical[code]
            else:
                result[code] = fallbacks[fb_i % len(fallbacks)]
                fb_i += 1
        return result

    @staticmethod
    def _classmap_to_rgb(class_map, mask, inhibit, palette, nodata_color=(0.93, 0.93, 0.93),
                          inhibit_color=(0.55, 0.55, 0.55)):
        H, W = class_map.shape
        rgb = np.full((H, W, 3), nodata_color, dtype=np.float32)
        rgb[inhibit] = inhibit_color
        for code, color in palette.items():
            sel = (class_map == code) & mask & ~inhibit
            rgb[sel] = color
        return rgb

    @staticmethod
    def _binary_to_rgb(binary_map, mask, inhibit, built_color=(0.85, 0.10, 0.10),
                        bg_color=(0.93, 0.93, 0.93), inhibit_color=(0.55, 0.55, 0.55)):
        H, W = binary_map.shape
        rgb = np.full((H, W, 3), bg_color, dtype=np.float32)
        rgb[inhibit] = inhibit_color
        rgb[(binary_map == 1) & mask & ~inhibit] = built_color
        return rgb


    def _build_report(self, out_folder, Y, years, future_years,
                      ensemble_results, prob_per_year,
                      mask, inhibit, metrics, validation,
                      posterior=None, shap_top_grid=None, corr_data=None,
                      landscape_per_year=None, pred_last=None,
                      SCM_final=None, multi_class=False, built_class=6,
                      convertible_classes=None, SCM_hist=None,
                      future_class_maps=None):
        """
        Executive PNG dashboard.

        Mode-aware behaviour:
        - binary mode   -> report maps use built-up representation
        - multi-class   -> report maps use multi-class land-cover representation
        """
        if not _HAS_PLT:
            return

        from matplotlib import patches
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import textwrap

        H, W = mask.shape
        last_future = int(future_years[-1])
        first_future = int(future_years[0])
        use_multiclass_report = bool(multi_class and SCM_final is not None and SCM_hist is not None)
        future_class_maps = future_class_maps or {}

        # Theme
        bg = "#F4F7FB"
        panel = "#FFFFFF"
        ink = "#102A43"
        muted = "#627D98"
        line = "#D9E2EC"
        blue = "#0B63CE"
        teal = "#00A884"
        cyan = "#00A6D6"
        orange = "#F59E0B"
        red = "#D64545"
        green = "#2F9E44"
        dark = "#0B2A4A"

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "figure.facecolor": bg,
            "axes.facecolor": panel,
            "axes.edgecolor": line,
            "axes.labelcolor": ink,
            "axes.titlecolor": ink,
            "axes.titleweight": "bold",
            "axes.titlesize": 9.6,
            "axes.labelsize": 8.0,
            "xtick.color": muted,
            "ytick.color": muted,
            "font.size": 8.4,
            "grid.color": "#E6EEF8",
            "grid.alpha": 0.85,
            "axes.spines.top": False,
            "axes.spines.right": False,
        })

        cmap_unc = LinearSegmentedColormap.from_list(
            "scape_unc", ["#F8FAFC", "#FFF2CC", "#F59E0B", "#B45309"]
        )

        fig = plt.figure(figsize=(24, 13.5), facecolor=bg)
        gs = fig.add_gridspec(20, 24, left=0.028, right=0.985, top=0.972, bottom=0.040,
                              wspace=0.55, hspace=1.10)

        def add_round_panel(ax, radius=0.030, fc=panel, ec=line, lw=0.9):
            rect = patches.FancyBboxPatch(
                (0, 0), 1, 1, transform=ax.transAxes,
                boxstyle=f"round,pad=0.010,rounding_size={radius}",
                facecolor=fc, edgecolor=ec, linewidth=lw,
                zorder=-10, clip_on=False)
            ax.add_patch(rect)

        def style_panel(ax, title=None, subtitle=None):
            ax.set_facecolor(panel)
            for sp in ax.spines.values():
                sp.set_edgecolor(line)
                sp.set_linewidth(0.8)
            if title:
                ax.text(0.000, 1.075, title, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=9.8, color=ink, weight="bold")
            if subtitle:
                ax.text(0.000, 1.020, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                        fontsize=7.1, color=muted)
            return ax

        def fmt_int(x):
            try:
                return f"{int(round(float(x))):,}"
            except Exception:
                return "-"

        def fmt_pct(x):
            try:
                return f"{float(x) * 100:.1f}%"
            except Exception:
                return "-"

        def draw_kpi(ax, label, value, sub, color):
            ax.set_axis_off()
            add_round_panel(ax, 0.030)
            ax.add_patch(patches.Rectangle((0.0, 0.0), 0.014, 1.0,
                                           transform=ax.transAxes, color=color,
                                           clip_on=False, zorder=-1))
            ax.text(0.060, 0.730, label.upper(), transform=ax.transAxes,
                    fontsize=7.3, color=muted, weight="bold", ha="left", va="center")
            ax.text(0.060, 0.420, value, transform=ax.transAxes,
                    fontsize=18.0, color=ink, weight="bold", ha="left", va="center")
            ax.text(0.060, 0.150, sub, transform=ax.transAxes,
                    fontsize=7.1, color=muted, ha="left", va="center")

        def binary_rgb(binary_map):
            rgb = np.full((H, W, 3), (0.965, 0.965, 0.965), dtype=np.float32)
            rgb[~mask] = (0.91, 0.93, 0.95)
            rgb[inhibit] = (0.62, 0.65, 0.68)
            rgb[(binary_map == 1) & mask & ~inhibit] = (0.83, 0.10, 0.10)
            return rgb

        class_codes = set([int(built_class)])
        if SCM_hist is not None:
            try:
                vals = np.unique(SCM_hist[np.isfinite(SCM_hist)])
                class_codes.update(int(v) for v in vals if int(v) != 0)
            except Exception:
                pass
        if SCM_final is not None:
            try:
                vals = np.unique(SCM_final[np.isfinite(SCM_final)])
                class_codes.update(int(v) for v in vals if int(v) != 0)
            except Exception:
                pass
        palette = self._modern_class_palette(sorted(class_codes), built_class)

        def render_mode_map(arr, mode):
            if mode == "class":
                return self._classmap_to_rgb(arr.astype(np.int32), mask, inhibit, palette)
            return binary_rgb(arr.astype(np.uint8))

        def show_map(ax, arr, title, subtitle=None, kind="binary", colorbar=False):
            style_panel(ax, title, subtitle)
            im = None
            if kind == "binary":
                ax.imshow(binary_rgb(arr), interpolation="nearest")
            elif kind == "class":
                ax.imshow(self._classmap_to_rgb(arr.astype(np.int32), mask, inhibit, palette), interpolation="nearest")
            elif kind == "uncertainty":
                im = ax.imshow(np.where(mask & ~inhibit, arr, np.nan), cmap=cmap_unc,
                               vmin=0, vmax=1, interpolation="nearest")
            elif kind == "validation":
                ax.imshow(arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
                sp.set_edgecolor(line)
            if colorbar and im is not None:
                cax = inset_axes(ax, width="72%", height="3.2%", loc="lower center",
                                 bbox_to_anchor=(0.0, -0.075, 1.0, 1.0),
                                 bbox_transform=ax.transAxes, borderpad=0)
                cb = fig.colorbar(im, cax=cax, orientation="horizontal")
                cb.ax.tick_params(labelsize=6.1, length=0, pad=1)
                cb.outline.set_visible(False)

        def safe_stack(year):
            return np.stack(ensemble_results[year], axis=0).astype(np.float32)

        def median_pattern(year):
            return (safe_stack(year).mean(axis=0) >= 0.5).astype(np.uint8)

        def wrap_text(s, width=88):
            return "\n".join(textwrap.wrap(str(s or ""), width=width))

        comp = validation.get("components", {}) if validation is not None else {}
        base_count = int(Y[-1].sum())
        last_stack = safe_stack(last_future)
        last_agreement = last_stack.mean(axis=0)
        last_uncertainty = (4.0 * last_agreement * (1.0 - last_agreement)).astype(np.float32)
        last_best = (last_agreement >= 0.5).astype(np.uint8)
        final_count = int(last_best.sum())
        growth_abs = final_count - base_count
        growth_pct = growth_abs / max(base_count, 1)
        uncertainty_mean = float(np.nanmean(last_uncertainty[mask & ~inhibit])) if np.any(mask & ~inhibit) else 0.0

        diagnosis = self._interpret_validation_status(comp) if validation is not None else {
            "status": "NO_VALIDATION",
            "title": "Validation was not executed",
            "interpretation": "The dashboard shows projected growth and uncertainty, but no historical allocation test was available.",
            "recommendation": "Enable historical validation to obtain FoM, hit, miss, false alarm, and calibration diagnostics.",
            "summary": "-",
        }
        status_color = {
            "GOOD": green, "MODERATE": orange, "OVER_PREDICTIVE": red,
            "UNDER_PREDICTIVE": orange, "WEAK_ALLOCATION": red, "NO_VALIDATION": muted,
        }.get(diagnosis.get("status"), muted)

        # Header with dedicated 2-row height to prevent overlap
        ax_header = fig.add_subplot(gs[0:2, :])
        ax_header.set_axis_off()
        ax_header.add_patch(patches.FancyBboxPatch(
            (0, 0.08), 1, 0.82, transform=ax_header.transAxes,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=dark, edgecolor=dark, zorder=0))
        ax_header.text(0.024, 0.66, "SCAPE-CA Spatial Growth Intelligence Dashboard",
                       transform=ax_header.transAxes, fontsize=15.5, color="white",
                       weight="bold", ha="left", va="center")
        ax_header.text(0.024, 0.34,
                       "Spatial Cellular Automata with Patch-based Evolution | Executive summary of validation, demand, uncertainty, and drivers",
                       transform=ax_header.transAxes, fontsize=7.6, color="#CFE8FF",
                       ha="left", va="center")
        ax_header.text(0.976, 0.70, "Firman Afrianto · Maya Safira",
                       transform=ax_header.transAxes, fontsize=7.7, color="#DDEBFF",
                       style="italic", ha="right", va="center")
        ax_header.text(0.976, 0.34,
                       f"Historical {years[0]}-{years[-1]} | Projection {first_future}-{last_future} | "
                       f"{'Multi-class' if use_multiclass_report else 'Built-up'} mode | Ensemble N={metrics.get('ensemble_n', 1)}",
                       transform=ax_header.transAxes, fontsize=7.2, color="#B6D4F2",
                       ha="right", va="center")

        # KPI row
        draw_kpi(fig.add_subplot(gs[2:4, 0:5]), "Base built cells", fmt_int(base_count), f"latest historical year {years[-1]}", blue)
        draw_kpi(fig.add_subplot(gs[2:4, 5:10]), "Projected built cells", fmt_int(final_count), f"median-pattern map {last_future}", teal)
        draw_kpi(fig.add_subplot(gs[2:4, 10:15]), "Net growth", f"+{fmt_int(growth_abs)}", f"{fmt_pct(growth_pct)} from baseline", orange)
        draw_kpi(fig.add_subplot(gs[2:4, 15:20]), "Validation FoM", f"{float(comp.get('FoM', 0)):.3f}", "Figure of Merit", green if float(comp.get("FoM", 0)) >= 0.15 else red)
        draw_kpi(fig.add_subplot(gs[2:4, 20:24]), "Uncertainty", f"{uncertainty_mean:.3f}", "mean ensemble disagreement", red if uncertainty_mean > 0.35 else cyan)

        hist_mode = "class" if use_multiclass_report else "binary"
        baseline_arr = SCM_hist[0] if use_multiclass_report else Y[0]
        ref_arr = SCM_hist[-1] if use_multiclass_report else Y[-1]

        show_map(fig.add_subplot(gs[5:12, 0:5]), baseline_arr,
                 f"Historical baseline {years[0]}",
                 "observed multi-class land cover" if use_multiclass_report else "observed built-up reference",
                 kind=hist_mode)
        show_map(fig.add_subplot(gs[5:12, 5:10]), ref_arr,
                 f"Historical reference {years[-1]}",
                 "latest observed multi-class land cover" if use_multiclass_report else "latest observed built-up",
                 kind=hist_mode)

        ax_val = fig.add_subplot(gs[5:12, 10:15])
        if validation is not None and pred_last is not None:
            b0 = Y[-2]
            b1 = Y[-1]
            rgb = np.full((H, W, 3), (0.965, 0.965, 0.965), dtype=np.float32)
            rgb[~mask] = (0.91, 0.93, 0.95)
            rgb[inhibit] = (0.62, 0.65, 0.68)
            rgb[(b0 == 1) & (b1 == 1) & mask] = (0.78, 0.20, 0.20)
            rgb[(b0 == 0) & (b1 == 1) & (pred_last == 1) & mask] = (0.18, 0.62, 0.32)
            rgb[(b0 == 0) & (b1 == 1) & (pred_last == 0) & mask] = (0.10, 0.39, 0.86)
            rgb[(b0 == 0) & (b1 == 0) & (pred_last == 1) & mask] = (0.95, 0.55, 0.13)
            show_map(ax_val, rgb, "Historical validation map", "red persistence | green hit | blue miss | orange false alarm", kind="validation")
        else:
            ax_val.set_axis_off(); add_round_panel(ax_val)
            ax_val.text(0.5, 0.5, "Validation map not available", ha="center", va="center", color=muted)

        ax_growth = fig.add_subplot(gs[5:12, 15:24])
        style_panel(ax_growth, "Built-up growth trajectory", "historical observations and projected ensemble band")
        hist_counts = [int(x.sum()) for x in Y]
        proj_counts = [int(safe_stack(y).mean(axis=0).sum()) for y in future_years]
        proj_low = [int(min(int(s.sum()) for s in ensemble_results[y])) for y in future_years]
        proj_high = [int(max(int(s.sum()) for s in ensemble_results[y])) for y in future_years]
        ax_growth.plot(years, hist_counts, marker="o", lw=2.6, color=blue, label="Historical")
        ax_growth.plot(future_years, proj_counts, marker="o", lw=2.6, color=teal, label="Projected median")
        ax_growth.fill_between(future_years, proj_low, proj_high, color=teal, alpha=0.18, label="Ensemble range")
        if posterior is not None:
            p_years = [y for y in future_years if y in posterior]
            if p_years:
                p_lo = [posterior[y]["lo95"] for y in p_years]
                p_hi = [posterior[y]["hi95"] for y in p_years]
                ax_growth.fill_between(p_years, p_lo, p_hi, color=orange, alpha=0.15, label="Demand 95% CrI")
        ax_growth.grid(True, axis="y")
        ax_growth.set_ylabel("Built-up cells")
        ax_growth.ticklabel_format(axis="y", style="plain")
        ax_growth.legend(frameon=False, fontsize=7.2, loc="upper left")

        # Bottom row: mode-aware final map + uncertainty + drivers + diagnostic
        final_mode = "class" if use_multiclass_report else "binary"
        final_map = SCM_final if use_multiclass_report else last_best
        final_subtitle = "projected multi-class land cover" if use_multiclass_report else "median-pattern projected built-up"
        show_map(fig.add_subplot(gs[14:20, 0:5]), final_map,
                 ("Projected land cover " if use_multiclass_report else "Projected built-up ") + str(last_future),
                 final_subtitle, kind=final_mode)
        show_map(fig.add_subplot(gs[14:20, 5:10]), last_uncertainty, f"Uncertainty {last_future}",
                 "4 x agreement x (1-agreement)", kind="uncertainty", colorbar=True)

        ax_feat = fig.add_subplot(gs[14:20, 10:15])
        style_panel(ax_feat, "Top spatial drivers", "maximum six drivers to avoid visual clutter")
        label_metric = "Delta FoM"
        if "permutation_importance" in metrics:
            ranking = metrics["permutation_importance"].get("ranking", [])[:6]
            labels = [str(r.get("feature", "-"))[:22] for r in ranking][::-1]
            vals = [float(r.get("delta_fom_mean", 0)) for r in ranking][::-1]
        elif "shap" in metrics:
            ranking = metrics["shap"].get("ranking", [])[:6]
            labels = [str(r.get("feature", "-"))[:22] for r in ranking][::-1]
            vals = [float(r.get("mean_abs_shap", 0)) for r in ranking][::-1]
            label_metric = "mean |SHAP|"
        else:
            labels, vals = [], []
        if vals:
            ax_feat.barh(range(len(vals)), vals, color=blue, alpha=0.86, height=0.62)
            ax_feat.set_yticks(range(len(labels)), labels, fontsize=6.8)
            ax_feat.set_xlabel(label_metric, fontsize=7.2)
            ax_feat.grid(True, axis="x")
            ax_feat.tick_params(axis="x", labelsize=6.8)
            ax_feat.margins(y=0.18)
        else:
            ax_feat.text(0.5, 0.5, "Feature contribution not available", ha="center", va="center", color=muted)
            ax_feat.set_xticks([]); ax_feat.set_yticks([])

        ax_diag = fig.add_subplot(gs[14:20, 15:24])
        ax_diag.set_axis_off(); add_round_panel(ax_diag, 0.025)
        ax_diag.text(0.040, 0.865, "AUTOMATED DIAGNOSTIC", transform=ax_diag.transAxes,
                     fontsize=8.1, color=muted, weight="bold", ha="left")
        ax_diag.text(0.040, 0.720, diagnosis.get("status", "-"), transform=ax_diag.transAxes,
                     fontsize=22, color=status_color, weight="bold", ha="left")
        ax_diag.text(0.040, 0.600, wrap_text(diagnosis.get("title", ""), 74), transform=ax_diag.transAxes,
                     fontsize=9.6, color=ink, weight="bold", ha="left", va="top")
        ax_diag.text(0.040, 0.440, wrap_text(diagnosis.get("interpretation", ""), 92), transform=ax_diag.transAxes,
                     fontsize=8.1, color="#334E68", ha="left", va="top", linespacing=1.22)
        ax_diag.text(0.040, 0.180, "Recommended action", transform=ax_diag.transAxes,
                     fontsize=8.0, color=muted, weight="bold", ha="left", va="top")
        ax_diag.text(0.040, 0.090, wrap_text(diagnosis.get("recommendation", ""), 94), transform=ax_diag.transAxes,
                     fontsize=7.8, color="#334E68", ha="left", va="top", linespacing=1.18)

        # separate projection snapshots, also mode-aware
        snap_path = os.path.join(out_folder, "projection_snapshots.png")
        try:
            n = len(future_years)
            cols = min(4, max(1, n))
            rows = int(math.ceil(n / cols))
            fig_s = plt.figure(figsize=(4.1 * cols, 4.4 * rows + 0.70), facecolor=bg)
            gs_s = fig_s.add_gridspec(rows + 1, cols, left=0.035, right=0.985, top=0.935, bottom=0.045,
                                      wspace=0.20, hspace=0.35, height_ratios=[0.22] + [1] * rows)
            ax_title = fig_s.add_subplot(gs_s[0, :]); ax_title.set_axis_off()
            snap_title = "SCAPE-CA Projected Land-Cover Snapshots" if use_multiclass_report else "SCAPE-CA Projected Built-up Snapshots"
            ax_title.text(0.0, 0.50, snap_title, transform=ax_title.transAxes,
                          fontsize=18, color=dark, weight="bold", ha="left", va="center")
            ax_title.text(1.0, 0.50, f"{first_future}-{last_future}", transform=ax_title.transAxes,
                          fontsize=10, color=muted, ha="right", va="center")
            for i, yr in enumerate(future_years):
                r = i // cols; c_col = i % cols
                ax = fig_s.add_subplot(gs_s[r + 1, c_col])
                if use_multiclass_report and int(yr) in future_class_maps:
                    arr = self._classmap_to_rgb(future_class_maps[int(yr)].astype(np.int32), mask, inhibit, palette)
                else:
                    arr = render_mode_map(median_pattern(yr), "binary")
                ax.imshow(arr, interpolation="nearest")
                ax.set_title(str(yr), fontsize=11, color=ink, weight="bold", pad=8)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_edgecolor(line); sp.set_linewidth(0.8)
            for j in range(len(future_years), rows * cols):
                ax = fig_s.add_subplot(gs_s[j // cols + 1, j % cols]); ax.set_axis_off()
            fig_s.savefig(snap_path, dpi=180, bbox_inches="tight", facecolor=bg)
            plt.close(fig_s)
        except Exception:
            pass

        png_path = os.path.join(out_folder, "report.png")
        fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor=bg)
        plt.close(fig)

        md_path = os.path.join(out_folder, "report.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown(metrics, validation, posterior, years, future_years,
                                            landscape_per_year, corr_data))


    def _read_pred_for_report(self, out_folder, year):
        path = os.path.join(out_folder, f"validation_pred_{year}.tif")
        with rasterio.open(path) as src:
            return src.read(1)

    # ---- diagnostic interpretation (verbatim from original) ----------

    def _interpret_validation_status(self, comp):
        fom = float(comp.get("FoM", 0.0))
        producer = float(comp.get("ProducerAcc", 0.0))
        user = float(comp.get("UserAcc", 0.0))
        false_alarms = int(comp.get("FalseAlarms", 0))
        misses = int(comp.get("Misses", 0))
        hits = int(comp.get("Hits", 0))

        if fom >= 0.30 and user >= 0.40:
            status = "GOOD"
            title = "Model cukup baik untuk simulasi skenario"
            interpretation = (
                "Model menunjukkan kemampuan alokasi perubahan yang relatif kuat. "
                "Nilai FoM dan User Accuracy menunjukkan bahwa prediksi perubahan cukup selektif "
                "dan dapat digunakan untuk eksplorasi skenario pertumbuhan."
            )
            recommendation = "Gunakan hasil sebagai baseline, lalu uji skenario demand dan kebijakan pola ruang."
        elif producer >= 0.80 and user < 0.10:
            status = "OVER_PREDICTIVE"
            title = "Model terlalu agresif atau over-predictive"
            interpretation = (
                "Model berhasil menangkap sebagian besar atau seluruh perubahan aktual, "
                "tetapi menghasilkan banyak false alarm. Ini berarti model sangat sensitif "
                "tetapi kurang selektif dalam menentukan lokasi perubahan."
            )
            recommendation = (
                "Naikkan SEED_THR, turunkan PATCH_MEAN, turunkan STEP_DEMAND, "
                "turunkan ITER_ADVANTAGE, turunkan TMC_SIGMA, dan gunakan training mode "
                "New built-up transition only."
            )
        elif producer < 0.40 and user >= 0.40:
            status = "UNDER_PREDICTIVE"
            title = "Model terlalu konservatif atau under-predictive"
            interpretation = (
                "Model relatif selektif, tetapi gagal menangkap banyak perubahan aktual. "
                "Hal ini mengindikasikan peluang transisi terlalu ketat atau driver belum cukup mewakili proses pertumbuhan."
            )
            recommendation = "Turunkan SEED_THR, naikkan CAND_MULT, tambah driver aksesibilitas, dan evaluasi ulang inhibiting factor."
        elif fom < 0.05:
            status = "WEAK_ALLOCATION"
            title = "Kemampuan alokasi spasial masih lemah"
            interpretation = (
                "Nilai FoM sangat rendah, sehingga lokasi perubahan yang diprediksi belum cocok dengan perubahan aktual. "
                "Kesalahan dapat berasal dari driver yang belum memadai, parameter alokasi yang terlalu longgar, "
                "atau domain validasi yang belum tepat."
            )
            recommendation = "Periksa mask validasi, candidate domain, driver transform, dan gunakan preset parameter konservatif."
        else:
            status = "MODERATE"
            title = "Model berada pada tingkat menengah"
            interpretation = (
                "Model memiliki sebagian kemampuan menangkap perubahan, tetapi masih membutuhkan kalibrasi untuk meningkatkan "
                "keseimbangan antara missed change dan false alarm."
            )
            recommendation = "Lakukan sensitivity analysis pada SEED_THR, PATCH_MEAN, STEP_DEMAND, dan demand mode."

        return {
            "status": status,
            "title": title,
            "interpretation": interpretation,
            "recommendation": recommendation,
            "summary": (
                f"Hits={hits:,}, Misses={misses:,}, False Alarms={false_alarms:,}, "
                f"FoM={fom:.3f}, Producer Accuracy={producer:.3f}, User Accuracy={user:.3f}."
            ),
        }

    # ---- PDF report (extended) ---------------------------------------


    def _build_pdf_report(self, out_folder, metrics, validation, years, future_years):
        """
        Professional PDF report builder for SCAPE-CA.

        The PDF is generated from the already-created report.md, enriched with
        metrics.json and visual outputs such as report.png and
        projection_snapshots.png when available. It uses only ReportLab so it is
        safe inside the QGIS Python environment and does not require Pandoc,
        wkhtmltopdf, or a browser engine.
        """
        if not _HAS_REPORTLAB:
            raise QgsProcessingException(
                "reportlab is not installed. Install it in QGIS Python, for example: "
                '"C:/Program Files/QGIS 3.40.13/apps/Python312/python.exe" -m pip install reportlab'
            )

        import html
        from reportlab.lib.pagesizes import landscape as _rl_landscape
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
        from reportlab.lib.utils import ImageReader

        pdf_path = os.path.join(out_folder, "report.pdf")
        md_path = os.path.join(out_folder, "report.md")
        metrics_path = os.path.join(out_folder, "metrics.json")

        # Prefer disk metrics when available. This allows rebuilding the PDF
        # from output artifacts even after the main modelling run.
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r", encoding="utf-8") as f:
                    disk_metrics = json.load(f)
                if isinstance(disk_metrics, dict):
                    merged = dict(metrics or {})
                    merged.update(disk_metrics)
                    metrics = merged
            except Exception:
                pass

        md_text = ""
        if os.path.exists(md_path):
            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    md_text = f.read()
            except Exception:
                md_text = ""

        page_size = _rl_landscape(A4)
        page_w, page_h = page_size
        margin_x = 1.05 * cm
        margin_top = 0.90 * cm
        margin_bottom = 0.78 * cm
        usable_w = page_w - 2 * margin_x
        usable_h = page_h - margin_top - margin_bottom

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=page_size,
            rightMargin=margin_x,
            leftMargin=margin_x,
            topMargin=margin_top,
            bottomMargin=margin_bottom,
            title="SCAPE-CA Diagnostic Report",
            author="Firman Afrianto, Maya Safira",
        )

        styles = getSampleStyleSheet()

        def add_style(name, parent, **kw):
            if name not in styles.byName:
                styles.add(ParagraphStyle(name=name, parent=styles[parent], **kw))
            return styles[name]

        S_TITLE = add_style("SCAPE_PDF_Title", "Title", fontName="Helvetica-Bold", fontSize=22, leading=25,
                            textColor=colors.HexColor("#082F49"), alignment=TA_LEFT, spaceAfter=5)
        S_SUB = add_style("SCAPE_PDF_Subtitle", "BodyText", fontSize=9.2, leading=11.5,
                          textColor=colors.HexColor("#486581"), alignment=TA_LEFT, spaceAfter=6)
        S_SECTION = add_style("SCAPE_PDF_Section", "Heading2", fontName="Helvetica-Bold", fontSize=12.4, leading=15,
                              textColor=colors.HexColor("#102A43"), spaceBefore=4, spaceAfter=5)
        S_BODY = add_style("SCAPE_PDF_Body", "BodyText", fontSize=7.8, leading=9.7,
                           textColor=colors.HexColor("#243B53"), alignment=TA_LEFT)
        S_SMALL = add_style("SCAPE_PDF_Small", "BodyText", fontSize=7.0, leading=8.6,
                            textColor=colors.HexColor("#334E68"), alignment=TA_LEFT)
        S_MUTED = add_style("SCAPE_PDF_Muted", "BodyText", fontSize=6.4, leading=7.7,
                            textColor=colors.HexColor("#6B7C93"), alignment=TA_CENTER)
        S_KPI_LABEL = add_style("SCAPE_PDF_KPILabel", "BodyText", fontName="Helvetica-Bold", fontSize=6.2, leading=7.4,
                                textColor=colors.HexColor("#627D98"), alignment=TA_CENTER)
        S_KPI_VALUE = add_style("SCAPE_PDF_KPIValue", "BodyText", fontName="Helvetica-Bold", fontSize=12.7, leading=14.3,
                                textColor=colors.HexColor("#102A43"), alignment=TA_CENTER)

        def clean(x):
            s = str(x if x is not None else "-")
            return (s.replace("—", "-").replace("–", "-").replace("→", "->")
                     .replace("±", "+/-").replace("≤", "<=").replace("≥", ">="))

        def esc(x):
            return html.escape(clean(x), quote=False)

        def para(x, style=None):
            return Paragraph(esc(x), style or S_SMALL)

        def fmt_int(x):
            try:
                return f"{int(round(float(x))):,}"
            except Exception:
                return "-"

        def fmt_float(x, nd=3):
            try:
                return f"{float(x):.{nd}f}"
            except Exception:
                return "-"

        def fmt_pct(x, nd=2):
            try:
                return f"{float(x) * 100.0:.{nd}f}%"
            except Exception:
                return "-"

        def parse_md_sections(text):
            sections = {}
            current = "Overview"
            buff = []
            for line in text.splitlines():
                if line.startswith("## "):
                    sections[current] = "\n".join(buff).strip()
                    current = line.replace("##", "", 1).strip()
                    buff = []
                elif not line.startswith("# "):
                    buff.append(line)
            sections[current] = "\n".join(buff).strip()
            return sections

        def md_table(section_text):
            lines = [ln.strip() for ln in section_text.splitlines() if ln.strip().startswith("|")]
            rows = []
            for ln in lines:
                cells = [c.strip().replace("**", "").replace("`", "") for c in ln.strip("|").split("|")]
                is_sep = all(set(c.replace(":", "").replace("-", "").strip()) == set() for c in cells)
                if not is_sep:
                    rows.append(cells)
            return rows

        md_sections = parse_md_sections(md_text)

        def table_style(header="#0B2A4A", zebra=True, n_rows=None):
            """Create a ReportLab TableStyle safely without referencing non-existent rows."""
            n_rows = int(n_rows or 0)
            cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.7),
                ("LEADING", (0, 0), (-1, -1), 8.1),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D9E2EC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
            if zebra and n_rows > 2:
                for i in range(1, n_rows, 2):
                    cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F7FAFC")))
            return TableStyle(cmds)

        def _as_list_row(row):
            """Return a safe list row for ReportLab tables."""
            if row is None:
                return ["-"]
            if isinstance(row, (list, tuple)):
                return list(row)
            return [row]

        def _normalise_rows_and_widths(rows, col_widths):
            """
            ReportLab can throw a non-informative 'list index out of range'
            when markdown-derived rows have uneven column counts. This makes
            every table rectangular before it is sent to ReportLab.
            """
            safe_rows = [_as_list_row(r) for r in (rows or [])]
            if not safe_rows:
                return [], []

            declared_n = len(col_widths or [])
            max_n = max(len(r) for r in safe_rows)
            ncols = declared_n if declared_n > 0 else max_n
            ncols = max(1, ncols)

            fixed = []
            for r in safe_rows:
                r = list(r)
                if len(r) > ncols:
                    r = r[:ncols - 1] + [" | ".join(clean(x) for x in r[ncols - 1:])]
                elif len(r) < ncols:
                    r = r + [""] * (ncols - len(r))
                fixed.append(r)

            widths = list(col_widths or [])
            if not widths:
                widths = [usable_w / ncols] * ncols
            elif len(widths) < ncols:
                remaining = max(usable_w - sum(widths), 1.0 * cm)
                widths = widths + [remaining / (ncols - len(widths))] * (ncols - len(widths))
            elif len(widths) > ncols:
                widths = widths[:ncols]
            return fixed, widths

        def add_table(story, rows, col_widths, header="#0B2A4A", max_rows=None):
            if not rows:
                return
            if max_rows is not None and len(rows) > max_rows + 1:
                rows = rows[:max_rows + 1]
            rows, col_widths = _normalise_rows_and_widths(rows, col_widths)
            if not rows:
                return
            rows = [[para(c) for c in r] for r in rows]
            t = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
            t.setStyle(table_style(header=header, n_rows=len(rows)))
            story.append(t)
            story.append(Spacer(1, 0.20 * cm))

        def kpi_card(label, value, note, color_hex):
            inner = Table([
                [Paragraph(esc(label).upper(), S_KPI_LABEL)],
                [Paragraph(esc(value), S_KPI_VALUE)],
                [Paragraph(esc(note), S_MUTED)],
            ], colWidths=[4.18 * cm], rowHeights=[0.34 * cm, 0.62 * cm, 0.43 * cm])
            inner.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("BOX", (0, 0), (-1, -1), 0.35, colors.HexColor("#D9E2EC")),
                ("LINEBEFORE", (0, 0), (0, -1), 3.4, colors.HexColor(color_hex)),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            return inner

        def add_scaled_image(story, image_path, title, caption=None):
            if not os.path.exists(image_path):
                return False
            story.append(PageBreak())
            story.append(Paragraph(esc(title), S_SECTION))
            if caption:
                story.append(Paragraph(esc(caption), S_SMALL))
                story.append(Spacer(1, 0.12 * cm))
            try:
                reader = ImageReader(image_path)
                iw, ih = reader.getSize()
                scale = min(usable_w / float(iw), (usable_h - 0.85 * cm) / float(ih))
                story.append(Image(image_path, width=iw * scale, height=ih * scale))
            except Exception:
                story.append(Image(image_path, width=usable_w, height=usable_h - 0.9 * cm))
            return True

        def page_decor(c, d):
            c.saveState()
            c.setStrokeColor(colors.HexColor("#D9E2EC"))
            c.setLineWidth(0.4)
            c.line(margin_x, page_h - 0.55 * cm, page_w - margin_x, page_h - 0.55 * cm)
            c.setFont("Helvetica", 6.4)
            c.setFillColor(colors.HexColor("#829AB1"))
            c.drawString(margin_x, 0.38 * cm, "SCAPE-CA Diagnostic Report")
            c.drawRightString(page_w - margin_x, 0.38 * cm, f"Page {d.page}")
            c.restoreState()

        comp = validation.get("components", {}) if validation is not None else {}
        if not comp and isinstance(metrics, dict):
            comp = metrics.get("components", {}) or metrics.get("validation", {}).get("components", {}) or {}
        diagnosis = self._interpret_validation_status(comp) if comp else {
            "status": "NO_VALIDATION",
            "title": "Validation was not executed",
            "interpretation": "Historical validation was not available in this run.",
            "recommendation": "Enable historical validation to obtain FoM, hit, miss, false alarm, and calibration diagnostics.",
            "summary": "-",
        }

        story = []

        title_box = Table([[
            Paragraph("SCAPE-CA v6 Diagnostic Run Report", S_TITLE),
            Paragraph("<b>Spatial Growth Intelligence</b><br/>validation, demand, uncertainty, drivers, and projected morphology", S_SUB),
        ]], colWidths=[14.5 * cm, usable_w - 14.5 * cm])
        title_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF4FF")),
            ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#B6D8FF")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(title_box)
        story.append(Spacer(1, 0.28 * cm))

        mode_label = "Multi-class" if metrics.get("multi_class") else "Built-up / binary"
        hist_label = f"{years[0]}-{years[-1]}" if years else "-"
        fut_label = f"{future_years[0]}-{future_years[-1]}" if future_years else "-"
        kpis = [[
            kpi_card("Historical Years", hist_label, "input time series", "#0B63CE"),
            kpi_card("Projection Years", fut_label, "simulation horizon", "#00A884"),
            kpi_card("FoM", fmt_float(comp.get("FoM", 0)), "allocation skill", "#2F9E44"),
            kpi_card("Producer Acc", fmt_float(comp.get("ProducerAcc", 0)), "observed change caught", "#7C3AED"),
            kpi_card("User Acc", fmt_float(comp.get("UserAcc", 0)), "prediction precision", "#F59E0B"),
            kpi_card("Mode", mode_label, "output representation", "#D64545"),
        ]]
        kpi_table = Table(kpis, colWidths=[4.22 * cm] * 6)
        kpi_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(kpi_table)
        story.append(Spacer(1, 0.25 * cm))

        status_bg = {
            "GOOD": "#E3FCEF", "MODERATE": "#FFF7E6", "OVER_PREDICTIVE": "#FFE3E3",
            "UNDER_PREDICTIVE": "#FFF7E6", "WEAK_ALLOCATION": "#FFE3E3", "NO_VALIDATION": "#F0F4F8",
        }.get(diagnosis.get("status"), "#F0F4F8")
        exec_text = (
            f"<b>Status:</b> {esc(diagnosis.get('status', '-'))}<br/>"
            f"<b>{esc(diagnosis.get('title', '-'))}</b><br/><br/>"
            f"{esc(diagnosis.get('interpretation', '-'))}<br/><br/>"
            f"<b>Recommended action:</b> {esc(diagnosis.get('recommendation', '-'))}<br/>"
            f"<b>Metric summary:</b> {esc(diagnosis.get('summary', '-'))}"
        )
        exec_box = Table([[Paragraph("Automated Diagnostic Interpretation", S_SECTION), Paragraph(exec_text, S_BODY)]],
                         colWidths=[6.2 * cm, usable_w - 6.2 * cm])
        exec_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor(status_bg)),
            ("BACKGROUND", (1, 0), (1, 0), colors.white),
            ("BOX", (0, 0), (-1, -1), 0.45, colors.HexColor("#D9E2EC")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        story.append(exec_box)
        story.append(Spacer(1, 0.25 * cm))

        story.append(Paragraph("Run Configuration", S_SECTION))
        driver_text = ", ".join([f"{d.get('name', '-')}/{d.get('transform', '-')}" for d in metrics.get("drivers", [])])
        run_rows = [
            ["Item", "Value", "Interpretation"],
            ["Target mode", metrics.get("target_mode", "unknown"), "Future built-cell demand estimation method."],
            ["Training target", metrics.get("training_target_mode_label", "unknown"), "Label strategy used for learning transition probability."],
            ["Neighborhood k", metrics.get("neighborhood_k", "-"), "Spatial contiguity pressure used by CA allocation."],
            ["Ensemble size", metrics.get("ensemble_n", "-"), "Number of stochastic simulation members."],
            ["Convertible classes", metrics.get("convertible_classes", []), "Source land-cover classes eligible for built-up conversion."],
            ["Drivers", driver_text or "-", "Driver rasters and applied transformations."],
            ["Runtime", f"{float(metrics.get('runtime_total_s', 0)):.1f} s", "Total processing duration."],
        ]
        if "cagr" in metrics:
            run_rows.insert(2, ["Auto CAGR", fmt_pct(metrics.get("cagr", 0)), "Historical growth rate estimated from built-up stack."])
        add_table(story, run_rows, [4.4 * cm, 8.1 * cm, usable_w - 12.5 * cm], header="#0B2A4A")

        if comp or "Validation" in md_sections:
            story.append(PageBreak())
            story.append(Paragraph("Validation Diagnostics", S_SECTION))
            rows = [["Metric", "Value", "Meaning"]]
            defs = [
                ("Hits", "Correctly predicted new built-up change"),
                ("Misses", "Observed change missed by the model"),
                ("FalseAlarms", "Predicted change that did not occur"),
                ("FoM", "Pontius Figure of Merit"),
                ("ProducerAcc", "Fraction of observed change captured"),
                ("UserAcc", "Fraction of predicted change that was correct"),
                ("ValidCells", "Total valid evaluation cells"),
                ("CandidateCells", "Non-built cells eligible for transition"),
                ("ObservedChangeCells", "Actual new built-up cells"),
                ("PredictedChangeCells", "Predicted new built-up cells"),
                ("FalseAlarmRatio", "Share of predicted changes that were false alarms"),
                ("OverpredictionFactor", "Predicted change divided by observed change"),
            ]
            for key, meaning in defs:
                if key in comp:
                    v = comp.get(key)
                    if isinstance(v, float):
                        v = fmt_float(v, 3)
                    elif isinstance(v, int):
                        v = fmt_int(v)
                    rows.append([key, v, meaning])
            if len(rows) == 1:
                rows = md_table(md_sections.get("Validation", "")) or rows
            add_table(story, rows, [5.0 * cm, 4.3 * cm, usable_w - 9.3 * cm], header="#0B63CE")

            if validation is not None and validation.get("multires_FoM"):
                story.append(Paragraph("Multi-resolution FoM", S_SECTION))
                rows = [["Window", "FoM", "Reading"]]
                for k, v in validation.get("multires_FoM", {}).items():
                    label = k.replace("FoM_w", "w=") + " cell"
                    rows.append([label, fmt_float(v, 3), "near-miss tolerant score" if label != "w=1 cell" else "cell-by-cell score"])
                add_table(story, rows, [5.0 * cm, 4.0 * cm, usable_w - 9.0 * cm], header="#006B54")

            if "calibration" in metrics or ("spatial_cv" in metrics and metrics.get("spatial_cv", {}).get("n_folds", 0) > 0):
                story.append(Paragraph("Calibration and Robustness", S_SECTION))
                rows = [["Diagnostic", "Value", "Interpretation"]]
                if "calibration" in metrics:
                    rows.append(["Brier score", fmt_float(metrics.get("calibration", {}).get("brier", 0), 4), "Lower value indicates better probability calibration."])
                if "spatial_cv" in metrics and metrics.get("spatial_cv", {}).get("n_folds", 0) > 0:
                    scv = metrics.get("spatial_cv", {})
                    rows.append(["Spatial block CV", f"{scv.get('n_folds', 0)} folds; FoM {fmt_float(scv.get('fom_mean', 0), 3)} +/- {fmt_float(scv.get('fom_std', 0), 3)}", "Spatially separated validation for more honest generalisation testing."])
                add_table(story, rows, [5.0 * cm, 7.0 * cm, usable_w - 12.0 * cm], header="#7C3AED")

        if "permutation_importance" in metrics or "shap" in metrics:
            story.append(PageBreak())
            story.append(Paragraph("Model Intelligence", S_SECTION))
            if "permutation_importance" in metrics:
                story.append(Paragraph("Permutation Driver Contribution", S_SECTION))
                rows = [["Rank", "Feature", "Delta FoM", "Std"]]
                for i, r in enumerate(metrics.get("permutation_importance", {}).get("ranking", [])[:15]):
                    rows.append([i + 1, r.get("feature", "-"), fmt_float(r.get("delta_fom_mean", 0), 4), fmt_float(r.get("delta_fom_std", 0), 4)])
                add_table(story, rows, [1.7 * cm, 13.0 * cm, 5.0 * cm, 4.8 * cm], header="#0B63CE")

            if "shap" in metrics:
                story.append(Paragraph("SHAP Attribution", S_SECTION))
                story.append(Paragraph(f"Top feature: <b>{esc(metrics.get('shap', {}).get('top_feature', '-'))}</b>. See shap_top_driver.tif for per-cell attribution when exported.", S_SMALL))
                story.append(Spacer(1, 0.10 * cm))
                rows = [["Rank", "Feature", "mean abs SHAP"]]
                for i, r in enumerate(metrics.get("shap", {}).get("ranking", [])[:15]):
                    rows.append([i + 1, r.get("feature", "-"), fmt_float(r.get("mean_abs_shap", 0), 4)])
                add_table(story, rows, [1.7 * cm, 16.0 * cm, 6.0 * cm], header="#263238")

        if metrics.get("demand_posterior") or metrics.get("landscape_per_year"):
            story.append(PageBreak())
            story.append(Paragraph("Demand and Projected Landscape Structure", S_SECTION))
            if metrics.get("demand_posterior"):
                story.append(Paragraph("Bayesian Demand Posterior", S_SECTION))
                rows = [["Year", "Median", "95% low", "95% high"]]
                for fy in sorted(metrics.get("demand_posterior", {}).keys(), key=lambda x: int(x)):
                    pf = metrics.get("demand_posterior", {}).get(fy, {})
                    rows.append([fy, fmt_int(pf.get("median", 0)), fmt_int(pf.get("lo95", 0)), fmt_int(pf.get("hi95", 0))])
                add_table(story, rows, [4.0 * cm, 6.0 * cm, 6.0 * cm, 6.0 * cm], header="#B45309")

            if metrics.get("landscape_per_year"):
                story.append(Paragraph("Landscape Metrics", S_SECTION))
                rows = [["Year", "NumPatches", "LPI (%)", "MPS (cells)", "EdgeDensity"]]
                for fy in sorted(metrics.get("landscape_per_year", {}).keys(), key=lambda x: int(x)):
                    lsm = metrics.get("landscape_per_year", {}).get(fy, {})
                    rows.append([fy, fmt_int(lsm.get("NumPatches", 0)), fmt_float(lsm.get("LPI_pct", 0), 2), fmt_float(lsm.get("MPS_cells", 0), 1), fmt_float(lsm.get("EdgeDensity", 0), 4)])
                add_table(story, rows, [3.0 * cm, 5.1 * cm, 5.0 * cm, 5.1 * cm, 5.1 * cm], header="#006B54")

        corr_warn = metrics.get("driver_correlation", {}).get("warnings", []) if isinstance(metrics.get("driver_correlation"), dict) else []
        if corr_warn:
            story.append(Paragraph("Driver Redundancy Warnings", S_SECTION))
            rows = [["Driver A", "Driver B", "r", "Note"]]
            for w in corr_warn[:20]:
                rows.append([w.get("a", "-"), w.get("b", "-"), fmt_float(w.get("r", 0), 3), "High absolute correlation can destabilise attribution."])
            add_table(story, rows, [6.0 * cm, 6.0 * cm, 3.0 * cm, usable_w - 15.0 * cm], header="#D64545")

        output_section = md_sections.get("Output files", "")
        if output_section:
            story.append(PageBreak())
            story.append(Paragraph("Output Inventory", S_SECTION))
            rows = [["Output", "Description"]]
            for ln in output_section.splitlines():
                ln = ln.strip()
                if not ln.startswith("- "):
                    continue
                ln = ln[2:].replace("`", "")
                if ":" in ln:
                    a, b = ln.split(":", 1)
                    rows.append([a.strip(), b.strip()])
                else:
                    rows.append([ln, "Generated output file"])
            add_table(story, rows, [8.5 * cm, usable_w - 8.5 * cm], header="#334E68", max_rows=20)

        add_scaled_image(story, os.path.join(out_folder, "report.png"), "Visual Dashboard", "Mode-aware dashboard exported by SCAPE-CA.")
        add_scaled_image(story, os.path.join(out_folder, "projection_snapshots.png"), "Projection Snapshots", "Projected built-up or multi-class land-cover snapshots for all future years.")

        if len(story) <= 4 and md_text:
            story.append(PageBreak())
            story.append(Paragraph("Markdown Narrative", S_SECTION))
            plain = " ".join([ln.replace("**", "").replace("`", "").strip() for ln in md_text.splitlines() if ln.strip() and not ln.strip().startswith("|")])
            story.append(Paragraph(esc(plain[:5000]), S_BODY))

        doc.build(story, onFirstPage=page_decor, onLaterPages=page_decor)
        return pdf_path

    # ---- narratives ---------------------------------------------------

    def _generate_narrative(self, metrics, validation, posterior, future_years):
        L = []
        L.append("INTERPRETATION")
        L.append("=" * 36)

        if validation is not None:
            comp = validation["components"]
            fom = comp["FoM"]
            L.append(f"Figure of Merit (FoM)  : {fom:.3f}")
            if fom >= 0.30:
                L.append("  -> Strong allocation skill.")
            elif fom >= 0.15:
                L.append("  -> Moderate; typical for real cities.")
            else:
                L.append("  -> Weak; consider adding drivers / longer history.")
            L.append(f"Producer / User acc    : {comp['ProducerAcc']:.3f} / {comp['UserAcc']:.3f}")
            mr = validation["multires_FoM"]
            L.append(f"FoM @1/3/5: {mr['FoM_w1']:.3f}/{mr['FoM_w3']:.3f}/{mr['FoM_w5']:.3f}")
            if (mr["FoM_w3"] - mr["FoM_w1"]) >= 0.05:
                L.append("  -> Errors mostly NEAR-MISSES (good)")
            else:
                L.append("  -> Errors not just near-misses")
            diagnosis = self._interpret_validation_status(comp)
            L.append(f"Diagnostic status      : {diagnosis['status']}")
            L.append(f"  -> {diagnosis['title']}")

        if "calibration" in metrics:
            bs = metrics["calibration"]["brier"]
            L.append(f"Brier score            : {bs:.4f}")
            if bs < 0.05:
                L.append("  -> Excellent calibration.")
            elif bs < 0.10:
                L.append("  -> Good calibration.")
            else:
                L.append("  -> Loose calibration.")

        if "spatial_cv" in metrics and metrics["spatial_cv"].get("n_folds", 0) > 0:
            scv = metrics["spatial_cv"]
            L.append(f"Spatial CV FoM         : {scv['fom_mean']:.3f} +/- {scv['fom_std']:.3f}")

        if "permutation_importance" in metrics:
            top = metrics["permutation_importance"]["ranking"][:3]
            L.append("Top drivers (permutation):")
            for i, r in enumerate(top):
                L.append(f"  {i+1}. {r['feature'][:22]} d={r['delta_fom_mean']:.4f}")

        if "shap" in metrics:
            top_shap = metrics["shap"]["ranking"][:3]
            L.append("Top drivers (SHAP):")
            for i, r in enumerate(top_shap):
                L.append(f"  {i+1}. {r['feature'][:22]} |s|={r['mean_abs_shap']:.4f}")

        if posterior is not None:
            L.append("Demand posterior (95% CrI):")
            for fy in future_years:
                pf = posterior[fy]
                L.append(f"  {fy}: {pf['median']:>7,} [{pf['lo95']:>7,}..{pf['hi95']:>7,}]")

        L.append("")
        L.append(f"Target mode            : {metrics.get('target_mode', 'unknown')}")
        L.append(f"Ensemble N             : {metrics.get('ensemble_n', 1)}")
        L.append(f"Neighborhood k         : {metrics.get('neighborhood_k', 3)}")
        if metrics.get("multi_class"):
            L.append(f"Multi-class mode       : ON {metrics.get('convertible_classes', [])}")
        if "driver_correlation" in metrics:
            n_w = len(metrics["driver_correlation"]["warnings"])
            L.append(f"Redundancy warnings    : {n_w}")
        L.append("Use agreement_<year>.tif and uncertainty_<year>.tif")
        L.append("for ensemble consensus and disagreement.")

        if "cagr" in metrics:
            L.append(f"Historical CAGR        : {metrics['cagr'] * 100:.2f}%/year")
        return "\n".join(L)

    def _generate_markdown(self, metrics, validation, posterior,
                            years, future_years, landscape_per_year, corr_data):
        L = []
        L.append("# SCAPE-CA v6 — Diagnostic Run Report\n")
        L.append(f"- Historical years: {years}")
        L.append(f"- Future years: {future_years}")
        L.append(f"- Target mode: **{metrics.get('target_mode', 'unknown')}**")
        L.append(f"- Training target mode: **{metrics.get('training_target_mode_label', 'unknown')}**")
        L.append(f"- Multi-class: **{'ON' if metrics.get('multi_class') else 'OFF'}**")
        if metrics.get("multi_class"):
            L.append(f"  - Convertible classes: {metrics.get('convertible_classes')}")
        L.append(f"- Neighborhood k: **{metrics.get('neighborhood_k', 3)}**")
        L.append(f"- Ensemble size: **{metrics.get('ensemble_n', 1)}**")
        L.append("- Drivers used:")
        for d in metrics["drivers"]:
            L.append(f"  - **{d['name']}** with transform `{d['transform']}`")
        if "cagr" in metrics:
            L.append(f"- Auto CAGR: **{metrics['cagr'] * 100:.2f}%/year**")
        L.append(f"- Total runtime: **{metrics.get('runtime_total_s', 0):.1f} s**\n")

        if validation is not None:
            comp = validation["components"]
            mr = validation["multires_FoM"]
            L.append("## Validation\n")
            L.append("| Metric | Value | Meaning |")
            L.append("|---|---:|---|")
            L.append(f"| Hits | {comp['Hits']:,} | Correctly predicted change |")
            L.append(f"| Misses | {comp['Misses']:,} | Real change the model missed |")
            L.append(f"| False Alarms | {comp['FalseAlarms']:,} | Predicted change that did not happen |")
            L.append(f"| **FoM** | **{comp['FoM']:.3f}** | Pontius Figure of Merit |")
            L.append(f"| Producer Acc | {comp['ProducerAcc']:.3f} | Fraction of real change caught |")
            L.append(f"| User Acc | {comp['UserAcc']:.3f} | Fraction of predicted change correct |")
            if "ValidCells" in comp:
                L.append(f"| Valid Cells | {comp['ValidCells']:,} | Total valid evaluation cells |")
                L.append(f"| Candidate Cells | {comp['CandidateCells']:,} | Non-built cells eligible for transition |")
                L.append(f"| Observed Change Cells | {comp['ObservedChangeCells']:,} | Actual new built-up cells |")
                L.append(f"| Predicted Change Cells | {comp['PredictedChangeCells']:,} | Predicted new built-up cells |")
                L.append(f"| False Alarm Ratio | {comp['FalseAlarmRatio']:.3f} | Share of predicted changes that were false alarms |")
                L.append(f"| Overprediction Factor | {comp['OverpredictionFactor']:.3f} | Predicted / observed change |")
            L.append("")

            diagnosis = self._interpret_validation_status(comp)
            L.append("## Automated Diagnostic Interpretation\n")
            L.append(f"**Diagnostic status:** `{diagnosis['status']}`\n")
            L.append(f"**Interpretation:** {diagnosis['title']}\n")
            L.append(diagnosis["interpretation"] + "\n")
            L.append("**Recommended action:**\n")
            L.append(diagnosis["recommendation"] + "\n")
            L.append("**Metric summary:**\n")
            L.append(diagnosis["summary"] + "\n")

            L.append("## Multi-resolution FoM\n")
            L.append("| Window | FoM |")
            L.append("|---|---:|")
            for k, v in mr.items():
                L.append(f"| {k.replace('FoM_w', 'w=')} cell | {v:.3f} |")
            L.append("")

        if "calibration" in metrics:
            bs = metrics["calibration"]["brier"]
            L.append("## Probability calibration\n")
            L.append(f"- Brier score: **{bs:.4f}**.\n")

        if "spatial_cv" in metrics and metrics["spatial_cv"].get("n_folds", 0) > 0:
            scv = metrics["spatial_cv"]
            L.append("## Spatial block CV\n")
            L.append(f"- Folds: **{scv['n_folds']}**, FoM = **{scv['fom_mean']:.3f} +/- {scv['fom_std']:.3f}**\n")

        if "permutation_importance" in metrics:
            L.append("## Permutation driver contribution\n")
            L.append("| Rank | Feature | DeltaFoM | std |")
            L.append("|---:|---|---:|---:|")
            for i, r in enumerate(metrics["permutation_importance"]["ranking"][:15]):
                L.append(f"| {i+1} | `{r['feature']}` | {r['delta_fom_mean']:.4f} | {r['delta_fom_std']:.4f} |")
            L.append("")

        if "shap" in metrics:
            L.append("## SHAP attribution (mean |SHAP|)\n")
            L.append("| Rank | Feature | mean |SHAP| |")
            L.append("|---:|---|---:|")
            for i, r in enumerate(metrics["shap"]["ranking"][:15]):
                L.append(f"| {i+1} | `{r['feature']}` | {r['mean_abs_shap']:.4f} |")
            L.append(f"\nTop feature: **{metrics['shap']['top_feature']}** (see `shap_top_driver.tif`).\n")

        if posterior is not None:
            L.append("## Bayesian demand posterior\n")
            L.append("| Year | Median | 95% lo | 95% hi |")
            L.append("|---|---:|---:|---:|")
            for fy in future_years:
                pf = posterior[fy]
                L.append(f"| {fy} | {pf['median']:,} | {pf['lo95']:,} | {pf['hi95']:,} |")
            L.append("")

        if landscape_per_year:
            L.append("## Landscape metrics (projected)\n")
            L.append("| Year | NumPatches | LPI (%) | MPS (cells) | EdgeDensity |")
            L.append("|---|---:|---:|---:|---:|")
            for fy in sorted(landscape_per_year.keys()):
                lsm = landscape_per_year[fy]
                L.append(f"| {fy} | {lsm['NumPatches']} | {lsm['LPI_pct']:.2f} | "
                            f"{lsm['MPS_cells']:.1f} | {lsm['EdgeDensity']:.4f} |")
            L.append("")

        if corr_data is not None and corr_data["warnings"]:
            L.append("## Driver redundancy warnings\n")
            for w in corr_data["warnings"]:
                L.append(f"- `{w['a']}` ~ `{w['b']}`: r = {w['r']:+.3f}")
            L.append("\n> Highly correlated drivers (|r| > 0.85) can destabilise permutation/SHAP attributions.\n")

        L.append("## Output files\n")
        L.append("- `landcover_<year>.tif`: simulated land-cover map per future year; in multi-class mode, non-converted classes are preserved")
        L.append("- `agreement_<year>.tif`: ensemble agreement, 0..1")
        L.append("- `uncertainty_<year>.tif`: ensemble uncertainty, 4*p*(1-p)")
        L.append("- `prob_<year>.tif`: model probability surface")
        L.append("- `validation_pred_<lastyear>.tif`: hindcast map")
        L.append("- `shap_top_driver.tif`: per-cell SHAP attribution (top driver)")
        L.append("- `report.png`, `report.md`, `metrics.json`, `report.pdf`")
        L.append("- `plan_allowed_mask.tif`: only when PLAN_RASTER is supplied")
        L.append("- `inhibit_combined_mask.tif`: union of all inhibit layers, only when at least one INHIBIT raster is supplied")
        return "\n".join(L) + "\n"