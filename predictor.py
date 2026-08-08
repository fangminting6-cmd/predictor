import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrow
from sklearn.pipeline import Pipeline

# ============================================================
# 0. Page configuration
# ============================================================
st.set_page_config(
    page_title="Individual ACL Loading Prediction",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

POS_COLOR = "#9C1A1C"   # Positive SHAP contribution: increases ACL force
NEG_COLOR = "#48A597"   # Negative SHAP contribution: decreases ACL force
TITLE_COLOR = "#1A5276"
BUNDLE_FILE = "acl_xgboost_web_bundle.pkl"

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.grid": False,
    "axes.unicode_minus": False
})

FEATURE_LABELS = {
    "HFA": "Hip Flexion Angle (HFA)",
    "HRA": "Hip Rotation Angle (HRA)",
    "HAA": "Hip Adduction Angle (HAA)",
    "KFA": "Knee Flexion Angle (KFA)",
    "ITR": "Internal Tibial Rotation (ITR)",
    "KVA": "Knee Varus/Valgus Angle (KVA)",
    "ADF": "Ankle Dorsiflexion Angle (ADF)",
    "FPA": "Foot Progression Angle (FPA)",
    "TFA": "Trunk Flexion Angle (TFA)",
    "H/Q": "Hamstring/Quadriceps Ratio (H/Q)"
}

FEATURE_UNITS = {
    "HFA": "°", "HRA": "°", "HAA": "°", "KFA": "°", "ITR": "°",
    "KVA": "°", "ADF": "°", "FPA": "°", "TFA": "°", "H/Q": "ratio"
}

@st.cache_resource
def load_assets():
    bundle = joblib.load(BUNDLE_FILE)
    required = ["pipeline", "model", "feature_names"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise KeyError(f"Missing fields in model bundle: {missing}")

    pipeline = bundle["pipeline"]
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    processed_feature_names = list(bundle.get("processed_feature_names", feature_names))
    explainer = shap.TreeExplainer(model)
    return bundle, pipeline, model, explainer, feature_names, processed_feature_names


def scalar_base_value(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def fmt_feature_value(name, value):
    return f"{value:.2f}"


def make_waterfall_figure(shap_values, feature_values, feature_names, base_value):
    """Custom SHAP waterfall so colors always match the Force Plot."""
    shap_values = np.asarray(shap_values, dtype=float).reshape(-1)
    feature_values = np.asarray(feature_values, dtype=float).reshape(-1)

    order = np.argsort(np.abs(shap_values))[::-1]
    vals = shap_values[order]
    names = [feature_names[j] for j in order]
    data = feature_values[order]
    prediction = float(base_value + shap_values.sum())

    current = prediction
    starts, ends = [], []
    for phi in vals:
        nxt = current - phi
        starts.append(nxt)
        ends.append(current)
        current = nxt

    all_x = [base_value, prediction] + starts + ends
    x_min, x_max = min(all_x), max(all_x)
    raw_span = max(x_max - x_min, 1e-6)
    margin = raw_span * 0.13

    fig, ax = plt.subplots(figsize=(8, 5.5))
    y = np.arange(len(vals))

    for row, (start, end, phi) in enumerate(zip(starts, ends, vals)):
        color = POS_COLOR if phi > 0 else NEG_COLOR
        dx = end - start
        head_length = min(abs(dx) * 0.35, raw_span * 0.018)
        head_length = max(head_length, raw_span * 0.0025)

        arrow = FancyArrow(
            start, row, dx, 0,
            width=0.52,
            head_width=0.70,
            head_length=head_length,
            length_includes_head=True,
            linewidth=0,
            facecolor=color,
            edgecolor=color,
            zorder=3
        )
        ax.add_patch(arrow)

        if row < len(vals) - 1:
            ax.plot(
                [start, start], [row + 0.35, row + 0.65],
                color="#BFC5C9", linestyle="--", linewidth=0.8, zorder=2
            )

        label = f"{phi:+.2f}"
        if abs(dx) >= raw_span * 0.075:
            ax.text((start + end) / 2, row, label, ha="center", va="center",
                    fontsize=11, color="white", zorder=4)
        else:
            text_x = end + (raw_span * 0.012 if phi > 0 else -raw_span * 0.012)
            ax.text(text_x, row, label, ha="left" if phi > 0 else "right",
                    va="center", fontsize=10.5, color=color, zorder=4)

    ylabels = [
        rf"$\mathit{{{fmt_feature_value(name, value)}}}$ = {name}"
        for name, value in zip(names, data)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=12)
    ax.invert_yaxis()

    ax.axvline(base_value, color="#B9BFC3", linestyle="--", linewidth=0.9, zorder=1)
    ax.axvline(prediction, color="#222222", linestyle="--", linewidth=1.0, zorder=1)
    ax.text(prediction, -0.83, rf"$f(x)$ = {prediction:.3f}", ha="center", va="bottom",
            fontsize=12.5, fontweight="bold")
    ax.text(base_value, len(vals) - 0.05, rf"$E[f(X)]$ = {base_value:.3f}", ha="center",
            va="top", fontsize=11.5, color="#8C8C8C")

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(len(vals) - 0.15, -1.05)
    ax.grid(axis="y", linestyle=(0, (1, 4)), linewidth=0.7, alpha=0.35)
    ax.tick_params(axis="x", labelsize=10.5, length=4, width=1)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.0)
    fig.tight_layout()
    return fig


def recolor_force_plot(fig):
    ax = fig.gca()
    target_pos = mcolors.to_rgb("#FF0051")
    target_neg = mcolors.to_rgb("#008BFB")

    def match_color(c):
        if c is None:
            return None
        try:
            rgb = mcolors.to_rgb(c)
            if sum((a - b) ** 2 for a, b in zip(rgb, target_pos)) < 0.05:
                return POS_COLOR
            if sum((a - b) ** 2 for a, b in zip(rgb, target_neg)) < 0.05:
                return NEG_COLOR
        except Exception:
            pass
        return None

    for obj in ax.findobj():
        if hasattr(obj, "get_color") and hasattr(obj, "set_color"):
            try:
                new_c = match_color(obj.get_color())
                if new_c:
                    obj.set_color(new_c)
            except Exception:
                pass
        if hasattr(obj, "get_facecolor") and hasattr(obj, "set_facecolor"):
            try:
                fc = obj.get_facecolor()
                if isinstance(fc, np.ndarray) and fc.size >= 3:
                    fc = fc[0] if fc.ndim == 2 else fc
                new_c = match_color(fc)
                if new_c:
                    obj.set_facecolor(new_c)
            except Exception:
                pass
        if hasattr(obj, "get_edgecolor") and hasattr(obj, "set_edgecolor"):
            try:
                ec = obj.get_edgecolor()
                if isinstance(ec, np.ndarray) and ec.size >= 3:
                    ec = ec[0] if ec.ndim == 2 else ec
                new_c = match_color(ec)
                if new_c:
                    obj.set_edgecolor(new_c)
            except Exception:
                pass
    return fig


st.markdown("""
<style>
    .stApp { background-color: #FFFFFF; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
    .sci-title { color:#1A5276; font-family:'Times New Roman',serif; font-size:2.55rem;
                 font-weight:800; text-align:center; margin-bottom:0.15rem; }
    .sci-subtitle { color:#7F8C8D; font-family:Arial,sans-serif; text-align:center;
                    font-size:1rem; margin-bottom:1.8rem; }
    .result-card { background-color:#F8F9FA; border:1px solid #EAECEE; border-radius:10px;
                   padding:22px 24px; border-left:6px solid #1A5276; margin-top:10px; margin-bottom:12px; }
    .label-text { color:#2E4053; font-size:0.84rem; text-transform:uppercase; font-weight:bold;
                  letter-spacing:0.08rem; }
    .result-row { display:flex; align-items:baseline; flex-wrap:wrap; gap:24px; margin-top:5px; }
    .value-text { color:#111111; font-family:'Times New Roman',serif; font-size:3.15rem;
                  font-weight:800; line-height:1.05; }
    .unit-text { color:#666666; font-size:1.35rem; font-weight:600; margin-left:5px; }
    .status-text { font-family:Arial,sans-serif; font-size:1.45rem; font-weight:800; letter-spacing:0.02rem; }
    .small-note { color:#7F8C8D; font-size:0.83rem; margin-top:9px; line-height:1.45; }
    div[data-testid="stMetric"] { background:#FAFAFA; border:1px solid #ECECEC; padding:12px; border-radius:8px; }
    .stNumberInput { margin-bottom:-0.15rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='sci-title'>Individual ACL Loading Prediction During Badminton Wide Lunge</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='sci-subtitle'>XGBoost-based biomechanical prediction with individual SHAP interpretation</p>",
            unsafe_allow_html=True)

try:
    bundle, pipeline, model, explainer, feature_names, processed_feature_names = load_assets()
except Exception as exc:
    st.error(
        f"Model bundle could not be loaded: {exc}\n\n"
        f"Place `{BUNDLE_FILE}` in the same folder as this Streamlit script."
    )
    st.stop()

defaults = bundle.get("feature_defaults", {})
ranges = bundle.get("feature_ranges", {})
metrics = bundle.get("metrics", {})
baseline = scalar_base_value(bundle.get("base_value", explainer.expected_value))
threshold = float(bundle.get("high_load_threshold", np.nan))
threshold_percentile = bundle.get("threshold_percentile", 75)

col_left, col_right = st.columns([0.95, 1.20], gap="large")

with col_left:
    st.markdown("### 📋 Individual biomechanical inputs")

    with st.form("acl_prediction_form"):
        input_values = {}
        c1, c2 = st.columns(2)

        for idx, name in enumerate(feature_names):
            target_col = c1 if idx % 2 == 0 else c2
            default_val = float(defaults.get(name, 0.0))
            r = ranges.get(name, {})
            rmin, rmax = r.get("min"), r.get("max")
            help_text = None
            if rmin is not None and rmax is not None:
                help_text = (
                    f"Training-data range: {float(rmin):.2f} to {float(rmax):.2f} "
                    f"{FEATURE_UNITS.get(name, '')}"
                )

            with target_col:
                input_values[name] = st.number_input(
                    FEATURE_LABELS.get(name, name), value=default_val, step=0.01,
                    format="%.2f", help=help_text, key=f"input_{name}"
                )

        submitted = st.form_submit_button("Predict ACL Force", type="primary", use_container_width=True)

    if not submitted:
        st.info("Enter the individual biomechanical variables and click **Predict ACL Force**.")
        st.stop()

    input_data = pd.DataFrame(
        [[input_values[name] for name in feature_names]], columns=feature_names
    )

    out_of_range = []
    for name in feature_names:
        if name in ranges:
            v = float(input_values[name])
            low = float(ranges[name].get("min", -np.inf))
            high = float(ranges[name].get("max", np.inf))
            if v < low or v > high:
                out_of_range.append(f"{name}={v:.2f} (training range {low:.2f}–{high:.2f})")

    if out_of_range:
        st.warning(
            "The following values are outside the training-data range. "
            "The prediction is therefore an extrapolation:\n\n- " + "\n- ".join(out_of_range)
        )

    prediction = float(pipeline.predict(input_data)[0])

    if np.isfinite(threshold):
        is_high = prediction >= threshold
        status_label = "HIGH RELATIVE LOAD" if is_high else "WITHIN REFERENCE RANGE"
        status_color = POS_COLOR if is_high else NEG_COLOR
        threshold_note = (
            f"Reference threshold: {threshold:.3f} BW "
            f"({threshold_percentile}th percentile of the training ACL-force distribution)"
        )
    else:
        status_label = "PREDICTION COMPLETE"
        status_color = TITLE_COLOR
        threshold_note = "No high-load reference threshold was stored in the model bundle."

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label-text">Predicted peak ACL force</div>
            <div class="result-row">
                <span class="value-text">{prediction:.3f}<span class="unit-text"> BW</span></span>
                <span class="status-text" style="color:{status_color};">{status_label}</span>
            </div>
            <div class="small-note">
                {threshold_note}<br>
                SHAP baseline E[f(X)]: {baseline:.3f} BW
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Test R²", f"{metrics.get('r2', np.nan):.3f}" if "r2" in metrics else "—")
    m2.metric("Test RMSE", f"{metrics.get('rmse', np.nan):.3f} BW" if "rmse" in metrics else "—")
    m3.metric("Test MAE", f"{metrics.get('mae', np.nan):.3f} BW" if "mae" in metrics else "—")

    with st.expander("ℹ️ Model and interpretation notes"):
        st.markdown(
            """
            - **Algorithm:** XGBoost regression.
            - **Output:** predicted peak ACL force normalized to body weight (BW).
            - **Positive SHAP contribution:** increases the predicted ACL force.
            - **Negative SHAP contribution:** decreases the predicted ACL force.
            - The relative-load label is a **research reference classification**, not a clinical diagnosis.
            """
        )

    report = input_data.copy()
    report["Predicted_ACL_Force_BW"] = prediction
    report["Reference_Status"] = status_label
    if np.isfinite(threshold):
        report["Reference_Threshold_BW"] = threshold

    st.download_button(
        "📥 Export individual prediction (CSV)",
        data=report.to_csv(index=False).encode("utf-8-sig"),
        file_name="individual_acl_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )

with col_right:
    st.markdown("### 🔍 Individual model interpretation")

    preprocessor = Pipeline(pipeline.steps[:-1])
    transformed = preprocessor.transform(input_data)
    processed_input = pd.DataFrame(transformed, columns=processed_feature_names)

    try:
        shap_row = explainer.shap_values(processed_input, check_additivity=False)
        shap_row = np.asarray(shap_row)[0]
    except TypeError:
        shap_row = np.asarray(explainer.shap_values(processed_input))[0]

    current_base = scalar_base_value(explainer.expected_value)
    tab1, tab2 = st.tabs(["Waterfall plot", "Force plot"])

    with tab1:
        wf_fig = make_waterfall_figure(
            shap_values=shap_row,
            feature_values=processed_input.iloc[0].values,
            feature_names=processed_feature_names,
            base_value=current_base
        )
        st.pyplot(wf_fig, clear_figure=True, use_container_width=True)
        st.caption(
            "Waterfall plot: deep red indicates contributions that increase predicted ACL force; "
            "teal indicates contributions that decrease predicted ACL force."
        )

    with tab2:
        force_features = processed_input.iloc[0].round(3)
        shap_fig = shap.force_plot(
            current_base, shap_row, force_features,
            matplotlib=True, show=False
        )
        force_fig = shap_fig if shap_fig is not None else plt.gcf()
        force_fig.set_size_inches(13, 3.2)
        force_fig = recolor_force_plot(force_fig)
        st.pyplot(force_fig, clear_figure=True, use_container_width=True)
        st.caption(
            "Force plot: the prediction is decomposed from the SHAP baseline into positive "
            "and negative feature contributions."
        )

st.markdown(
    """
    <br><hr>
    <div style="color:#95A5A6; font-size:0.78rem; font-family:'Times New Roman', serif;">
        Research-use prototype for biomechanical ACL-loading prediction. 
        Predictions outside the training-data range should be interpreted cautiously.
    </div>
    """,
    unsafe_allow_html=True
)
