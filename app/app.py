import csv
import streamlit as st
import streamlit.components.v1 as st_components
import pandas as pd
import numpy as np
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import Descriptors
from pathlib import Path
from bokeh.embed import components as bokeh_components
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import CDN

from descriptors import smiles_to_ecfp4, compute_tsne, get_mol_b64_image, compute_similarity_matrix
from splitters import (
    ActivityCliffSplitter, 
    ScaffoldSplitter, 
    RandomSplitter,
    KNNFailureSplitter,
    SubstructureDistanceSplitter,
    ProxySortedSplitter
)
from ml_models import train_and_eval_regressor, run_bias_sweep

st.set_page_config(page_title="Biased Split", layout="wide")

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "standardized"
    / "target_CHEMBL203-1.IC50.csv"
)

def detect_separator(source) -> str:
    if isinstance(source, (str, Path)):
        sample = Path(source).read_text(encoding="utf-8", errors="replace")[:4096]
    else:
        position = source.tell()
        sample = source.read(4096)
        source.seek(position)
        if isinstance(sample, bytes):
            sample = sample.decode("utf-8", errors="replace")

    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def read_dataset(source, separator=None):
    return pd.read_csv(source, sep=separator or detect_separator(source))


@st.cache_data(show_spinner=False)
def load_sample_dataset():
    np.random.seed(42)
    df = read_dataset(DATA_PATH)
    sample_smiles = df["standardized_smiles"]
    pchembl = df["pchembl_value"]
    return pd.DataFrame({"canonical_smiles": sample_smiles, "pchembl_value": pchembl})

@st.cache_data(show_spinner=False)
def prepare_molecular_features(smiles_list, image_limit=500):
    fps = np.array([smiles_to_ecfp4(s) for s in smiles_list])
    if len(fps) < 3:
        tsne_coords = np.zeros((len(fps), 2))
        if len(fps) == 2:
            tsne_coords[:, 0] = [-1.0, 1.0]
    else:
        tsne_coords = compute_tsne(fps, perplexity=min(30, max(2, (len(fps) - 1) // 3)))
    b64_imgs = [
        get_mol_b64_image(s) if index < image_limit else ""
        for index, s in enumerate(smiles_list)
    ]
    return fps, tsne_coords, b64_imgs

st.sidebar.title("Configuration & Parameters")

uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
separator_input = st.sidebar.text_input(
    "CSV separator (optional)",
    value="",
    max_chars=1,
    help="Leave blank to detect automatically. Examples: ; or tab",
)
if separator_input and separator_input in {"\n", "\r"}:
    st.sidebar.error("Enter a single-character separator, such as ; or |.")
    st.stop()

if uploaded_file:
    df = read_dataset(uploaded_file, separator=separator_input or None)
else:
    st.sidebar.info("Using built-in sample dataset.")
    df = load_sample_dataset()

smi_col = st.sidebar.selectbox("SMILES Column", df.columns, index=0)
act_col = st.sidebar.selectbox(
    "Activity Column (pIC50 / pChEMBL)",
    df.columns,
    index=1 if len(df.columns) > 1 else 0,
)

df = df.dropna(subset=[smi_col, act_col]).copy()
df[act_col] = pd.to_numeric(df[act_col], errors="coerce")
df = df.dropna(subset=[act_col]).reset_index(drop=True)
if df.empty:
    st.error("The selected SMILES and activity columns contain no usable rows.")
    st.stop()

# Added the new split types to the dropdown
split_type = st.sidebar.selectbox(
    "Splitting Method",
    [
        "Activity Cliff Split", 
        "kNN Failure Split",
        "Substructure Distance Split",
        "Proxy Sorted Split (cLogP)",
        "Murcko Scaffold Split", 
        "Random Split"
    ],
)
test_frac = st.sidebar.slider("Test Set Fraction", 0.1, 0.4, 0.2, 0.05)

# --- Dynamic UI Parameters based on Split Type ---
intended_bias, sim_thresh, act_thresh, n_neighbors = 0.0, 0.7, 1.0, 3
ideal_min, ideal_max = 2.0, 3.0

if split_type in ["Activity Cliff Split", "kNN Failure Split", "Substructure Distance Split", "Proxy Sorted Split (cLogP)"]:
    intended_bias = st.sidebar.slider("Intended Bias", 0.0, 1.0, 0.5, 0.05)

if split_type in ["Activity Cliff Split", "kNN Failure Split", "Substructure Distance Split"]:
    sim_thresh = st.sidebar.slider("Similarity Threshold", 0.5, 0.9, 0.7, 0.05)

if split_type in ["Activity Cliff Split", "kNN Failure Split"]:
    act_thresh = st.sidebar.slider("Activity Cliff Gap (|Δy|)", 0.5, 2.5, 1.0, 0.25)

if split_type == "kNN Failure Split":
    n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 10, 3, 1)

if split_type == "Proxy Sorted Split (cLogP)":
    ideal_min = st.sidebar.number_input("Ideal cLogP Min", value=2.0)
    ideal_max = st.sidebar.number_input("Ideal cLogP Max", value=3.0)

ml_algo = st.sidebar.selectbox("Regression Model", ["Random Forest", "Ridge Regression"])

# --- Data Prep ---
smiles_data = df[smi_col].astype(str).tolist()
activity_data = df[act_col].astype(float).values
progress = st.progress(0, text="Preparing dataset...")
progress.progress(15, text=f"Preparing {len(smiles_data):,} molecular structures...")
fps, tsne_coords, b64_images = prepare_molecular_features(smiles_data)
progress.progress(65, text="Molecular features ready. Calculating split...")

# --- Splitting Logic Execution ---
if split_type == "Activity Cliff Split":
    splitter = ActivityCliffSplitter(
        similarity_threshold=sim_thresh, activity_threshold=act_thresh, test_fraction=test_frac
    )
    train_idx, test_idx, effective_bias = splitter.split(smiles_data, activity_data, intended_bias=intended_bias)

elif split_type == "kNN Failure Split":
    splitter = KNNFailureSplitter(
        similarity_threshold=sim_thresh, activity_threshold=act_thresh, n_neighbors=n_neighbors, test_fraction=test_frac
    )
    train_idx, test_idx, effective_bias = splitter.split_for_intended_bias(smiles_data, activity_data, intended_bias, random_seed=42)

elif split_type == "Substructure Distance Split":
    tversky_matrix = compute_similarity_matrix(fps, method="tversky")
    splitter = SubstructureDistanceSplitter(similarity_threshold=sim_thresh, test_fraction=test_frac)
    train_idx, test_idx, effective_bias = splitter.split_for_intended_bias(smiles_data, tversky_matrix, activity_data, intended_bias, random_seed=42)

elif split_type == "Proxy Sorted Split (cLogP)":
    def compute_logp(smi):
        mol = Chem.MolFromSmiles(smi)
        return Descriptors.MolLogP(mol) if mol else 0.0
    proxy_values = np.array([compute_logp(s) for s in smiles_data])
    splitter = ProxySortedSplitter(proxy_function=compute_logp, ideal_range_min=ideal_min, ideal_range_max=ideal_max, test_fraction=test_frac)
    train_idx, test_idx, effective_bias = splitter.split_for_intended_bias(smiles_data, proxy_values, activity_data, intended_bias, random_seed=42)

elif split_type == "Murcko Scaffold Split":
    splitter = ScaffoldSplitter(test_fraction=test_frac)
    train_idx, test_idx, effective_bias = splitter.split(smiles_data)

else:
    splitter = RandomSplitter(test_fraction=test_frac)
    train_idx, test_idx, effective_bias = splitter.split(smiles_data)


# --- UI and Plotting ---
partition_labels = np.array(["Unassigned"] * len(df))
partition_labels[train_idx] = "Train"
partition_labels[test_idx] = "Test"

model, preds, metrics = train_and_eval_regressor(
    fps[train_idx], activity_data[train_idx],
    fps[test_idx], activity_data[test_idx],
    model_type=ml_algo,
)
progress.progress(100, text="Analysis complete.")

st.title("Biased Split")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Dataset Size", f"{len(df)} compounds")
kpi2.metric("Train / Test Ratio", f"{len(train_idx)} / {len(test_idx)}")
kpi3.metric(
    "Effective Bias",
    f"{effective_bias:.3f}",
    delta=f"{effective_bias - intended_bias:.3f}" if "Split" in split_type and split_type not in ["Murcko Scaffold Split", "Random Split"] else None,
)
kpi4.metric(f"{ml_algo} Test R²", f"{metrics['R2']:.3f}")

tab_tsne, tab_dist, tab_benchmark = st.tabs(
    [
        "Interactive t-SNE Chemical Space",
        "Property & Cliff Distributions",
        "Model Performance vs. Bias Sweep",
    ]
)

with tab_tsne:
    st.subheader("t-SNE Chemical Space Projection (Hover to inspect structure)")
    plot_df = pd.DataFrame({
        "tSNE_1": tsne_coords[:, 0], "tSNE_2": tsne_coords[:, 1],
        "Partition": partition_labels, "Activity": activity_data,
        "SMILES": smiles_data, "MoleculeImage": b64_images,
    })
    plot_df["Color"] = plot_df["Partition"].map({"Train": "#1f77b4", "Test": "#ff7f0e"})

    source = ColumnDataSource(plot_df)
    fig_tsne = figure(
        height=680,
        sizing_mode="stretch_width",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_axis_label="t-SNE Dimension 1",
        y_axis_label="t-SNE Dimension 2",
        toolbar_location="above",
    )
    fig_tsne.scatter(
        x="tSNE_1", y="tSNE_2", source=source, size=8,
        color="Color",
        alpha=0.8, legend_field="Partition",
    )
    fig_tsne.add_tools(HoverTool(
        tooltips="""
            <div>
                <div><strong>@SMILES</strong></div>
                <div>Partition: @Partition</div>
                <div>Activity: @Activity{0.00}</div>
                <div><img src="@MoleculeImage" width="150" height="150"></div>
            </div>
        """,
    ))
    fig_tsne.legend.location = "top_left"
    bokeh_script, bokeh_div = bokeh_components(fig_tsne)
    st_components.html(
        f"{CDN.render_js()}\n{bokeh_script}\n{bokeh_div}",
        height=700,
        scrolling=False,
    )

with tab_dist:
    col_dist1, col_dist2 = st.columns(2)
    with col_dist1:
        st.subheader("Bioactivity Distribution by Partition")
        fig_hist = px.histogram(
            plot_df, x="Activity", color="Partition", barmode="overlay",
            color_discrete_map={"Train": "#1f77b4", "Test": "#ff7f0e"},
            nbins=30, opacity=0.7,
        )
        fig_hist.update_layout(xaxis_title="Activity Value", yaxis_title="Compound Count", height=400)
        st.plotly_chart(fig_hist, width="stretch")

    with col_dist2:
        st.subheader("Test Predictions vs. True Activity")
        eval_df = pd.DataFrame({"True": activity_data[test_idx], "Predicted": preds})
        fig_scatter = px.scatter(eval_df, x="True", y="Predicted", opacity=0.7)
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash", color="grey"),
            x0=eval_df["True"].min(), y0=eval_df["True"].min(),
            x1=eval_df["True"].max(), y1=eval_df["True"].max(),
        )
        fig_scatter.update_layout(
            height=400, xaxis_title="True Activity", yaxis_title="Predicted Activity",
            title=f"RMSE: {metrics['RMSE']:.3f} | MAE: {metrics['MAE']:.3f}",
        )
        st.plotly_chart(fig_scatter, width="stretch")

with tab_benchmark:
    st.subheader("Generalization Decay Across Intended Bias Levels")
    st.markdown(
        "Evaluates the predictive performance of the chosen QSAR regressor when progressively more "
        "test compounds lie across a difficult/biased split relative to the training set."
    )

    sweep_supported = split_type not in ["Murcko Scaffold Split", "Random Split"]
    dataset_signature = hash(
        (
            tuple(df.columns),
            tuple(pd.util.hash_pandas_object(df, index=True).tolist()),
        )
    )
    sweep_config = (
        dataset_signature,
        smi_col,
        act_col,
        split_type,
        test_frac,
        sim_thresh,
        act_thresh,
        n_neighbors,
        ideal_min,
        ideal_max,
        ml_algo,
    )

    if st.session_state.get("bias_sweep_config") != sweep_config:
        st.session_state.pop("bias_sweep_results", None)

    if not sweep_supported:
        st.info("Bias sweeps apply only to split methods with an intended-bias parameter.")

    if st.button(
        "Run Full Bias Sweep (0.0 to 1.0)",
        disabled=not sweep_supported,
    ):
        with st.spinner("Sweeping biases and retraining regressors..."):
            biases = np.linspace(0.0, 1.0, 11)
            sweep_results = run_bias_sweep(
                smiles_data, activity_data, fps,
                splitter, # Passes whatever splitter is currently selected in the UI
                biases, model_type=ml_algo,
            )
            st.session_state["bias_sweep_results"] = sweep_results
            st.session_state["bias_sweep_config"] = sweep_config

    if "bias_sweep_results" in st.session_state:
        sweep_df = pd.DataFrame(st.session_state["bias_sweep_results"])

        c1, c2 = st.columns(2)
        with c1:
            fig_perf = px.line(
                sweep_df, x="Effective Bias", y=["R2", "RMSE", "MAE"], markers=True,
                title="Test Set Performance vs. Effective Bias",
            )
            fig_perf.update_layout(height=420, xaxis_title="Effective Bias", yaxis_title="Score")
            st.plotly_chart(fig_perf, width="stretch")

        with c2:
            fig_bias_map = px.line(
                sweep_df, x="Intended Bias", y="Effective Bias", markers=True,
                title="Intended vs. Realized Effective Bias",
            )
            fig_bias_map.add_shape(type="line", line=dict(dash="dot", color="red"), x0=0, y0=0, x1=1, y1=1)
            fig_bias_map.update_layout(height=420)
            st.plotly_chart(fig_bias_map, width="stretch")

        best_index = sweep_df["RMSE"].idxmin()
        best_result = sweep_df.loc[best_index]
        st.success(
            f"Best bias level by lowest RMSE: intended bias "
            f"{best_result['Intended Bias']:.2f} "
            f"(effective bias {best_result['Effective Bias']:.3f}), "
            f"RMSE {best_result['RMSE']:.3f}, R² {best_result['R2']:.3f}."
        )

        def highlight_best(row):
            color = "background-color: #d9f2e6; font-weight: 600"
            return [color if row.name == best_index else "" for _ in row]

        display_df = sweep_df.style.format({
            "Intended Bias": "{:.2f}", "Effective Bias": "{:.3f}",
            "R2": "{:.3f}", "RMSE": "{:.3f}", "MAE": "{:.3f}",
        }).apply(highlight_best, axis=1)
        st.dataframe(display_df, width="stretch")