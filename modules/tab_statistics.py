"""
Tab 3: Statistics - cumulative overview of all measurements in the validation log.
"""

from __future__ import annotations

import streamlit as st

from .config import (
    DENSITY_MAX,
    DENSITY_MIN,
    DENSITY_STEP,
    PATTERNS,
    TEMP_MAX,
    TEMP_MIN,
    TEMP_STEP,
)
from .metrics import compute_summary, tier_accuracy
from .plotting import error_histogram, predicted_vs_measured_scatter
from .predictor import LightGBMUTSModel, PlaceholderUTSModel
from .validation_log import clear_log, delete_entry, load_log, update_entry


def render(model: LightGBMUTSModel | PlaceholderUTSModel) -> None:
    st.subheader("Validation statistics")
    df = load_log()

    if df.empty:
        st.info(
            "No measurements logged yet. Go to the 'Validation' tab "
            "and enter your first measured UTS."
        )
        return

    _render_kpis(df)
    st.divider()
    _render_tier_accuracy(df)
    st.divider()
    _render_charts(df)
    st.divider()
    _render_log_table(df, model)


def _render_kpis(df) -> None:
    summary = compute_summary(df)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Number of measurements", summary["n"])
    col2.metric("RMSE", f"{summary['rmse']:.3f} MPa")
    col3.metric("MAE", f"{summary['mae']:.3f} MPa")
    r2_val = summary["r2"]
    col4.metric("R²", f"{r2_val:.3f}" if r2_val == r2_val else "—")


def _render_tier_accuracy(df) -> None:
    st.markdown("##### Accuracy by tier (from the report)")
    tiers = tier_accuracy(df)
    cols = st.columns(len(tiers))
    for col, (name, pct) in zip(cols, tiers.items()):
        col.metric(name, f"{pct:.1f} %")


def _render_charts(df) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(predicted_vs_measured_scatter(df), use_container_width=True)
    with col2:
        st.plotly_chart(error_histogram(df), use_container_width=True)


def _format_row_option(i: int, row) -> str:
    return (
        f"#{i}  |  {row['timestamp']}  |  {row['pattern']} "
        f"d={row['density_pct']}% T={row['temp_c']}°C  |  "
        f"pred={row['predicted_uts']:.2f}  meas={row['measured_uts']:.2f}"
    )


def _render_edit_form(df, model: LightGBMUTSModel | PlaceholderUTSModel) -> None:
    st.markdown("##### Edit a single measurement")
    options = [_format_row_option(i, row) for i, row in df.iterrows()]
    selected = st.selectbox(
        "Select a measurement to edit",
        options=list(range(len(df))),
        format_func=lambda i: options[i],
        key="edit_selector",
    )
    row = df.iloc[selected]

    with st.form(key="edit_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            new_density = st.number_input(
                "Infill density (%)",
                min_value=DENSITY_MIN,
                max_value=DENSITY_MAX,
                value=float(row["density_pct"]),
                step=DENSITY_STEP,
                key="edit_density",
            )
        with col2:
            new_temp = st.number_input(
                "Nozzle temp (°C)",
                min_value=TEMP_MIN,
                max_value=TEMP_MAX,
                value=float(row["temp_c"]),
                step=TEMP_STEP,
                key="edit_temp",
            )
        with col3:
            new_pattern = st.selectbox(
                "Pattern",
                options=PATTERNS,
                index=PATTERNS.index(row["pattern"]),
                key="edit_pattern",
            )

        col4, col5 = st.columns([1, 2])
        with col4:
            new_measured = st.number_input(
                "Measured UTS (MPa)",
                min_value=0.0,
                max_value=50.0,
                value=float(row["measured_uts"]),
                step=0.1,
                key="edit_measured",
            )
        with col5:
            new_note = st.text_input(
                "Note",
                value=str(row.get("note", "")),
                key="edit_note",
            )

        recompute = st.checkbox(
            "Recompute prediction from parameters (if they changed)",
            value=True,
            key="edit_recompute",
            help="If unchecked, predicted_uts stays as it was when the measurement was first saved.",
        )

        submitted = st.form_submit_button("💾 Save changes", type="primary")

    if submitted:
        if recompute:
            pred = model.predict(new_density, new_temp, new_pattern)
            new_predicted = pred.uts_mpa
        else:
            new_predicted = float(row["predicted_uts"])

        update_entry(
            int(selected),
            density_pct=float(new_density),
            temp_c=float(new_temp),
            pattern=new_pattern,
            predicted_uts=new_predicted,
            measured_uts=float(new_measured),
            note=new_note,
        )
        st.success(f"Measurement #{selected} updated.")
        st.rerun()


def _render_log_table(df, model: LightGBMUTSModel | PlaceholderUTSModel) -> None:
    st.markdown("##### Raw measurement log")
    st.dataframe(df, use_container_width=True, hide_index=False)

    with st.expander("✏️ Edit a single measurement"):
        _render_edit_form(df, model)

    st.markdown("##### Delete a single measurement")
    col_idx, col_btn = st.columns([3, 1])
    with col_idx:
        options = [
            f"#{i}  |  {row['timestamp']}  |  {row['pattern']} "
            f"d={row['density_pct']}% T={row['temp_c']}°C  |  "
            f"pred={row['predicted_uts']:.2f}  meas={row['measured_uts']:.2f}"
            for i, row in df.iterrows()
        ]
        selected = st.selectbox(
            "Select row to delete",
            options=list(range(len(df))),
            format_func=lambda i: options[i],
            key="delete_selector",
        )
    with col_btn:
        st.write("")
        st.write("")
        if st.button("🗑️ Delete", type="secondary", use_container_width=True):
            delete_entry(int(selected))
            st.success(f"Measurement #{selected} deleted.")
            st.rerun()

    st.divider()
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Delete ALL measurements", type="secondary"):
            clear_log()
            st.success("Log cleared.")
            st.rerun()
    with col2:
        st.download_button(
            "⬇️ Download log as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="validation_log.csv",
            mime="text/csv",
        )
