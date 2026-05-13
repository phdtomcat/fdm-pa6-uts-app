"""
Main Streamlit entry point. All logic lives in modules/* - this file
only assembles tabs and the sidebar.
"""

from __future__ import annotations

import streamlit as st

from modules import sidebar, tab_prediction, tab_statistics, tab_validation
from modules.predictor import load_model


def main() -> None:
    st.set_page_config(
        page_title="Dog Bone UTS Predictor",
        page_icon="🦴",
        layout="wide",
    )

    model = load_model()
    sidebar.render(model)

    st.title("Tensile strength prediction for 3D-printed PA6 dog-bone specimens")
    st.markdown(
        "ML model trained on FDM tensile tests. The app predicts UTS from print "
        "parameters and records actual measurements to validate the model."
    )

    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "🧪 Validation", "📊 Statistics"])
    with tab1:
        tab_prediction.render(model)
    with tab2:
        tab_validation.render(model)
    with tab3:
        tab_statistics.render(model)


if __name__ == "__main__":
    main()
