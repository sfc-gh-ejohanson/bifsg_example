from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from callbacks import Callbacks
from code_exporter import create_notebook
from common import get_col_types
from histograms import AutoHistogram
from model_metrics import ModelMetrics
from preprocessing import AutoPreProcessor
from snowflake.ml.modeling.impute import SimpleImputer
from snowflake.ml.modeling.linear_model import (
    ElasticNet,
    LinearRegression,
    LogisticRegression,
)
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)
from snowflake.ml.modeling.xgboost import XGBClassifier, XGBRegressor
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session
from streamlit.components.v1 import html
from utils import get_databases, get_feature_importance_df, get_schemas, get_tables
from scipy import stats

AVATAR_PATH = str(Path(__file__).parent / "resources" / "Snowflake_ICON_Chat.png")
probabilitiesDataframe = pd.DataFrame()

def set_state(state: int):
    st.session_state["app_state"] = state
    if state not in st.session_state["recorded_steps"]:
        st.session_state["recorded_steps"].append(state)


def create_metric_card(label, value):
    return f"""
             <span class="property_container">
                <span class="property_title">{label}</span>
                <span class="property_pill_current">{value}</span>
            </span>
                """


class TopMenu:
    def __init__(self) -> None:
        header_menu_c = st.container(border=False, height=60)
        header_menu = header_menu_c.columns(3)
        header_menu[0].button(
            "Select Dataset",
            key="btn_select",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 0 else "secondary",
            disabled=0 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[0],
        )
        header_menu[2].button(
            "Bias Testing",
            key="btn_modeling",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 2 else "secondary",
            disabled=2 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[2],
        )
        header_menu[1].button(
            "BIFSG",
            key="btn_bifsg",
            use_container_width=True,
            type="primary" if st.session_state["app_state"] == 1 else "secondary",
            disabled=1 not in st.session_state["recorded_steps"],
            on_click=set_state,
            args=[1],
        )


class AutoMLModeling:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.step_data = {}

    def render_ml_builder(self):
        with st.container(height=101, border=False):
            if st.button("🔍"):
                if st.session_state["dataset"]:
                    AutoHistogram(
                        df=st.session_state["dataset"],
                        name=f"{st.session_state.get('aml_mpa.sel_db','-')}.{st.session_state.get('aml_mpa.sel_schema','-')}.{st.session_state.get('aml_mpa.sel_table','-')}",
                    ).render_grid()
                else:
                    st.toast("You must select a dataset before.")

        TopMenu()
        if st.session_state["app_state"] > -1:
            dataset_chat = st.chat_message(
                name="assistant",
                avatar=AVATAR_PATH,
            )
            with dataset_chat:
                st.write("Let's begin by selecting a source dataset.")
                with st.popover(
                    "Dataset Selection",
                    disabled=not (st.session_state["app_state"] == 0),
                    use_container_width=True,
                ):
                    context_menu_cols = st.columns((1, 2))
                    databases = get_databases(self.session)
                    db = context_menu_cols[0].selectbox(
                        "Source Database",
                        index=None,
                        options=databases,
                        placeholder="Select a database",
                        key="aml_mpa.sel_db",
                    )
                    if db:
                        st.session_state["context"]["database"] = db
                        st.session_state["context"]["schemas"] = get_schemas(
                            self.session, db
                        )
                    else:
                        st.session_state["context"]["schemas"] = []

                    schema = context_menu_cols[0].selectbox(
                        "Source Schema",
                        st.session_state["context"].get("schemas", []),
                        index=None,
                        placeholder="Select a schema",
                        key="aml_mpa.sel_schema",
                    )

                    if schema:
                        st.session_state["context"]["tables"] = get_tables(
                            self.session, db, schema
                        )
                    else:
                        st.session_state["context"]["tables"] = []

                    table = context_menu_cols[0].selectbox(
                        "Source Table",
                        st.session_state["context"].get("tables", []),
                        index=None,
                        placeholder="Select a table",
                        key="aml_mpa.sel_table",
                        on_change=Callbacks.set_dataset,
                        args=[self.session, db, schema, "aml_mpa.sel_table"],
                    )
                    if all([db, schema, table]):
                        context_menu_cols[1].dataframe(
                            st.session_state["dataset"].limit(5),
                            hide_index=True,
                            use_container_width=True,
                        )
                        st.container(border=False, height=10)
                        dataset_chat_cols = dataset_chat.columns(3)
                        dataset_chat_cols[2].button(
                            "Next",
                            use_container_width=True,
                            type="primary",
                            on_click=set_state,
                            args=[1],
                            disabled=not (st.session_state["app_state"] == 0),
                        )

        if 1 in st.session_state["recorded_steps"] and st.session_state["dataset"]:
            preproc_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with preproc_chat:
                st.write("Now, let's pre-process the source dataset and run BIFSG")
                st.info(
                    "Click the :mag: icon above to dig deeper into your data. If you have null values, add a **SimpleImputer** step. If you have string features, add a **OneHotEncoder** step. "
                )
                with st.expander(
                    "Pre-Processing Options",
                    expanded=st.session_state["app_state"] == 1,
                ):
                    st.header("Preprocessing Options")
                    st.caption(":red[*] required fields")
                    feature_cols = st.multiselect(
                        "Select the feature columns.:red[*]",
                        options=st.session_state["dataset"].columns,
                    )
                    target_col = st.selectbox(
                        "Select the target column.:red[*]",
                        st.session_state["dataset"].columns,
                        index=None,
                    )
                    session = st.connection('snowflake').session()

# Change the query to point to your table
                    # query = """
                    # call ml_sidekick.test_data.surgeo_udf_no_values()
                    # """
                    

                    query2 = """
                    select * from ml_sidekick.test_data.probabilities limit 10000;
                    """

                    # data2 = session.sql(query2).collect()
                    # probabilitiesDataframe = pd.DataFrame(data2)

                    # probabilitiesDataframe['Max_Value_Numeric'] = probabilitiesDataframe.max(axis=1, numeric_only=True)
                    # probabilitiesDataframe['Max_Value_Type'] = probabilitiesDataframe.idxmax(axis=1, numeric_only=True)
                    # with st.status("Running BIFSG...", expanded=True) as status:
                    #     st.dataframe(probabilitiesDataframe)
                    #     status.update(
                    #         label="Download complete!", state="complete", expanded=False
                    #     )
                    with st.spinner("Running BIFSG...", show_time=True):
                        query = """
                        call ml_sidekick.test_data.sp_process_data()
                        """
                        data = session.sql(query).collect()
                        data2 = session.sql(query2).collect()
                        probabilitiesDataframe = pd.DataFrame(data2)
                        probabilitiesDataframe['Max_Value_Numeric'] = probabilitiesDataframe.max(axis=1, numeric_only=True)
                        probabilitiesDataframe['Max_Value_Type'] = probabilitiesDataframe.idxmax(axis=1, numeric_only=True)
                    
                    st.success("Done!")

                    # st.dataframe(data2)
                    st.dataframe(probabilitiesDataframe)

                    number = st.number_input(
                        "Type a number as a percentage", value=None, placeholder="Select a threshold percentage..."
                    )

                    blackGroup = pd.DataFrame()
                    whiteGroup = pd.DataFrame()
                    apiGroup = pd.DataFrame()
                    nativeGroup = pd.DataFrame()
                    multipleGroup = pd.DataFrame()
                    hispanicGroup = pd.DataFrame()

                    blackGroup = probabilitiesDataframe[(probabilitiesDataframe['Max_Value_Type'] == 'BLACK') & (probabilitiesDataframe['Max_Value_Numeric'] >= number)]
                    whiteGroup = probabilitiesDataframe[(probabilitiesDataframe['Max_Value_Type'] == 'WHITE') & (probabilitiesDataframe['Max_Value_Numeric'] >= number)]
                    apiGroup = probabilitiesDataframe[(probabilitiesDataframe['Max_Value_Type'] == 'API') & (probabilitiesDataframe['Max_Value_Numeric'] >= number)]
                    nativeGroup = probabilitiesDataframe[(probabilitiesDataframe['Max_Value_Type'] == 'NATIVE') & (probabilitiesDataframe['Max_Value_Numeric'] >= number)]
                    multipleGroup = probabilitiesDataframe[(probabilitiesDataframe['Max_Value_Type'] == 'MULTIPLE') & (probabilitiesDataframe['Max_Value_Numeric'] >= number)]
                    hispanicGroup = probabilitiesDataframe[(probabilitiesDataframe['Max_Value_Type'] == 'HISPANIC') & (probabilitiesDataframe['Max_Value_Numeric'] >= number)]

                    # st.radio(
                    #     "Set Threshold visibility 👉",
                    #     key="visibility",
                    #     options=["50%", "70%", "90%"],
                    # )
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["BLACK", "WHITE", "API", "NATIVE", "MULTIPLE", "HISPANIC"])
                    with tab1:
                        st.header("BLACK")
                        st.dataframe(blackGroup)
                    with tab2:
                        st.header("WHITE")
                        st.dataframe(whiteGroup)
                    with tab3:
                        st.header("API")
                        st.dataframe(apiGroup)
                    with tab4:
                        st.header("NATIVE")
                        st.dataframe(nativeGroup)
                    with tab5:
                        st.header("MULTIPLE")
                        st.dataframe(multipleGroup)
                    with tab6:
                        st.header("HISPANIC")
                        st.dataframe(hispanicGroup)
                    

                    st.button(
                                "Next",
                                use_container_width=True,
                                type="primary",
                                on_click=set_state,
                                args=[2],
                                key="pproc_nxt",
                            )
        if 2 in st.session_state["recorded_steps"] and st.session_state["dataset"]:
            modeling_chat = st.chat_message(name="assistant", avatar=AVATAR_PATH)
            with modeling_chat:
                st.write("Choose your bias testing options")
                with st.expander(
                    "Modeling", expanded=st.session_state["app_state"] == 2
                ):
                    st.header("Bias Testing Options")
                    model_types = [
                        {
                            "type": "T Test",
                            "models": [
                                "Scipy",
                            ],
                        },
                        {
                            "type": "Regression",
                            "models": [
                                "XGBRegressor",
                                "LinearRegression",
                                "ElasticNet",
                            ],
                        },
                        {
                            "type": "Classification",
                            "models": [
                                "XGBClassifier",
                                "LogisticRegression",
                            ],
                        },
                    ]

                    model_type = st.radio(
                        "Model Type",
                        options=[i.get("type") for i in model_types],
                        horizontal=True,
                    )
                    available_models = [
                        i for i in model_types if i.get("type") == model_type
                    ][0].get("models")
                    model_selections = st.selectbox("Model", options=available_models)
                    if bool(model_selections):
                        fit_menu = st.columns(4, gap="large")
                        show_metrics = fit_menu[0].toggle(
                            "Retrieve Model Metrics", value=True
                        )
                        fit_btn = fit_menu[1].button(
                            "Fit Model & Run Prediction(s)", use_container_width=True
                        )
                        # Perform independent samples t-test
                        

                        

                        # query = """
                        # select * from ml_sidekick.test_data.probabilities;
                        # """

                        with st.spinner("Running T Test...", show_time=True):
                            query2 = """
                            select * from ml_sidekick.test_data.hartford_fake_data;
                            """

                            # data = session.sql(query).collect()
                            data2 = session.sql(query2).collect()
                            # probabilitiesDataframe = pd.DataFrame(data)
                            probabilitiesFakeData = pd.DataFrame(data2)
                            # Extract the columns for the t-test
                        

                            column_df4 = probabilitiesFakeData['FEATURE_13']

                            # Perform the independent samples t-test
                            # equal_var=True assumes equal variances; set to False for Welch's t-test
                            t_statistic_black, p_value_black = stats.ttest_ind(probabilitiesDataframe['BLACK'], column_df4, equal_var=True, nan_policy='omit')
                            t_statistic_white, p_value_white = stats.ttest_ind(probabilitiesDataframe['WHITE'], column_df4, equal_var=True, nan_policy='omit')
                            t_statistic_api, p_value_api = stats.ttest_ind(probabilitiesDataframe['API'], column_df4, equal_var=True, nan_policy='omit')
                            t_statistic_native, p_value_native = stats.ttest_ind(probabilitiesDataframe['NATIVE'], column_df4, equal_var=True, nan_policy='omit')
                            t_statistic_multiple, p_value_multiple = stats.ttest_ind(probabilitiesDataframe['MULTIPLE'], column_df4, equal_var=True, nan_policy='omit')
                            t_statistic_hispanic, p_value_hispanic = stats.ttest_ind(probabilitiesDataframe['HISPANIC'], column_df4, equal_var=True, nan_policy='omit')
                            st.write(t_statistic_black)
                            st.write(p_value_black)
                            df_results = pd.DataFrame()
                            df_results['BLACK'] = [t_statistic_black]
                            df_results['WHITE'] = [t_statistic_white]
                            df_results['API'] = t_statistic_api
                            df_results['NATIVE'] = t_statistic_native
                            df_results['MULTIPLE'] = t_statistic_multiple
                            df_results['HISPANIC'] = t_statistic_hispanic
                            st.dataframe(df_results)
                        st.success("Done!")

                        
                        # results_columns = st.columns(6)
                        # for k,v in df_results.interpolate().iterrows():
                        #     with results_columns[k]:
                        #         st.metric
