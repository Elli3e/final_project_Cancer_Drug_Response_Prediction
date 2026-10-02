import streamlit as st
import numpy as np 
import pandas as pd
import xgboost as xgb
import sqlite3
from pickle import load



st.title("🧬 Cancer Drug Response Prediction")

st.markdown(
    """
    Predict cancer drug sensitivity using molecular and patient-level features.
    
    Select a drug, provide patient information, and estimate the predicted 
    **LN_IC50** and **IC50**.
    """
)

# load the model
@st.cache_resource
def load_model():
    with open("XGboost_LN_IC50_cancer_drug_prediction.sav", "rb") as file:
        return load(file)

@st.cache_resource
def load_feature_columns():
    with open("feature_columns.pkl", "rb") as file:
        return load(file)

model = load_model()
feature_columns = load_feature_columns()

# load reference_drug into the app
@st.cache_data
def load_reference_drugs():
        return pd.read_csv("20 drugs Pk.csv")

@st.cache_data
def load_x_train():

    return pd.read_csv("x_train.csv")

# Select reference drug
reference_drug= load_reference_drugs()
reference_drug_names = (reference_drug.loc[reference_drug["Cmax_M"].notna(),"Name"].dropna().tolist())
reference_drug["Name"] = (
    reference_drug["Name"]
    .astype(str)
    .str.strip()
)

# load x_train
x_train= load_x_train()

x_train["DRUG_NAME"] = (
    x_train["DRUG_NAME"]
    .astype(str)
    .str.strip()
)

# Create drug → pathway lookup
drug_pathway = (
    x_train.drop_duplicates("DRUG_NAME")
    .set_index("DRUG_NAME")["PATHWAY_NAME"]
)

all_drug_names = sorted(x_train["DRUG_NAME"].dropna().unique())

st.subheader("💊 Drug Selection")

drug_name = st.selectbox("Select a drug", all_drug_names)

def prepare_user_data():
    user_data={}

    st.subheader("🧪 Concentration Range")
    col1, col2 = st.columns(2)

    with col1:
        min_concentration = st.number_input(
            "Minimum concentration (µM)",
            min_value=0.0001,
            value=0.0001,
            step=0.0001,
            format="%.4f"
        )

    with col2:
        max_concentration = st.number_input(
            "Maximum concentration (µM)",
            min_value=0.0001,
            value=0.1,
            step=0.01,
            format="%.4f"
        )

    if max_concentration < min_concentration:
        st.error("Maximum concentration must be greater than or equal to minimum concentration.")


    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
            ethnicity = st.selectbox(
                "Ethnicity",
                ["Arabic", "Black", "East Asian", "Other",
                "South Asian", "Unknown", "White"]
            )

    with col2:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Unknown"]
            )

    st.subheader("🧬 Gene mutations")
    TP53_mutation_count = st.number_input("TP53 mutation count", min_value=0, step=1)
    KRAS_mutation_count = st.number_input("KRAS mutation count", min_value=0, step=1)
    KMT2C_mutation_count = st.number_input("KMT2C mutation count", min_value=0, step=1)
    PTEN_mutation_count = st.number_input("PTEN mutation count", min_value=0, step=1)
    KMT2D_mutation_count = st.number_input("KMT2D mutation count", min_value=0, step=1)
    RB1_mutation_count = st.number_input("RB1 mutation count", min_value=0, step=1)

    user_data["MIN_CONC"] = min_concentration
    user_data["MAX_CONC"] = max_concentration
    user_data["ethnicity"] = ethnicity
    user_data["gender"] = gender
    user_data["DRUG_NAME"] = drug_name

    user_data["TP53_mutation_count"] = TP53_mutation_count
    user_data["KRAS_mutation_count"] = KRAS_mutation_count
    user_data["KMT2C_mutation_count"] = KMT2C_mutation_count
    user_data["PTEN_mutation_count"] = PTEN_mutation_count
    user_data["KMT2D_mutation_count"] = KMT2D_mutation_count
    user_data["RB1_mutation_count"] = RB1_mutation_count

    user_data["TP53_mutated"] = 1 if TP53_mutation_count > 0 else 0
    user_data["KRAS_mutated"] = 1 if KRAS_mutation_count > 0 else 0
    user_data["KMT2C_mutated"] =  1 if KMT2C_mutation_count > 0 else 0
    user_data["PTEN_mutated"] = 1 if PTEN_mutation_count > 0 else 0
    user_data["KMT2D_mutated"] = 1 if KMT2D_mutation_count > 0 else 0
    user_data["RB1_mutated"] = 1 if RB1_mutation_count > 0 else 0

    user_data["PATHWAY_NAME"] = drug_pathway.get(drug_name, "Other")

    data_option = st.radio("Molecular data input", ["Use default values","Enter my own VAF and expression values", "Upload my own CSV or excel for multiple patients"])
    
    if data_option == "Enter my own VAF and expression values":
        st.caption("VAF: enter values only when mutation count > 0.")
        st.subheader("🧬 Molecular Data")

        TP53_min_VAF = st.number_input( "TP53 minimum VAF",min_value=0.0, max_value=1.0,value=0.0, step=0.01)
        TP53_max_VAF = st.number_input( "TP53 maximum VAF", min_value=0.0,max_value=1.0, value=0.0, step=0.01)
        TP53_expression = st.number_input("TP53 expression", value=0.0,step=0.1)

        KRAS_min_VAF = st.number_input( "KRAS minimum VAF",min_value=0.0, max_value=1.0,value=0.0, step=0.01)
        KRAS_max_VAF = st.number_input( "KRAS maximum VAF", min_value=0.0,max_value=1.0, value=0.0, step=0.01)
        KRAS_expression = st.number_input("KRAS expression", value=0.0,step=0.1)

        KMT2C_min_VAF = st.number_input( "KMT2C minimum VAF",min_value=0.0, max_value=1.0,value=0.0, step=0.01)
        KMT2C_max_VAF = st.number_input( "KMT2C maximum VAF", min_value=0.0,max_value=1.0, value=0.0, step=0.01)
        KMT2C_expression = st.number_input("KMT2C expression", value=0.0,step=0.1)

        PTEN_min_VAF = st.number_input( "PTEN minimum VAF",min_value=0.0, max_value=1.0,value=0.0, step=0.01)
        PTEN_max_VAF = st.number_input( "PTEN maximum VAF", min_value=0.0,max_value=1.0, value=0.0, step=0.01)
        PTEN_expression = st.number_input("PTEN expression", value=0.0,step=0.1)

        KMT2D_min_VAF = st.number_input( "KMT2D minimum VAF",min_value=0.0, max_value=1.0,value=0.0, step=0.01)
        KMT2D_max_VAF = st.number_input( "KMT2D maximum VAF", min_value=0.0,max_value=1.0, value=0.0, step=0.01)
        KMT2D_expression = st.number_input("KMT2D expression", value=0.0,step=0.1)

        RB1_min_VAF = st.number_input( "RB1 minimum VAF",min_value=0.0, max_value=1.0,value=0.0, step=0.01)
        RB1_max_VAF = st.number_input( "RB1 maximum VAF", min_value=0.0,max_value=1.0, value=0.0, step=0.01)
        RB1_expression = st.number_input("RB1 expression", value=0.0,step=0.1)

        user_data["TP53_min_VAF"]= TP53_min_VAF
        user_data["TP53_max_VAF"]= TP53_max_VAF
        user_data["TP53_expression"]= TP53_expression

        user_data["KRAS_min_VAF"] = KRAS_min_VAF
        user_data["KRAS_max_VAF"]= KRAS_max_VAF
        user_data["KRAS_expression"] = KRAS_expression

        user_data["KMT2C_min_VAF"]= KMT2C_min_VAF
        user_data["KMT2C_max_VAF"] = KMT2C_max_VAF
        user_data["KMT2C_expression"] = KMT2C_expression

        user_data["PTEN_min_VAF"] = PTEN_min_VAF
        user_data["PTEN_max_VAF"]= PTEN_max_VAF
        user_data["PTEN_expression"] = PTEN_expression

        user_data["KMT2D_min_VAF"] = KMT2D_min_VAF
        user_data["KMT2D_max_VAF"] = KMT2D_max_VAF
        user_data["KMT2D_expression"] = KMT2D_expression


        user_data["RB1_min_VAF"] = RB1_min_VAF
        user_data["RB1_max_VAF"] = RB1_max_VAF
        user_data["RB1_expression"] = RB1_expression
        
    elif data_option == "Use default values":
    
        if TP53_mutation_count > 0:
            user_data["TP53_min_VAF"] = x_train["TP53_min_VAF"].median()
            user_data["TP53_max_VAF"] = x_train["TP53_max_VAF"].median()
        else:
            user_data["TP53_min_VAF"] = 0
            user_data["TP53_max_VAF"] = 0

        if KRAS_mutation_count > 0:
            user_data["KRAS_min_VAF"] = x_train["KRAS_min_VAF"].median()
            user_data["KRAS_max_VAF"] = x_train["KRAS_max_VAF"].median()
        else:
            user_data["KRAS_max_VAF"] =0
            user_data["KRAS_min_VAF"] = 0

        if KMT2C_mutation_count>0:
            user_data["KMT2C_min_VAF"] = x_train["KMT2C_min_VAF"].median()
            user_data["KMT2C_max_VAF"] = x_train["KMT2C_max_VAF"].median()
        else:
            user_data["KMT2C_min_VAF"]=0
            user_data["KMT2C_max_VAF"]=0

        if PTEN_mutation_count>0:
            user_data["PTEN_min_VAF"] = x_train["PTEN_min_VAF"].median()
            user_data["PTEN_max_VAF"] = x_train["PTEN_max_VAF"].median()
        else:
            user_data["PTEN_min_VAF"]=0
            user_data["PTEN_max_VAF"]=0

        if KMT2D_mutation_count>0:
            user_data["KMT2D_min_VAF"] = x_train["KMT2D_min_VAF"].median()
            user_data["KMT2D_max_VAF"] = x_train["KMT2D_max_VAF"].median()
        else:
            user_data["KMT2D_min_VAF"]=0
            user_data["KMT2D_max_VAF"]=0

        if RB1_mutation_count>0:
            user_data["RB1_min_VAF"] = x_train["RB1_min_VAF"].median()
            user_data["RB1_max_VAF"] = x_train["RB1_max_VAF"].median()
        else:
            user_data["RB1_min_VAF"]=0
            user_data["RB1_max_VAF"]=0

        user_data["TP53_expression"] = x_train["TP53_expression"].median()
        user_data["KRAS_expression"] = x_train["KRAS_expression"].median()
        user_data["KMT2C_expression"] = x_train["KMT2C_expression"].median()
        user_data["PTEN_expression"] = x_train["PTEN_expression"].median()
        user_data["KMT2D_expression"] = x_train["KMT2D_expression"].median()
        user_data["RB1_expression"] = x_train["RB1_expression"].median()

    elif data_option== "Upload my own CSV or excel for multiple patients":

        st.subheader("📋 Patient Data Template")

        st.write(
            "Download a template, fill in your patient data, "
            "and then upload the completed file below.")

        col1, col2 = st.columns(2)

        with col1:
            with open("patient_input_template.xlsx", "rb") as file:
                st.download_button(
                    label="📥 Download Excel Template",
                    data=file,
                    file_name="patient_input_template.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with col2:
            with open("patient_input_template.csv", "rb") as file:
                st.download_button(
                    label="📥 Download CSV Template",
                    data=file,
                    file_name="patient_input_template.csv",
                    mime="text/csv")

        st.subheader("📤 Upload Completed Template")

        uploaded_file = st.file_uploader(
            "Upload your completed patient template",
            type=["xlsx", "csv"])
        
        if uploaded_file is not None:
            if  uploaded_file.name.endswith(".xlsx"):
                uploaded_df = pd.read_excel(uploaded_file)

            elif uploaded_file.name.endswith(".csv"):
                 uploaded_df = pd.read_csv(uploaded_file)

            return uploaded_df

    return user_data


    #conversion functions for LNIC50 
def conversion(LN_IC50):
        ic50_molar =  np.exp(LN_IC50)* 1e-6
        return  ic50_molar

def compare_ic50_cmax(predicted_ic50, cmax):

    ratio = cmax / predicted_ic50
    if 0.9 <= ratio <= 1.1:
        st.info(
            f"Clinical Cmax and predicted IC50 are approximately similar "
            f"({ratio:.2f}×)."
        )

    elif ratio > 1.1:
        st.success(
            f"Clinical Cmax is {ratio:.2f}× higher "
            f"than the predicted IC50."
        )

    else:
        ratio = predicted_ic50 / cmax

        st.warning(
            f"Predicted IC50 is {ratio:.2f}× higher "
            f"than clinical Cmax."
        )
        st.info("You may adjust the minimum and maximum concentration inputs"
    "  to evaluate the prediction over a different concentration range.")


user_data = prepare_user_data()

if isinstance(user_data, pd.DataFrame):
    uploaded_df = user_data
    # check the basic required columns 
    required_columns = [
    "patient_id",
    "DRUG_NAME",
    "ethnicity",
    "gender",

    "TP53_mutated",
    "TP53_mutation_count",
    "TP53_min_VAF",
    "TP53_max_VAF",
    "TP53_expression",

    "KRAS_mutated",
    "KRAS_mutation_count",
    "KRAS_min_VAF",
    "KRAS_max_VAF",
    "KRAS_expression",

    "KMT2C_mutated",
    "KMT2C_mutation_count",
    "KMT2C_min_VAF",
    "KMT2C_max_VAF",
    "KMT2C_expression",

    "PTEN_mutated",
    "PTEN_mutation_count",
    "PTEN_min_VAF",
    "PTEN_max_VAF",
    "PTEN_expression",

    "KMT2D_mutated",
    "KMT2D_mutation_count",
    "KMT2D_min_VAF",
    "KMT2D_max_VAF",
    "KMT2D_expression",

    "RB1_mutated",
    "RB1_mutation_count",
    "RB1_min_VAF",
    "RB1_max_VAF",
    "RB1_expression",

    "MIN_CONC",
    "MAX_CONC"
]

    missing_columns = [ col for col in required_columns if col not in uploaded_df.columns]

    if missing_columns:
        st.error(
            "Your file is missing these required columns: "
            + ", ".join(missing_columns))
        st.stop()

        # Convert numeric columns to numbers
    numeric_columns = [
        "MIN_CONC",
        "MAX_CONC"]
    
    mutation_genes = ["TP53", "KRAS", "KMT2C", "PTEN", "KMT2D", "RB1"]
    for gene in mutation_genes:
        numeric_columns.extend([
            f"{gene}_mutated",
            f"{gene}_mutation_count",
            f"{gene}_min_VAF",
            f"{gene}_max_VAF",
            f"{gene}_expression"
        ])

    for col in numeric_columns:
        uploaded_df[col] = pd.to_numeric(
            uploaded_df[col],
            errors="coerce"
        )
    # check the missing genes 
    mutation_genes = ["TP53", "KRAS", "KMT2C", "PTEN", "KMT2D", "RB1"]

    for gene in mutation_genes:

        col = f"{gene}_mutated"

        invalid = ~uploaded_df[col].isin([0, 1])

        if invalid.any():
            st.error(
                f"{col} must contain only 0 or 1.")
            st.stop()
    # check the mutations count 
    for gene in mutation_genes:

        col = f"{gene}_mutation_count"

        values = pd.to_numeric(uploaded_df[col],errors="coerce")

        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():

            st.error(
                f"{col} must contain non-negative whole numbers.")
            st.stop()

# check VAF
    for gene in mutation_genes:

        for suffix in ["min_VAF", "max_VAF"]:

            col = f"{gene}_{suffix}"

            values = pd.to_numeric(uploaded_df[col], errors="coerce" )

            if values.isna().any() or (values < 0).any() or (values > 1).any():

                st.error(
                    f"{col} must be between 0 and 1. "
                    "For example, 42% should be entered as 0.42.")
                st.stop()

# check mutation and mutations_count
    for gene in mutation_genes:

        inconsistent = (
            ((uploaded_df[f"{gene}_mutated"] == 0) &
            (uploaded_df[f"{gene}_mutation_count"] != 0))
            |
            ((uploaded_df[f"{gene}_mutated"] == 1) &
            (uploaded_df[f"{gene}_mutation_count"] == 0))
        )

        if inconsistent.any():

            st.error(
                f"{gene}_mutated and {gene}_mutation_count "
                "are inconsistent.")
            st.stop()
# check min VAF ≤ max VAF
    for gene in mutation_genes:

        if (
            uploaded_df[f"{gene}_min_VAF"]
            > uploaded_df[f"{gene}_max_VAF"]).any():

            st.error(
                f"For {gene}, min_VAF cannot be greater than max_VAF.")
            st.stop()

# check MIN_CONC ≥ 0, MAX_CONC ≥ 0, MIN_CONC ≤ MAX_CONC
    if (uploaded_df["MIN_CONC"] < 0).any():
        st.error("MIN_CONC cannot be negative.")
        st.stop()

    if (uploaded_df["MAX_CONC"] < 0).any():
        st.error("MAX_CONC cannot be negative.")
        st.stop()

    if (uploaded_df["MIN_CONC"] > uploaded_df["MAX_CONC"]).any():
        st.error(
            "MIN_CONC cannot be greater than MAX_CONC.")
        st.stop()

# Check the drug names
    invalid_drugs = uploaded_df.loc[
        ~uploaded_df["DRUG_NAME"].isin(all_drug_names),"DRUG_NAME"].unique()

    if len(invalid_drugs) > 0:
        st.error(
            "The following drug names are not recognized: "
            + ", ".join(invalid_drugs))
        st.stop()

    user_input = uploaded_df.copy()
else:
    user_input = pd.DataFrame([user_data])

    
user_input["PATHWAY_NAME"] = user_input["DRUG_NAME"].map(drug_pathway).fillna("Other")

user_input = pd.get_dummies(
    user_input,
    columns=["ethnicity", "gender", "DRUG_NAME", "PATHWAY_NAME"],
    dtype=int
)

user_input = user_input.reindex(
    columns=feature_columns,
    fill_value=0
)

if st.button("Predict Drug Response"):
    LN_IC50 =  model.predict(user_input)[0]
    predicted_ic50 = conversion(LN_IC50)


    st.subheader("Prediction Results")

    st.write(f"Predicted LN_IC50: {LN_IC50:.3f}")
    st.write(f"Predicted IC50: {predicted_ic50:.3e} M")

    # Check whether clinical Cmax is available

    drug_match = reference_drug[
    reference_drug["Name"].astype(str).str.strip()
    == drug_name.strip()]

    if len(drug_match) > 0 and pd.notna(drug_match.iloc[0]["Cmax_M"]):

        drug_info = drug_match.iloc[0]
        cmax = drug_info["Cmax_M"]

        st.subheader("Clinical Drug Information")

        st.write(
            "Clinical status:",
            drug_info["Clinical_status"]
        )
        st.write(
            "Dose:",
            drug_info["dose"]
        )
        st.write(
            "Route:",
            drug_info["route"]
        )
        st.write(
            "Cmax:",
            drug_info["Cmax"],
            drug_info["Cmax_unit"]
        )

        st.write("Cmax (M):", cmax)

        compare_ic50_cmax(predicted_ic50, cmax)

    else:

        st.info(
            "Reference clinical Cmax information is not available "
            "for this drug, so a clinical Cmax comparison cannot be performed."
        )


        