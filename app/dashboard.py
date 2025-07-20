import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Plateforme de Données Unifiée MTY", layout="wide")

st.title("Plateforme de Données Unifiée MTY")
st.subheader("Plateforme simulée pour l’analytique et l’apprentissage automatique dans les franchises")

df = pd.read_csv("data/features/final_dataset.csv")

# KPIs
st.metric("Magasins analysés", len(df))
st.metric("Magasins sous-performants", df["is_low_performing_store"].sum())
st.metric("Vente mensuelle moyenne", f"${df['total_month_sales'].mean():,.0f}")

# Visualisation
st.markdown("### Carte des magasins par ventes")
fig = px.scatter(df, x="sales_per_employee", y="avg_customer_rating",
                 color="is_low_performing_store",
                 hover_data=["store_id", "brand"])
st.plotly_chart(fig, use_container_width=True)

# Affichage des 3 magasins avec le plus bas rendement
st.subheader("Top 3 des magasins sous-performants")

low_perf_stores = df[df["is_low_performing_store"] == 1]
top3_low = low_perf_stores.sort_values("sales_per_employee").head(3)

st.table(top3_low[["store_id", "brand", "sales_per_employee", "avg_customer_rating"]])

# Métriques du modèle de ML
from ml.feature_engineering import get_model_metrics

df_preds = pd.read_csv("data/features/predictions.csv")
y_true = df_preds["actual"]
y_pred = df_preds["prediction"]

metrics = get_model_metrics(y_true, y_pred)

st.subheader(" Métriques du Modèle de Prédiction")
col1, col2, col3 = st.columns(3)
col1.metric("F1-score", f"{metrics['F1-score']:.2f}")
col2.metric("Précision", f"{metrics['Precision']:.2f}")
col3.metric("Rappel", f"{metrics['Recall']:.2f}")



# Tableau
st.markdown("### Données détaillées")
st.dataframe(df)

