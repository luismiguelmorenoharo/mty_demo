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

# Tableau
st.markdown("### Données détaillées")
st.dataframe(df)





