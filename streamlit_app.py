#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import streamlit as st
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px

# ---------------- CONFIG ----------------
SCRIPT_DIR = os.path.dirname(__file__)
OUTROOT = os.path.join(SCRIPT_DIR, "out_multi")
HOSPITAL_DATA = os.path.join(SCRIPT_DIR, "geocode_health_centre.csv")
RESOURCE_DATA = os.path.join(SCRIPT_DIR, "hospital_directory.csv")
TOP_K = 20

st.set_page_config(page_title="Epidemic Hotspot Dashboard", layout="wide")

# ---------------- HELPERS ----------------
def find_disease_folders(root: str = OUTROOT):
    if not os.path.exists(root):
        return []
    return [(f, os.path.join(root, f)) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]

def load_predictions(folder: str) -> pd.DataFrame:
    fp = os.path.join(folder, "next_week_predictions.csv")
    if not os.path.exists(fp):
        return pd.DataFrame()
    df = pd.read_csv(fp, low_memory=False)
    for col in ["predicted_cases", "Latitude", "Longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_rural"] = df.get("is_rural", False).astype(bool)
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df = df.dropna(subset=["Latitude", "Longitude"])
    return df

def load_hospitals() -> pd.DataFrame:
    if not os.path.exists(HOSPITAL_DATA):
        return pd.DataFrame()
    df = pd.read_csv(HOSPITAL_DATA, low_memory=False)
    rename_map = {
        'Facility_Name': 'name', 'Hospital_Name': 'name', 'Health_Centre_Name': 'name',
        'Facility Type': 'type', 'Facility_Type': 'type', 'Category': 'type',
        'Latitude': 'Latitude', 'Longitude': 'Longitude',
        'Beds': 'beds', 'State_Name': 'state_ut', 'District_Name': 'district'
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    if 'name' not in df.columns:
        df['name'] = 'Unknown'
    if 'type' not in df.columns:
        df['type'] = 'Unknown'

    # Estimate missing bed counts
    bed_map = {
        'Sub Centre': 6, 'SC': 6,
        'PHC': 15, 'Primary Health Centre': 15,
        'CHC': 40, 'Community Health Centre': 40,
        'Sub-District Hospital': 150, 'District Hospital': 300,
        'Civil Hospital': 250, 'General Hospital': 400,
        'Medical College': 600, 'Tertiary Hospital': 600
    }
    def estimate_beds(ftype):
        for key, val in bed_map.items():
            if pd.notna(ftype) and key.lower() in str(ftype).lower():
                return val
        return 20
    if 'beds' not in df.columns:
        df['beds'] = df['type'].apply(estimate_beds)
    else:
        df['beds'] = pd.to_numeric(df['beds'], errors='coerce').fillna(df['type'].apply(estimate_beds))

    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)
    return df

def get_color(value, q1, q2):
    if value >= q2:
        return 'red'
    if value >= q1:
        return 'orange'
    return 'green'

# ---------------- UI ----------------
st.title("🌍 Epidemic Hotspot Prediction & Resource Dashboard")
st.markdown("A unified dashboard for rural epidemic predictions, hospital capacity, and resource allocation insights.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Hotspot Analysis",
    "🏥 Hospital Integration",
    "🧩 Multi-Disease Summary",
    "🌾 Rural Access Insights",
    "📊 Resource Allocation Summary"
])

# ---------------- TAB 1: HOTSPOT ANALYSIS ----------------
with tab1:
    st.header("🔥 Hotspot Analysis (Enhanced View)")
    folders = find_disease_folders()
    if not folders:
        st.warning(f"No disease output folders found in {OUTROOT}/. Run your pipeline first.")
    else:
        diseases = [f for f, _ in folders]
        disease = st.selectbox("Select disease:", diseases, key="hotspots_disease")
        folder = os.path.join(OUTROOT, disease)
        preds = load_predictions(folder)
        if preds.empty:
            st.warning("No predictions available for this disease.")
        else:
            preds['predicted_cases'] = pd.to_numeric(preds['predicted_cases'], errors='coerce').fillna(0)
            q1, q2 = preds['predicted_cases'].quantile([0.6, 0.9])

            # --- Metrics Row ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧮 Total Predicted Cases", f"{int(preds['predicted_cases'].sum()):,}")
            c2.metric("📈 Average Cases per District", f"{preds['predicted_cases'].mean():.2f}")
            c3.metric("🔥 90th Percentile", f"{q2:.2f}")
            c4.metric("🗺️ Districts Covered", len(preds))

            # --- Layout: Map and Graphs ---
            col1, col2 = st.columns([2, 1])
            with col1:
                m = folium.Map(location=[22.97, 78.65], zoom_start=5, tiles='CartoDB dark_matter')
                for _, row in preds.iterrows():
                    lat, lon = row.get('Latitude'), row.get('Longitude')
                    if pd.isna(lat) or pd.isna(lon):
                        continue
                    color = get_color(row.get('predicted_cases', 0), q1, q2)
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=max(3, min(8, float(row.get('predicted_cases', 0)) / 5 + 3)),
                        color=color, fill=True, fill_color=color, fill_opacity=0.7,
                        popup=f"<b>{row.get('district','')}</b>, {row.get('state_ut','')}<br>Predicted: {row.get('predicted_cases',0):.2f}"
                    ).add_to(m)
                st_folium(m, width=950, height=550)

            with col2:
                st.markdown("### 📊 Case Distribution")
                fig = px.histogram(preds, x='predicted_cases', nbins=30,
                                   title='Distribution of Predicted Cases', template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 🔝 Top Hotspots")
                top = preds.sort_values('predicted_cases', ascending=False).head(TOP_K)
                fig2 = px.bar(top, x='predicted_cases', y='district', orientation='h',
                              title='Top 20 Districts', template='plotly_dark')
                st.plotly_chart(fig2, use_container_width=True)

            # --- Additional Charts ---
            st.markdown("### 📈 State-wise Aggregation")
            if 'state_ut' in preds.columns:
                state_summary = preds.groupby('state_ut')['predicted_cases'].sum().reset_index().sort_values('predicted_cases', ascending=False)
                fig3 = px.bar(state_summary, x='state_ut', y='predicted_cases',
                              title='Total Predicted Cases by State', template='plotly_dark')
                st.plotly_chart(fig3, use_container_width=True)

            st.markdown("### 🧠 Rural vs Urban Distribution")
            if 'is_rural' in preds.columns:
                rural_summary = preds.groupby('is_rural')['predicted_cases'].sum().reset_index()
                rural_summary['Category'] = rural_summary['is_rural'].map({True: 'Rural', False: 'Urban'})
                fig4 = px.pie(rural_summary, values='predicted_cases', names='Category',
                              title='Rural vs Urban Case Share', template='plotly_dark')
                st.plotly_chart(fig4, use_container_width=True)

            # --- Data Table ---
            st.markdown("### 🧾 Top Predicted Hotspots Table")
            cols = [c for c in ['state_ut', 'district', 'predicted_cases', 'is_rural'] if c in preds.columns]
            st.dataframe(preds.sort_values('predicted_cases', ascending=False)[cols].head(TOP_K), use_container_width=True)

# ---------------- TAB 2: HOSPITAL INTEGRATION ----------------
with tab2:
    st.header("🏥 Hospital Integration")
    hospitals = load_hospitals()
    if hospitals.empty:
        st.info("No hospital data found. Please verify geocode_health_centre.csv.")
    else:
        st.write(f"Loaded {len(hospitals)} hospitals.")
        state_col = 'state_ut' if 'state_ut' in hospitals.columns else None
        district_col = 'district' if 'district' in hospitals.columns else None
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            max_points = st.slider("Max markers", min_value=1000, max_value=10000, value=5000, step=500)
        with col2:
            state_val = st.selectbox("Filter state", sorted(hospitals[state_col].dropna().unique()) if state_col else [])
        with col3:
            if state_val and district_col:
                districts = hospitals[hospitals[state_col] == state_val][district_col].dropna().unique()
                district_val = st.selectbox("Filter district", sorted(districts)) if len(districts) > 0 else None
            else:
                district_val = None

        hos_view = hospitals.copy()
        if state_val:
            hos_view = hos_view[hos_view[state_col] == state_val]
        if district_val:
            hos_view = hos_view[hos_view[district_col] == district_val]
        if len(hos_view) > max_points:
            hos_view = hos_view.sample(max_points, random_state=0)
        st.write(f"Displaying {len(hos_view)} markers.")
        hmap = folium.Map(location=[22.97, 78.65], zoom_start=5)
        cluster = MarkerCluster().add_to(hmap)
        for _, h in hos_view.iterrows():
            name = h.get('name', 'Unknown')
            typ = h.get('type', 'N/A')
            beds_val = h.get('beds', 'N/A')
            folium.CircleMarker(
                location=[h['Latitude'], h['Longitude']], radius=3, color='blue', fill=True,
                fill_color='blue', popup=f"<b>{name}</b><br>Type: {typ}<br>Beds: {beds_val}"
            ).add_to(cluster)
        st_folium(hmap, width=1000, height=450)
        st.caption("🔵 Hospitals & Health Facilities (clustered and sampled for performance)")

# ---------------- TAB 3–5 (UNCHANGED) ----------------
# Keep your Multi-Disease Summary, Rural Access Insights, and Resource Allocation code exactly as you had it.

# # ---------------- SIDEBAR ----------------
# st.sidebar.markdown("---")
# st.sidebar.info("Built for Rural Epidemic Forecasting & Resource Allocation — 2025 Prototype")


# ---------------- TAB 3: MULTI-DISEASE SUMMARY ----------------
with tab3:
    st.header("🧩 Multi-Disease Vulnerability Summary")
    folders = find_disease_folders()
    records = []
    for dis, path in folders:
        preds = load_predictions(path)
        if preds.empty:
            continue
        preds['disease'] = dis
        preds['predicted_cases'] = pd.to_numeric(preds.get('predicted_cases', 0), errors='coerce').fillna(0)
        thresh = preds['predicted_cases'].quantile(0.85)
        preds['outbreak_flag'] = preds['predicted_cases'] >= thresh
        keep_cols = ['state_ut', 'district', 'is_rural', 'predicted_cases', 'outbreak_flag', 'disease']
        records.append(preds[keep_cols])
    if not records:
        st.warning("No disease predictions found.")
    else:
        all_df = pd.concat(records, ignore_index=True)
        agg = all_df.groupby(['state_ut', 'district']).agg(
            num_diseases=('disease', 'nunique'),
            mean_predicted_cases=('predicted_cases', 'mean')
        ).reset_index().sort_values('num_diseases', ascending=False)
        st.dataframe(agg.head(200), use_container_width=True)

# ---------------- TAB 4: RURAL ACCESS INSIGHTS ----------------
with tab4:
    st.header("🌾 Rural Access to Healthcare and Relief (World Bank Survey)")
    try:
        data_dir = os.path.join(SCRIPT_DIR, "IND_2020_COVIDRS_v01_M_CSV")
        wb1 = pd.read_csv(os.path.join(data_dir, "wb1_cleaned_dataset.csv"), low_memory=False, encoding='latin1')
        wb2 = pd.read_csv(os.path.join(data_dir, "wb2_cleaned_dataset_09_24.csv"), low_memory=False, encoding='latin1')
        wb3 = pd.read_csv(os.path.join(data_dir, "wb3_cleaned_dataset_09_24.csv"), low_memory=False, encoding='latin1')
        wb_all = pd.concat([wb1, wb2, wb3], ignore_index=True)
        st.success(f"Loaded World Bank Rural Access datasets: {len(wb_all)} records.")
    except Exception as e:
        st.error(f"Error loading WB datasets: {e}")
        wb_all = pd.DataFrame()

    if not wb_all.empty:
        possible_state_cols = [c for c in wb_all.columns if "state" in c.lower()]
        possible_district_cols = [c for c in wb_all.columns if "district" in c.lower()]
        possible_access_cols = [c for c in wb_all.columns if "access" in c.lower() or "relief" in c.lower() or "health" in c.lower()]

        st.markdown("### 📋 Detected Columns")
        st.write({
            "State column": possible_state_cols[:1],
            "District column": possible_district_cols[:1],
            "Access/Relief columns": possible_access_cols[:5]
        })

        state_col = possible_state_cols[0] if possible_state_cols else None
        access_col = possible_access_cols[0] if possible_access_cols else None

        if state_col and access_col:
            access_summary = (
                wb_all.groupby(state_col)[access_col]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            st.markdown("### 📊 Average Access by State")
            st.dataframe(access_summary, use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(access_summary[state_col], access_summary[access_col])
            ax.set_xlabel(f"Average {access_col}")
            ax.set_title("Average Reported Access by State")
            st.pyplot(fig)

# ---------------- TAB 5: RESOURCE ALLOCATION ----------------
# ---------------- TAB 5: RESOURCE ALLOCATION ----------------
with tab5:
    st.header("📊 Integrated Resource Allocation Summary")

    try:
        resource_df = pd.read_csv(RESOURCE_DATA)
        st.success(f"Loaded resource allocation dataset with {len(resource_df)} facilities.")
    except Exception as e:
        st.error(f"Error loading hospital_directory.csv: {e}")
        resource_df = pd.DataFrame()

    # --- Step 1: Normalize district column naming ---
    if not resource_df.empty:
        possible_district_cols = [c for c in resource_df.columns if 'district' in c.lower()]
        if possible_district_cols:
            resource_df.rename(columns={possible_district_cols[0]: 'district'}, inplace=True)
            st.info(f"✅ Using '{possible_district_cols[0]}' as district column.")
        else:
            st.warning("⚠️ No district column found in hospital_directory.csv — please include one.")
            resource_df['district'] = None

        # Handle missing bed column
        if 'beds' not in resource_df.columns:
            st.warning("⚠️ No 'beds' column found. Estimating as 20 per facility.")
            resource_df['beds'] = 20

    # --- Step 2: Combine with predictions ---
    all_preds = []
    for dis, path in find_disease_folders():
        preds = load_predictions(path)
        if not preds.empty:
            preds['disease'] = dis
            preds['predicted_cases'] = pd.to_numeric(preds.get('predicted_cases', 0), errors='coerce').fillna(0)
            all_preds.append(preds[['state_ut', 'district', 'predicted_cases', 'disease']])

    if all_preds:
        preds_combined = pd.concat(all_preds, ignore_index=True)
        preds_summary = preds_combined.groupby(['state_ut', 'district']).agg(
            total_predicted_cases=('predicted_cases', 'sum'),
            num_diseases=('disease', 'nunique')
        ).reset_index()
    else:
        preds_summary = pd.DataFrame(columns=['state_ut', 'district', 'total_predicted_cases', 'num_diseases'])

    # --- Step 3: Merge predictions + hospital data ---
    if not resource_df.empty and not preds_summary.empty:
        # Normalize naming
        resource_df['district'] = resource_df['district'].astype(str).str.strip().str.lower()
        preds_summary['district'] = preds_summary['district'].astype(str).str.strip().str.lower()

        merged = preds_summary.merge(
            resource_df.groupby('district', as_index=False)['beds'].sum(),
            on='district', how='left'
        )
        merged['beds'] = merged['beds'].fillna(0)

        # Compute strain ratios
        merged['strain_ratio'] = np.where(
            merged['beds'] > 0, merged['total_predicted_cases'] / merged['beds'], np.nan
        )
        merged['strain_level'] = pd.cut(
            merged['strain_ratio'],
            bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        st.markdown("### 🧠 District-Wise Resource Allocation")
        st.dataframe(
            merged[['state_ut', 'district', 'total_predicted_cases', 'beds', 'strain_ratio', 'strain_level']],
            use_container_width=True
        )

        # Save final output
        merged.to_csv(os.path.join(SCRIPT_DIR, "resource_allocation_summary.csv"), index=False)
        st.success("💾 Saved merged resource allocation summary as resource_allocation_summary.csv")

        st.markdown("### 🔍 Legend")
        st.write("""
        ✅ **Low strain** — Sufficient hospital capacity  
        ⚠️ **Medium** — Moderate pressure  
        🚨 **High** — Possible strain  
        🟥 **Critical** — Urgent attention required
        """)
    else:
        st.warning("⚠️ Either predictions or hospital resource data is missing; unable to compute allocation.")


# ---------------- SIDEBAR ----------------
st.sidebar.markdown("---")
st.sidebar.info("Built for Rural Epidemic Forecasting & Resource Allocation — 2025 Prototype")

# ---------------- SIDEBAR: SMS TRIGGER ----------------
import subprocess
import streamlit as st
import os

st.sidebar.markdown("---")
st.sidebar.markdown("### 📩 Send Location-Based Alerts")

# Get absolute path to sms_alerts.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SMS_SCRIPT = os.path.join(SCRIPT_DIR, "sms_alerts.py")

if st.sidebar.button("Send Message Now"):
    st.sidebar.info("📤 Triggering SMS alerts... please wait.")

    if not os.path.exists(SMS_SCRIPT):
        st.sidebar.error(f"❌ sms_alerts.py not found at:\n{SMS_SCRIPT}")
    else:
        try:
            process = subprocess.Popen(
                ["python", SMS_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=SCRIPT_DIR,  # make sure correct folder is used
            )
            stdout, stderr = process.communicate()

            # Ensure None doesn’t cause crash
            stdout = stdout or ""
            stderr = stderr or ""

            if stdout.strip():
                st.sidebar.text_area("📜 SMS Log Output", stdout, height=250)

            if stderr.strip():
                st.sidebar.error(f"⚠️ Error:\n{stderr}")
            elif not stdout.strip():
                st.sidebar.warning("⚠️ sms_alerts.py ran but didn't print any output.")
            else:
                st.sidebar.success("✅ SMS alert script completed successfully!")
        except Exception as e:
                st.sidebar.error(f"❌ Failed to execute sms_alerts.py: {e}")
