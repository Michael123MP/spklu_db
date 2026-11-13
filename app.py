import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely.geometry import Point

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="WebGIS Persebaran SPKLU Jakarta",
    layout="wide",
)

@st.cache_data
def load_data():
    df_kel = pd.read_csv('spklu_jakarta.csv')
    try:
        df_points = pd.read_csv('spklu_titik_individu.csv')
    except:
        df_points = df_kel.copy() # Fallback jika file titik terpisah belum ada
    return df_kel, df_points

df_cleaned_final, df_spklu_points = load_data()

min_gap = df_cleaned_final['gap_score'].min()
median_gap = df_cleaned_final['gap_score'].median()
max_gap = df_cleaned_final['gap_score'].max()
mean_gap = df_cleaned_final['gap_score'].mean()

# --- INISIALISASI SESSION STATE ---
if 'selected_kel' not in st.session_state:
    st.session_state.selected_kel = df_cleaned_final.iloc[0]['kelurahan']
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0  # 0: Eksplorasi, 1: Detail Wilayah

# --- HEADER COMMAND CENTER ---
st.markdown("""
    <div style="background: linear-gradient(90deg, #0e3d24 0%, #13221e 100%); padding: 15px 25px; border-radius: 8px; color: white; display: flex; justify-content: space-between; align-items: center; border: 1px solid #1f3d33;">
        <div>
            <h2 style="margin: 0; color: #4ade80; font-size: 22px;">⚡ SPKLU-GIS Jakarta Command Center</h2>
            <p style="margin: 0; font-size: 12px; color: #94a3b8;">Spatial Decision Support System • Robust Spatial Click Handler</p>
        </div>
        <div style="font-size: 13px; background: #0b1311; padding: 8px 15px; border-radius: 20px; border: 1px solid #22c55e;">
             <b>System Status:</b> Live (Spatial Point-in-Polygon)
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- TOP METRICS ---
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric('📉 MIN GAP', f'{min_gap:.2f}', 'Aman')
col_m2.metric('📈 MEDIAN (Threshold)', f'{median_gap:.2f}', 'Batas Normal')
col_m3.metric('⚡ MAX GAP', f'{max_gap:,.2f}', 'Prioritas Darurat!')

st.markdown('---')

# --- LAYOUT UTAMA 3 KOLOM ---
col_left, col_center, col_right = st.columns([1.2, 2.2, 1.3])

# --- KOLOM KIRI: 2 MENU UTAMA (EKSPLORASI & DETAIL WILAYAH) ---
with col_left:
    st.subheader('📑 Panel Navigasi Wilayah')

    tabs_label = ["1. Eksplorasi Kelurahan", "2. Detail & Fitur Wilayah"]
    selected_tab_label = st.radio(
        "Menu Navigasi", 
        tabs_label, 
        index=st.session_state.active_tab, 
        horizontal=True, 
        label_visibility="collapsed"
    )
    st.session_state.active_tab = tabs_label.index(selected_tab_label)
    st.markdown("---")

    # ==========================================
    # TAB 1: EKSPLORASI KELURAHAN
    # ==========================================
    if st.session_state.active_tab == 0:
        st.subheader('🔍 Cari Kelurahan')
        search_box = st.text_input('Ketik nama kelurahan...')

        df_filtered = df_cleaned_final.copy()
        if search_box:
            df_filtered = df_filtered[df_filtered['kelurahan'].str.contains(search_box, case=False, na=False)]

        with st.container(height=400):
            for idx, row in df_filtered.iterrows():
                nama_kel = row.get('kelurahan', 'Unknown')
                gap_val = row.get('gap_score', 0)
                badge = '🔴' if gap_val > median_gap else '🟢'

                if st.button(f"{badge} {nama_kel} (Gap: {gap_val:.1f})", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.selected_kel = nama_kel
                    st.session_state.active_tab = 1  # Pindah otomatis ke Tab Detail Wilayah
                    st.rerun()

    # ==========================================
    # TAB 2: DETAIL & FITUR WILAYAH KELURAHAN
    # ==========================================
    elif st.session_state.active_tab == 1:
        st.subheader('📌 Detail Wilayah')

        matched_rows = df_cleaned_final[df_cleaned_final['kelurahan'].str.lower() == str(st.session_state.selected_kel).lower()]
        if matched_rows.empty:
            matched_rows = df_cleaned_final.iloc[[0]]
        selected_row = matched_rows.iloc[0]

        nama_wilayah = selected_row.get('kelurahan', 'Unknown')
        kec_name = selected_row.get('kecamatan_right', selected_row.get('kecamatan', '-'))
        gap_val = selected_row.get('gap_score', 0)
        status_label = "Prioritas Tinggi" if gap_val > median_gap else "Normal / Aman"
        status_color = "#ef4444" if gap_val > median_gap else "#22c55e"

        st.markdown(f"""
            <div style="background-color: #13221e; border: 1px solid #1f3d33; padding: 12px; border-radius: 8px; margin-bottom: 10px; color: #f8fafc;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #4ade80; font-size: 15px;">{nama_wilayah}</h4>
                    <span style="background-color: {status_color}22; color: {status_color}; border: 1px solid {status_color}; padding: 2px 8px; border-radius: 12px; font-size: 10px; font-weight: bold;">{status_label}</span>
                </div>
                <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">Kecamatan: {kec_name}</div>
                <hr style="margin: 8px 0; border-color: #1f3d33;">
                <div style="font-size: 11px; color: #94a3b8; font-weight: bold;">GAP SCORE KELURAHAN</div>
                <h2 style="color: #ffffff; margin: 2px 0 0 0;">{gap_val:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<p style="font-size: 11px; font-weight: bold; margin-bottom: 2px; color: #cbd5e1;">📊 Perbandingan vs Rata-rata Jakarta</p>', unsafe_allow_html=True)
        df_comp = pd.DataFrame({
            'Kategori': [nama_wilayah, 'Rata-rata Jakarta'],
            'Gap Score': [gap_val, mean_gap]
        })
        st.bar_chart(df_comp.set_index('Kategori'), height=100)

        st.markdown('<p style="font-size: 11px; font-weight: bold; margin-top: 6px; margin-bottom: 2px; color: #cbd5e1;">🔍 Variabel Fitur (X) Khusus Wilayah Ini:</p>', unsafe_allow_html=True)

        # Filter ketat hanya kolom numerik fitur X hasil feature engineering
        exclude_cols = ['kelurahan', 'kecamatan', 'kecamatan_right', 'latitude', 'longitude', 'geometry', 'gap_score', 'Unnamed: 0', 'fid', 'kel_desa']
        feature_cols = [c for c in df_cleaned_final.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_cleaned_final[c])]

        feat_data = {col: selected_row[col] for col in feature_cols if col in selected_row}
        if feat_data:
            df_feat = pd.DataFrame(list(feat_data.items()), columns=['Variabel Fitur (X)', 'Nilai'])
            st.dataframe(df_feat, use_container_width=True, hide_index=True, height=160)

# --- KOLOM TENGAH: PETA DUAL-LAYER & SPATIAL CLICK HANDLER ---
with col_center:
    st.subheader('🗺️ Peta Wilayah & Pop-up Atribut SPKLU')

    try:
        gdf = gpd.read_file('Provinsi_DKI_Jakarta-KEL_DESA.geojson') 
        df_clean_to_merge = df_cleaned_final.copy()
        if 'geometry' in df_clean_to_merge.columns:
            df_clean_to_merge = df_clean_to_merge.drop(columns=['geometry'])

        merge_col = 'kel_desa' if 'kel_desa' in gdf.columns else ('kelurahan' if 'kelurahan' in gdf.columns else gdf.columns[0])
        gdf_merged = gdf.merge(df_clean_to_merge, left_on=merge_col, right_on='kelurahan', how='inner')

        if 'geometry' in gdf_merged.columns:
            gdf_merged = gpd.GeoDataFrame(gdf_merged, geometry='geometry')
        elif 'geometry_x' in gdf_merged.columns:
            gdf_merged = gdf_merged.rename(columns={'geometry_x': 'geometry'})
            gdf_merged = gdf_merged.set_geometry('geometry')

        gdf_merged = gdf_merged.dropna(subset=['geometry'])

        # Inisialisasi Peta
        m = folium.Map(
            location=[-6.2088, 106.8456], zoom_start=11, tiles='OpenStreetMap'
        )

        # Layer 1: Poligon Wilayah Kelurahan
        def style_function(feature):
            props = feature['properties']
            kel_name = props.get('kelurahan') or props.get('kel_desa') or props.get('KELURAHAN') or props.get('NAMOBJ')
            gap = props.get('gap_score', 0)
            color_fill = '#22c55e' if gap <= median_gap else '#ef4444'

            is_selected = (str(kel_name).strip().lower() == str(st.session_state.selected_kel).strip().lower())

            return {
                'fillColor': color_fill,
                'color': '#000000' if is_selected else '#ffffff',
                'weight': 3 if is_selected else 1,
                'fillOpacity': 0.65 if is_selected else 0.45
            }

        folium.GeoJson(
            gdf_merged,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['kelurahan', 'gap_score'] if 'kelurahan' in gdf_merged.columns else [gdf_merged.columns[0], 'gap_score'],
                aliases=['Kelurahan:', 'Gap Score:'],
                localize=True,
                sticky=True,
                direction="auto",
                style="background: #0b1311; color: #ffffff; font-family: sans-serif; font-size: 12px; padding: 6px; border-radius: 4px;"
            )
        ).add_to(m)

        # Layer 2: Titik SPKLU Individu dengan Pop-up HTML Lengkap
        for idx, row in df_spklu_points.iterrows():
            lat = row.get('latitude')
            lon = row.get('longitude')

            if pd.notna(lat) and pd.notna(lon):
                html_content = f"""
                <div style="font-family: sans-serif; font-size: 11px; width: 260px; max-height: 250px; overflow-y: auto;">
                    <b style="color: #0e3d24; font-size: 13px;">⚡ {row.get('nama_spklu', 'Stasiun SPKLU')}</b><hr style="margin: 4px 0;">
                    <table style="width: 100%; border-collapse: collapse;">
                """
                for col_name, val in row.items():
                    if col_name not in ['geometry', 'latitude', 'longitude']:
                        html_content += f"""
                        <tr style="border-bottom: 1px solid #eee;">
                            <td style="padding: 3px; color: #555;"><b>{col_name}</b></td>
                            <td style="padding: 3px; color: #111; text-align: right;">{val}</td>
                        </tr>
                        """
                html_content += "</table></div>"

                popup_obj = folium.Popup(html_content, max_width=300)

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color='#ffffff',
                    weight=1.5,
                    fill=True,
                    fill_color='#f59e0b',
                    fill_opacity=0.9,
                    popup=popup_obj,
                    tooltip=f"Klik untuk lihat detail: {row.get('nama_spklu', 'SPKLU')}"
                ).add_to(m)

        # Render Peta dan tangkap koordinat klik menggunakan st_folium
        map_output = st_folium(m, width='100%', height=550, returned_objects=['last_clicked'])

        # Tangani klik peta menggunakan Point-in-Polygon (Analisis Spasial)
        if map_output and map_output.get('last_clicked'):
            clicked_coords = map_output['last_clicked']
            c_lat = clicked_coords['lat']
            c_lon = clicked_coords['lng']
            click_point = Point(c_lon, c_lat)  # Format Shapely: (Longitude, Latitude)

            # Cari poligon kelurahan mana yang mengandung titik klik tersebut
            found_kel = None
            for _, poly_row in gdf_merged.iterrows():
                geom = poly_row.get('geometry')
                if geom and geom.contains(click_point):
                    found_kel = poly_row.get('kelurahan')
                    break

            if found_kel and found_kel != st.session_state.selected_kel:
                st.session_state.selected_kel = found_kel
                st.session_state.active_tab = 1  # Pindah otomatis ke Tab Detail Wilayah
                st.rerun()

    except Exception as e:
        st.warning(f"Gagal memuat peta multi-layer. Detail error: {e}")

# --- KOLOM KANAN: PERFORMA MODEL ---
with col_right:
    st.subheader('📊 Performa Model')

    df_perf = pd.DataFrame({
        "Model": ["XGBoost", "RandomForest", "DecisionTree", "LinearReg"],
        "R² Score": ["0.725", "0.716", "-0.076", "0.453"],
        "RMSE": ["0.028", "0.029", "0.056", "0.039"]
    })
    st.dataframe(df_perf, use_container_width=True, hide_index=True)

    st.markdown('---')
    st.subheader('📈 Distribusi Gap Score Global')
    st.bar_chart(df_cleaned_final['gap_score'], height=220)
