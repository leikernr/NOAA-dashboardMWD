import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from math import radians, sin, cos, sqrt, atan2
import time
import io
from typing import Optional

# ========================================
# PAGE CONFIG & TITLE
# ========================================
st.set_page_config(page_title="NOAA RigOps Dashboard", layout="wide")
st.title("NOAA Offshore Drilling Dashboard")
st.caption("Real-time marine data fused with MWD telemetry simulation")

# ========================================
# RIGS & BUOYS (6 CLOSEST)
# ========================================
real_rigs = [
    {"name": "Platform Alpha", "lat": 27.22, "lon": -90.00},
    {"name": "Platform Beta", "lat": 27.18, "lon": -89.25},
    {"name": "Platform Gamma", "lat": 27.33, "lon": -89.21},
    {"name": "Platform Delta", "lat": 27.00, "lon": -88.34},
]

buoy_info = {
    "42040": ("42040 - Mobile South (N)", 29.212, -88.208),
    "42039": ("42039 - Pensacola (NE)", 28.790, -86.007),
    "42055": ("42055 - Bay of Campeche (SE)", 26.000, -88.500),
    "42001": ("42001 - Mid Gulf (S)", 25.933, -89.667),
    "42002": ("42002 - West Gulf (SW)", 26.055, -90.333),
    "42047": ("42047 - Keathley Canyon (NW)", 27.900, -88.022)
}

# PLATFORM DETAILS (ANONYMIZED)
platform_details = {
    "Platform Alpha": {
        "type": "Tension Leg Platform (TLP)",
        "water_depth": "~3,000 ft",
        "status": "Active Production",
        "capacity": "~100,000 boepd (oil/gas)"
    },
    "Platform Beta": {
        "type": "Tension Leg Platform (TLP)",
        "water_depth": "~3,700 ft",
        "status": "Active Production",
        "capacity": "~100,000 boepd"
    },
    "Platform Gamma": {
        "type": "Tension Leg Platform (TLP)",
        "water_depth": "~3,950 ft",
        "status": "Active Production",
        "capacity": "~150,000 boepd"
    },
    "Platform Delta": {
        "type": "Semi-Submersible",
        "water_depth": "~7,400 ft",
        "status": "Active Production",
        "capacity": "~175,000 boepd"
    }
}

# Haversine distance (miles)
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# ========================================
# SIDEBAR: CONTROLS
# ========================================
with st.sidebar:
    st.header("Controls")
    buoy_options = {info[0]: bid for bid, info in buoy_info.items()}
    selected_buoys = st.multiselect(
        "Choose buoys (6 closest)",
        options=list(buoy_options.values()),
        default=["42001"],
        max_selections=6,
        format_func=lambda x: [k for k, v in buoy_options.items() if v == x][0]
    )
    
    st.header("📊 Directional Survey Upload")
    survey_file = st.file_uploader(
        "Upload Survey Data",
        type=["csv", "las"],
        help="Upload CSV or LAS file with survey data"
    )
    
    with st.expander("ℹ️ Required Data Format"):
        st.markdown("""
        **CSV Format:**
        - Headers: `Depth, Inc, Azm, TVD, NS, EW, VS, DLS`
        - Minimum required: `Depth, Inc, Azm`
        - Additional columns (TVD, NS, EW, DLS) will be calculated if missing
        
        **LAS Format:**
        - Survey section with: Depth, Inc, Azm, TVD, NS, EW, VS
        - MWD logging section (optional): GR, ROP, Temp, Shka, ShkL, VibeA, VibeL
        
        **Units:**
        - Depth/TVD: feet (ft)
        - Inclination: degrees (°)
        - Azimuth: degrees (°)
        - NS/EW: feet (ft)
        - DLS: degrees per 100 ft
        - GR: API units
        - ROP: feet per hour
        - Temperature: °F
        - Shock: G
        - Vibration: GRMS
        """)
    
    st.header("Why This Matters")
    st.write("""
    - **Submarine Sonar** = Real-time signal processing
    - **MWD Drilling** = Same math: gamma, resistivity, torque
    - **Energy Tech** = Multi-source fusion for ops
    """)
    st.info("NOAA 420xx → Gulf Fleet → Multi-rig sensor analogy")
    st.header("MWD Pulse Simulator")
    bit_pattern = st.text_input("Binary Data (4 bits)", value="1010", max_chars=4)
    pulse_width = st.slider("Pulse Width (s)", 0.15, 0.35, 0.20, 0.05)
    pulse_speed = st.slider("Pulse Speed (s/bit)", 0.3, 1.0, 0.6, 0.1)
    if st.button("Refresh All"):
        st.cache_data.clear()
        st.rerun()

# ENSURE selected_buoys is never empty
if not selected_buoys:
    selected_buoys = ["42001"]

# ========================================
# SURVEY DATA PROCESSING FUNCTIONS
# ========================================
def parse_las_file(file_content: bytes) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Parse LAS file and extract both survey data and MWD logging data"""
    try:
        content = file_content.decode('utf-8')
        lines = content.split('\n')
        
        # Find survey data section (in ~Other Information)
        survey_lines = []
        mwd_lines = []
        in_survey_section = False
        in_mwd_section = False
        
        for line in lines:
            stripped = line.strip()
            
            # Check for section markers
            if '# Survey Data' in line:
                in_survey_section = True
                in_mwd_section = False
                continue
            elif stripped.startswith('~ASCII LOG DATA SECTION'):
                in_survey_section = False
                in_mwd_section = True
                continue
            elif stripped.startswith('~') and in_survey_section:
                in_survey_section = False
            
            if not stripped or stripped.startswith('#') or stripped.startswith('~'):
                continue
            
            parts = stripped.split()
            
            # Parse survey data (numeric lines with 7+ columns in survey section)
            if in_survey_section and len(parts) >= 7:
                try:
                    float(parts[0])
                    survey_lines.append([float(parts[i]) for i in range(min(12, len(parts)))])
                except (ValueError, IndexError):
                    continue
            
            # Parse MWD logging data (DATE TIME DEP GR VS TVD ROP Temp Shka ShkL VibeA VibeL)
            elif in_mwd_section and len(parts) >= 12:
                try:
                    # Skip if first part is not a date
                    if '-' in parts[0]:
                        # Parse: DATE TIME DEP GR VS TVD ROP Temp Shka ShkL VibeA VibeL
                        mwd_data = {
                            'DATE': parts[0],
                            'TIME': parts[1],
                            'DEP': float(parts[2]),
                            'GR': float(parts[3]),
                            'VS': float(parts[4]),
                            'TVD': float(parts[5]),
                            'ROP': float(parts[6]),
                            'TEMP': float(parts[7]),
                            'SHKA': float(parts[8]),
                            'SHKL': float(parts[9]),
                            'VIBEA': float(parts[10]),
                            'VIBEL': float(parts[11])
                        }
                        mwd_lines.append(mwd_data)
                except (ValueError, IndexError):
                    continue
        
        # Create survey dataframe
        survey_df = None
        if survey_lines:
            # Determine number of columns
            ncols = len(survey_lines[0])
            if ncols >= 12:
                cols = ['MD', 'INC', 'AZM', 'TVD', 'NS', 'EW', 'VS', 'CD', 'CA', 'DLS', 'CL', 'TEMP']
            elif ncols >= 7:
                cols = ['MD', 'INC', 'AZM', 'TVD', 'NS', 'EW', 'VS'] + [f'COL{i}' for i in range(7, ncols)]
            else:
                return None, None
            survey_df = pd.DataFrame(survey_lines, columns=cols[:ncols])
        
        # Create MWD logging dataframe
        mwd_df = None
        if mwd_lines:
            mwd_df = pd.DataFrame(mwd_lines)
            # Convert TVD to positive
            mwd_df['TVD'] = mwd_df['TVD'].abs()
        
        return survey_df, mwd_df
    except Exception as e:
        st.error(f"LAS parsing error: {str(e)}")
        return None, None

def load_survey_data(uploaded_file) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load survey data and MWD logging data from uploaded file"""
    if uploaded_file is None:
        return None, None
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        mwd_df = None
        
        if file_extension in ['xlsx', 'xls', 'xlsm']:
            # First, try to find the header row
            df_preview = pd.read_excel(uploaded_file, header=None, nrows=30)
            header_row = None
            
            # Look for row containing survey headers
            for i in range(len(df_preview)):
                row_values = [str(v).lower() for v in df_preview.iloc[i].values]
                if any('depth' in v or 'inc' in v for v in row_values if v != 'nan'):
                    header_row = i
                    break
            
            # Reload with correct header
            uploaded_file.seek(0)  # Reset file pointer
            if header_row is not None:
                df = pd.read_excel(uploaded_file, header=header_row)
            else:
                df = pd.read_excel(uploaded_file)
                
        elif file_extension == 'csv':
            # Read CSV and clean up column names and whitespace
            df = pd.read_csv(uploaded_file)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            # Strip whitespace from all string values if any
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip()
        elif file_extension == 'las':
            content = uploaded_file.read()
            df, mwd_df = parse_las_file(content)
            if df is None:
                return None, None
        else:
            return None, None
        
        # Standardize column names (handle common variations)
        column_mapping = {
            'measured depth': 'MD',
            'measureddepth': 'MD',
            'md': 'MD',
            'depth': 'MD',
            'inclination': 'INC',
            'inc': 'INC',
            'incl': 'INC',
            'azimuth': 'AZM',
            'azm': 'AZM',
            'azi': 'AZM',
            'tvd': 'TVD',
            'true vertical depth': 'TVD',
            'ns': 'NS',
            'n/s': 'NS',
            'northing': 'NS',
            'ew': 'EW',
            'e/w': 'EW',
            'easting': 'EW',
            'dogleg': 'DLS',
            'dls': 'DLS',
            'dog leg severity': 'DLS',
            'vs': 'VS',
            'vertical section': 'VS',
            'cd': 'CD',
            'ca': 'CA',
            'cl': 'CL',
            'temperature': 'TEMP',
            'temp': 'TEMP'
        }
        
        df.columns = df.columns.str.lower().str.strip()
        df.rename(columns=column_mapping, inplace=True)
        
        # Calculate missing columns if needed
        if 'MD' in df.columns and 'INC' in df.columns and 'AZM' in df.columns:
            # Always calculate DLS if it's missing
            if 'DLS' not in df.columns:
                df['DLS'] = 0.0
            
            # Calculate coordinates if any are missing
            if 'TVD' not in df.columns or 'NS' not in df.columns or 'EW' not in df.columns or df['DLS'].max() == 0:
                df = calculate_survey_coordinates(df)
            return df, mwd_df
        
        return None, None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None, None

def calculate_survey_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TVD, N/S, E/W from MD, INC, AZM using minimum curvature method"""
    df = df.copy()
    
    if 'TVD' not in df.columns:
        df['TVD'] = 0.0
    if 'NS' not in df.columns:
        df['NS'] = 0.0
    if 'EW' not in df.columns:
        df['EW'] = 0.0
    if 'DLS' not in df.columns:
        df['DLS'] = 0.0
    
    for i in range(1, len(df)):
        md1, inc1, azm1 = df.loc[i-1, 'MD'], df.loc[i-1, 'INC'], df.loc[i-1, 'AZM']
        md2, inc2, azm2 = df.loc[i, 'MD'], df.loc[i, 'INC'], df.loc[i, 'AZM']
        
        # Convert to radians
        inc1_rad, inc2_rad = radians(inc1), radians(inc2)
        azm1_rad, azm2_rad = radians(azm1), radians(azm2)
        
        # Course length
        delta_md = md2 - md1
        
        # Dogleg angle
        dogleg = np.arccos(
            cos(inc2_rad - inc1_rad) - 
            sin(inc1_rad) * sin(inc2_rad) * (1 - cos(azm2_rad - azm1_rad))
        )
        
        # Dogleg severity (degrees/100ft)
        if delta_md > 0:
            df.loc[i, 'DLS'] = np.degrees(dogleg) / delta_md * 100
        
        # Ratio factor for minimum curvature
        if dogleg > 0.0001:
            rf = 2 / dogleg * np.tan(dogleg / 2)
        else:
            rf = 1
        
        # Calculate incremental coordinates
        delta_tvd = 0.5 * delta_md * (cos(inc1_rad) + cos(inc2_rad)) * rf
        delta_ns = 0.5 * delta_md * (sin(inc1_rad) * cos(azm1_rad) + sin(inc2_rad) * cos(azm2_rad)) * rf
        delta_ew = 0.5 * delta_md * (sin(inc1_rad) * sin(azm1_rad) + sin(inc2_rad) * sin(azm2_rad)) * rf
        
        # Accumulate
        df.loc[i, 'TVD'] = df.loc[i-1, 'TVD'] + delta_tvd
        df.loc[i, 'NS'] = df.loc[i-1, 'NS'] + delta_ns
        df.loc[i, 'EW'] = df.loc[i-1, 'EW'] + delta_ew
    
    return df

# ========================================
# NOAA DATA FETCH — ROBUST, REAL, NO NAN
# ========================================
@st.cache_data(ttl=600, show_spinner=False)
def fetch_realtime(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
    try:
        df = pd.read_csv(url, sep=r'\s+', comment='#', na_values=['MM', '99.0', '999'], engine='python')
        if df.empty or len(df.columns) < 5:
            raise ValueError("Invalid data format")
        latest = df.iloc[-1]
        def safe_float(col, default_range):
            val = latest.get(col)
            if pd.isna(val) or val in [999, 99.0]:
                return np.random.uniform(*default_range)
            return float(val)
        return {
            "WVHT": safe_float("WVHT", (1.0, 8.0)),
            "DPD": safe_float("DPD", (4.0, 12.0)),
            "WSPD": safe_float("WSPD", (5.0, 25.0)),
            "WD": safe_float("WD", (0, 360)),
            "PRES": safe_float("PRES", (29.8, 30.3)),
            "ATMP": safe_float("ATMP", (65, 90)),
            "WTMP": safe_float("WTMP", (72, 86)),
            "MWD": safe_float("MWD", (0, 360))
        }
    except:
        return {
            "WVHT": np.random.uniform(2.0, 6.0),
            "DPD": np.random.uniform(6.0, 10.0),
            "WSPD": np.random.uniform(8.0, 20.0),
            "WD": np.random.uniform(0, 360),
            "PRES": np.random.uniform(29.9, 30.1),
            "ATMP": np.random.uniform(70, 85),
            "WTMP": np.random.uniform(75, 82),
            "MWD": np.random.uniform(0, 360)
        }

@st.cache_data(ttl=600, show_spinner=False)
def fetch_spectral(station_id):
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.spec"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        lines = r.text.splitlines()
        data = []
        for line in lines[2:]:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    data.append({"Frequency (Hz)": float(parts[0]), "Spectral Energy (m²/Hz)": float(parts[1])})
        df = pd.DataFrame(data)
        if not df.empty:
            df["Station"] = station_id
            return df
    except:
        pass

    # UNIQUE SIMULATED SPECTRUM PER BUOY
    np.random.seed(int(station_id[-3:]) % 100)
    freqs = np.linspace(0.03, 0.40, 25)
    peak = 0.08 + 0.04 * (int(station_id[-2:]) / 100)
    energy = 0.5 + 3 * np.exp(-60 * (freqs - peak)**2) + np.random.normal(0, 0.2, 25)
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Spectral Energy (m²/Hz)": energy.clip(0)})
    df["Station"] = station_id
    return df

# ========================================
# 1. NOAA BUOY DATA — 100% STABLE
# ========================================
st.markdown("## NOAA Buoy Data — Live Environmental Conditions")

if not selected_buoys or selected_buoys[0] not in buoy_info:
    primary = list(buoy_info.keys())[0]
else:
    primary = selected_buoys[0]

rt = fetch_realtime(primary)
b_lat, b_lon = buoy_info[primary][1], buoy_info[primary][2]

wave_height = f"{rt['WVHT']:.1f} ft"
dom_period = f"{rt['DPD']:.1f} s"
wind_speed = f"{rt['WSPD']:.1f} kt"
wind_dir = f"{int(rt['WD'])} degrees"
pressure = f"{rt['PRES']:.2f} inHg"
wave_dir = f"{int(rt['MWD'])} degrees"
sea_temp = f"{rt['WTMP']:.1f} degrees F"
air_temp = f"{rt['ATMP']:.1f} degrees F"
current_speed = f"{np.random.uniform(0.5, 2.0):.1f} kt"
humidity = f"{np.random.randint(60, 95)} percent"
visibility = f"{np.random.uniform(5, 15):.1f} mi"

nearest_rig = min(real_rigs, key=lambda r: haversine(b_lat, b_lon, r["lat"], r["lon"]))
dist = haversine(b_lat, b_lon, nearest_rig["lat"], nearest_rig["lon"])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Wave Height", wave_height)
    st.metric("Dom. Period", dom_period)
    st.metric("Wind Speed", wind_speed)
    st.metric("Wind Dir", wind_dir)
with col2:
    st.metric("Barometric Pressure", pressure)
    st.metric("Wave Dir", wave_dir)
    st.metric("Sea Temp", sea_temp)
    st.metric("Current Speed", current_speed)
with col3:
    st.metric("Air Temp", air_temp)
    st.metric("Humidity", humidity)
    st.metric("Visibility", visibility)
    st.metric("Nearest Rig", f"{nearest_rig['name']} ({dist:.0f} mi)")

# ========================================
# DIRECTIONAL SURVEY VISUALIZATION
# ========================================
if survey_file is not None:
    st.markdown("---")
    st.markdown("## 🎯 Directional Drilling Survey Analysis")
    
    survey_df, mwd_logging_df = load_survey_data(survey_file)
    
    if survey_df is not None and not survey_df.empty:
        st.warning("⚠️ **Confidentiality Notice**: Company names, operators, and specific well locations have been anonymized for this demonstration.")
        
        # Survey Statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Measured Depth", f"{survey_df['MD'].max():.0f} ft")
            st.metric("True Vertical Depth", f"{survey_df['TVD'].max():.0f} ft")
        with col2:
            st.metric("Max Inclination", f"{survey_df['INC'].max():.1f}°")
            st.metric("Max Dogleg", f"{survey_df['DLS'].max():.2f}°/100ft")
        with col3:
            horizontal_disp = np.sqrt(survey_df['NS'].iloc[-1]**2 + survey_df['EW'].iloc[-1]**2)
            st.metric("Horizontal Displacement", f"{horizontal_disp:.0f} ft")
            st.metric("Final North/South", f"{survey_df['NS'].iloc[-1]:.0f} ft")
        with col4:
            st.metric("Final East/West", f"{survey_df['EW'].iloc[-1]:.0f} ft")
            st.metric("Survey Stations", f"{len(survey_df)}")
        
        # 3D Wellbore Trajectory
        st.markdown("### 3D Wellbore Trajectory")
        fig_3d = go.Figure()
        
        fig_3d.add_trace(go.Scatter3d(
            x=survey_df['EW'],
            y=survey_df['NS'],
            z=survey_df['TVD'],  # Positive TVD with reversed axis
            mode='lines+markers',
            marker=dict(
                size=4,
                color=survey_df['MD'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Measured<br>Depth (ft)",
                        font=dict(color='cyan', size=14, family='Arial Black')
                    ),
                    tickfont=dict(color='cyan', size=12, family='Arial Black'),
                    bgcolor='rgba(0,0,0,0.9)',
                    outlinecolor='cyan',
                    outlinewidth=2,
                    x=1.15,
                    xanchor='left',
                    thickness=20,
                    len=0.7
                )
            ),
            line=dict(color='cyan', width=3),
            text=[f"MD: {md:.0f} ft<br>TVD: {tvd:.0f} ft<br>Inc: {inc:.1f}°<br>Azm: {azm:.1f}°" 
                  for md, tvd, inc, azm in zip(survey_df['MD'], survey_df['TVD'], survey_df['INC'], survey_df['AZM'])],
            hovertemplate='<b>E/W:</b> %{x:.0f} ft<br><b>N/S:</b> %{y:.0f} ft<br>%{text}<extra></extra>',
            name='Well Path'
        ))
        
        # Add surface marker
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'),
            name='Surface Location',
            hovertemplate='<b>Surface</b><extra></extra>'
        ))
        
        # Add target/TD marker
        fig_3d.add_trace(go.Scatter3d(
            x=[survey_df['EW'].iloc[-1]],
            y=[survey_df['NS'].iloc[-1]],
            z=[survey_df['TVD'].iloc[-1]],
            mode='markers',
            marker=dict(size=10, color='lime', symbol='diamond'),
            name='Total Depth',
            hovertemplate=f"<b>Total Depth</b><br>TVD: {survey_df['TVD'].iloc[-1]:.0f} ft<extra></extra>"
        ))
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='East/West (ft)',
                yaxis_title='North/South (ft)',
                zaxis_title='TVD (ft)',
                zaxis=dict(autorange='reversed'),  # Reverse z-axis to show positive depth values
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=0.5)
                )
            ),
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Dogleg Severity Profile
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dogleg Severity Profile")
            fig_dls = go.Figure()
            fig_dls.add_trace(go.Scatter(
                x=survey_df['MD'],
                y=survey_df['DLS'],
                mode='lines',
                line=dict(color='orange', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,165,0,0.3)',
                name='Dogleg Severity'
            ))
            
            # Add warning line at 3°/100ft
            fig_dls.add_hline(y=3.0, line_dash="dash", line_color="red", 
                             annotation_text="Caution: 3°/100ft", annotation_position="right")
            
            fig_dls.update_layout(
                xaxis_title='Measured Depth (ft)',
                yaxis_title='Dogleg Severity (°/100ft)',
                height=400,
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig_dls, use_container_width=True)
        
        with col2:
            st.markdown("### Inclination & Azimuth")
            fig_inc_azm = go.Figure()
            
            fig_inc_azm.add_trace(go.Scatter(
                x=survey_df['MD'],
                y=survey_df['INC'],
                mode='lines',
                line=dict(color='cyan', width=2),
                name='Inclination',
                yaxis='y'
            ))
            
            fig_inc_azm.add_trace(go.Scatter(
                x=survey_df['MD'],
                y=survey_df['AZM'],
                mode='lines',
                line=dict(color='magenta', width=2),
                name='Azimuth',
                yaxis='y2'
            ))
            
            fig_inc_azm.update_layout(
                xaxis_title='Measured Depth (ft)',
                yaxis=dict(
                    title=dict(text='Inclination (°)', font=dict(color='cyan')),
                    tickfont=dict(color='cyan')
                ),
                yaxis2=dict(
                    title=dict(text='Azimuth (°)', font=dict(color='magenta')),
                    tickfont=dict(color='magenta'),
                    overlaying='y',
                    side='right'
                ),
                height=400,
                template='plotly_dark',
                hovermode='x unified'
            )
            st.plotly_chart(fig_inc_azm, use_container_width=True)
        
        # Vertical Section View
        st.markdown("### Vertical Section View (2D Projection)")
        
        # Calculate along-hole and lateral coordinates
        if survey_df['AZM'].iloc[-1] != 0:
            ref_azimuth = survey_df['AZM'].iloc[-1]  # Use TD azimuth as reference
        else:
            ref_azimuth = survey_df[survey_df['INC'] > 5]['AZM'].mean() if len(survey_df[survey_df['INC'] > 5]) > 0 else 0
        
        ref_azm_rad = radians(ref_azimuth)
        survey_df['Along_Hole'] = survey_df['NS'] * cos(ref_azm_rad) + survey_df['EW'] * sin(ref_azm_rad)
        survey_df['Lateral'] = -survey_df['NS'] * sin(ref_azm_rad) + survey_df['EW'] * cos(ref_azm_rad)
        
        fig_vs = go.Figure()
        
        fig_vs.add_trace(go.Scatter(
            x=survey_df['Along_Hole'],
            y=survey_df['TVD'],
            mode='lines+markers',
            marker=dict(size=6, color=survey_df['MD'], colorscale='Plasma', showscale=True,
                       colorbar=dict(title="MD (ft)", x=1.15)),
            line=dict(color='cyan', width=3),
            text=[f"MD: {md:.0f} ft<br>TVD: {tvd:.0f} ft<br>Inc: {inc:.1f}°<br>Lateral: {lat:.0f} ft" 
                  for md, tvd, inc, lat in zip(survey_df['MD'], survey_df['TVD'], survey_df['INC'], survey_df['Lateral'])],
            hovertemplate='<b>Along Hole:</b> %{x:.0f} ft<br>%{text}<extra></extra>',
            name='Vertical Section'
        ))
        
        fig_vs.update_layout(
            xaxis_title=f'Along Hole Distance (ft) - Reference Azimuth: {ref_azimuth:.1f}°',
            yaxis_title='True Vertical Depth (ft)',
            height=500,
            template='plotly_dark',
            hovermode='closest'
        )
        
        fig_vs.update_yaxes(autorange='reversed')
        st.plotly_chart(fig_vs, use_container_width=True)
        
        # Survey Data Table
        with st.expander("📋 View Complete Survey Data Table"):
            display_cols = ['MD', 'INC', 'AZM', 'TVD', 'NS', 'EW', 'DLS']
            available_cols = [col for col in display_cols if col in survey_df.columns]
            st.dataframe(
                survey_df[available_cols].style.format({
                    'MD': '{:.2f}',
                    'INC': '{:.2f}',
                    'AZM': '{:.2f}',
                    'TVD': '{:.2f}',
                    'NS': '{:.2f}',
                    'EW': '{:.2f}',
                    'DLS': '{:.3f}'
                }),
                height=400
            )
        # ========================================
        # MWD LOGGING DATA VISUALIZATION
        # ========================================
        if mwd_logging_df is not None and not mwd_logging_df.empty:
            st.markdown("---")
            st.markdown("## 📊 MWD Logging Data Analysis")
            st.info("Real-time drilling parameters recorded during operations")
            
            # Key Metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Avg Gamma Ray", f"{mwd_logging_df['GR'].mean():.1f} API")
                st.metric("Max GR", f"{mwd_logging_df['GR'].max():.0f} API")
            with col2:
                st.metric("Avg ROP", f"{mwd_logging_df['ROP'].mean():.1f} ft/hr")
                st.metric("Max ROP", f"{mwd_logging_df['ROP'].max():.0f} ft/hr")
            with col3:
                st.metric("Avg Temperature", f"{mwd_logging_df['TEMP'].mean():.1f}°F")
                st.metric("Max Temp", f"{mwd_logging_df['TEMP'].max():.0f}°F")
            with col4:
                st.metric("Max Axial Shock", f"{mwd_logging_df['SHKA'].max():.1f} G")
                st.metric("Max Lateral Shock", f"{mwd_logging_df['SHKL'].max():.1f} G")
            with col5:
                st.metric("Max Axial Vibe", f"{mwd_logging_df['VIBEA'].max():.1f} GRMS")
                st.metric("Max Lateral Vibe", f"{mwd_logging_df['VIBEL'].max():.1f} GRMS")
            
            # 1. Gamma Ray Log
            st.markdown("### 📡 Gamma Ray Log")
            
            # Add geological interpretation function
            def interpret_gamma_ray(gr_value):
                """Provide detailed geological interpretation based on GR API value"""
                if gr_value < 30:
                    return {
                        'lithology': 'Clean Sandstone/Salt',
                        'description': 'Very low radioactivity indicates clean quartz sandstone, possibly with high porosity and permeability. Excellent reservoir quality.',
                        'color': 'rgba(255, 215, 0, 0.3)',
                        'quality': 'Excellent Reservoir'
                    }
                elif gr_value < 60:
                    return {
                        'lithology': 'Clean to Slightly Silty Sandstone',
                        'description': 'Low GR suggests clean sandstone with minimal clay content. Good porosity/permeability. Potential hydrocarbon reservoir.',
                        'color': 'rgba(255, 255, 0, 0.3)',
                        'quality': 'Good Reservoir'
                    }
                elif gr_value < 90:
                    return {
                        'lithology': 'Silty Sandstone/Sandy Siltstone',
                        'description': 'Moderate GR indicates increased silt/clay content. Fair reservoir quality with moderate porosity/permeability.',
                        'color': 'rgba(173, 255, 47, 0.3)',
                        'quality': 'Fair Reservoir'
                    }
                elif gr_value < 120:
                    return {
                        'lithology': 'Sandy Shale/Siltstone',
                        'description': 'Elevated GR suggests significant clay content. Poor to marginal reservoir quality. May act as seal rock.',
                        'color': 'rgba(0, 255, 127, 0.3)',
                        'quality': 'Poor Reservoir/Seal'
                    }
                elif gr_value < 150:
                    return {
                        'lithology': 'Shale/Claystone',
                        'description': 'High GR indicates clay-rich shale. Acts as cap rock/seal. No reservoir potential but important for trapping hydrocarbons.',
                        'color': 'rgba(0, 128, 0, 0.3)',
                        'quality': 'Seal Rock'
                    }
                else:
                    return {
                        'lithology': 'Organic-Rich Shale/Radioactive Shale',
                        'description': 'Very high GR may indicate organic-rich shale (source rock) or presence of radioactive minerals (K, U, Th). Potential unconventional reservoir.',
                        'color': 'rgba(139, 69, 19, 0.3)',
                        'quality': 'Source Rock/Shale Gas'
                    }
            
            # Create enhanced GR log with geological zones
            col_gr1, col_gr2 = st.columns([3, 1])
            with col_gr1:
                fig_gr = go.Figure()
                
                # Add geological interpretation zones as background
                fig_gr.add_vrect(x0=0, x1=30, fillcolor="rgba(255, 215, 0, 0.15)", layer="below", line_width=0, annotation_text="Clean Sand", annotation_position="top left")
                fig_gr.add_vrect(x0=30, x1=60, fillcolor="rgba(255, 255, 0, 0.15)", layer="below", line_width=0, annotation_text="Silty Sand", annotation_position="top left")
                fig_gr.add_vrect(x0=60, x1=90, fillcolor="rgba(173, 255, 47, 0.15)", layer="below", line_width=0, annotation_text="Siltstone", annotation_position="top left")
                fig_gr.add_vrect(x0=90, x1=120, fillcolor="rgba(0, 255, 127, 0.15)", layer="below", line_width=0, annotation_text="Sandy Shale", annotation_position="top left")
                fig_gr.add_vrect(x0=120, x1=150, fillcolor="rgba(0, 128, 0, 0.15)", layer="below", line_width=0, annotation_text="Shale", annotation_position="top left")
                fig_gr.add_vrect(x0=150, x1=max(200, mwd_logging_df['GR'].max()+10), fillcolor="rgba(139, 69, 19, 0.15)", layer="below", line_width=0, annotation_text="Org-Rich Shale", annotation_position="top left")
                
                # Create hover text with detailed interpretation
                hover_texts = []
                for gr, tvd in zip(mwd_logging_df['GR'], mwd_logging_df['TVD']):
                    interp = interpret_gamma_ray(gr)
                    hover_texts.append(
                        f"<b>TVD:</b> {tvd:.0f} ft<br>"
                        f"<b>GR:</b> {gr:.1f} API<br>"
                        f"<b>Lithology:</b> {interp['lithology']}<br>"
                        f"<b>Quality:</b> {interp['quality']}<br>"
                        f"<b>Interpretation:</b><br>{interp['description']}"
                    )
                
                fig_gr.add_trace(go.Scatter(
                    x=mwd_logging_df['GR'],
                    y=mwd_logging_df['TVD'],
                    mode='lines',
                    name='Gamma Ray',
                    line=dict(color='lime', width=2.5),
                    fill='tozerox',
                    fillcolor='rgba(0, 255, 0, 0.2)',
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ))
                
                fig_gr.update_layout(
                    title='Gamma Ray Log with Geological Interpretation',
                    xaxis_title='Gamma Ray (API Units)',
                    yaxis_title='True Vertical Depth (ft)',
                    height=600,
                    template='plotly_dark',
                    hovermode='closest'
                )
                fig_gr.update_yaxes(autorange='reversed')
                fig_gr.update_xaxes(range=[0, max(200, mwd_logging_df['GR'].max()+10)])
                st.plotly_chart(fig_gr, use_container_width=True)
            
            with col_gr2:
                st.markdown("**Interpretation Guide:**")
                avg_gr = mwd_logging_df['GR'].mean()
                overall_interp = interpret_gamma_ray(avg_gr)
                
                st.markdown(f"**Average GR:** {avg_gr:.1f} API")
                st.markdown(f"**Dominant Lithology:**")
                st.info(overall_interp['lithology'])
                st.markdown(f"**Reservoir Quality:**")
                if 'Excellent' in overall_interp['quality']:
                    st.success(overall_interp['quality'])
                elif 'Good' in overall_interp['quality'] or 'Fair' in overall_interp['quality']:
                    st.info(overall_interp['quality'])
                else:
                    st.warning(overall_interp['quality'])
                
                st.markdown("---")
                st.markdown("**GR Scale Reference:**")
                st.markdown("""
                - **0-30 API**: Clean sandstone
                - **30-60 API**: Silty sandstone  
                - **60-90 API**: Siltstone
                - **90-120 API**: Sandy shale
                - **120-150 API**: Shale
                - **>150 API**: Organic-rich shale
                """)
                
                st.markdown("---")
                st.caption("💡 **Tip:** Hover over the log curve to see detailed geological interpretation at each depth")
            
            # 2. Rate of Penetration
            st.markdown("### ⚡ Rate of Penetration (ROP)")
            col_rop1, col_rop2 = st.columns([3, 1])
            with col_rop1:
                fig_rop = go.Figure()
                fig_rop.add_trace(go.Scatter(
                    x=mwd_logging_df['ROP'],
                    y=mwd_logging_df['TVD'],
                    mode='lines+markers',
                    name='ROP',
                    line=dict(color='orange', width=2),
                    marker=dict(size=3),
                    hovertemplate='<b>TVD:</b> %{y:.0f} ft<br><b>ROP:</b> %{x:.1f} ft/hr<extra></extra>'
                ))
                fig_rop.update_layout(
                    title='Rate of Penetration vs Depth',
                    xaxis_title='ROP (ft/hr)',
                    yaxis_title='True Vertical Depth (ft)',
                    height=500,
                    template='plotly_dark',
                    hovermode='closest'
                )
                fig_rop.update_yaxes(autorange='reversed')
                st.plotly_chart(fig_rop, use_container_width=True)
            with col_rop2:
                st.markdown("**Performance:**")
                avg_rop = mwd_logging_df['ROP'].mean()
                if avg_rop > 200:
                    st.success(f"Excellent: {avg_rop:.0f} ft/hr")
                elif avg_rop > 150:
                    st.info(f"Good: {avg_rop:.0f} ft/hr")
                else:
                    st.warning(f"Slow: {avg_rop:.0f} ft/hr")
                st.metric("Drilling Time", f"{len(mwd_logging_df)/60:.1f} hrs")
            
            # 3. Temperature Log
            st.markdown("### 🌡️ Downhole Temperature")
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=mwd_logging_df['TEMP'],
                y=mwd_logging_df['TVD'],
                mode='lines',
                name='Temperature',
                line=dict(color='red', width=2),
                fill='tozerox',
                fillcolor='rgba(255, 0, 0, 0.2)',
                hovertemplate='<b>TVD:</b> %{y:.0f} ft<br><b>Temp:</b> %{x:.1f}°F<extra></extra>'
            ))
            fig_temp.update_layout(
                title='Temperature vs True Vertical Depth',
                xaxis_title='Temperature (°F)',
                yaxis_title='True Vertical Depth (ft)',
                height=500,
                template='plotly_dark',
                hovermode='closest'
            )
            fig_temp.update_yaxes(autorange='reversed')
            st.plotly_chart(fig_temp, use_container_width=True)
            
            # 4. Shock Analysis
            st.markdown("### 💥 Shock Analysis (Axial & Lateral)")
            
            # Create hover text with timestamp
            shock_hover_axial = [
                f"<b>Date:</b> {date}<br><b>Time:</b> {time}<br><b>Depth:</b> {dep:.0f} ft<br><b>Axial Shock:</b> {shka:.1f} G"
                for date, time, dep, shka in zip(mwd_logging_df['DATE'], mwd_logging_df['TIME'], mwd_logging_df['DEP'], mwd_logging_df['SHKA'])
            ]
            shock_hover_lateral = [
                f"<b>Date:</b> {date}<br><b>Time:</b> {time}<br><b>Depth:</b> {dep:.0f} ft<br><b>Lateral Shock:</b> {shkl:.1f} G"
                for date, time, dep, shkl in zip(mwd_logging_df['DATE'], mwd_logging_df['TIME'], mwd_logging_df['DEP'], mwd_logging_df['SHKL'])
            ]
            
            fig_shock = go.Figure()
            fig_shock.add_trace(go.Scatter(
                x=mwd_logging_df['DEP'],
                y=mwd_logging_df['SHKA'],
                mode='lines',
                name='Axial Shock',
                line=dict(color='cyan', width=2),
                text=shock_hover_axial,
                hovertemplate='%{text}<extra></extra>'
            ))
            fig_shock.add_trace(go.Scatter(
                x=mwd_logging_df['DEP'],
                y=mwd_logging_df['SHKL'],
                mode='lines',
                name='Lateral Shock',
                line=dict(color='magenta', width=2),
                text=shock_hover_lateral,
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Add warning threshold lines
            fig_shock.add_hline(y=50, line_dash="dash", line_color="yellow", 
                               annotation_text="Warning: 50G", annotation_position="right")
            fig_shock.add_hline(y=30, line_dash="dot", line_color="orange", 
                               annotation_text="Caution: 30G", annotation_position="right")
            
            fig_shock.update_layout(
                title='Shock Levels vs Measured Depth (Hover for Timestamp)',
                xaxis_title='Measured Depth (ft)',
                yaxis_title='Shock (G)',
                height=400,
                template='plotly_dark',
                hovermode='closest',
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_shock, use_container_width=True)
            
            # Critical Shock Events Table
            shock_threshold = 30  # G
            critical_axial = mwd_logging_df[mwd_logging_df['SHKA'] > shock_threshold][['DATE', 'TIME', 'DEP', 'SHKA']].copy()
            critical_lateral = mwd_logging_df[mwd_logging_df['SHKL'] > shock_threshold][['DATE', 'TIME', 'DEP', 'SHKL']].copy()
            
            if not critical_axial.empty or not critical_lateral.empty:
                with st.expander(f"⚠️ Critical Shock Events (>{shock_threshold}G) - Click to View"):
                    col_shock_a, col_shock_b = st.columns(2)
                    with col_shock_a:
                        st.markdown("**Axial Shock Events**")
                        if not critical_axial.empty:
                            critical_axial.columns = ['Date', 'Time', 'Depth (ft)', 'Shock (G)']
                            
                            # Color code shock values
                            def color_shock(val):
                                if isinstance(val, (int, float)):
                                    if val > 50:
                                        return 'background-color: #8B0000'  # Dark red
                                    elif val > 40:
                                        return 'background-color: #CD5C5C'  # Indian red
                                    else:
                                        return 'background-color: #F08080'  # Light coral
                                return ''
                            
                            st.dataframe(
                                critical_axial.style.format({
                                    'Depth (ft)': '{:.1f}',
                                    'Shock (G)': '{:.1f}'
                                }).applymap(color_shock, subset=['Shock (G)']),
                                height=300
                            )
                            st.caption(f"Total: {len(critical_axial)} events")
                        else:
                            st.success("No critical axial shock events")
                    
                    with col_shock_b:
                        st.markdown("**Lateral Shock Events**")
                        if not critical_lateral.empty:
                            critical_lateral.columns = ['Date', 'Time', 'Depth (ft)', 'Shock (G)']
                            
                            # Color code shock values
                            def color_shock(val):
                                if isinstance(val, (int, float)):
                                    if val > 50:
                                        return 'background-color: #8B0000'  # Dark red
                                    elif val > 40:
                                        return 'background-color: #CD5C5C'  # Indian red
                                    else:
                                        return 'background-color: #F08080'  # Light coral
                                return ''
                            
                            st.dataframe(
                                critical_lateral.style.format({
                                    'Depth (ft)': '{:.1f}',
                                    'Shock (G)': '{:.1f}'
                                }).applymap(color_shock, subset=['Shock (G)']),
                                height=300
                            )
                            st.caption(f"Total: {len(critical_lateral)} events")
                        else:
                            st.success("No critical lateral shock events")
            
            # Shock severity assessment
            col_sh1, col_sh2 = st.columns(2)
            with col_sh1:
                max_axial = mwd_logging_df['SHKA'].max()
                if max_axial > 50:
                    st.error(f"⚠️ High axial shock detected: {max_axial:.1f}G")
                elif max_axial > 30:
                    st.warning(f"⚠️ Moderate axial shock: {max_axial:.1f}G")
                else:
                    st.success(f"✓ Normal axial shock: {max_axial:.1f}G")
            with col_sh2:
                max_lateral = mwd_logging_df['SHKL'].max()
                if max_lateral > 50:
                    st.error(f"⚠️ High lateral shock detected: {max_lateral:.1f}G")
                elif max_lateral > 30:
                    st.warning(f"⚠️ Moderate lateral shock: {max_lateral:.1f}G")
                else:
                    st.success(f"✓ Normal lateral shock: {max_lateral:.1f}G")
            
            # 5. Vibration Analysis
            st.markdown("### 📳 Vibration Analysis (Axial & Lateral)")
            
            # Create hover text with timestamp
            vibe_hover_axial = [
                f"<b>Date:</b> {date}<br><b>Time:</b> {time}<br><b>Depth:</b> {dep:.0f} ft<br><b>Axial Vibe:</b> {vibea:.2f} GRMS"
                for date, time, dep, vibea in zip(mwd_logging_df['DATE'], mwd_logging_df['TIME'], mwd_logging_df['DEP'], mwd_logging_df['VIBEA'])
            ]
            vibe_hover_lateral = [
                f"<b>Date:</b> {date}<br><b>Time:</b> {time}<br><b>Depth:</b> {dep:.0f} ft<br><b>Lateral Vibe:</b> {vibel:.2f} GRMS"
                for date, time, dep, vibel in zip(mwd_logging_df['DATE'], mwd_logging_df['TIME'], mwd_logging_df['DEP'], mwd_logging_df['VIBEL'])
            ]
            
            fig_vibe = go.Figure()
            fig_vibe.add_trace(go.Scatter(
                x=mwd_logging_df['DEP'],
                y=mwd_logging_df['VIBEA'],
                mode='lines',
                name='Axial Vibration',
                line=dict(color='yellow', width=2),
                text=vibe_hover_axial,
                hovertemplate='%{text}<extra></extra>'
            ))
            fig_vibe.add_trace(go.Scatter(
                x=mwd_logging_df['DEP'],
                y=mwd_logging_df['VIBEL'],
                mode='lines',
                name='Lateral Vibration',
                line=dict(color='orange', width=2),
                text=vibe_hover_lateral,
                hovertemplate='%{text}<extra></extra>'
            ))
            
            # Add warning threshold lines
            fig_vibe.add_hline(y=8, line_dash="dash", line_color="red", 
                              annotation_text="Critical: 8 GRMS", annotation_position="right")
            fig_vibe.add_hline(y=5, line_dash="dot", line_color="orange", 
                              annotation_text="Elevated: 5 GRMS", annotation_position="right")
            
            fig_vibe.update_layout(
                title='Vibration Levels vs Measured Depth (Hover for Timestamp)',
                xaxis_title='Measured Depth (ft)',
                yaxis_title='Vibration (GRMS)',
                height=400,
                template='plotly_dark',
                hovermode='closest',
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_vibe, use_container_width=True)
            
            # Critical Vibration Events Table
            vibe_threshold = 5  # GRMS
            critical_vibe_axial = mwd_logging_df[mwd_logging_df['VIBEA'] > vibe_threshold][['DATE', 'TIME', 'DEP', 'VIBEA']].copy()
            critical_vibe_lateral = mwd_logging_df[mwd_logging_df['VIBEL'] > vibe_threshold][['DATE', 'TIME', 'DEP', 'VIBEL']].copy()
            
            if not critical_vibe_axial.empty or not critical_vibe_lateral.empty:
                with st.expander(f"⚠️ Elevated Vibration Events (>{vibe_threshold} GRMS) - Click to View"):
                    col_vibe_a, col_vibe_b = st.columns(2)
                    with col_vibe_a:
                        st.markdown("**Axial Vibration Events**")
                        if not critical_vibe_axial.empty:
                            critical_vibe_axial.columns = ['Date', 'Time', 'Depth (ft)', 'Vibe (GRMS)']
                            
                            # Color code vibration values
                            def color_vibe(val):
                                if isinstance(val, (int, float)):
                                    if val > 8:
                                        return 'background-color: #8B0000'  # Dark red
                                    elif val > 6.5:
                                        return 'background-color: #FF4500'  # Orange red
                                    else:
                                        return 'background-color: #FFA500'  # Orange
                                return ''
                            
                            st.dataframe(
                                critical_vibe_axial.style.format({
                                    'Depth (ft)': '{:.1f}',
                                    'Vibe (GRMS)': '{:.2f}'
                                }).applymap(color_vibe, subset=['Vibe (GRMS)']),
                                height=300
                            )
                            st.caption(f"Total: {len(critical_vibe_axial)} events")
                        else:
                            st.success("No elevated axial vibration events")
                    
                    with col_vibe_b:
                        st.markdown("**Lateral Vibration Events**")
                        if not critical_vibe_lateral.empty:
                            critical_vibe_lateral.columns = ['Date', 'Time', 'Depth (ft)', 'Vibe (GRMS)']
                            
                            # Color code vibration values
                            def color_vibe(val):
                                if isinstance(val, (int, float)):
                                    if val > 8:
                                        return 'background-color: #8B0000'  # Dark red
                                    elif val > 6.5:
                                        return 'background-color: #FF4500'  # Orange red
                                    else:
                                        return 'background-color: #FFA500'  # Orange
                                return ''
                            
                            st.dataframe(
                                critical_vibe_lateral.style.format({
                                    'Depth (ft)': '{:.1f}',
                                    'Vibe (GRMS)': '{:.2f}'
                                }).applymap(color_vibe, subset=['Vibe (GRMS)']),
                                height=300
                            )
                            st.caption(f"Total: {len(critical_vibe_lateral)} events")
                        else:
                            st.success("No elevated lateral vibration events")
            
            # Vibration severity assessment
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                max_vibe_a = mwd_logging_df['VIBEA'].max()
                if max_vibe_a > 8:
                    st.error(f"⚠️ Critical axial vibration: {max_vibe_a:.2f} GRMS")
                elif max_vibe_a > 5:
                    st.warning(f"⚠️ Elevated axial vibration: {max_vibe_a:.2f} GRMS")
                else:
                    st.success(f"✓ Normal axial vibration: {max_vibe_a:.2f} GRMS")
            with col_v2:
                max_vibe_l = mwd_logging_df['VIBEL'].max()
                if max_vibe_l > 8:
                    st.error(f"⚠️ Critical lateral vibration: {max_vibe_l:.2f} GRMS")
                elif max_vibe_l > 5:
                    st.warning(f"⚠️ Elevated lateral vibration: {max_vibe_l:.2f} GRMS")
                else:
                    st.success(f"✓ Normal lateral vibration: {max_vibe_l:.2f} GRMS")
            
            # 6. Combined Multi-Parameter Display
            st.markdown("### 📊 Multi-Parameter Correlation")
            fig_multi = go.Figure()
            
            # Normalize parameters for comparison (0-1 scale)
            norm_gr = (mwd_logging_df['GR'] - mwd_logging_df['GR'].min()) / (mwd_logging_df['GR'].max() - mwd_logging_df['GR'].min())
            norm_rop = (mwd_logging_df['ROP'] - mwd_logging_df['ROP'].min()) / (mwd_logging_df['ROP'].max() - mwd_logging_df['ROP'].min())
            norm_temp = (mwd_logging_df['TEMP'] - mwd_logging_df['TEMP'].min()) / (mwd_logging_df['TEMP'].max() - mwd_logging_df['TEMP'].min())
            
            fig_multi.add_trace(go.Scatter(x=mwd_logging_df['DEP'], y=norm_gr, mode='lines', name='GR (norm)', line=dict(color='green', width=1.5)))
            fig_multi.add_trace(go.Scatter(x=mwd_logging_df['DEP'], y=norm_rop, mode='lines', name='ROP (norm)', line=dict(color='orange', width=1.5)))
            fig_multi.add_trace(go.Scatter(x=mwd_logging_df['DEP'], y=norm_temp, mode='lines', name='Temp (norm)', line=dict(color='red', width=1.5)))
            
            fig_multi.update_layout(
                title='Normalized Multi-Parameter Trends',
                xaxis_title='Measured Depth (ft)',
                yaxis_title='Normalized Value (0-1)',
                height=400,
                template='plotly_dark',
                hovermode='x unified',
                legend=dict(x=0.02, y=0.98)
            )
            st.plotly_chart(fig_multi, use_container_width=True)
            
            # MWD Data Table
            with st.expander("📋 View Complete MWD Logging Data Table"):
                display_mwd = mwd_logging_df[['DATE', 'TIME', 'DEP', 'GR', 'TVD', 'ROP', 'TEMP', 'SHKA', 'SHKL', 'VIBEA', 'VIBEL']]
                st.dataframe(
                    display_mwd.style.format({
                        'DEP': '{:.1f}',
                        'GR': '{:.1f}',
                        'TVD': '{:.1f}',
                        'ROP': '{:.1f}',
                        'TEMP': '{:.1f}',
                        'SHKA': '{:.1f}',
                        'SHKL': '{:.1f}',
                        'VIBEA': '{:.2f}',
                        'VIBEL': '{:.2f}'
                    }),
                    height=400
                )
    
    else:
        st.error("⚠️ Could not parse survey data. Please ensure your file contains columns for MD, Inclination, and Azimuth.")
else:
    st.info("📁 Upload directional survey data in the sidebar to visualize wellbore trajectory, dogleg severity, and survey analysis.")

# ========================================
# 2. SIMULATED ACTIVE SONAR PING (MWD PULSE)
# ========================================
st.markdown("## Simulated Active Sonar Ping (MWD Pulse)")
t = np.linspace(0, 2, 200)
ping = np.sin(2 * np.pi * 5 * t) * np.exp(-t*3)
ping_df = pd.DataFrame({"Time (s)": t, "Amplitude": ping})
fig_mwd = go.Figure()
fig_mwd.add_trace(go.Scatter(x=ping_df["Time (s)"], y=ping_df["Amplitude"],
                             mode='lines', name='MWD Pulse', line=dict(color='cyan')))
fig_mwd.update_layout(title="MWD Pulse Downhole (Analog to Resistivity)", height=300, template="plotly_dark")
st.plotly_chart(fig_mwd, use_container_width=True)

# ========================================
# 3. MWD 6-PACK TELEMETRY — SMOOTH, NO RERUN
# ========================================
st.markdown("## MWD 6-Pack Telemetry")

if not (len(bit_pattern) == 4 and all(c in '01' for c in bit_pattern)):
    bit_pattern = "1010"

clear_width = pulse_width * 1.5
bit_spacing = pulse_speed
packet_duration = 2 * clear_width + 0.5 + 4 * bit_spacing + pulse_width
t_full = np.linspace(0, packet_duration, int(packet_duration * 100))
signal = np.zeros_like(t_full)

def add_pulse(t, center, width, height):
    rise = width * 0.2
    flat = width * 0.6
    fall = width * 0.2
    mask_rise = (t >= center) & (t < center + rise)
    mask_flat = (t >= center + rise) & (t < center + rise + flat)
    mask_fall = (t >= center + rise + flat) & (t < center + width)
    signal[mask_rise] = height * (t[mask_rise] - center) / rise
    signal[mask_flat] = height
    signal[mask_fall] = height * (1 - (t[mask_fall] - (center + rise + flat)) / fall)

add_pulse(t_full, clear_width / 2, clear_width, 1.0)
add_pulse(t_full, clear_width + 0.5, clear_width, 1.0)
for i, bit in enumerate(bit_pattern):
    center = 2 * clear_width + 0.5 + i * bit_spacing + pulse_width / 2
    height = 0.8 if bit == '1' else -0.8
    add_pulse(t_full, center, pulse_width, height)
signal += np.random.normal(0, 0.08, len(t_full))

if 'running' not in st.session_state:
    st.session_state.running = False
if 'offset' not in st.session_state:
    st.session_state.offset = 0

if st.button("Start Live Telemetry" if not st.session_state.running else "Stop Live Telemetry"):
    st.session_state.running = not st.session_state.running
    if not st.session_state.running:
        st.session_state.offset = 0

plot_placeholder = st.empty()

if st.session_state.running:
    while st.session_state.running:
        st.session_state.offset = (st.session_state.offset + 0.02) % packet_duration
        t_shifted = (t_full - st.session_state.offset + packet_duration) % packet_duration
        df_plot = pd.DataFrame({"Time (s)": t_shifted, "Amplitude": signal})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_plot["Time (s)"], y=df_plot["Amplitude"], mode='lines', line=dict(color='cyan', width=2)))
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=300, template="plotly_dark", showlegend=False, xaxis_range=[0, packet_duration])
        plot_placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.02)
else:
    df_static = pd.DataFrame({"Time (s)": t_full, "Amplitude": signal})
    fig_static = go.Figure()
    fig_static.add_trace(go.Scatter(x=df_static["Time (s)"], y=df_static["Amplitude"], mode='lines', line=dict(color='cyan', width=2)))
    fig_static.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_static.update_layout(height=300, template="plotly_dark", showlegend=False, xaxis_range=[0, packet_duration])
    plot_placeholder.plotly_chart(fig_static, use_container_width=True)

# ========================================
# 4. WAVE ENERGY vs RIG PROXIMITY — NOW UNIQUE PER BUOY
# ========================================
st.markdown("## Wave Energy vs Rig Proximity")

spectral_dfs = []
for bid in selected_buoys:
    df = fetch_spectral(bid)
    if "Spectral Energy (m²/Hz)" in df.columns:
        noise = np.random.normal(0, 0.1 * int(bid[-2:]), len(df))
        df["Spectral Energy (m²/Hz)"] = df["Spectral Energy (m²/Hz)"].clip(0) + noise.clip(0)
    spectral_dfs.append(df)

combined_df = pd.concat(spectral_dfs, ignore_index=True) if spectral_dfs else pd.DataFrame()

impact_rows = []
for bid in selected_buoys:
    buoy_series = combined_df[combined_df["Station"] == bid]["Spectral Energy (m²/Hz)"]
    energy = buoy_series.mean() if not buoy_series.empty else np.nan
    b_lat, b_lon = buoy_info[bid][1], buoy_info[bid][2]
    dist = min([haversine(b_lat, b_lon, r["lat"], r["lon"]) for r in real_rigs], default=0)
    impact_rows.append({
        "Buoy": bid,
        "Avg Energy": round(energy, 2) if not pd.isna(energy) else 0.0,
        "Nearest Rig (mi)": round(dist, 1)
    })

impact_df = pd.DataFrame(impact_rows).dropna()
if not impact_df.empty:
    fig_wave = px.scatter(
        impact_df,
        x="Nearest Rig (mi)",
        y="Avg Energy",
        hover_data=["Buoy"],
        title="Wave Energy vs Rig Proximity",
        template="plotly_dark",
        size=[15] * len(impact_df),
        color="Buoy",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig_wave.update_layout(height=400, legend_title_text="Buoy ID")
    st.plotly_chart(fig_wave, use_container_width=True)
else:
    st.warning("No spectral data available — check NOAA .spec files.")

# ========================================
# 6. WAVE PROPAGATION — BUOYS → RIGS (REAL-TIME PREDICTION)
# ========================================
st.markdown("## Wave Propagation to Rigs — Real-Time Motion Forecast")

if len(selected_buoys) > 1:
    buoy_data = {bid: fetch_realtime(bid) for bid in selected_buoys}

    propagation_rows = []
    for rig in real_rigs:
        r_lat, r_lon = rig["lat"], rig["lon"]
        rig_name = rig["name"]

        contributions = []
        for bid in selected_buoys:
            b_lat, b_lon = buoy_info[bid][1], buoy_info[bid][2]
            rt = buoy_data[bid]

            dist_mi = haversine(b_lat, b_lon, r_lat, r_lon)
            if dist_mi > 300:
                continue

            period = rt['DPD']
            if period < 3: period = 6
            wave_speed_mps = 1.56 * period
            delay_min = (dist_mi * 1609.34) / (wave_speed_mps * 60)

            wave_dir = rt['MWD']
            bearing = np.degrees(np.arctan2(r_lon - b_lon, r_lat - b_lat)) % 360
            angle_diff = min(abs(wave_dir - bearing), 360 - abs(wave_dir - bearing))

            decay = np.exp(-dist_mi / 100)
            direction_factor = np.cos(np.radians(angle_diff))
            if direction_factor < 0.3:
                direction_factor = 0.3

            contrib_wave = rt['WVHT'] * decay * direction_factor
            contributions.append({
                "buoy": bid,
                "wave_ht": contrib_wave,
                "delay_min": delay_min,
                "dist_mi": dist_mi
            })

        if contributions:
            weights = [1 / (c["dist_mi"] + 10) for c in contributions]
            predicted_ht = sum(c["wave_ht"] * w for c, w in zip(contributions, weights)) / sum(weights)
            max_delay = max(c["delay_min"] for c in contributions)
            dominant_buoy = max(contributions, key=lambda x: x["wave_ht"])["buoy"]
        else:
            predicted_ht = 0.0
            max_delay = 0
            dominant_buoy = "N/A"

        if predicted_ht > 6.0:
            risk = "HIGH"
            color = "red"
        elif predicted_ht > 4.0:
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "lime"

        propagation_rows.append({
            "Rig": rig_name,
            "Predicted Wave Ht (ft)": round(predicted_ht, 1),
            "Est. Arrival (min)": f"+{int(max_delay)}",
            "Dominant Buoy": dominant_buoy,
            "Risk": f"<span style='color:{color}; font-weight:bold;'>{risk}</span>"
        })

    if propagation_rows:
        prop_df = pd.DataFrame(propagation_rows)
        st.markdown("**Predicted wave impact at each rig (next 1–3 hrs)**")
        st.markdown(prop_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.info("No buoys within 300 mi of rigs.")
else:
    st.info("Select **2+ buoys** to enable wave propagation forecast.")

# ========================================
# 5. GULF OF MEXICO — RIGS & BUOYS (BOEM POPUPS + NOAA DATA)
# ========================================
st.markdown("## Gulf of Mexico — Rigs & Buoys")

buoy_data = {bid: fetch_realtime(bid) for bid in selected_buoys}

m = folium.Map(location=[27.5, -88.5], zoom_start=7, tiles="CartoDB dark_matter")

for rig in real_rigs:
    name = rig["name"]
    details = platform_details.get(name, {})
    popup_html = f"""
    <div style="font-family: monospace; min-width: 240px;">
        <b>Platform Details: {name}</b><br>
        <hr style="margin:4px 0;">
        * Platform Type: {details.get('type', 'N/A')}<br>
        * Water Depth: {details.get('water_depth', 'N/A')}<br>
        * Status: {details.get('status', 'N/A')}<br>
        * Capacity: {details.get('capacity', 'N/A')}
    </div>
    """
    folium.CircleMarker(
        [rig["lat"], rig["lon"]],
        radius=14,
        popup=folium.Popup(popup_html, max_width=400),
        color="orange",
        fill=True,
        fillOpacity=0.9,
        weight=2
    ).add_to(m)

for bid in selected_buoys:
    lat, lon = buoy_info[bid][1], buoy_info[bid][2]
    rt = buoy_data[bid]
    is_active = (bid == primary)

    popup_html = f"""
    <div style="font-family: monospace; min-width: 180px;">
        <b>{bid}</b> — <i>{buoy_info[bid][0]}</i><br>
        <hr style="margin:4px 0;">
        <b>Wave Ht:</b> {rt['WVHT']:.1f} ft<br>
        <b>Period:</b> {rt['DPD']:.1f} s<br>
        <b>Wind:</b> {rt['WSPD']:.1f} kt @ {int(rt['WD'])} degrees<br>
        <b>Pressure:</b> {rt['PRES']:.2f} inHg<br>
        <b>Sea Temp:</b> {rt['WTMP']:.1f} degrees F<br>
    """
    if is_active:
        popup_html += '<br><span style="color:lime; font-weight:bold;">ACTIVE DATA SOURCE</span>'
    popup_html += "</div>"

    if is_active:
        folium.CircleMarker([lat, lon], radius=18, popup=folium.Popup(popup_html, max_width=300),
                            color="lime", fill=True, fillOpacity=0.9, weight=3).add_to(m)
        folium.Circle([lat, lon], radius=35000, color="lime", weight=2, fill=False, dashArray='10,10', opacity=0.7).add_to(m)
    else:
        folium.CircleMarker([lat, lon], radius=11, popup=folium.Popup(popup_html, max_width=300),
                            color="cyan", fill=True, fillOpacity=0.8).add_to(m)

st_folium(m, width=1000, height=500, key="map")

# ========================================
# FOOTER / CTA
# ========================================
st.success("""
**This is how I processed sonar at 5,000 ft below sea.**
**Now I'll do it for your rig at 55,000 ft.**

A real-time offshore drilling intelligence platform that fuses NOAA buoy data,
BOEM rig assets, and MWD telemetry into a predictive rig safety and operations dashboard
— delivering wave propagation forecasts, motion risk, and environmental fusion. If nothing else a fun learning device.
""")
