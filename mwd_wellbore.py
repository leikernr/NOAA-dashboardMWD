import re
from typing import Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _standardize_survey_col(name: str) -> str:
    cu = name.upper().strip()
    if 'DEPTH' in cu or cu == 'MD':
        return 'MD'
    if 'INC' in cu:
        return 'INC'
    if 'AZM' in cu or 'AZI' in cu:
        return 'AZM'
    if 'TVD' in cu:
        return 'TVD'
    if cu in ('VS', 'VSEC', 'VERT_SECT'):
        return 'VS'
    if 'N/' in cu or 'NORTH' in cu or cu == 'NS':
        return 'NS'
    if 'E/' in cu or 'EAST' in cu or cu == 'EW':
        return 'EW'
    if 'DLS' in cu:
        return 'DLS'
    if 'TRACK' in cu:
        return 'TRACK'
    if 'SURV' in cu:
        return 'SURVEY'
    return cu


def _standardize_log_col(name: str) -> str:
    cu = name.upper().strip()
    if cu in ('DEPTH', 'DEPT', 'DEP', 'MD'):
        return 'DEP'
    if cu in ('GR', 'GR_MWD', 'GAMMA', 'GAMA'):
        return 'GR'
    if 'TVD' in cu:
        return 'TVD'
    if 'ROP' in cu:
        return 'ROP'
    if 'TEMP' in cu:
        return 'TEMP'
    if cu in ('VS', 'VSEC'):
        return 'VS'
    if 'SHKA' in cu or 'SHOCKA' in cu or 'SHK_A' in cu:
        return 'SHKA'
    if 'SHKL' in cu or 'SHOCKL' in cu or 'SHK_L' in cu:
        return 'SHKL'
    if 'VIBEA' in cu or 'VIBA' in cu:
        return 'VIBEA'
    if 'VIBEL' in cu or 'VIBL' in cu:
        return 'VIBEL'
    return cu


def parse_las_file(file_content: bytes) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Parse LAS 2.0 survey data (~Other) and MWD logging curves (~Ascii)."""
    try:
        content = file_content.decode('utf-8', errors='ignore')
        lines = content.splitlines()

        survey_header_cols = None
        survey_records = []
        numeric_survey_rows = []
        mwd_lines = []
        ascii_rows = []
        curve_names = []

        in_survey_section = False
        in_mwd_section = False
        in_curve_section = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            upper = stripped.upper()

            if stripped.startswith('~'):
                in_survey_section = False
                in_curve_section = upper.startswith('~C')
                in_mwd_section = upper.startswith('~A')
                continue

            if in_curve_section and not stripped.startswith('#'):
                m = re.match(r'^([^.:]+)', stripped)
                if m:
                    curve_names.append(m.group(1).strip())
                continue

            if '# SURVEY DATA' in upper or '#SURVEY DATA' in upper:
                in_survey_section = True
                in_mwd_section = False
                continue

            if in_survey_section and stripped.startswith('#'):
                if any(k in upper for k in ('DEPTH', 'INCL', 'AZM', 'TVD')):
                    raw_cols = re.sub(r'[#]', '', stripped).split()
                    survey_header_cols = [_standardize_survey_col(c) for c in raw_cols]
                continue

            if stripped.startswith('#'):
                continue

            parts = stripped.split()

            if in_survey_section and len(parts) >= 7:
                if survey_header_cols is not None:
                    if len(parts) > len(survey_header_cols):
                        if len(parts) == len(survey_header_cols) + 1 and parts[1].upper() == 'TIE' and parts[2].upper() == 'IN':
                            parts = [parts[0], 'Tie In'] + parts[3:]
                        else:
                            excess = len(parts) - len(survey_header_cols)
                            merged = ' '.join(parts[1:1 + excess + 1])
                            parts = [parts[0], merged] + parts[1 + excess + 1:]
                    if len(parts) >= len(survey_header_cols):
                        survey_records.append(parts[:len(survey_header_cols)])
                else:
                    try:
                        float(parts[0])
                        numeric_survey_rows.append([float(parts[i]) for i in range(min(12, len(parts)))])
                    except (ValueError, IndexError):
                        continue

            elif in_mwd_section:
                if len(parts) >= 12 and '-' in parts[0]:
                    try:
                        mwd_lines.append({
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
                            'VIBEL': float(parts[11]),
                        })
                    except (ValueError, IndexError):
                        continue
                elif curve_names and len(parts) >= len(curve_names):
                    try:
                        ascii_rows.append([float(parts[i]) for i in range(len(curve_names))])
                    except (ValueError, IndexError):
                        continue

        survey_df = None
        if survey_records and survey_header_cols:
            survey_df = pd.DataFrame(survey_records, columns=survey_header_cols)
            for col in ['MD', 'INC', 'AZM', 'TVD', 'VS', 'NS', 'EW', 'DLS']:
                if col in survey_df.columns:
                    survey_df[col] = pd.to_numeric(survey_df[col], errors='coerce')
            survey_df = survey_df.dropna(subset=['MD'])
        elif numeric_survey_rows:
            ncols = len(numeric_survey_rows[0])
            if ncols >= 12:
                cols = ['MD', 'INC', 'AZM', 'TVD', 'NS', 'EW', 'VS', 'CD', 'CA', 'DLS', 'CL', 'TEMP']
            elif ncols >= 7:
                cols = ['MD', 'INC', 'AZM', 'TVD', 'NS', 'EW', 'VS'] + [f'COL{i}' for i in range(7, ncols)]
            else:
                return None, None
            survey_df = pd.DataFrame(numeric_survey_rows, columns=cols[:ncols])

        mwd_df = None
        if mwd_lines:
            mwd_df = pd.DataFrame(mwd_lines)
        elif ascii_rows and curve_names:
            std_cols = [_standardize_log_col(c) for c in curve_names]
            mwd_df = pd.DataFrame(ascii_rows, columns=std_cols)
            mwd_df = mwd_df.mask(mwd_df <= -900.0, np.nan)
            if 'DEP' in mwd_df.columns:
                if 'TVD' not in mwd_df.columns:
                    mwd_df['TVD'] = mwd_df['DEP']
                for str_col in ('DATE', 'TIME'):
                    if str_col not in mwd_df.columns:
                        mwd_df[str_col] = 'N/A'
                for num_col in ('GR', 'VS', 'ROP', 'TEMP', 'SHKA', 'SHKL', 'VIBEA', 'VIBEL'):
                    if num_col not in mwd_df.columns:
                        mwd_df[num_col] = 0.0
            else:
                mwd_df = None

        if mwd_df is not None and 'TVD' in mwd_df.columns:
            mwd_df['TVD'] = mwd_df['TVD'].abs()

        return survey_df, mwd_df
    except Exception as e:
        st.error(f"LAS parsing error: {str(e)}")
        return None, None


def render_wellbore_trajectory(survey_df: pd.DataFrame) -> None:
    """Render wellbore trajectory in WebGL-free 3D SVG or 2D Plan/Section views."""
    st.markdown("### 3D Wellbore Trajectory")

    view_mode = st.radio(
        "Trajectory Display Mode",
        options=[
            "🌐 3D View (WebGL-Free SVG)",
            "📐 2D Plan & Section Views (SVG)",
        ],
        horizontal=True,
        key="wellbore_trajectory_view_mode",
    )

    if "2D Plan & Section" in view_mode:
        col_plan, col_sect = st.columns(2)

        with col_plan:
            fig_plan = go.Figure()
            fig_plan.add_trace(go.Scatter(
                x=survey_df['EW'],
                y=survey_df['NS'],
                mode='lines+markers',
                name='Well Path',
                line=dict(color='cyan', width=3),
                marker=dict(
                    size=5,
                    color=survey_df['MD'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='MD (ft)'),
                ),
                customdata=survey_df[['MD', 'TVD', 'INC', 'AZM']].values,
                hovertemplate=(
                    '<b>E/W:</b> %{x:.0f} ft<br><b>N/S:</b> %{y:.0f} ft<br>'
                    '<b>MD:</b> %{customdata[0]:.0f} ft<br><b>TVD:</b> %{customdata[1]:.0f} ft<br>'
                    '<b>Inc:</b> %{customdata[2]:.1f}° | <b>Azm:</b> %{customdata[3]:.1f}°<extra></extra>'
                ),
            ))
            fig_plan.add_trace(go.Scatter(
                x=[0], y=[0], mode='markers', name='Surface Location',
                marker=dict(size=11, color='red', symbol='diamond'),
            ))
            fig_plan.add_trace(go.Scatter(
                x=[survey_df['EW'].iloc[-1]], y=[survey_df['NS'].iloc[-1]],
                mode='markers', name='Total Depth',
                marker=dict(size=11, color='lime', symbol='diamond'),
            ))
            fig_plan.update_layout(
                title='Plan View (Top-Down: East/West vs North/South)',
                xaxis_title='East/West (ft)',
                yaxis_title='North/South (ft)',
                height=520,
                template='plotly_dark',
                yaxis=dict(scaleanchor='x', scaleratio=1),
            )
            st.plotly_chart(fig_plan, use_container_width=True)

        with col_sect:
            fig_sect = go.Figure()
            x_col = 'VS' if 'VS' in survey_df.columns and survey_df['VS'].notna().any() else 'NS'
            fig_sect.add_trace(go.Scatter(
                x=survey_df[x_col],
                y=survey_df['TVD'],
                mode='lines+markers',
                name='Vertical Section',
                line=dict(color='orange', width=3),
                marker=dict(
                    size=5,
                    color=survey_df['INC'],
                    colorscale='Plasma',
                    showscale=True,
                    colorbar=dict(title='Inc (°)'),
                ),
                customdata=survey_df[['MD', 'AZM']].values,
                hovertemplate=(
                    f'<b>{x_col}:</b> %{{x:.0f}} ft<br><b>TVD:</b> %{{y:.0f}} ft<br>'
                    '<b>MD:</b> %{customdata[0]:.0f} ft<br><b>Azm:</b> %{customdata[1]:.1f}°<extra></extra>'
                ),
            ))
            fig_sect.update_layout(
                title=f'Section View (TVD vs {x_col})',
                xaxis_title='Vertical Section (ft)' if x_col == 'VS' else 'North/South (ft)',
                yaxis_title='TVD (ft)',
                yaxis=dict(autorange='reversed'),
                height=520,
                template='plotly_dark',
            )
            st.plotly_chart(fig_sect, use_container_width=True)
        return

    # 3D View controls
    ctrl1, ctrl2, ctrl3 = st.columns([1.2, 1.4, 1.4])
    with ctrl1:
        preset = st.selectbox(
            "Camera Angle Preset",
            ["Isometric (NE)", "Isometric (SW)", "Side Profile (East)", "Top-Down Plan"],
            index=0,
            key="trajectory_camera_preset",
        )
    preset_angles = {
        "Isometric (NE)": (35, 22),
        "Isometric (SW)": (215, 22),
        "Side Profile (East)": (0, 0),
        "Top-Down Plan": (0, 90),
    }
    def_az, def_el = preset_angles[preset]
    with ctrl2:
        azim_deg = st.slider("Rotate Azimuth (°)", 0, 360, def_az, 5, key="trajectory_azim_slider")
    with ctrl3:
        elev_deg = st.slider("Tilt Elevation (°)", 0, 90, def_el, 2, key="trajectory_elev_slider")

    use_native_webgl = st.checkbox(
        "Use Experimental Native WebGL Orbit (disable if browser shows WebGL error)",
        value=False,
        key="trajectory_use_native_webgl",
    )

    if use_native_webgl:
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Scatter3d(
            x=survey_df['EW'],
            y=survey_df['NS'],
            z=survey_df['TVD'],
            mode='lines+markers',
            marker=dict(
                size=4,
                color=survey_df['MD'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Measured Depth (ft)'),
            ),
            line=dict(color='cyan', width=3),
            text=[
                f"MD: {md:.0f} ft<br>TVD: {tvd:.0f} ft<br>Inc: {inc:.1f}°<br>Azm: {azm:.1f}°"
                for md, tvd, inc, azm in zip(survey_df['MD'], survey_df['TVD'], survey_df['INC'], survey_df['AZM'])
            ],
            hovertemplate='<b>E/W:</b> %{x:.0f} ft<br><b>N/S:</b> %{y:.0f} ft<br>%{text}<extra></extra>',
            name='Well Path',
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            marker=dict(size=10, color='red', symbol='diamond'), name='Surface Location',
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=[survey_df['EW'].iloc[-1]], y=[survey_df['NS'].iloc[-1]], z=[survey_df['TVD'].iloc[-1]],
            mode='markers', marker=dict(size=10, color='lime', symbol='diamond'), name='Total Depth',
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='East/West (ft)',
                yaxis_title='North/South (ft)',
                zaxis_title='TVD (ft)',
                zaxis=dict(autorange='reversed'),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.5)),
            ),
            height=600,
            template='plotly_dark',
            showlegend=True,
        )
        st.plotly_chart(fig_3d, use_container_width=True)
        return

    # Pure SVG 3D Projection (zero WebGL dependency)
    ew = survey_df['EW'].values.astype(float)
    ns = survey_df['NS'].values.astype(float)
    tvd = survey_df['TVD'].values.astype(float)

    lat_max = max(float(np.nanmax(np.abs(ew))), float(np.nanmax(np.abs(ns))), 100.0)
    tvd_max = max(float(np.nanmax(tvd)), 100.0)

    nx = ew / lat_max
    ny = ns / lat_max
    nz = 1.0 - 2.0 * (tvd / tvd_max)

    az = np.radians(azim_deg)
    el = np.radians(elev_deg)

    def proj(x, y, z):
        x1 = x * np.cos(az) - y * np.sin(az)
        y1 = x * np.sin(az) + y * np.cos(az)
        return x1, z * np.cos(el) + y1 * np.sin(el)

    fig_svg = go.Figure()

    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    bx, by = [], []
    for i, j in edges:
        px_i, py_i = proj(corners[i, 0], corners[i, 1], corners[i, 2])
        px_j, py_j = proj(corners[j, 0], corners[j, 1], corners[j, 2])
        bx.extend([px_i, px_j, None])
        by.extend([py_i, py_j, None])
    fig_svg.add_trace(go.Scatter(
        x=bx, y=by, mode='lines',
        line=dict(color='rgba(160,180,210,0.3)', width=1, dash='dot'),
        hoverinfo='skip', showlegend=False,
    ))

    shx, shy = proj(nx, ny, np.full_like(nz, -1.0))
    fig_svg.add_trace(go.Scatter(
        x=shx, y=shy, mode='lines', name='Floor Projection',
        line=dict(color='rgba(255,255,255,0.25)', width=2, dash='dash'),
        hoverinfo='skip',
    ))

    sx, sy = proj(nx, ny, nz)
    fig_svg.add_trace(go.Scatter(
        x=sx, y=sy, mode='lines+markers', name='Well Path',
        line=dict(color='cyan', width=3),
        marker=dict(
            size=5,
            color=survey_df['MD'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(text='Measured<br>Depth (ft)', font=dict(color='cyan', size=13)),
                tickfont=dict(color='cyan', size=11),
            ),
        ),
        customdata=survey_df[['MD', 'TVD', 'INC', 'AZM', 'EW', 'NS']].values,
        hovertemplate=(
            '<b>E/W:</b> %{customdata[4]:.0f} ft | <b>N/S:</b> %{customdata[5]:.0f} ft<br>'
            '<b>MD:</b> %{customdata[0]:.0f} ft | <b>TVD:</b> %{customdata[1]:.0f} ft<br>'
            '<b>Inc:</b> %{customdata[2]:.1f}° | <b>Azm:</b> %{customdata[3]:.1f}°<extra></extra>'
        ),
    ))

    fig_svg.add_trace(go.Scatter(
        x=[sx[0]], y=[sy[0]], mode='markers+text', name='Surface Location',
        marker=dict(size=11, color='red', symbol='diamond'),
        text=['Surface'], textposition='top center',
    ))
    fig_svg.add_trace(go.Scatter(
        x=[sx[-1]], y=[sy[-1]], mode='markers+text', name='Total Depth',
        marker=dict(size=11, color='lime', symbol='diamond'),
        text=[f"TD ({survey_df['TVD'].iloc[-1]:.0f} ft)"], textposition='bottom center',
    ))

    lbl_pts = [
        (1.12, 0, -1, f'+E ({lat_max:.0f} ft)'),
        (-1.12, 0, -1, f'-W (-{lat_max:.0f} ft)'),
        (0, 1.12, -1, f'+N ({lat_max:.0f} ft)'),
        (0, -1.12, -1, f'-S (-{lat_max:.0f} ft)'),
        (-1.12, -1.12, 1.05, 'TVD: 0 ft'),
        (-1.12, -1.12, -1.05, f'TVD: {tvd_max:.0f} ft'),
    ]
    lx, ly, lt = [], [], []
    for px0, py0, pz0, txt in lbl_pts:
        pxs, pys = proj(px0, py0, pz0)
        lx.append(pxs)
        ly.append(pys)
        lt.append(txt)
    fig_svg.add_trace(go.Scatter(
        x=lx, y=ly, mode='text', text=lt,
        textfont=dict(color='#94a3b8', size=11),
        hoverinfo='skip', showlegend=False,
    ))

    fig_svg.update_layout(
        title='3D Wellbore Trajectory (Pure SVG — WebGL-Free)',
        height=600,
        template='plotly_dark',
        showlegend=True,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.8, 1.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1, range=[-1.6, 1.6]),
    )
    st.plotly_chart(fig_svg, use_container_width=True)
