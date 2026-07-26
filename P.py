import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')



# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Oil & Gold Predictor - Iran War Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# COUNTRY COORDINATES FOR MAP
# ============================================

COUNTRY_COORDS = {
    'Iran': {'lat': 32.4279, 'lon': 53.6880, 'emoji': '🇮🇷'},
    'Iraq': {'lat': 33.2232, 'lon': 43.6793, 'emoji': '🇮🇶'},
    'Israel': {'lat': 31.0461, 'lon': 34.8516, 'emoji': '🇮🇱'},
    'Lebanon': {'lat': 33.8547, 'lon': 35.8623, 'emoji': '🇱🇧'},
    'Syria': {'lat': 34.8021, 'lon': 38.9968, 'emoji': '🇸🇾'},
    'Turkey': {'lat': 38.9637, 'lon': 35.2433, 'emoji': '🇹🇷'},
    'Yemen': {'lat': 15.5527, 'lon': 48.5164, 'emoji': '🇾🇪'},
    'Saudi Arabia': {'lat': 23.8859, 'lon': 45.0792, 'emoji': '🇸🇦'},
    'UAE': {'lat': 23.4241, 'lon': 53.8478, 'emoji': '🇦🇪'},
    'Jordan': {'lat': 30.5852, 'lon': 36.2384, 'emoji': '🇯🇴'},
    'Kuwait': {'lat': 29.3117, 'lon': 47.4818, 'emoji': '🇰🇼'},
    'Bahrain': {'lat': 26.0667, 'lon': 50.5577, 'emoji': '🇧🇭'},
    'Oman': {'lat': 21.4735, 'lon': 55.9754, 'emoji': '🇴🇲'},
    'Qatar': {'lat': 25.3548, 'lon': 51.1839, 'emoji': '🇶🇦'},
    'Palestine': {'lat': 31.9522, 'lon': 35.2332, 'emoji': '🇵🇸'}
}

COUNTRY_EVENT_COLS = {
    'Iran': 'Iran_Events',
    'Iraq': 'Iraq_Events',
    'Israel': 'Israel_Events',
    'Lebanon': 'Lebanon_Events',
    'Syria': 'Syria_Events',
    'Turkey': 'Turkey_Events',
    'Yemen': 'Yemen_Events',
    'Saudi Arabia': 'Saudi_Arabia_Events',
    'UAE': 'United_Arab_Emirates_Events',
    'Jordan': 'Jordan_Events',
    'Kuwait': 'Kuwait_Events',
    'Bahrain': 'Bahrain_Events',
    'Oman': 'Oman_Events',
    'Qatar': 'Qatar_Events',
    'Palestine': 'Palestine_Events'
}

# ============================================
# CUSTOM CSS - PREMIUM DARK THEME
# ============================================

def load_css():
    st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #0a0e1a 0%, #1a1f2e 40%, #0d1b2a 100%);
        }
        
        /* Glassmorphism Cards */
        .card {
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(20px);
            border-radius: 24px;
            padding: 28px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
            margin: 12px 0;
            position: relative;
            overflow: hidden;
        }
        .card:hover {
            transform: translateY(-8px) scale(1.01);
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            border-color: rgba(255,215,0,0.15);
        }
        
        /* Main Title */
        .main-title {
            font-size: 4rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FFD700, #FF8C00, #FFD700, #FF6B00);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 20px 0 10px 0;
            animation: shimmer 4s ease-in-out infinite;
        }
        @keyframes shimmer {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .sub-title {
            text-align: center;
            color: rgba(255,255,255,0.6);
            font-size: 1.2rem;
            margin-bottom: 30px;
            letter-spacing: 2px;
        }
        .sub-title span {
            color: #FFD700;
            font-weight: 600;
        }
        
        /* Section Titles */
        .section-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #FFD700;
            border-left: 5px solid #FFD700;
            padding-left: 18px;
            margin: 35px 0 25px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        /* Metric Cards */
        .metric-card {
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 20px 15px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.06);
            transition: all 0.4s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            background: rgba(255,255,255,0.07);
            border-color: rgba(255,215,0,0.15);
        }
        .metric-icon {
            font-size: 2.8rem;
            display: block;
        }
        .metric-value {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFD700, #FF6B00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-label {
            color: rgba(255,255,255,0.6);
            font-size: 0.85rem;
            margin-top: 4px;
        }
        
        /* Feature Cards */
        .feature-card {
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 16px 12px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.05);
            cursor: pointer;
            transition: all 0.4s ease;
        }
        .feature-card:hover {
            background: rgba(255,215,0,0.06);
            border-color: rgba(255,215,0,0.2);
            transform: translateY(-5px) scale(1.03);
        }
        .feature-card .icon {
            font-size: 2.5rem;
            display: block;
        }
        .feature-card .name {
            color: #FFD700;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .feature-card .value {
            color: rgba(255,255,255,0.5);
            font-size: 0.7rem;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #FFD700, #FF6B00);
            color: #0a0e1a;
            font-weight: 700;
            border: none;
            border-radius: 14px;
            padding: 14px 35px;
            transition: all 0.4s ease;
            box-shadow: 0 4px 20px rgba(255,215,0,0.15);
        }
        .stButton > button:hover {
            transform: scale(1.05) translateY(-3px);
            box-shadow: 0 8px 40px rgba(255,215,0,0.3);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 12px;
            padding: 10px 25px;
            color: rgba(255,255,255,0.5);
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #FFD700, #FF6B00);
            color: #0a0e1a;
        }
        
        /* Prediction Box */
        .prediction-box {
            background: linear-gradient(135deg, rgba(255,215,0,0.05), rgba(255,107,0,0.05));
            border-radius: 24px;
            padding: 35px;
            text-align: center;
            border: 2px solid rgba(255,215,0,0.15);
        }
        .prediction-value {
            font-size: 3.5rem;
            font-weight: 900;
            background: linear-gradient(135deg, #FFD700, #FF6B00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .prediction-label {
            color: rgba(255,255,255,0.6);
            font-size: 1.1rem;
        }
        .prediction-icon {
            font-size: 4rem;
            display: block;
        }
        
        /* Badges */
        .badge {
            display: inline-block;
            padding: 4px 14px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-success { background: rgba(76,175,80,0.2); color: #4CAF50; }
        .badge-warning { background: rgba(255,193,7,0.2); color: #FFC107; }
        .badge-danger { background: rgba(244,67,54,0.2); color: #F44336; }
        .badge-info { background: rgba(33,150,243,0.2); color: #2196F3; }
        .badge-gold { background: rgba(255,215,0,0.15); color: #FFD700; }
        
        /* Text */
        .description {
            color: rgba(255,255,255,0.8);
            font-size: 1.05rem;
            line-height: 1.9;
        }
        .highlight { color: #FFD700; font-weight: 600; }
        .highlight-orange { color: #FF6B00; font-weight: 600; }
        .text-muted { color: rgba(255,255,255,0.4); }
        
        /* Divider */
        .custom-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(255,215,0,0.3), transparent);
            margin: 25px 0;
        }
        
        /* Slide Indicator */
        .slide-indicator {
            text-align: center;
            color: rgba(255,255,255,0.2);
            font-size: 0.85rem;
            margin-top: 30px;
            letter-spacing: 2px;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: rgba(10,14,26,0.95);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, #FFD700, #FF6B00);
            border-radius: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================
import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

@st.cache_data
def load_and_preprocess_data():
    """Load data and perform full feature engineering pipeline"""
    
    # 1. استخدام مسار مطلق لمنع خطأ FileNotFoundError
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'Merged.xlsx')
    
    df = pd.read_excel(file_path)
    
    # STEP 1: Date Unification
    if 'WEEK (acled)' in df.columns:
        df['week_start'] = pd.to_datetime(df['WEEK (acled)'])
    
    date_cols_to_drop = ['WEEK (acled)', 'week_range(gdelt)', 'start_date (yahoo)', 
                         'end_date', 'date_range', 'observation_date (waad)', 'Date']
    existing_drop = [col for col in date_cols_to_drop if col in df.columns]
    df.drop(columns=existing_drop, inplace=True, errors='ignore')
    
    # STEP 2: Remove Redundant Features
    columns_to_drop = ['wti_price', 'Total_Events', 'total_geopolitical_event_volume',
                       'FEDFUNDS', 'OPEC_Production_mbpd']
    existing_drop = [col for col in columns_to_drop if col in df.columns]
    df.drop(columns=existing_drop, inplace=True, errors='ignore')
    
    # STEP 3: Merge Country Events
    lebanon = 'Lebanon_Events' if 'Lebanon_Events' in df.columns else None
    palestine = 'Palestine_Events' if 'Palestine_Events' in df.columns else None
    jordan = 'Jordan_Events' if 'Jordan_Events' in df.columns else None
    israel = 'Israel_Events' if 'Israel_Events' in df.columns else None
    
    if all([lebanon, palestine, jordan, israel]):
        df['Lebanon_Palestien_Jordan_Israel_Events'] = (
            df[lebanon].fillna(0) + df[palestine].fillna(0) + df[jordan].fillna(0) + df[israel].fillna(0)
        )
        df.drop(columns=[lebanon, palestine, jordan, israel], inplace=True, errors='ignore')
    
    turkey = 'Turkey_Events' if 'Turkey_Events' in df.columns else None
    syria = 'Syria_Events' if 'Syria_Events' in df.columns else None
    
    if all([turkey, syria]):
        df['Turkey_And_Syria_Events'] = df[turkey].fillna(0) + df[syria].fillna(0)
        df.drop(columns=[turkey, syria], inplace=True, errors='ignore')
    
    # STEP 4: War Intensity Clustering
    war_features = ['Fatalities', 'Battles_Events', 'Explosions_Events', 'Strategic_Events']
    existing_war = [col for col in war_features if col in df.columns]
    
    if len(existing_war) >= 3:
        # 1. تنظيف القيم المفقودة نهائياً قبل التدريب
        df[existing_war] = df[existing_war].fillna(0)
        
        X_war = df[existing_war].copy()
        
        # 2. التأكد التام من عدم وجود أعداد نهائية أو NaN
        X_war = X_war.dropna()
        
        war_scaler = StandardScaler()
        X_war_scaled = war_scaler.fit_transform(X_war)
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        
        # تعيين النتائج للمؤشرات الصحيحة
        df.loc[X_war.index, 'War_Cluster'] = kmeans.fit_predict(X_war_scaled)
        
        cluster_severity = df.groupby('War_Cluster')[existing_war].mean().sum(axis=1).sort_values()
        cluster_order = cluster_severity.index.tolist()
        
        cluster_labels = {cluster_order[0]: 'Low', cluster_order[1]: 'Medium', cluster_order[2]: 'High'}
        df['War_Intensity'] = df['War_Cluster'].map(cluster_labels)
        
        intensity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        df['War_Intensity_Code'] = df['War_Intensity'].map(intensity_map)
        
        df = df.sort_values('week_start').reset_index(drop=True)
        df['War_Intensity_Lag1'] = df['War_Intensity_Code'].shift(1).fillna(0)

        
   
    
    # STEP 5: Lag Features
    if 'gold_price' in df.columns:
        df['Gold_Lag_1'] = df['gold_price'].shift(1)
        df['Gold_Lag_2'] = df['gold_price'].shift(2)
        df['Gold_Lag_4'] = df['gold_price'].shift(4)
    
    if 'target_brent_price' in df.columns:
        df['Brent_Lag_1'] = df['target_brent_price'].shift(1)
        df['Brent_Lag_2'] = df['target_brent_price'].shift(2)
    
    return df

# ============================================
# TRAIN MODELS
# ============================================

@st.cache_resource
def train_models(df):
    models = {}
    
    # Gold Model
    gold_features = ['Gold_Lag_1', 'Gold_Lag_2', 'Gold_Lag_4', 'War_Intensity_Lag1',
                     'usd_index', 'defense_stocks', 'sp500_etf', 'energy_stocks',
                     'oil_volatility', 'market_volatility', 'US_Crude_Inventories_mb']
    
    existing_gold = [col for col in gold_features if col in df.columns]
    
    if len(existing_gold) >= 5 and 'gold_price' in df.columns:
        df_gold = df.dropna(subset=existing_gold + ['gold_price']).copy()
        
        if len(df_gold) > 30:
            X_gold = df_gold[existing_gold]
            y_gold = df_gold['gold_price']
            
            split = int(len(df_gold) * 0.8)
            X_train, X_test = X_gold.iloc[:split], X_gold.iloc[split:]
            y_train, y_test = y_gold.iloc[:split], y_gold.iloc[split:]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            ridge = Ridge(alpha=10)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            
            models['ridge'] = {
                'model': ridge,
                'scaler': scaler,
                'features': existing_gold,
                'score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
    
    # Oil Model
    oil_features = ['Brent_Lag_1', 'Brent_Lag_2', 'War_Intensity_Lag1',
                    'usd_index', 'defense_stocks', 'sp500_etf', 'energy_stocks',
                    'oil_volatility', 'market_volatility', 'US_Crude_Inventories_mb']
    
    existing_oil = [col for col in oil_features if col in df.columns]
    
    if len(existing_oil) >= 5 and 'target_brent_price' in df.columns:
        df_oil = df.dropna(subset=existing_oil + ['target_brent_price']).copy()
        
        if len(df_oil) > 30:
            X_oil = df_oil[existing_oil]
            y_oil = df_oil['target_brent_price']
            
            split = int(len(df_oil) * 0.8)
            X_train, X_test = X_oil.iloc[:split], X_oil.iloc[split:]
            y_train, y_test = y_oil.iloc[:split], y_oil.iloc[split:]
            
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            
            models['rf'] = {
                'model': rf,
                'features': existing_oil,
                'score': r2_score(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
    
    return models

# ============================================
# MAP FUNCTIONS
# ============================================

def prepare_map_data(df, year=None, intensity_filter=None):
    """Prepare data for the interactive map"""
    
    if year is not None and 'week_start' in df.columns:
        df_filtered = df[df['week_start'].dt.year == year].copy()
    else:
        df_filtered = df.copy()
    
    if intensity_filter and 'War_Intensity' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['War_Intensity'].isin(intensity_filter)]
    
    map_data = []
    
    for country, col in COUNTRY_EVENT_COLS.items():
        if col in df_filtered.columns:
            events = df_filtered[col].sum()
            
            if country in COUNTRY_COORDS:
                coords = COUNTRY_COORDS[country]
                map_data.append({
                    'country': country,
                    'emoji': coords['emoji'],
                    'events': int(events),
                    'lat': coords['lat'],
                    'lon': coords['lon'],
                    'size': np.sqrt(events + 1) * 8 + 5
                })
    
    return pd.DataFrame(map_data)


def create_interactive_map(df_map, year):
    """Create an interactive Plotly map"""
    
    if len(df_map) == 0:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scattermapbox(
        lat=df_map['lat'],
        lon=df_map['lon'],
        mode='markers+text',
        marker=dict(
            size=df_map['size'],
            color=df_map['events'],
            colorscale='Portland',
            showscale=True,
            colorbar=dict(title="Events", titleside="right", thickness=20, len=0.6),
            opacity=0.85,
            line=dict(width=2, color='rgba(255,255,255,0.3)')
        ),
        text=df_map['emoji'],
        textposition='top center',
        textfont=dict(size=22, color='white'),
        hovertext=df_map.apply(
            lambda row: f"<b>{row['emoji']} {row['country']}</b><br>Events: {row['events']:,}", 
            axis=1
        ),
        hoverinfo='text'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='dark',
            center=dict(lat=30, lon=45),
            zoom=4.2,
        ),
        height=600,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(
            text=f"🌍 Middle East Geopolitical Events - {year}",
            font=dict(size=22, color='#FFD700')
        )
    )
    
    return fig


def display_geopolitical_map(df):
    """Display the interactive map"""
    
    st.markdown("""
    <div class="section-title">🗺️ Geopolitical Map of the Middle East</div>
    """, unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if 'week_start' in df.columns:
            years = sorted(df['week_start'].dt.year.unique())
            selected_year = st.selectbox("📅 Select Year", years, index=len(years)-1)
        else:
            selected_year = None
    
    with col2:
        if 'War_Intensity' in df.columns:
            intensity_options = ['Low', 'Medium', 'High']
            selected_intensities = st.multiselect(
                "⚡ War Intensity", intensity_options, default=intensity_options
            )
        else:
            selected_intensities = None
    
    # Prepare and display map
    df_map = prepare_map_data(df, year=selected_year, intensity_filter=selected_intensities)
    
    if len(df_map) > 0:
        fig = create_interactive_map(df_map, selected_year)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ No data available for the selected filters")
    
    # Statistics
    st.markdown("### 📊 Event Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_events = df_map['events'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{total_events:,}</div>
            <div class="metric-label">Total Events</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        active_countries = len(df_map[df_map['events'] > 0])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🏳️</div>
            <div class="metric-value">{active_countries}</div>
            <div class="metric-label">Active Countries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if len(df_map) > 0:
            max_events = df_map['events'].max()
            max_country = df_map[df_map['events'] == max_events]['country'].iloc[0]
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔥</div>
                <div class="metric-value">{max_country}</div>
                <div class="metric-label">Most Active</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        avg_events = df_map['events'].mean() if len(df_map) > 0 else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{avg_events:.1f}</div>
            <div class="metric-label">Avg/Country</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Detailed table
    with st.expander("📋 Detailed Country Breakdown", expanded=False):
        display_df = df_map[['emoji', 'country', 'events']].copy()
        display_df = display_df.sort_values('events', ascending=False)
        display_df.columns = ['', 'Country', 'Events']
        
        max_ev = display_df['Events'].max() if len(display_df) > 0 else 1
        display_df['Intensity'] = display_df['Events'].apply(
            lambda x: '🔴 High' if x > max_ev * 0.6 else ('🟡 Medium' if x > max_ev * 0.2 else '🟢 Low')
        )
        
        st.dataframe(display_df, use_container_width=True)
    
    # Time series chart
    st.markdown("### 📈 Event Trends Over Time")
    
    countries_with_events = [c for c in COUNTRY_EVENT_COLS.keys() 
                            if COUNTRY_EVENT_COLS[c] in df.columns]
    
    selected_countries = st.multiselect(
        "Select countries", countries_with_events,
        default=['Iran', 'Israel', 'Lebanon'][:min(3, len(countries_with_events))]
    )
    
    if selected_countries and 'week_start' in df.columns:
        df_ts = df[df['week_start'].dt.year == selected_year].copy() if selected_year else df.copy()
        
        fig_ts = go.Figure()
        colors = ['#FF6B00', '#FFD700', '#FF4444', '#4CAF50', '#2196F3', '#9C27B0']
        
        for i, country in enumerate(selected_countries):
            col = COUNTRY_EVENT_COLS.get(country)
            if col in df_ts.columns:
                weekly = df_ts.groupby('week_start')[col].sum().reset_index()
                fig_ts.add_trace(go.Scatter(
                    x=weekly['week_start'],
                    y=weekly[col],
                    name=f"{COUNTRY_COORDS.get(country, {}).get('emoji', '')} {country}",
                    line=dict(color=colors[i % len(colors)], width=2.5),
                    mode='lines+markers'
                ))
        
        fig_ts.update_layout(
            template='plotly_dark',
            height=350,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            xaxis_title="Date",
            yaxis_title="Events",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_ts, use_container_width=True)
    
    st.markdown("""
    <div class="slide-indicator">📍 Slide 4 of 6</div>
    """, unsafe_allow_html=True)

# ============================================
# DISPLAY FUNCTIONS FOR OTHER SLIDES
# ============================================

def display_overview(df):
    """Display overview slide"""
    
    st.markdown('<div class="main-title">🌍 Oil & Gold Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Predicting <span>Oil</span> & <span>Gold</span> Prices in Light of the <span>Iran War</span></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
            <p class="description">
                This project analyzes and predicts <span class="highlight">Brent Oil</span> and 
                <span class="highlight">Gold</span> prices using geopolitical event data from the 
                Middle East region, with special focus on the <span class="highlight-orange">Iran War</span> 
                and its impact on financial markets.
            </p>
            <div class="custom-divider"></div>
            <p class="description">
                <strong>🛠️ Technologies:</strong><br>
                <span class="badge badge-info">🐍 Python</span>
                <span class="badge badge-gold">📊 Streamlit</span>
                <span class="badge badge-success">🤖 Scikit-learn</span>
                <span class="badge badge-warning">📈 Plotly</span>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:4rem;">🌍</div>
            <h3 style="color:#FFD700;">Middle East</h3>
            <p style="color:#aaa;">Geopolitical Analysis</p>
            <div class="custom-divider"></div>
            <div style="display:flex; justify-content:space-around;">
                <div><span style="color:#FF6B00;">🛢️</span> Oil</div>
                <div><span style="color:#FFD700;">🥇</span> Gold</div>
                <div><span style="color:#FF4444;">💥</span> Conflict</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown('<div class="section-title">📊 Quick Stats</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        latest_brent = df['target_brent_price'].iloc[-1] if 'target_brent_price' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🛢️</div>
            <div class="metric-value">${latest_brent:.2f}</div>
            <div class="metric-label">Latest Brent Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latest_gold = df['gold_price'].iloc[-1] if 'gold_price' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">🥇</div>
            <div class="metric-value">${latest_gold:.2f}</div>
            <div class="metric-label">Latest Gold Price</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_ovx = df['oil_volatility'].mean() if 'oil_volatility' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📈</div>
            <div class="metric-value">{avg_ovx:.2f}</div>
            <div class="metric-label">Avg Oil Volatility</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_events = df['Battles_Events'].mean() if 'Battles_Events' in df.columns else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">⚔️</div>
            <div class="metric-value">{avg_events:.0f}</div>
            <div class="metric-label">Avg Weekly Battles</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price trends chart
    st.markdown('<div class="section-title">📈 Price Trends</div>', unsafe_allow_html=True)
    
    if 'week_start' in df.columns and 'target_brent_price' in df.columns:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df['week_start'], y=df['target_brent_price'],
                       name="Brent Oil", line=dict(color="#FF6B00", width=3)),
            secondary_y=False
        )
        
        if 'gold_price' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['week_start'], y=df['gold_price'] / 10,
                           name="Gold (÷10)", line=dict(color="#FFD700", width=2, dash="dash")),
                secondary_y=True
            )
        
        fig.update_layout(template="plotly_dark", height=400, hovermode="x unified")
        fig.update_yaxes(title_text="Brent Oil ($)", secondary_y=False)
        fig.update_yaxes(title_text="Gold / 10 ($)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    <div class="slide-indicator">📍 Slide 1 of 6</div>
    """, unsafe_allow_html=True)


def display_feature_engineering(df):
    """Display feature engineering steps"""
    
    st.markdown("""
    <div class="section-title">🔬 Feature Engineering</div>
    """, unsafe_allow_html=True)
    
    # Step 1: Date Alignment
    with st.expander("📅 Step 1: Date Alignment", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FF6B00;">❌ Before</h4>
                <ul style="color:#ccc;">
                    <li>Multiple date columns from different sources</li>
                    <li>Dates not aligned across datasets</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FFD700;">✅ After</h4>
                <ul style="color:#ccc;">
                    <li>Unified <span class="highlight">week_start</span> column</li>
                    <li>All dates aligned to weekly intervals</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Step 2: Remove Redundant
    with st.expander("🗑️ Step 2: Remove Redundant Features", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FF6B00;">❌ Removed</h4>
                <ul style="color:#ccc;">
                    <li><span class="highlight">wti_price</span> (95% correlated)</li>
                    <li><span class="highlight">Total_Events</span> (sum of others)</li>
                    <li><span class="highlight">FEDFUNDS</span> (low correlation)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card">
                <h4 style="color:#FFD700;">✅ Result</h4>
                <ul style="color:#ccc;">
                    <li>Reduced from {len(df.columns)} features</li>
                    <li>Reduced multicollinearity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Step 3: Merge Country Events
    with st.expander("🔗 Step 3: Merge Country Events", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FF6B00;">❌ Before</h4>
                <ul style="color:#ccc;">
                    <li>Lebanon, Palestine, Jordan, Israel (separate)</li>
                    <li>Turkey, Syria (separate)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FFD700;">✅ After</h4>
                <ul style="color:#ccc;">
                    <li><span class="highlight">Lebanon_Palestien_Jordan_Israel_Events</span></li>
                    <li><span class="highlight">Turkey_And_Syria_Events</span></li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Step 4: War Intensity
    with st.expander("⚡ Step 4: War Intensity Clustering", expanded=True):
        if 'War_Intensity' in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, names='War_Intensity', title='War Intensity Distribution',
                            color='War_Intensity',
                            color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFC107', 'High': '#F44336'})
                fig.update_layout(template="plotly_dark", height=300)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                counts = df['War_Intensity'].value_counts()
                st.markdown(f"""
                <div class="card">
                    <h4 style="color:#FFD700;">📊 Statistics</h4>
                    <ul style="color:#ccc;">
                        <li>🟢 Low: {counts.get('Low', 0)} weeks</li>
                        <li>🟡 Medium: {counts.get('Medium', 0)} weeks</li>
                        <li>🔴 High: {counts.get('High', 0)} weeks</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Step 5: Lag Features
    with st.expander("📈 Step 5: Lag Features", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FFD700;">🥇 Gold Lags</h4>
                <ul style="color:#ccc;">
                    <li><span class="highlight">Gold_Lag_1</span>: 1 week ago</li>
                    <li><span class="highlight">Gold_Lag_2</span>: 2 weeks ago</li>
                    <li><span class="highlight">Gold_Lag_4</span>: 4 weeks ago</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="card">
                <h4 style="color:#FF6B00;">🛢️ Oil Lags</h4>
                <ul style="color:#ccc;">
                    <li><span class="highlight">Brent_Lag_1</span>: 1 week ago</li>
                    <li><span class="highlight">Brent_Lag_2</span>: 2 weeks ago</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Summary
    st.markdown('<div class="section-title">📊 Final Dataset</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📋</div>
            <div class="metric-value">{len(df.columns)}</div>
            <div class="metric-label">Total Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Rows</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        if 'week_start' in df.columns:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📅</div>
                <div class="metric-value">{df['week_start'].min().strftime('%Y-%m')} → {df['week_start'].max().strftime('%Y-%m')}</div>
                <div class="metric-label">Date Range</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="slide-indicator">📍 Slide 3 of 6</div>
    """, unsafe_allow_html=True)


def display_prediction_models(df, models):
    """Display prediction models"""
    
    st.markdown("""
    <div class="section-title">📈 Prediction Models</div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🥇 Gold Model", "🛢️ Oil Model"])
    
    with tab1:
        st.markdown("### 🥇 Gold Price Prediction - Ridge Regression")
        
        if 'ridge' in models:
            m = models['ridge']
            st.markdown(f"""
            <div class="card">
                <p class="description">
                    📊 Model Performance:<br>
                    <span class="highlight">R² Score:</span> {m['score']:.3f} &nbsp;|&nbsp;
                    <span class="highlight">MAE:</span> ${m['mae']:.2f} &nbsp;|&nbsp;
                    <span class="highlight">RMSE:</span> ${m['rmse']:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Input fields
            st.markdown("#### 📝 Enter Values")
            
            col1, col2 = st.columns(2)
            with col1:
                gold_lag1 = st.number_input("Gold_Lag_1", value=float(df['gold_price'].iloc[-1]))
                gold_lag2 = st.number_input("Gold_Lag_2", value=float(df['gold_price'].iloc[-2]))
                gold_lag4 = st.number_input("Gold_Lag_4", value=float(df['gold_price'].iloc[-4]))
                war_intensity = st.selectbox("War_Intensity_Lag1", [0, 1, 2], index=1)
                usd_index = st.number_input("USD Index", value=float(df['usd_index'].iloc[-1]))
            
            with col2:
                defense_stocks = st.number_input("Defense Stocks", value=float(df['defense_stocks'].iloc[-1]))
                sp500_etf = st.number_input("S&P 500 ETF", value=float(df['sp500_etf'].iloc[-1]))
                energy_stocks = st.number_input("Energy Stocks", value=float(df['energy_stocks'].iloc[-1]))
                oil_volatility = st.number_input("Oil Volatility", value=float(df['oil_volatility'].iloc[-1]))
                market_volatility = st.number_input("Market Volatility", value=float(df['market_volatility'].iloc[-1]))
                crude_inventories = st.number_input("US Crude Inventories", value=float(df['US_Crude_Inventories_mb'].iloc[-1]))
            
            if st.button("🔮 Predict Gold Price", key="gold_predict"):
                try:
                    features = np.array([[gold_lag1, gold_lag2, gold_lag4, war_intensity,
                                         usd_index, defense_stocks, sp500_etf, energy_stocks,
                                         oil_volatility, market_volatility, crude_inventories]])
                    features_scaled = m['scaler'].transform(features)
                    prediction = m['model'].predict(features_scaled)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-icon">🥇</div>
                        <div class="prediction-label">Predicted Gold Price</div>
                        <div class="prediction-value">${prediction[0]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Gold model not available. Please train the model first.")
    
    with tab2:
        st.markdown("### 🛢️ Oil Price Prediction - Random Forest")
        
        if 'rf' in models:
            m = models['rf']
            st.markdown(f"""
            <div class="card">
                <p class="description">
                    📊 Model Performance:<br>
                    <span class="highlight">R² Score:</span> {m['score']:.3f} &nbsp;|&nbsp;
                    <span class="highlight">MAE:</span> ${m['mae']:.2f} &nbsp;|&nbsp;
                    <span class="highlight">RMSE:</span> ${m['rmse']:.2f}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📝 Enter Values")
            
            col1, col2 = st.columns(2)
            with col1:
                brent_lag1 = st.number_input("Brent_Lag_1", value=float(df['target_brent_price'].iloc[-1]))
                brent_lag2 = st.number_input("Brent_Lag_2", value=float(df['target_brent_price'].iloc[-2]))
                war_intensity_oil = st.selectbox("War_Intensity_Lag1", [0, 1, 2], index=1, key="oil_war")
                usd_index_oil = st.number_input("USD Index", value=float(df['usd_index'].iloc[-1]), key="oil_usd")
            
            with col2:
                defense_stocks_oil = st.number_input("Defense Stocks", value=float(df['defense_stocks'].iloc[-1]), key="oil_defense")
                sp500_etf_oil = st.number_input("S&P 500 ETF", value=float(df['sp500_etf'].iloc[-1]), key="oil_sp500")
                energy_stocks_oil = st.number_input("Energy Stocks", value=float(df['energy_stocks'].iloc[-1]), key="oil_energy")
                oil_volatility_oil = st.number_input("Oil Volatility", value=float(df['oil_volatility'].iloc[-1]), key="oil_vol")
                market_volatility_oil = st.number_input("Market Volatility", value=float(df['market_volatility'].iloc[-1]), key="oil_market")
                crude_inventories_oil = st.number_input("US Crude Inventories", value=float(df['US_Crude_Inventories_mb'].iloc[-1]), key="oil_crude")
            
            if st.button("🔮 Predict Oil Price", key="oil_predict"):
                try:
                    features = np.array([[brent_lag1, brent_lag2, war_intensity_oil,
                                         usd_index_oil, defense_stocks_oil, sp500_etf_oil, energy_stocks_oil,
                                         oil_volatility_oil, market_volatility_oil, crude_inventories_oil]])
                    prediction = m['model'].predict(features)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-icon">🛢️</div>
                        <div class="prediction-label">Predicted Brent Oil Price</div>
                        <div class="prediction-value">${prediction[0]:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("⚠️ Oil model not available. Please train the model first.")
    
    st.markdown("""
    <div class="slide-indicator">📍 Slide 5 of 6</div>
    """, unsafe_allow_html=True)


def display_results(models):
    """Display results and insights"""
    
    st.markdown("""
    <div class="section-title">📉 Results & Insights</div>
    """, unsafe_allow_html=True)
    
    # Model Comparison
    st.markdown("### 📊 Model Performance Comparison")
    
    comparison_data = []
    for name, m in models.items():
        comparison_data.append({
            'Model': 'Ridge (Gold)' if name == 'ridge' else 'Random Forest (Oil)',
            'R² Score': f"{m['score']:.3f}",
            'MAE': f"${m['mae']:.2f}",
            'RMSE': f"${m['rmse']:.2f}"
        })
    
    if comparison_data:
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
    
    # Key Insights
    st.markdown("### 💡 Key Insights")
    
    st.markdown("""
    <div class="card">
        <h4 style="color:#FFD700;">📌 Geopolitical Impact</h4>
        <ul style="color:#ccc;">
            <li>🔴 <span class="highlight">High war intensity</span> correlates with increased oil volatility</li>
            <li>🟡 <span class="highlight">Gold</span> shows strong safe-haven behavior during conflicts</li>
            <li>🟢 <span class="highlight">Defense stocks</span> tend to rise during geopolitical tensions</li>
        </ul>
    </div>
    <div class="card">
        <h4 style="color:#FFD700;">📌 Model Insights</h4>
        <ul style="color:#ccc;">
            <li>📈 <span class="highlight">Random Forest</span> performs better for oil price prediction</li>
            <li>📉 <span class="highlight">Ridge Regression</span> works well for gold with lag features</li>
            <li>⚡ <span class="highlight">War intensity</span> is a significant predictor for both assets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="slide-indicator">📍 Slide 6 of 6</div>
    """, unsafe_allow_html=True)

# ============================================
# MAIN APP
# ============================================

def main():
    # Load data
    df = load_and_preprocess_data()
    models = train_models(df)
    
    # Sidebar
    st.sidebar.markdown("""
    <div style="text-align:center; padding:10px 0;">
        <div style="font-size:3rem;">🌍</div>
        <div style="color:#FFD700; font-weight:700; font-size:1.2rem;">Oil & Gold</div>
        <div style="color:#aaa; font-size:0.85rem;">Predictor</div>
    </div>
    <div class="custom-divider"></div>
    """, unsafe_allow_html=True)
    
    slides = [
        "🏠 Overview",
        "🔬 Feature Engineering",
        "🗺️ Geopolitical Map",
        "📈 Prediction Models",
        "📉 Results & Insights"
    ]
    
    selected_slide = st.sidebar.radio("📌 Navigation", slides)
    
    # Display selected slide
    if selected_slide == "🏠 Overview":
        display_overview(df)
    elif selected_slide == "🔬 Feature Engineering":
        display_feature_engineering(df)
    elif selected_slide == "🗺️ Geopolitical Map":
        display_geopolitical_map(df)
    elif selected_slide == "📈 Prediction Models":
        display_prediction_models(df, models)
    elif selected_slide == "📉 Results & Insights":
        display_results(models)

if __name__ == "__main__":
    main()