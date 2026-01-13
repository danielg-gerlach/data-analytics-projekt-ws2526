"""
NYC Taxi Fare & Tip Predictor - Streamlit Application

Diese Anwendung nutzt Machine Learning Modelle zur Vorhersage von:
1. Fahrtpreis (Fare Amount) - Basierend auf Distanz (Haversine), Zeit, Location, Borough
2. Trinkgeld (Tip Amount) - Basierend auf Distanz, Zeit, Wetter (OHNE Fare Amount)

Architektur: Unabhängige Prediction Pipeline
- Fare Model: Random Forest mit Borough-Features
- Tip Model: Random Forest mit Wetter-Features (fairer Baseline-Vergleich)

Die Modelle werden aus HuggingFace Repo geladen. 
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, date
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# Hugging Face für Model-Download
from huggingface_hub import hf_hub_download

# Haversine-Funktion importieren
from utils.distance import haversine_distance, meters_to_miles

# Hugging Face Repository
HF_REPO_ID = "dnltre/taxi-nyc-models"

# =============================================================================
# KONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="NYC Taxi Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pfade
BASE_PATH = Path(__file__).parent
MODELS_PATH = BASE_PATH / 'models' / 'final'
DATA_PATH = BASE_PATH / 'data'
WEATHER_PATH = BASE_PATH / 'weather'

# Fallback auf alte Modelle falls final nicht existiert
if not MODELS_PATH.exists() or not (MODELS_PATH / 'fare_model.pkl').exists():
    MODELS_PATH = BASE_PATH / 'models'

# =============================================================================
# HILFSFUNKTIONEN - DATA LOADING
# =============================================================================

@st.cache_resource
def load_models():
    """Lade beide ML-Modelle und deren Feature-Listen - lokal oder von Hugging Face"""
    
    # Prüfe ob lokale Modelle existieren
    local_exists = MODELS_PATH.exists() and (MODELS_PATH / 'fare_model.pkl').exists()
    
    if local_exists:
        # Lokaler Modus (für Entwicklung)
        try:
            with open(MODELS_PATH / 'fare_model.pkl', 'rb') as f:
                fare_model = pickle.load(f)
            with open(MODELS_PATH / 'fare_features.pkl', 'rb') as f:
                fare_features = pickle.load(f)
            with open(MODELS_PATH / 'tip_model.pkl', 'rb') as f:
                tip_model = pickle.load(f)
            with open(MODELS_PATH / 'tip_features.pkl', 'rb') as f:
                tip_features = pickle.load(f)
            return fare_model, fare_features, tip_model, tip_features
        except Exception:
            pass  # Fallback zu Hugging Face
    
    # Download von Hugging Face (für Deployment / Streamlit Cloud)
    try:
        fare_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="fare_model.pkl")
        fare_features_path = hf_hub_download(repo_id=HF_REPO_ID, filename="fare_features.pkl")
        tip_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="tip_model.pkl")
        tip_features_path = hf_hub_download(repo_id=HF_REPO_ID, filename="tip_features.pkl")
        
        with open(fare_model_path, 'rb') as f:
            fare_model = pickle.load(f)
        with open(fare_features_path, 'rb') as f:
            fare_features = pickle.load(f)
        with open(tip_model_path, 'rb') as f:
            tip_model = pickle.load(f)
        with open(tip_features_path, 'rb') as f:
            tip_features = pickle.load(f)
        
        return fare_model, fare_features, tip_model, tip_features
        
    except Exception as e:
        st.error(f"Fehler beim Laden der Modelle: {e}")
        st.error("Tipp: Prüfe die Hugging Face Repo-ID oder führe die Notebooks lokal aus.")
        st.stop()

@st.cache_data
def load_model_info():
    """Lade Modell-Metriken aus den Info-Dateien - lokal oder von Hugging Face"""
    try:
        # Versuche lokal
        if MODELS_PATH.exists() and (MODELS_PATH / 'fare_model_info.pkl').exists():
            with open(MODELS_PATH / 'fare_model_info.pkl', 'rb') as f:
                fare_info = pickle.load(f)
            with open(MODELS_PATH / 'tip_model_info.pkl', 'rb') as f:
                tip_info = pickle.load(f)
        else:
            # Download von Hugging Face
            fare_info_path = hf_hub_download(repo_id=HF_REPO_ID, filename="fare_model_info.pkl")
            tip_info_path = hf_hub_download(repo_id=HF_REPO_ID, filename="tip_model_info.pkl")
            
            with open(fare_info_path, 'rb') as f:
                fare_info = pickle.load(f)
            with open(tip_info_path, 'rb') as f:
                tip_info = pickle.load(f)
        
        return fare_info, tip_info
    except Exception as e:
        st.warning(f"Modell-Info konnte nicht geladen werden: {e}")
        return None, None

@st.cache_data
def load_zone_data():
    """Lade Taxi-Zonen Daten mit GPS-Koordinaten - lokal oder von Hugging Face"""
    try:
        # Versuche lokal
        if DATA_PATH.exists() and (DATA_PATH / 'taxi_zone_lookup.csv').exists():
            zones = pd.read_csv(DATA_PATH / 'taxi_zone_lookup.csv')
            coords = pd.read_csv(DATA_PATH / 'taxi_zone_coordinates.csv')
        else:
            # Download von Hugging Face
            zones_path = hf_hub_download(repo_id=HF_REPO_ID, filename="taxi_zone_lookup.csv")
            coords_path = hf_hub_download(repo_id=HF_REPO_ID, filename="taxi_zone_coordinates.csv")
            zones = pd.read_csv(zones_path)
            coords = pd.read_csv(coords_path)
        
        zones['display_name'] = zones['Borough'] + ' - ' + zones['Zone']
        zones = zones.merge(coords, on='LocationID', how='left')
        
        return zones
    except Exception as e:
        st.error(f"Fehler beim Laden der Zone-Daten: {e}")
        st.stop()

@st.cache_data
def load_weather_data():
    """Lade Wetterdaten für Januar 2023 - lokal oder von Hugging Face"""
    try:
        # Versuche lokal
        if WEATHER_PATH.exists() and (WEATHER_PATH / 'weather_data.parquet').exists():
            weather_df = pd.read_parquet(WEATHER_PATH / 'weather_data.parquet')
        else:
            # Download von Hugging Face
            weather_path = hf_hub_download(repo_id=HF_REPO_ID, filename="weather_data.parquet")
            weather_df = pd.read_parquet(weather_path)
        
        # Finde Datetime-Spalte
        datetime_col = None
        for col in weather_df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                datetime_col = col
                break
        
        if datetime_col:
            weather_df['datetime'] = pd.to_datetime(weather_df[datetime_col])
        else:
            weather_df['datetime'] = weather_df.index
        
        weather_df['hour'] = weather_df['datetime'].dt.floor('H')
        
        # Identifiziere Wetter-Spalten
        weather_cols = {}
        mappings = {
            'temperature': ['temp', 'temperature'],
            'humidity': ['humidity', 'relative_humidity'],
            'rain': ['rain', 'precipitation', 'precip'],
            'wind_speed': ['wind_speed', 'windspeed', 'wind']
        }
        
        for target, possible in mappings.items():
            for col in weather_df.columns:
                if any(p in col.lower() for p in possible):
                    weather_cols[target] = col
                    break
        
        # Aggregiere auf Stundenbasis
        agg_dict = {weather_cols[key]: 'mean' for key in weather_cols if key in weather_cols}
        weather_hourly = weather_df.groupby('hour').agg(agg_dict).reset_index()
        weather_hourly.columns = ['hour'] + list(weather_cols.keys())
        
        return weather_hourly
    except Exception as e:
        st.warning(f"Wetterdaten konnten nicht geladen werden: {e}")
        return None

@st.cache_data
def load_top_locations():
    """Lade Top Locations für das Tip-Modell - lokal oder von Hugging Face"""
    try:
        # Versuche lokal
        if MODELS_PATH.exists() and (MODELS_PATH / 'tip_locations.pkl').exists():
            with open(MODELS_PATH / 'tip_locations.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            # Download von Hugging Face
            locations_path = hf_hub_download(repo_id=HF_REPO_ID, filename="tip_locations.pkl")
            with open(locations_path, 'rb') as f:
                return pickle.load(f)
    except:
        # Fallback: Standard Top 20 Locations
        return {
            'top_pickup': [237, 161, 236, 162, 186, 170, 164, 230, 48, 142,
                          263, 239, 234, 163, 249, 68, 246, 114, 229, 79],
            'top_dropoff': [237, 161, 236, 162, 186, 170, 164, 230, 48, 142,
                           263, 239, 234, 163, 249, 68, 246, 114, 229, 79]
        }

@st.cache_data
def load_route_statistics():
    """Lade historische Routen-Statistiken - lokal oder von Hugging Face"""
    try:
        # Versuche lokal
        if DATA_PATH.exists() and (DATA_PATH / 'route_price_statistics.csv').exists():
            route_stats = pd.read_csv(DATA_PATH / 'route_price_statistics.csv')
        else:
            # Download von Hugging Face
            route_stats_path = hf_hub_download(repo_id=HF_REPO_ID, filename="route_price_statistics.csv")
            route_stats = pd.read_csv(route_stats_path)
        return route_stats
    except Exception as e:
        st.warning(f"Routen-Statistiken konnten nicht geladen werden: {e}")
        return None

def get_route_statistics(route_stats_df, pickup_id, dropoff_id):
    """Hole historische Statistiken für eine bestimmte Route"""
    if route_stats_df is None:
        return None
    
    route_data = route_stats_df[
        (route_stats_df['PULocationID'] == pickup_id) & 
        (route_stats_df['DOLocationID'] == dropoff_id)
    ]
    
    if len(route_data) == 0:
        return None
    
    row = route_data.iloc[0]
    return {
        'mean_fare': row['mean_fare_amount'],
        'std_fare': row['std_fare_amount'],
        'median_fare': row['median_fare_amount'],
        'mean_tip': row['mean_tip_amount'],
        'std_tip': row['std_tip_amount'],
        'median_tip': row['median_tip_amount'],
        'mean_price_per_mile': row['mean_price_per_mile'],
        'trip_count': int(row['trip_count']),
        'route_name': row['route']
    }

# =============================================================================
# HILFSFUNKTIONEN - BERECHNUNGEN
# =============================================================================

def calculate_distance_haversine(pickup_id, dropoff_id, zones_df):
    """
    Berechne Distanz zwischen zwei Locations mit Haversine-Formel.
    Gibt die Distanz in Meilen zurueck.
    """
    pickup_row = zones_df[zones_df['LocationID'] == pickup_id]
    dropoff_row = zones_df[zones_df['LocationID'] == dropoff_id]
    
    if pickup_row.empty or dropoff_row.empty:
        return 2.5  # Fallback
    
    pickup_lat = pickup_row['latitude'].values[0]
    pickup_lon = pickup_row['longitude'].values[0]
    dropoff_lat = dropoff_row['latitude'].values[0]
    dropoff_lon = dropoff_row['longitude'].values[0]
    
    if pd.isna(pickup_lat) or pd.isna(dropoff_lat):
        return 2.5  # Fallback
    
    # Haversine-Distanz berechnen (in Metern)
    distance_m = haversine_distance(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    
    # Konvertiere zu Meilen
    distance_miles = meters_to_miles(distance_m)
    
    # Minimum 0.5 Meilen (für sehr kurze Fahrten)
    return max(0.5, distance_miles)

def calculate_duration(distance, hour, is_rush_hour):
    """Berechne geschaetzte Fahrtdauer basierend auf Distanz und Verkehrslage"""
    # Durchschnittsgeschwindigkeiten (mph)
    if is_rush_hour:
        avg_speed = 12.0  # Langsamer im Rush Hour
    elif 22 <= hour or hour <= 5:
        avg_speed = 20.0  # Schneller nachts
    else:
        avg_speed = 15.0  # Normal tagsueber
    
    duration_minutes = (distance / avg_speed) * 60
    return max(5, duration_minutes)  # Mindestens 5 Minuten

def get_weather_for_datetime(weather_df, target_datetime):
    """Hole Wetterdaten für einen bestimmten Zeitpunkt"""
    if weather_df is None:
        # Fallback: Durchschnittswerte für Januar
        return {
            'temperature': 2.0,
            'humidity': 65.0,
            'rain': 0.0,
            'wind_speed': 15.0
        }
    
    target_hour = pd.to_datetime(target_datetime).floor('H')
    
    # Suche passende Stunde
    weather_row = weather_df[weather_df['hour'] == target_hour]
    
    if len(weather_row) > 0:
        return {
            'temperature': weather_row['temperature'].values[0],
            'humidity': weather_row['humidity'].values[0],
            'rain': weather_row['rain'].values[0],
            'wind_speed': weather_row['wind_speed'].values[0]
        }
    else:
        # Fallback auf Durchschnitt
        return {
            'temperature': weather_df['temperature'].mean(),
            'humidity': weather_df['humidity'].mean(),
            'rain': weather_df['rain'].mean(),
            'wind_speed': weather_df['wind_speed'].mean()
        }

def predict_fare_and_tip(
    pickup_id, dropoff_id, pickup_datetime, passenger_count,
    fare_model, fare_features, tip_model, tip_features,
    zones_df, weather_df, top_locations
):
    """
    Zwei-Stufen-Vorhersage: Fare -> Tip
    
    Returns:
        dict mit allen Vorhersagen und Zwischenwerten
    """
    # Feature Engineering
    hour = pickup_datetime.hour
    day_of_week = pickup_datetime.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
    
    # Distanz mit Haversine berechnen
    trip_distance = calculate_distance_haversine(pickup_id, dropoff_id, zones_df)
    trip_duration = calculate_duration(trip_distance, hour, is_rush_hour)
    
    # Flughafen-Check
    airport_zones = [1, 132, 138]
    is_airport_trip = 1 if (pickup_id in airport_zones or dropoff_id in airport_zones) else 0
    
    # Borough-Mapping für One-Hot-Encoding
    def get_borough_for_location(location_id):
        """Hole Borough für eine LocationID"""
        borough_row = zones_df[zones_df['LocationID'] == location_id]
        if len(borough_row) > 0:
            return borough_row['Borough'].values[0]
        return 'Unknown'
    
    pickup_borough = get_borough_for_location(pickup_id)
    dropoff_borough = get_borough_for_location(dropoff_id)
    
    # STUFE 1: Fare Prediction
    # Erstelle DataFrame mit allen Basis-Features
    fare_input_data = {
        'calculated_distance': [trip_distance],
        'passenger_count': [passenger_count],
        'hour_of_day': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'is_rush_hour': [is_rush_hour],
        'is_airport_trip': [is_airport_trip]
    }
    
    # Borough One-Hot-Encoding für alle möglichen Boroughs
    all_boroughs = ['Brooklyn', 'EWR', 'Manhattan', 'Queens', 'Staten Island', 'Unknown']
    
    # Pickup Borough Features
    for borough in all_boroughs:
        fare_input_data[f'PU_Borough_{borough}'] = [1 if pickup_borough == borough else 0]
    
    # Dropoff Borough Features  
    for borough in all_boroughs:
        fare_input_data[f'DO_Borough_{borough}'] = [1 if dropoff_borough == borough else 0]
    
    # Erstelle DataFrame nur mit den Features, die das Modell erwartet
    fare_input_filtered = pd.DataFrame({
        col: fare_input_data.get(col, [0]) for col in fare_features
    })
    
    predicted_fare = fare_model.predict(fare_input_filtered)[0]
    
    # STUFE 2: Tip Prediction
    weather = get_weather_for_datetime(weather_df, pickup_datetime)
    
    # Location Grouping
    top_pickup = top_locations.get('top_pickup', [])
    top_dropoff = top_locations.get('top_dropoff', [])
    
    pu_grouped = pickup_id if pickup_id in top_pickup else 0
    do_grouped = dropoff_id if dropoff_id in top_dropoff else 0
    
    # Tip-Modell nutzt NICHT fare_amount (fairer Vergleich mit Baseline)
    tip_input = pd.DataFrame({
        'trip_distance': [trip_distance],
        'passenger_count': [passenger_count],
        'temperature': [weather['temperature']],
        'humidity': [weather['humidity']],
        'rain': [weather['rain']],
        'wind_speed': [weather['wind_speed']],
        'hour_of_day': [hour],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'PULocationID_grouped': [pu_grouped],
        'DOLocationID_grouped': [do_grouped]
    })
    
    # Nur Features verwenden, die das Modell kennt
    tip_input_filtered = tip_input[[col for col in tip_features if col in tip_input.columns]]
    
    # Fehlende Features mit 0 auffüllen
    for col in tip_features:
        if col not in tip_input_filtered.columns:
            tip_input_filtered[col] = 0
    
    tip_input_filtered = tip_input_filtered[tip_features]
    predicted_tip = tip_model.predict(tip_input_filtered)[0]
    
    # Ergebnis zusammenstellen
    total = predicted_fare + predicted_tip
    profitability = total / trip_distance if trip_distance > 0 else 0
    
    return {
        'fare_amount': predicted_fare,
        'tip_amount': predicted_tip,
        'total_amount': total,
        'trip_distance': trip_distance,
        'trip_duration': trip_duration,
        'profitability_score': profitability,
        'weather': weather,
        'hour': hour,
        'is_weekend': is_weekend,
        'is_rush_hour': is_rush_hour
    }

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Lade Daten und Modelle (mit Spinner beim ersten Laden)
    with st.spinner("Lade Modelle und Daten..."):
        fare_model, fare_features, tip_model, tip_features = load_models()
        fare_info, tip_info = load_model_info()
        zones_df = load_zone_data()
        weather_df = load_weather_data()
        top_locations = load_top_locations()
        route_stats_df = load_route_statistics()
    
    # Header
    st.title("🚕 NYC Taxi Fare & Tip Predictor")
    st.markdown("""
    **Machine Learning basierte Vorhersage von Fahrtpreisen und Trinkgeldern**
    
    Diese Anwendung nutzt zwei trainierte Random Forest Modelle:
    - **Fare Model**: Vorhersage des Fahrtpreises basierend auf Route, Zeit und Verkehrslage
    - **Tip Model**: Vorhersage des Trinkgelds unter Berücksichtigung von Wetterdaten
    """)
    
    st.divider()
    
    # Sidebar - Input Parameter
    with st.sidebar:
        st.header("Fahrt-Parameter")
        
        # Location Auswahl
        st.subheader("Route")
        
        # Pickup Location
        midtown_matches = zones_df[zones_df['Zone'] == 'Midtown Center']
        default_pickup = int(midtown_matches.index[0]) if len(midtown_matches) > 0 else 0
        pickup_zone = st.selectbox(
            "Pickup Location",
            options=zones_df['display_name'].tolist(),
            index=default_pickup
        )
        pickup_id = zones_df[zones_df['display_name'] == pickup_zone]['LocationID'].values[0]
        
        # Dropoff Location
        jfk_matches = zones_df[zones_df['Zone'] == 'JFK Airport']
        default_dropoff = int(jfk_matches.index[0]) if len(jfk_matches) > 0 else 10
        dropoff_zone = st.selectbox(
            "Dropoff Location",
            options=zones_df['display_name'].tolist(),
            index=default_dropoff
        )
        dropoff_id = zones_df[zones_df['display_name'] == dropoff_zone]['LocationID'].values[0]
        
        st.divider()
        
        # Zeitpunkt
        st.subheader("Zeitpunkt")
        
        # Datum (beschränkt auf Januar 2023)
        pickup_date = st.date_input(
            "Datum",
            value=date(2023, 1, 15),
            min_value=date(2023, 1, 1),
            max_value=date(2023, 1, 31),
            help="Nur Januar 2023 verfügbar (Wetterdaten-Limitierung)"
        )
        
        pickup_hour = st.slider(
            "Uhrzeit",
            min_value=0,
            max_value=23,
            value=14,
            format="%d:00 Uhr"
        )
        
        pickup_datetime = datetime.combine(pickup_date, datetime.min.time().replace(hour=pickup_hour))
        
        st.divider()
        
        # Weitere Parameter
        st.subheader("Weitere Details")
        passenger_count = st.slider(
            "Anzahl Passagiere",
            min_value=1,
            max_value=6,
            value=1
        )
        
        st.divider()
        
        # Predict Button
        predict_button = st.button(
            "Vorhersage starten",
            type="primary",
            use_container_width=True
        )
    
    # Main Content
    if predict_button:
        with st.spinner("Berechne Vorhersagen..."):
            try:
                # Prediction durchfuehren
                result = predict_fare_and_tip(
                    pickup_id=pickup_id,
                    dropoff_id=dropoff_id,
                    pickup_datetime=pickup_datetime,
                    passenger_count=passenger_count,
                    fare_model=fare_model,
                    fare_features=fare_features,
                    tip_model=tip_model,
                    tip_features=tip_features,
                    zones_df=zones_df,
                    weather_df=weather_df,
                    top_locations=top_locations
                )
                
                # Speichere in Session State
                st.session_state['result'] = result
                st.session_state['pickup_zone'] = pickup_zone
                st.session_state['dropoff_zone'] = dropoff_zone
                st.session_state['pickup_datetime'] = pickup_datetime
                st.session_state['passenger_count'] = passenger_count
                st.session_state['pickup_id'] = pickup_id
                st.session_state['dropoff_id'] = dropoff_id
                
                # Hole historische Statistiken für diese Route
                route_stats = get_route_statistics(route_stats_df, pickup_id, dropoff_id)
                st.session_state['route_stats'] = route_stats
                
            except Exception as e:
                st.error(f"Fehler bei der Vorhersage: {e}")
                st.stop()
    
    # Zeige Ergebnisse
    if 'result' in st.session_state:
        result = st.session_state['result']
        
        st.success("Vorhersage erfolgreich!")
        
        # Route Info
        st.markdown(f"### Route: {st.session_state['pickup_zone']} -> {st.session_state['dropoff_zone']}")
        st.caption(f"{st.session_state['pickup_datetime'].strftime('%A, %d. %B %Y um %H:00 Uhr')}")
        
        # Karte mit Start- und Endpunkt
        pickup_id = zones_df[zones_df['display_name'] == st.session_state['pickup_zone']]['LocationID'].values[0]
        dropoff_id = zones_df[zones_df['display_name'] == st.session_state['dropoff_zone']]['LocationID'].values[0]
        
        pickup_coords = zones_df[zones_df['LocationID'] == pickup_id][['latitude', 'longitude']].values[0]
        dropoff_coords = zones_df[zones_df['LocationID'] == dropoff_id][['latitude', 'longitude']].values[0]
        
        if not pd.isna(pickup_coords[0]) and not pd.isna(dropoff_coords[0]):
            map_data = pd.DataFrame({
                'lat': [pickup_coords[0], dropoff_coords[0]],
                'lon': [pickup_coords[1], dropoff_coords[1]],
                'type': ['Start', 'Ziel'],
                'location': [st.session_state['pickup_zone'], st.session_state['dropoff_zone']]
            })
            
            st.map(map_data, zoom=10, use_container_width=True)
        
        st.divider()
        
        # Hauptmetriken
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Fahrtpreis (Fare)",
                value=f"${result['fare_amount']:.2f}",
                help="Vorhergesagt durch Fare Regression Model"
            )
        
        with col2:
            st.metric(
                label="Trinkgeld (Tip)",
                value=f"${result['tip_amount']:.2f}",
                help="Vorhergesagt durch Tip Regression Model"
            )
        
        with col3:
            st.metric(
                label="Gesamt (Total)",
                value=f"${result['total_amount']:.2f}",
                help="Gesamtbetrag inkl. Trinkgeld"
            )
        
        with col4:
            st.metric(
                label="Profitabilität",
                value=f"${result['profitability_score']:.2f}/mi",
                help="Einnahmen pro Meile (Total / Distanz)"
            )
        
        st.divider()
        
        # Details in Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Fahrt-Details", "Historischer Vergleich", "Wetterbedingungen", "Modell-Info"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### Fahrtinformationen")
                st.write(f"**Distanz (Haversine):** {result['trip_distance']:.2f} Meilen")
                st.write(f"**Passagiere:** {st.session_state['passenger_count']}")
                st.write(f"**Wochentag:** {'Wochenende' if result['is_weekend'] else 'Werktag'}")
                st.write(f"**Rush Hour:** {'Ja' if result['is_rush_hour'] else 'Nein'}")
            
            with col_b:
                st.markdown("#### Kostenaufschlüsselung")
                fare_pct = (result['fare_amount'] / result['total_amount']) * 100
                tip_pct = (result['tip_amount'] / result['total_amount']) * 100
                
                st.write(f"**Fahrtpreis:** ${result['fare_amount']:.2f} ({fare_pct:.1f}%)")
                st.write(f"**Trinkgeld:** ${result['tip_amount']:.2f} ({tip_pct:.1f}%)")
                st.write(f"**Gesamt:** ${result['total_amount']:.2f} (100%)")
                
                # Einfaches Balkendiagramm
                st.progress(tip_pct / 100, text=f"Trinkgeld-Anteil: {tip_pct:.1f}%")
        
        with tab2:
            st.markdown("#### Vergleich: ML-Vorhersage vs. Historische Daten")
            
            route_stats = st.session_state.get('route_stats')
            
            if route_stats:
                st.success(f"Historische Daten verfügbar: **{route_stats['trip_count']:,} Fahrten** auf dieser Route")
                
                col_pred, col_hist = st.columns(2)
                
                with col_pred:
                    st.markdown("##### ML-Vorhersage")
                    st.metric("Fahrtpreis", f"${result['fare_amount']:.2f}")
                    st.metric("Trinkgeld", f"${result['tip_amount']:.2f}")
                    st.metric("Gesamt", f"${result['total_amount']:.2f}")
                
                with col_hist:
                    st.markdown("##### Historischer Durchschnitt")
                    st.metric(
                        "Fahrtpreis (Ø)", 
                        f"${route_stats['mean_fare']:.2f}",
                        delta=f"±${route_stats['std_fare']:.2f}",
                        delta_color="off"
                    )
                    st.metric(
                        "Trinkgeld (Ø)", 
                        f"${route_stats['mean_tip']:.2f}",
                        delta=f"±${route_stats['std_tip']:.2f}",
                        delta_color="off"
                    )
                    hist_total = route_stats['mean_fare'] + route_stats['mean_tip']
                    st.metric("Gesamt (Ø)", f"${hist_total:.2f}")
                
                st.divider()
                
                # Abweichungsanalyse
                st.markdown("##### Abweichungsanalyse")
                
                fare_diff = result['fare_amount'] - route_stats['mean_fare']
                fare_diff_pct = (fare_diff / route_stats['mean_fare']) * 100 if route_stats['mean_fare'] > 0 else 0
                
                tip_diff = result['tip_amount'] - route_stats['mean_tip']
                tip_diff_pct = (tip_diff / route_stats['mean_tip']) * 100 if route_stats['mean_tip'] > 0 else 0
                
                col_f, col_t = st.columns(2)
                
                with col_f:
                    if abs(fare_diff_pct) < 10:
                        st.info(f"Fare-Vorhersage liegt **{fare_diff_pct:+.1f}%** vom Durchschnitt (im Normalbereich)")
                    elif fare_diff_pct > 0:
                        st.warning(f"Fare-Vorhersage liegt **{fare_diff_pct:+.1f}%** über dem Durchschnitt")
                    else:
                        st.warning(f"Fare-Vorhersage liegt **{fare_diff_pct:+.1f}%** unter dem Durchschnitt")
                
                with col_t:
                    if abs(tip_diff_pct) < 15:
                        st.info(f"Tip-Vorhersage liegt **{tip_diff_pct:+.1f}%** vom Durchschnitt (im Normalbereich)")
                    elif tip_diff_pct > 0:
                        st.warning(f"Tip-Vorhersage liegt **{tip_diff_pct:+.1f}%** über dem Durchschnitt")
                    else:
                        st.warning(f"Tip-Vorhersage liegt **{tip_diff_pct:+.1f}%** unter dem Durchschnitt")
                
                st.divider()
                
                # Zusätzliche historische Infos
                st.markdown("##### Weitere historische Kennzahlen")
                col_m1, col_m2, col_m3 = st.columns(3)
                
                with col_m1:
                    st.metric("Median Fare", f"${route_stats['median_fare']:.2f}")
                with col_m2:
                    st.metric("Median Tip", f"${route_stats['median_tip']:.2f}")
                with col_m3:
                    st.metric("Ø Preis/Meile", f"${route_stats['mean_price_per_mile']:.2f}")
                
            else:
                st.warning("Keine historischen Daten für diese Route verfügbar.")
                st.info("Diese Route kommt in den Trainingsdaten nicht häufig genug vor, um verlässliche Statistiken zu berechnen.")
        
        with tab3:
            st.markdown("#### Wetterbedingungen zum Fahrzeitpunkt")
            
            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            
            with col_w1:
                st.metric(
                    "Temperatur",
                    f"{result['weather']['temperature']:.1f} C"
                )
            
            with col_w2:
                st.metric(
                    "Luftfeuchtigkeit",
                    f"{result['weather']['humidity']:.0f}%"
                )
            
            with col_w3:
                st.metric(
                    "Niederschlag",
                    f"{result['weather']['rain']:.1f} mm"
                )
            
            with col_w4:
                st.metric(
                    "Windgeschwindigkeit",
                    f"{result['weather']['wind_speed']:.1f} km/h"
                )
            
            st.info("Wetterdaten werden vom Tip Prediction Model berücksichtigt. Schlechtes Wetter kann zu höheren Trinkgeldern führen.")
        
        with tab4:
            st.markdown("#### Verwendete Machine Learning Modelle")
            
            # Dynamische Anzeige der Modell-Metriken aus den Info-Dateien
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                st.markdown("**Fare Prediction Model**")
                if fare_info:
                    st.write(f"- Typ: {fare_info.get('model_type', 'Random Forest')}")
                    st.write(f"- Target: {fare_info.get('target', 'fare_amount')}")
                    st.write(f"- Training Samples: {fare_info.get('training_samples', 'N/A'):,}")
                    st.write(f"- Test Samples: {fare_info.get('test_samples', 'N/A'):,}")
                    st.divider()
                    st.markdown("**Performance (Testdaten):**")
                    st.metric("MAE", f"${fare_info.get('rf_mae', 0):.2f}")
                    st.metric("RMSE", f"${float(fare_info.get('rf_rmse', 0)):.2f}")
                    st.metric("R²", f"{fare_info.get('rf_r2', 0):.4f}")
                else:
                    st.write("- Typ: Random Forest Regressor")
                    st.write("- Keine Metriken verfügbar")
            
            with col_m2:
                st.markdown("**Tip Prediction Model**")
                if tip_info:
                    st.write(f"- Typ: {tip_info.get('model_type', 'Random Forest')}")
                    st.write(f"- Target: {tip_info.get('target', 'tip_amount')}")
                    st.write(f"- Training Samples: {tip_info.get('training_samples', 'N/A'):,}")
                    st.write(f"- Test Samples: {tip_info.get('test_samples', 'N/A'):,}")
                    st.divider()
                    st.markdown("**Performance (Testdaten):**")
                    st.metric("MAE", f"${tip_info.get('test_mae', 0):.2f}")
                    st.metric("RMSE", f"${float(tip_info.get('test_rmse', 0)):.2f}")
                    st.metric("R²", f"{tip_info.get('test_r2', 0):.4f}")
                    if tip_info.get('improvement_vs_baseline'):
                        st.metric("vs. Baseline", f"+{tip_info.get('improvement_vs_baseline', 0):.1f}%")
                else:
                    st.write("- Typ: Random Forest Regressor")
                    st.write("- Keine Metriken verfügbar")
            
            st.divider()
            st.markdown("""
            **Architektur & Besonderheiten**
            - Fare und Tip werden unabhängig voneinander vorhergesagt
            - Tip-Modell nutzt NICHT den Fare Amount (fairer Baseline-Vergleich)
            - Wetterdaten fließen nur ins Tip-Modell ein
            - Distanzberechnung via Haversine-Formel (Luftlinie)
            """)
            
            st.success("Alle Modelle erfolgreich geladen und einsatzbereit")
    
    else:
        # Zeige Platzhalter bei erstem Laden
        st.info("Bitte wähle die Fahrtparameter in der Sidebar und klicke auf 'Vorhersage starten'")
        
        st.markdown("""
        ### Wie funktioniert diese App?
        
        1. **Wähle Route:** Pickup und Dropoff Location aus über 260 NYC Zonen
        2. **Wähle Zeitpunkt:** Datum und Uhrzeit der Fahrt (Januar 2023)
        3. **Weitere Details:** Anzahl der Passagiere
        4. **Vorhersage:** Die App berechnet automatisch:
           - Distanz basierend auf Haversine-Funktion
           - Geschätzte Fahrtdauer basierend auf Verkehrslage
           - Wetterbedingungen für den gewählten Zeitpunkt
           - Fahrtpreis mittels ML-Modell
           - Erwartetes Trinkgeld mittels ML-Modell
           - Profitabilitäts-Score (Einnahmen pro Meile)
        
        ### Anwendungsfälle
        
        - **Für Taxi-Fahrer:** Einschätzung der Gesamteinnahmen vor Fahrtantritt
        - **Für Fahrgäste:** Transparenz über erwartete Kosten
        - **Für Analysten:** Demonstration von ML-Pipelines in Produktion
        """)
    
    # Footer
    st.divider()
    st.caption("📦 Modelle & Daten werden von [Hugging Face](https://huggingface.co/dnltre/taxi-nyc-models) geladen · HFT Stuttgart · Data Analytics WiSe 25/26")

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()
