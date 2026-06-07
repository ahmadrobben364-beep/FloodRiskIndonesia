import ee
import json
import streamlit as st
import datetime

service_account = st.secrets["gcp_service_account"]

credentials = ee.ServiceAccountCredentials(
    service_account["client_email"],
    key_data=json.dumps(dict(service_account))
)

ee.Initialize(
    credentials=credentials,
    project="ndvisaya"
)

# =====================================================
# EE TO FOLIUM
# =====================================================

def add_ee_layer(map_obj, ee_image, vis_params, name):

    map_id = ee.Image(ee_image).getMapId(vis_params)

    folium.raster_layers.TileLayer(
        tiles=map_id["tile_fetcher"].url_format,
        attr="Google Earth Engine",
        name=name,
        overlay=True,
        control=True
    ).add_to(map_obj)

# =====================================================
# STREAMLIT
# =====================================================

st.set_page_config(
    page_title="Flood Risk Indonesia",
    layout="wide"
)

st.markdown("""
<div style="
background-color:#1E3A5F;
padding:20px;
border-radius:10px;
text-align:center;
">

<h1 style="color:white;">
🌊 Flood Risk Monitoring Indonesia
</h1>

<h3 style="color:white;">
Oleh Ahmad Adreand Robben
</h3>

<p style="color:white;">
Mahasiswa Program Studi Geologi - Universitas Indonesia
</p>

<p style="color:white;">
Website ini dikembangkan untuk memetakan tingkat kerentanan banjir dan
risiko banjir di Indonesia menggunakan data satelit, topografi,
dan curah hujan near real-time dari Google Earth Engine.
</p>

</div>
""", unsafe_allow_html=True)

# =====================================================
# INDONESIA
# =====================================================

indonesia = (
    ee.FeatureCollection("FAO/GAUL/2015/level0")
    .filter(
        ee.Filter.eq(
            "ADM0_NAME",
            "Indonesia"
        )
    )
)

geom = indonesia.geometry()

# =====================================================
# DATE
# =====================================================

today = datetime.date.today()

end_date = (
    today -
    datetime.timedelta(days=3)
)

start_date = (
    end_date -
    datetime.timedelta(days=30)
)

normal_start = (
    end_date -
    datetime.timedelta(days=180)
)

normal_end = (
    end_date -
    datetime.timedelta(days=30)
)

rain_start = (
    end_date -
    datetime.timedelta(days=7)
)

# =====================================================
# SENTINEL MASK
# =====================================================

def maskS2(img):

    qa = img.select("QA60")

    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11

    mask = (
        qa.bitwiseAnd(cloudBitMask)
        .eq(0)
        .And(
            qa.bitwiseAnd(cirrusBitMask)
            .eq(0)
        )
    )

    return img.updateMask(mask)

# =====================================================
# RAINFALL
# =====================================================

rain = (
    ee.ImageCollection(
        "NASA/GPM_L3/IMERG_V07"
    )
    .filterBounds(geom)
    .filterDate(
        str(rain_start),
        str(end_date)
    )
    .select("precipitation")
    .sum()
    .clip(geom)
)

# =====================================================
# NDWI
# =====================================================

normal_img = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(geom)
    .filterDate(
        str(normal_start),
        str(normal_end)
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            70
        )
    )
    .map(maskS2)
    .median()
    .clip(geom)
)

current_img = (
    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )
    .filterBounds(geom)
    .filterDate(
        str(start_date),
        str(end_date)
    )
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            70
        )
    )
    .map(maskS2)
    .median()
    .clip(geom)
)

ndwi_now = (
    current_img.normalizedDifference(
        ["B3", "B8"]
    )
)

ndwi_normal = (
    normal_img.normalizedDifference(
        ["B3", "B8"]
    )
)

delta_ndwi = (
    ndwi_now
    .subtract(
        ndwi_normal
    )
    .rename(
        "DeltaNDWI"
    )
)

# =====================================================
# TOPOGRAPHY
# =====================================================

elevation = (
    ee.Image(
        "USGS/SRTMGL1_003"
    )
    .clip(geom)
)

slope = ee.Terrain.slope(
    elevation
)

flow_acc = (
    ee.Image(
        "WWF/HydroSHEDS/15ACC"
    )
    .clip(geom)
)

# =====================================================
# LANDUSE
# =====================================================

landuse = (
    ee.ImageCollection(
        "ESA/WorldCover/v200"
    )
    .first()
    .select("Map")
    .clip(geom)
)

landmask = (
    landuse.neq(80)
)

urban = landuse.eq(50)

cropland = landuse.eq(40)

wetland = landuse.eq(90)

landuse_score = (
    urban.multiply(1.0)
    .add(
        cropland.multiply(0.7)
    )
    .add(
        wetland.multiply(0.8)
    )
)

# =====================================================
# NORMALIZATION
# =====================================================

rain_norm = (
    rain
    .unitScale(0, 100)
    .clamp(0, 1)
)

delta_norm = (
    delta_ndwi
    .unitScale(-0.05, 0.05)
    .clamp(0, 1)
)

flow_norm = (
    flow_acc
    .unitScale(0, 5000)
    .clamp(0, 1)
)

elev_norm = (
    ee.Image(1)
    .subtract(
        elevation.divide(3000)
    )
    .clamp(0, 1)
)

slope_norm = (
    ee.Image(1)
    .subtract(
        slope.divide(45)
    )
    .clamp(0, 1)
)

# =====================================================
# SUSCEPTIBILITY
# =====================================================

susceptibility = (
    elev_norm.multiply(0.35)
    .add(
        slope_norm.multiply(0.20)
    )
    .add(
        flow_norm.multiply(0.30)
    )
    .add(
        landuse_score.multiply(0.15)
    )
)

susceptibility = (
    susceptibility.updateMask(
        landmask
    )
)

# =====================================================
# FLOOD RISK
# =====================================================

risk = (
    rain_norm.multiply(0.50)
    .add(
        delta_norm.multiply(0.25)
    )
    .add(
        susceptibility.multiply(0.25)
    )
)

risk = (
    risk
    .updateMask(
        landmask
    )
    .rename(
        "FloodRisk"
    )
)

# =====================================================
# RISK CLASS
# =====================================================

risk_class = (
    ee.Image(0)
    .where(
        risk.gte(0.30),
        1
    )
    .where(
        risk.gte(0.40),
        2
    )
    .where(
        risk.gte(0.55),
        3
    )
)

risk_class = (
    risk_class.updateMask(
        landmask
    )
)

# =====================================================
# VISUALIZATION
# =====================================================

sus_vis = {
    "min": 0,
    "max": 1,
    "palette": [
        "green",
        "yellow",
        "orange",
        "red"
    ]
}

risk_vis = {
    "min": 0,
    "max": 3,
    "palette": [
        "green",
        "yellow",
        "orange",
        "red"
    ]
}

# =====================================================
# SELECT MAP
# =====================================================

# =====================================================
# MAP
# =====================================================

m = folium.Map(
    location=[-2.5, 118],
    zoom_start=5,
    tiles="CartoDB positron"
)

# FLOOD SUSCEPTIBILITY
# =====================================================

st.markdown("---")

st.header("Flood Susceptibility Indonesia")

m1 = folium.Map(
    location=[-2.5,118],
    zoom_start=5,
    tiles="CartoDB positron"
)

add_ee_layer(
    m1,
    susceptibility,
    sus_vis,
    "Flood Susceptibility"
)

folium.LayerControl().add_to(m1)

st_folium(
    m1,
    width=1400,
    height=600,
    key="susceptibility"
)

st.markdown("""
### Legenda

🟩 Rendah

🟨 Sedang

🟧 Tinggi

🟥 Sangat Tinggi

### Penjelasan Peta

Flood Susceptibility menggambarkan tingkat kerentanan fisik suatu wilayah
terhadap banjir berdasarkan karakteristik topografi dan penggunaan lahan.

Wilayah dataran rendah, lereng landai, daerah dengan akumulasi aliran tinggi,
serta kawasan terbangun dan lahan basah cenderung memiliki nilai
kerentanan yang lebih tinggi.

### Metode

Flood Susceptibility dihitung menggunakan:

- Elevation (35%)
- Slope (20%)
- Flow Accumulation (30%)
- Land Use (15%)

Persamaan:

Susceptibility =
0.35 × Elevation +
0.20 × Slope +
0.30 × Flow Accumulation +
0.15 × Land Use
""")

st.markdown("---")

st.header("Flood Risk Indonesia")

m2 = folium.Map(
    location=[-2.5,118],
    zoom_start=5,
    tiles="CartoDB positron"
)

add_ee_layer(
    m2,
    risk_class,
    risk_vis,
    "Flood Risk"
)

folium.LayerControl().add_to(m2)

st_folium(
    m2,
    width=1400,
    height=600,
    key="risk"
)

st.markdown("""
### Legenda

🟩 Aman

🟨 Waspada

🟧 Siaga

🟥 Bahaya

### Penjelasan Peta

Flood Risk menunjukkan tingkat potensi banjir saat ini
berdasarkan kombinasi kondisi fisik wilayah, curah hujan,
dan perubahan kondisi kelembapan permukaan.

Peta ini bersifat dinamis dan akan berubah mengikuti data terbaru.

### Metode

Flood Risk dihitung menggunakan:

- Rainfall IMERG (50%)
- Delta NDWI (25%)
- Flood Susceptibility (25%)

Persamaan:

Risk =
0.50 × Rainfall +
0.25 × Delta NDWI +
0.25 × Susceptibility

Klasifikasi:

🟩 Aman (<0.30)

🟨 Waspada (0.30–0.40)

🟧 Siaga (0.40–0.55)

🟥 Bahaya (>0.55)
""")

st.markdown("---")

st.header("📚 Data dan Library")

st.markdown("""
### Data

- NASA GPM IMERG V07
- Sentinel-2 SR Harmonized
- SRTM DEM
- HydroSHEDS Flow Accumulation
- ESA WorldCover v200

### Library

- Google Earth Engine
- Streamlit
- Folium
- Streamlit-Folium
- Python

### Pengembang

**Oleh Ahmad Adreand Robben**

Mahasiswa Program Studi Geologi

Universitas Indonesia
""")