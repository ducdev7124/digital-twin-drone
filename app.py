import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH TRANG WEB
# ==========================================
st.set_page_config(page_title="Digital Twin Nông Nghiệp", layout="wide", page_icon="🌱")

# ==========================================
# MODULE 1: TẠO DỮ LIỆU MÔ PHỎNG (CACHE ĐỂ TỐI ƯU HIỆU NĂNG)
# ==========================================
@st.cache_data
def generate_timeseries_data():
    """Sinh dữ liệu chuỗi thời gian 100 ngày cho Tab Giám sát"""
    np.random.seed(42)
    days = np.arange(1, 101)
    
    # Nhiệt độ dao động 20 - 40 độ C
    temperature = np.random.normal(26, 4, 100) + np.sin(days/5)*4
    # Độ ẩm đất dao động 20 - 40%
    soil_moisture = np.random.normal(30, 5, 100) + np.cos(days/5)*3
    
    # Sinh khối tăng dần tuyến tính lên khoảng 100g
    biomass_growth = np.random.normal(1.0, 0.1, 100)
    biomass = np.cumsum(biomass_growth)
    
    df_ts = pd.DataFrame({
        'Day': days,
        'Temperature': temperature,
        'SoilMoisture': soil_moisture,
        'Biomass': biomass
    })
    return df_ts

@st.cache_data
def generate_ai_training_data(n_samples=500):
    """Sinh dữ liệu tĩnh cho Tab AI dự báo"""
    np.random.seed(99)
    data = []
    for _ in range(n_samples):
        avg_temp = np.random.normal(26, 4)
        avg_moist = np.random.normal(45, 12)
        sunlight = np.random.normal(8, 2)
        
        temp_factor = 1.0 - abs(avg_temp - 27) / 20.0
        moist_factor = 1.0 - abs(avg_moist - 50) / 40.0
        base_ndvi = 0.3 + (temp_factor * moist_factor * 0.6)
        ndvi = np.clip(np.random.normal(base_ndvi, 0.05), 0.1, 0.95)
        
        yield_kg = 5000 * ndvi * (sunlight/8) + np.random.normal(0, 200)
        data.append([avg_temp, avg_moist, sunlight, ndvi, yield_kg])
        
    cols = ['Temperature', 'Soil_Moisture', 'Sunlight', 'NDVI', 'Yield']
    return pd.DataFrame(data, columns=cols)

@st.cache_resource
def train_model(df):
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return model, rmse, r2, X.columns

# ==========================================
# KHỞI TẠO DỮ LIỆU
# ==========================================
df_ts = generate_timeseries_data()
df_ai = generate_ai_training_data()
rf_model, rf_rmse, rf_r2, feature_names = train_model(df_ai)

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🌱 Mục tiêu: Xây dựng Dashboard tích hợp Sensor ảo, Drone 3D và AI dự báo năng suất")
st.markdown("---")

# TẠO 3 TAB CHÍNH
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Giám sát", "🤖 AI Dự Báo", "🚁 Thiết kế Drone"])

# ==========================================
# TAB 1: DASHBOARD GIÁM SÁT (TIME-SERIES)
# ==========================================
with tab1:
    st.header("1. Drone View (NDVI & RGB)")
    
    # Thanh trượt chọn ngày
    selected_day = st.slider("Chọn ngày quan sát:", min_value=1, max_value=100, value=100)
    
    # Lấy dữ liệu của ngày hiện tại
    current_data = df_ts[df_ts['Day'] == selected_day].iloc[0]
    current_biomass = current_data['Biomass']
    
    # Tính toán NDVI trung bình của ngày hôm đó (Scale theo Biomass)
    avg_ndvi = min(current_biomass / 100.0 + np.random.normal(0, 0.02), 0.98)
    avg_ndvi = max(avg_ndvi, 0.1) # Giới hạn không dưới 0.1
    
    # Nút radio chọn chế độ xem
    view_mode = st.radio("Chế độ hiển thị:", ["🔴 NDVI (Sức khỏe cây)", "🟢 RGB (Màu thực tế)"], horizontal=True)
    
    # Tạo mảng ảnh giả lập 30x30 Pixel
    grid_size = 30
    image_grid = np.random.normal(avg_ndvi, 0.05, (grid_size, grid_size))
    
    # Xử lý tạo "luống cày" ngang (Độc quyền không đụng hàng)
    for i in range(3, grid_size, 5):
        image_grid[i, :] = avg_ndvi - 0.35 # Giảm giá trị để tạo đường sáng màu/khác màu
        
    image_grid = np.clip(image_grid, 0, 1)

    # Vẽ bản đồ bằng Plotly
    if "NDVI" in view_mode:
        colorscale = "RdYlGn"
        title_map = f"Bản đồ NDVI Ngày {selected_day} (TB: {avg_ndvi:.2f})"
    else:
        colorscale = "Greens"
        title_map = f"Bản đồ RGB Ngày {selected_day}"

    fig_map = px.imshow(image_grid, color_continuous_scale=colorscale, zmin=0, zmax=1, title=title_map)
    fig_map.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Tọa độ X",
        yaxis_title="Tọa độ Y",
        coloraxis_colorbar=dict(title="Chỉ số")
    )
    # Tùy chỉnh Hover Tooltip giống hệt ảnh mẫu
    fig_map.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>color: %{z:.6f}<extra></extra>")
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")
    
    # BẢN ĐỒ MÔI TRƯỜNG
    st.header("2. Giám sát Môi trường")
    fig_env = go.Figure()
    fig_env.add_trace(go.Scatter(x=df_ts['Day'][:selected_day], y=df_ts['Temperature'][:selected_day], mode='lines', name='Nhiệt độ (°C)', line=dict(color='#63b3ed')))
    fig_env.add_trace(go.Scatter(x=df_ts['Day'][:selected_day], y=df_ts['SoilMoisture'][:selected_day], mode='lines', name='Độ ẩm (%)', line=dict(color='#2b6cb0')))
    
    fig_env.update_layout(
        template="plotly_dark",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.85),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_env, use_container_width=True)

    st.markdown("---")
    
    # BẢN ĐỒ SINH TRƯỞNG (BIOMASS)
    st.header("3. Sinh trưởng (Biomass) 🔗")
    fig_bio = go.Figure()
    fig_bio.add_trace(go.Scatter(x=df_ts['Day'], y=df_ts['Biomass'], fill='tozeroy', mode='none', name='Sinh khối', fillcolor='rgba(99, 179, 237, 0.6)'))
    
    # Thêm đường line báo "Ngày hiện tại"
    fig_bio.add_shape(type="line", x0=selected_day, y0=0, x1=selected_day, y1=max(df_ts['Biomass']),
                      line=dict(color="White", width=2, dash="dash"))
    fig_bio.add_annotation(x=selected_day, y=max(df_ts['Biomass']), text="Ngày hiện tại", showarrow=False, xanchor="left", xshift=5, font=dict(color="white"))
    
    fig_bio.update_layout(
        template="plotly_dark",
        xaxis_title="Ngày",
        yaxis_title="Khối lượng (g)",
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_bio, use_container_width=True)


# ==========================================
# TAB 2: AI DỰ BÁO (WHAT-IF SCENARIO)
# ==========================================
with tab2:
    st.header("Trí tuệ nhân tạo (Random Forest Regressor)")
    
    col1, col2 = st.columns(2)
    col1.metric(label="Độ chính xác (R-squared)", value=f"{rf_r2*100:.2f} %")
    col2.metric(label="Sai số trung bình (RMSE)", value=f"{rf_rmse:.1f} kg/ha")
    
    st.markdown("### 🛠️ Kịch bản giả định (What-if Analysis)")
    st.info("Nhập các thông số môi trường giả định để AI dự đoán năng suất cuối mùa.")
    
    c1, c2 = st.columns(2)
    with c1:
        sim_temp = st.slider("Nhiệt độ môi trường (°C)", 10.0, 45.0, 26.0)
        sim_moist = st.slider("Độ ẩm đất (%)", 10.0, 90.0, 45.0)
    with c2:
        sim_sun = st.slider("Thời gian chiếu sáng (h/ngày)", 2.0, 14.0, 8.0)
        sim_ndvi = st.slider("Chỉ số NDVI kỳ vọng", 0.1, 1.0, 0.7)
    
    input_data = pd.DataFrame([[sim_temp, sim_moist, sim_sun, sim_ndvi]], columns=feature_names)
    predicted_yield = rf_model.predict(input_data)[0]
    
    st.success(f"🌾 **Năng suất ước tính theo kịch bản:** {predicted_yield:,.2f} kg/ha")
    
    st.markdown("### 📈 Mức độ ảnh hưởng (Feature Importance)")
    importances = rf_model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 3))
    # Tùy chỉnh màu nền biểu đồ matplotlib để hợp với darkmode
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    sns.barplot(x=importances, y=feature_names, ax=ax, palette="viridis")
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    st.pyplot(fig)


# ==========================================
# TAB 3: THIẾT KẾ DRONE
# ==========================================
with tab3:
    st.header("🚁 Mô hình Động học UAV (Quadcopter)")
    st.write("Hệ thống giả lập Drone thu thập dữ liệu quang học dựa trên cấu hình Quadcopter 4 cánh quạt độc lập.")
    
    col_img, col_text = st.columns([1, 2])
    with col_text:
        st.markdown("""
        **Cấu hình phần cứng tham khảo:**
        * Khung (Frame): 450 mm
        * Động cơ (Motor): Brushless 2212–920KV
        * Vi điều khiển: Flight Controller Pixhawk
        * Cảm biến: Camera đa phổ (Multispectral)
        """)
        
        st.markdown("**Phương trình cân bằng lực:**")
        st.latex(r"F_{lift} = \sum_{i=1}^{4} k_f \omega_i^2 > m \cdot g")
        st.caption("Trong đó $k_f$ là hằng số lực nâng, $\omega$ là tốc độ góc rotor, $m$ là khối lượng Drone.")
