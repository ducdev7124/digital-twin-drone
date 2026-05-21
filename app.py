import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# CẤU HÌNH TRANG WEB VÀ CSS TÙY CHỈNH
# ==========================================
st.set_page_config(
    page_title="Digital Twin Nông Nghiệp", 
    layout="wide", 
    page_icon="🌱",
    initial_sidebar_state="expanded"
)

# Custom CSS để làm đẹp giao diện các Metric và Header
st.markdown("""
<style>
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 5% 5% 5% 10%;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1E1E1E;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODULE 1: TẠO DỮ LIỆU MÔ PHỎNG (CACHE ĐỂ TỐI ƯU)
# ==========================================
@st.cache_data
def generate_timeseries_data():
    """Sinh dữ liệu chuỗi thời gian 100 ngày cho Tab Giám sát"""
    np.random.seed(42)
    days = np.arange(1, 101)
    
    # Nhiệt độ dao động 20 - 40 độ C, có chu kỳ ngày đêm
    temperature = np.random.normal(26, 4, 100) + np.sin(days/5)*4
    # Độ ẩm đất dao động 20 - 40%, có chu kỳ tưới
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
    """Sinh dữ liệu tĩnh cho Tab AI dự báo dựa trên hàm toán học"""
    np.random.seed(99)
    data = []
    for _ in range(n_samples):
        # Giả lập thông số môi trường ngẫu nhiên
        avg_temp = np.random.normal(26, 5)
        avg_moist = np.random.normal(50, 15)
        sunlight = np.random.normal(8, 2)
        
        # NDVI được mô phỏng có tính độc lập cao hơn để AI không bị thiên vị
        base_ndvi = 0.4 + (sunlight * 0.02) + np.random.normal(0, 0.1)
        ndvi = np.clip(base_ndvi, 0.1, 0.95)
        
        # CÔNG THỨC NĂNG SUẤT ĐÃ ĐƯỢC CÂN BẰNG LẠI TRỌNG SỐ:
        # 1. NDVI (Sức khỏe lá): Đóng góp nền tảng
        base_yield = 2000 + (2500 * ndvi)
        
        # 2. Nhiệt độ: Lý tưởng là 27 độ, lệch càng nhiều phạt càng nặng (Mô phỏng Stress nhiệt)
        temp_penalty = 120 * abs(avg_temp - 27)
        
        # 3. Độ ẩm: Lý tưởng là 60%, lệch càng nhiều phạt càng nặng (Mô phỏng hạn hán / ngập úng)
        moist_penalty = 60 * abs(avg_moist - 60)
        
        # 4. Ánh sáng: Đóng góp thêm vào quá trình quang hợp
        sun_bonus = 150 * sunlight
        
        # Năng suất cuối cùng = Nền tảng + Ánh sáng - Phạt nhiệt độ - Phạt độ ẩm
        yield_kg = base_yield + sun_bonus - temp_penalty - moist_penalty + np.random.normal(0, 150)
        yield_kg = max(yield_kg, 500) # Đảm bảo năng suất không bị âm
        
        data.append([avg_temp, avg_moist, sunlight, ndvi, yield_kg])
        
    cols = ['Temperature', 'Soil_Moisture', 'Sunlight', 'NDVI', 'Yield']
    return pd.DataFrame(data, columns=cols)

@st.cache_resource
def train_model(df):
    """Huấn luyện mô hình Random Forest Regressor"""
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return model, rmse, r2, X.columns, X.mean()

# ==========================================
# KHỞI TẠO DỮ LIỆU & HUẤN LUYỆN AI
# ==========================================
df_ts = generate_timeseries_data()
df_ai = generate_ai_training_data()
rf_model, rf_rmse, rf_r2, feature_names, X_mean = train_model(df_ai)

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🌱 Digital Twin Nông Nghiệp: Mô phỏng & Dự báo năng suất")
st.markdown("Hệ thống giả lập môi trường, thu thập dữ liệu bằng Drone ảo và sử dụng AI để ra quyết định.")
st.markdown("---")

# TẠO 3 TAB CHÍNH
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Giám sát", "🤖 Phân tích Kịch bản (What-if)", "🚁 Thiết kế Drone"])

# ==========================================
# TAB 1: DASHBOARD GIÁM SÁT (TIME-SERIES)
# ==========================================
with tab1:
    col_map, col_chart = st.columns([1.2, 2])
    
    with col_map:
        st.subheader("1. Drone View (NDVI & RGB)")
        # Thanh trượt chọn ngày
        selected_day = st.slider("Chọn ngày quan sát (Mô phỏng thời gian thực):", min_value=1, max_value=100, value=100)
        
        # Lấy dữ liệu của ngày hiện tại
        current_data = df_ts[df_ts['Day'] == selected_day].iloc[0]
        current_biomass = current_data['Biomass']
        
        # Tính toán NDVI trung bình của ngày hôm đó (Scale theo Biomass)
        avg_ndvi = min(current_biomass / 100.0 + np.random.normal(0, 0.02), 0.98)
        avg_ndvi = max(avg_ndvi, 0.1) # Giới hạn không dưới 0.1
        
        # Nút radio chọn chế độ xem
        view_mode = st.radio("Chế độ hiển thị Camera:", ["🔴 NDVI (Phổ hồng ngoại - Sức khỏe cây)", "🟢 RGB (Màu quang học thực tế)"], horizontal=False)
        
        # Tạo mảng ảnh giả lập 30x30 Pixel
        grid_size = 30
        image_grid = np.random.normal(avg_ndvi, 0.05, (grid_size, grid_size))
        
        # Xử lý tạo "luống cày" ngang (Mô phỏng khoảng trống giữa các hàng cây)
        for i in range(3, grid_size, 5):
            image_grid[i, :] = avg_ndvi - 0.35 
            
        image_grid = np.clip(image_grid, 0, 1)

        # Vẽ bản đồ bằng Plotly
        if "NDVI" in view_mode:
            colorscale = "RdYlGn"
            title_map = f"Bản đồ phân bố NDVI - Ngày {selected_day} <br><sup>Chỉ số trung bình: {avg_ndvi:.2f}</sup>"
        else:
            colorscale = "Greens"
            title_map = f"Bản đồ quang học RGB - Ngày {selected_day}"

        fig_map = px.imshow(image_grid, color_continuous_scale=colorscale, zmin=0, zmax=1, title=title_map)
        fig_map.update_layout(
            template="plotly_dark",
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_colorbar=dict(title="Chỉ số", thicknessmode="pixels", thickness=15)
        )
        fig_map.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig_map.update_traces(hovertemplate="Tọa độ: (%{x}, %{y})<br>Chỉ số: %{z:.3f}<extra></extra>")
        st.plotly_chart(fig_map, use_container_width=True)

    with col_chart:
        st.subheader("2. Giám sát Vi khí hậu & Sinh trưởng")
        
        # BẢN ĐỒ MÔI TRƯỜNG
        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(x=df_ts['Day'][:selected_day], y=df_ts['Temperature'][:selected_day], mode='lines', name='Nhiệt độ (°C)', line=dict(color='#ff9f43', width=2)))
        fig_env.add_trace(go.Scatter(x=df_ts['Day'][:selected_day], y=df_ts['SoilMoisture'][:selected_day], mode='lines', name='Độ ẩm đất (%)', line=dict(color='#00d2d3', width=2)))
        
        fig_env.update_layout(
            height=280,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_env, use_container_width=True)
        
        # BẢN ĐỒ SINH TRƯỞNG (BIOMASS)
        fig_bio = go.Figure()
        fig_bio.add_trace(go.Scatter(x=df_ts['Day'], y=df_ts['Biomass'], fill='tozeroy', mode='lines', name='Sinh khối tích lũy', line=dict(color='#10ac84'), fillcolor='rgba(16, 172, 132, 0.3)'))
        
        # Thêm đường line báo "Ngày hiện tại"
        fig_bio.add_shape(type="line", x0=selected_day, y0=0, x1=selected_day, y1=max(df_ts['Biomass']),
                          line=dict(color="#ee5253", width=2, dash="dash"))
        fig_bio.add_annotation(x=selected_day, y=max(df_ts['Biomass'])*0.9, text="Ngày quan sát", showarrow=True, arrowhead=1, ax=-40, ay=0, font=dict(color="white"))
        
        fig_bio.update_layout(
            height=280,
            template="plotly_dark",
            yaxis_title="Sinh khối (g)",
            margin=dict(l=20, r=20, t=20, b=20),
            hovermode="x"
        )
        st.plotly_chart(fig_bio, use_container_width=True)

# ==========================================
# TAB 2: AI DỰ BÁO (WHAT-IF SCENARIO TỐI ƯU HÓA)
# ==========================================
with tab2:
    st.header("🧠 Trí tuệ nhân tạo (Random Forest Regressor)")
    st.markdown("Mô hình AI học từ dữ liệu lịch sử để dự đoán năng suất dựa trên sự thay đổi của môi trường.")
    
    # Hiển thị độ tin cậy của AI
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Thuật toán sử dụng", value="Random Forest")
    m2.metric(label="Độ chính xác (R-squared)", value=f"{rf_r2*100:.2f} %")
    m3.metric(label="Sai số trung bình (RMSE)", value=f"{rf_rmse:.1f} kg/ha")
    
    st.markdown("---")
    st.subheader("🛠️ Kịch bản giả định (What-if Analysis)")
    st.info("Kịch bản What-if cho phép bạn thay đổi thông số môi trường giả định và xem tác động (Tăng/Giảm) so với năng suất cơ sở (Baseline).")
    
    # Tính toán Baseline (Mức cơ sở lý tưởng)
    baseline_inputs = pd.DataFrame([[27.0, 60.0, 8.0, 0.75]], columns=feature_names)
    baseline_yield = rf_model.predict(baseline_inputs)[0]
    
    # Khu vực điều khiển What-if
    c1, c2, c3 = st.columns([1, 1, 1.5])
    
    with c1:
        st.markdown("**1. Thông số vi khí hậu**")
        sim_temp = st.slider("Nhiệt độ môi trường (°C)", 15.0, 40.0, 27.0, 0.5)
        sim_moist = st.slider("Độ ẩm đất (%)", 10.0, 90.0, 60.0, 1.0)
        
    with c2:
        st.markdown("**2. Điều kiện quang hợp**")
        sim_sun = st.slider("Thời gian chiếu sáng (h/ngày)", 2.0, 14.0, 8.0, 0.5)
        sim_ndvi = st.slider("Chỉ số NDVI kỳ vọng", 0.1, 1.0, 0.75, 0.05)
        
    with c3:
        st.markdown("**3. Kết quả mô phỏng hệ thống**")
        # Dự đoán với thông số mới
        sim_inputs = pd.DataFrame([[sim_temp, sim_moist, sim_sun, sim_ndvi]], columns=feature_names)
        simulated_yield = rf_model.predict(sim_inputs)[0]
        
        # Tính toán chênh lệch (Delta)
        yield_delta = simulated_yield - baseline_yield
        delta_pct = (yield_delta / baseline_yield) * 100
        
        st.metric(
            label="🌾 NĂNG SUẤT DỰ KIẾN (WHAT-IF)", 
            value=f"{simulated_yield:,.0f} kg/ha", 
            delta=f"{yield_delta:,.0f} kg/ha ({delta_pct:.1f}%) so với tiêu chuẩn",
            delta_color="normal"
        )
        
        if yield_delta > 0:
            st.success("Tín hiệu tốt! Điều kiện giả định giúp tăng năng suất mùa màng.")
        elif yield_delta < -500:
            st.error("Cảnh báo! Điều kiện khắc nghiệt làm sụt giảm nghiêm trọng năng suất.")
        else:
            st.warning("Năng suất giảm nhẹ. Cần điều chỉnh hệ thống tưới tiêu.")

    st.markdown("---")
    st.subheader("📈 Mức độ ảnh hưởng của các biến (Feature Importance)")
    
    importances = rf_model.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': ['Nhiệt độ', 'Độ ẩm đất', 'Chiếu sáng', 'Chỉ số NDVI'],
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)
    
    fig_imp = px.bar(
        df_importance, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        color='Importance',
        color_continuous_scale='viridis',
        text_auto='.1%'
    )
    fig_imp.update_layout(
        template='plotly_dark',
        xaxis_title="Trọng số ảnh hưởng (0 - 1.0)",
        yaxis_title="Yếu tố đầu vào",
        height=300,
        margin=dict(l=20, r=20, t=30, b=20),
        coloraxis_showscale=False # Ẩn thanh màu bên cạnh cho gọn
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ==========================================
# TAB 3: THIẾT KẾ DRONE ĐỘNG HỌC
# ==========================================
with tab3:
    st.header("🚁 Mô hình Động học UAV (Quadcopter) trong Digital Twin")
    st.write("Hệ thống giả lập Drone làm nhiệm vụ bay quét bản đồ quang học định kỳ để lấy chỉ số NDVI mà không cần triển khai thiết bị vật lý tốn kém.")
    
    col_img, col_text = st.columns([1.5, 2])
    
    with col_img:
        # Thay vì chỉ để text trống, dùng card markdown để tạo khung nổi bật
        st.markdown("""
        <div style="background-color: #2d3436; padding: 30px; border-radius: 10px; text-align: center; border: 1px dashed #74b9ff;">
            <h1 style="font-size: 60px;">🚁</h1>
            <h4 style="color: #74b9ff;">UAV Virtual Sensor</h4>
            <p>Mô phỏng luồng dữ liệu Multispectral Camera<br>Cập nhật liên tục vào hệ thống</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_text:
        st.markdown("### Thông số thiết kế bản sao số (Digital Twin)")
        st.markdown("""
        **Cấu hình phần cứng giả lập:**
        * **Khung (Frame):** 450 mm (Đảm bảo độ cân bằng khí động học)
        * **Động cơ (Motor):** Brushless 2212–920KV
        * **Vi điều khiển:** Flight Controller Pixhawk mô phỏng
        * **Cảm biến:** Camera đa phổ (Multispectral) - Phản hồi giá trị NIR & RED.
        """)
        
        st.info("Để Drone duy trì trạng thái bay tĩnh (Hover) quét cánh đồng, tổng lực nâng sinh ra từ 4 động cơ phải cân bằng hoặc lớn hơn trọng lượng máy bay.")
        
        st.markdown("**Phương trình cân bằng lực động học:**")
        st.latex(r"F_{lift} = \sum_{i=1}^{4} k_f \omega_i^2 \ge m \cdot g")
        st.caption("Trong đó: $k_f$ là hằng số lực nâng của cánh quạt, $\omega$ là tốc độ góc rotor, $m$ là tổng khối lượng Drone, $g$ là gia tốc trọng trường.")
