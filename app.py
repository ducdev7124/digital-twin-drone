import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# MODULE 1: GIẢ LẬP MÔI TRƯỜNG VẬT LÝ (PHYSICAL SPACE)
# ==========================================
def generate_agricultural_data(n_samples=500):
    """
    Sinh tập dữ liệu mô phỏng các mùa vụ khác nhau.
    """
    np.random.seed(42)
    data = []
    
    for _ in range(n_samples):
        # Mô phỏng vi khí hậu trung bình của một mùa vụ
        avg_temp = np.random.normal(26, 4)       # Nhiệt độ TB 26 độ C
        avg_moist = np.random.normal(45, 12)     # Độ ẩm đất TB 45%
        sunlight_hours = np.random.normal(8, 2)  # Giờ nắng TB 8h/ngày
        
        # Mô hình sinh trưởng sinh khối (Biomass Accumulation)
        # Sinh khối cao nhất khi nhiệt độ ~25-28 độ, độ ẩm ~50%
        temp_factor = 1.0 - abs(avg_temp - 27) / 20.0
        moist_factor = 1.0 - abs(avg_moist - 50) / 40.0
        
        # Biến đổi NDVI từ sinh khối (Giả lập dữ liệu Drone)
        base_ndvi = 0.3 + (temp_factor * moist_factor * 0.6)
        ndvi = np.clip(np.random.normal(base_ndvi, 0.05), 0.1, 0.95)
        
        # Tính toán năng suất cuối vụ (kg/ha)
        yield_kg = 5000 * ndvi * (sunlight_hours/8) + np.random.normal(0, 200)
        
        data.append([avg_temp, avg_moist, sunlight_hours, ndvi, yield_kg])
        
    cols = ['Temperature', 'Soil_Moisture', 'Sunlight', 'NDVI', 'Yield']
    return pd.DataFrame(data, columns=cols)

# ==========================================
# MODULE 2: XÂY DỰNG BẢN SAO SỐ (VIRTUAL TWIN & AI)
# ==========================================
@st.cache_resource # Lưu cache để mô hình không phải train lại mỗi khi kéo thanh trượt
def train_digital_twin(df):
    """
    Huấn luyện mô hình Random Forest đóng vai trò AI cho Digital Twin
    """
    X = df.drop('Yield', axis=1)
    y = df['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Khởi tạo mô hình rừng quyết định
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Đánh giá chất lượng mô hình
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    return rf_model, rmse, r2, X.columns

# ==========================================
# MODULE 3: STREAMLIT DASHBOARD (APPLICATION LAYER)
# ==========================================
def run_dashboard():
    # Cấu hình trang web
    st.set_page_config(page_title="Agri Digital Twin", layout="wide", page_icon="🌱")
    st.title("🌱 Mô phỏng Digital Twin - Dự báo Năng suất Cây trồng")
    st.markdown("**Đồ án tốt nghiệp - Nhóm sinh viên Cơ điện tử HUTECH**")
    
    # 1. Chuẩn bị dữ liệu và mô hình
    df = generate_agricultural_data()
    model, rmse, r2, features = train_digital_twin(df)
    
    # Hiển thị Metrics (Độ chính xác AI)
    st.subheader("🤖 Hiệu suất Mô hình AI (Random Forest)")
    col1, col2 = st.columns(2)
    col1.metric(label="Độ chính xác (R-squared)", value=f"{r2*100:.2f} %")
    col2.metric(label="Sai số trung bình (RMSE)", value=f"{rmse:.1f} kg/ha")
    
    st.markdown("---")
    
    # 2. Giao diện Phân tích Kịch bản (What-if Analysis) ở Cột trái (Sidebar)
    st.sidebar.header("🛠️ Kịch bản Môi trường (What-if)")
    st.sidebar.write("Kéo thanh trượt để điều chỉnh các thông số giả định và quan sát tác động đến năng suất.")
    
    sim_temp = st.sidebar.slider("Nhiệt độ môi trường (°C)", 10.0, 45.0, 26.0)
    sim_moist = st.sidebar.slider("Độ ẩm đất (%)", 10.0, 90.0, 45.0)
    sim_sun = st.sidebar.slider("Thời gian chiếu sáng (h/ngày)", 2.0, 14.0, 8.0)
    sim_ndvi = st.sidebar.slider("Chỉ số NDVI hiện tại (Drone giả lập)", 0.1, 1.0, 0.7)
    
    # 3. Thực thi dự báo thời gian thực
    input_data = pd.DataFrame([[sim_temp, sim_moist, sim_sun, sim_ndvi]], columns=features)
    predicted_yield = model.predict(input_data)[0]
    
    # Hiển thị kết quả chính
    st.subheader("📊 Kết quả Dự báo Thời gian thực")
    st.success(f"Năng suất ước tính theo kịch bản: **{predicted_yield:,.2f} kg/ha**")
    
    # 4. Biểu đồ Feature Importance
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📈 Mức độ ảnh hưởng của các yếu tố (Feature Importance)")
    st.write("Biểu đồ thể hiện yếu tố nào đang tác động mạnh nhất đến năng suất mùa màng trong bản sao số.")
    
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=importances, y=features, ax=ax, palette="viridis")
    ax.set_title("Trọng số đóng góp của từng biến số")
    ax.set_xlabel("Mức độ quan trọng")
    ax.set_ylabel("Thông số đầu vào")
    st.pyplot(fig)

# Khởi chạy hệ thống
if __name__ == "__main__":
    run_dashboard()