from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import base64

app = Flask(__name__)

# Hàm thực hiện phân cụm và tạo biểu đồ
def cluster_and_plot(data, n_clusters=3):
    # Tiền xử lý dữ liệu
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Thực hiện K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    
    # Tạo scatter plot với phụ lục
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Màu sắc cho từng cụm
    for i in range(n_clusters):
        plt.scatter(data_scaled[labels == i, 0], data_scaled[labels == i, 1], 
                    label=f'Cụm {i+1}', s=50, color=colors[i])
    
    plt.title('Kết quả phân cụm khách hàng', fontsize=14, pad=15)
    plt.xlabel('Tuổi (Chuẩn hóa)', fontsize=12)
    plt.ylabel('Thu nhập hàng năm (Chuẩn hóa)', fontsize=12)
    plt.legend(title="Cụm", loc="upper right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu biểu đồ vào bộ nhớ
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra file tải lên
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Đọc file CSV
        try:
            data = pd.read_csv(file)
            data = data.dropna()  # Loại bỏ hàng có giá trị thiếu
            
            # Giả sử dữ liệu có ít nhất 2 cột
            if data.shape[1] < 2:
                return "Dữ liệu phải có ít nhất 2 cột."
            
            # Thực hiện phân cụm và tạo biểu đồ
            plot_url = cluster_and_plot(data.iloc[:, :2])
            
            return render_template('result.html', plot_url=plot_url)
        except Exception as e:
            return f"Đã có lỗi xảy ra: {str(e)}"
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)