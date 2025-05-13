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
def cluster_and_plot(data, full_data, n_clusters=3):
    # Tiền xử lý dữ liệu
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Thực hiện K-Means trên dữ liệu chuẩn hóa
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)
    
    # Tính toán thông tin mô tả cho từng cụm
    data_with_labels = full_data.copy()
    data_with_labels['Cluster'] = labels
    cluster_descriptions = []
    for i in range(n_clusters):
        cluster_data = data_with_labels[data_with_labels['Cluster'] == i]
        avg_age = cluster_data['Age'].mean()
        avg_income = cluster_data['AnnualIncome'].mean()
        avg_spending = cluster_data['SpendingScore'].mean()
        num_customers = len(cluster_data)
        gender_dist = cluster_data['Gender'].value_counts().to_dict()
        female_count = gender_dist.get('Female', 0)
        male_count = gender_dist.get('Male', 0)
        
        # Tạo mô tả ngữ cảnh
        age_desc = "Trẻ" if avg_age < 30 else "Trung niên" if avg_age < 50 else "Lớn tuổi"
        income_desc = "Thấp" if avg_income < 50 else "Trung bình" if avg_income < 80 else "Cao"
        spending_desc = "Thấp" if avg_spending < 4 else "Trung bình" if avg_spending < 7 else "Cao"
        context = f"Khách hàng {age_desc.lower()}, thu nhập {income_desc.lower()}, chi tiêu {spending_desc.lower()}"
        
        description = {
            'context': context,
            'avg_age': f"{avg_age:.1f}",
            'avg_income': f"{avg_income:.1f}",
            'avg_spending': f"{avg_spending:.1f}",
            'female_count': female_count,
            'male_count': male_count
        }
        cluster_descriptions.append(description)
    
    # Tạo scatter plot với dữ liệu gốc (không chuẩn hóa)
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Màu sắc cho từng cụm
    for i in range(n_clusters):
        plt.scatter(data_with_labels[data_with_labels['Cluster'] == i]['Age'],
                    data_with_labels[data_with_labels['Cluster'] == i]['AnnualIncome'],
                    label=f'Cụm {i+1}', s=50, color=colors[i])
    
    plt.title('Kết quả phân cụm khách hàng', fontsize=14, pad=15)
    plt.xlabel('Tuổi', fontsize=12)
    plt.ylabel('Thu nhập hàng năm (triệu đồng)', fontsize=12)
    plt.legend(title="Cụm", loc="upper right", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Lưu biểu đồ vào bộ nhớ
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    # Chuẩn bị dữ liệu bảng
    table_data = data_with_labels[['Age', 'AnnualIncome', 'Cluster']].to_dict('records')
    
    return plot_url, cluster_descriptions, table_data

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
            
            # Kiểm tra xem các cột cần thiết có tồn tại không
            required_columns = ['Age', 'AnnualIncome', 'SpendingScore', 'Gender']
            if not all(col in data.columns for col in required_columns):
                return "File CSV phải chứa các cột 'Age', 'AnnualIncome', 'SpendingScore', và 'Gender'."
            
            # Lấy dữ liệu từ các cột 'Age' và 'AnnualIncome' để phân cụm
            clustering_data = data[['Age', 'AnnualIncome']]
            
            # Truyền toàn bộ dữ liệu để tính toán mô tả cụm
            plot_url, cluster_descriptions, table_data = cluster_and_plot(clustering_data, data)
            
            return render_template('result.html', plot_url=plot_url, cluster_descriptions=cluster_descriptions, table_data=table_data)
        except Exception as e:
            return f"Đã có lỗi xảy ra: {str(e)}"
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)