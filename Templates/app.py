import os
from flask import Flask, request, jsonify, render_template, session
import joblib
import numpy as np

# Khởi tạo Flask app
template_dir = os.path.abspath(r'D:\HUIT\Học Máy\NhomEEE_DoAnCuoiKy\Demo\Templates')
app = Flask(__name__, template_folder=template_dir, static_folder='static')
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Route chính hiển thị trang chủ
@app.route('/')
def home():
    return render_template('index.html')

# Route xử lý form submit
@app.route('/submit', methods=['POST'])
def submit():
    # Lấy dữ liệu từ form
    data = request.form
    age = int(data['age'])
    sex = int(data['sex'])
    cp = int(data['cp'])
    trestbps = int(data['trestbps'])
    chol = int(data['chol'])
    fbs = int(data['fbs'])
    restecg = int(data['restecg'])
    thalach = int(data['thalach'])
    exang = int(data['exang'])
    oldpeak = float(data['oldpeak'])
    slope = int(data['slope'])
    ca = int(data['ca'])
    thal = int(data['thal'])

    # Lấy mô hình từ session
    selected_model = session.get('selected_model', 'random_forest')
    model_path = get_model_path(selected_model)

    # Load mô hình tương ứng
    model = joblib.load(model_path)

    # Tạo vector chứa dữ liệu
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    # Dự đoán bệnh tim
    prediction = model.predict(features)

    # Chuyển đổi dự đoán sang kết quả chuẩn đoán
    if prediction[0] == 0:
        diagnosis = 'Không mắc bệnh tim'
    else:
        diagnosis = 'Có nguy cơ mắc bệnh tim'

    # Trả về kết quả dưới dạng JSON
    return jsonify({'diagnosis': diagnosis})

# Route để lưu trữ lựa chọn thuật toán vào session
@app.route('/select_model', methods=['POST'])
def select_model():
    selected_model = request.form.get('algorithm', 'random_forest')
    session['selected_model'] = selected_model
    return jsonify({'selected_model': selected_model})

def get_model_path(selected_model):
    models_folder = r'D:\HUIT\Học Máy\NhomEEE_DoAnCuoiKy\Demo\Models'
    model_files = {
        'random_forest': 'random_model.pkl',
        'logistic_regression': 'logistic_model.pkl',
        'decision_tree': 'decision_model.pkl',
        'knn': 'knn_model.pkl'
    }
    return os.path.join(models_folder, model_files[selected_model])

# Thêm phần này để chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
