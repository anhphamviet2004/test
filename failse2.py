
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Đọc file CSV 
df = pd.read_csv("D:/python/sklearn-env/LogisticRegression_TuyenDung.csv")
df.columns = df.columns.str.strip()  # Xóa ký tự trắng và \n

# Lấy dữ liệu đặc trưng và nhãn
X = df[['Kinh nghiệm (năm)']].values
y = df['Kết quả tuyển dụng (1:Được tuyển, 0:Không tuyển)'].values.reshape(-1, 1)

# Tách dữ liệu thành train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =========================== CÁCH 1: THỦ CÔNG ===========================

# Thêm bias (hệ số chặn) cho cả train và test
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_b = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Khởi tạo tham số
theta = np.zeros((X_train_b.shape[1], 1))

# Hàm sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Huấn luyện logistic regression bằng gradient descent
def train_logistic(X, y, theta, lr=0.01, epochs=10000):
    m = len(y)
    for _ in range(epochs):
        h = sigmoid(X @ theta)
        gradient = X.T @ (h - y) / m
        theta -= lr * gradient
    return theta

# Huấn luyện mô hình
theta = train_logistic(X_train_b, y_train, theta)

# In hệ số
print("=== THỦ CÔNG - HỒI QUY LOGISTIC ===")
print("Hệ số chặn (theta0):", theta[0][0])
print("Hệ số góc (theta1):", theta[1][0])

# Dự đoán trên tập test
y_test_probs = sigmoid(X_test_b @ theta)
y_test_pred = (y_test_probs >= 0.5).astype(int)

# In một số kết quả test
print("\nDự đoán trên tập test (Thủ công):")
for i in range(min(5, len(y_test))):
    print(f"Thực tế: {y_test[i][0]}, Xác suất: {y_test_probs[i][0]:.2f}, Dự đoán: {y_test_pred[i][0]}")

# =========================== CÁCH 2: DÙNG THƯ VIỆN ===========================

model = LogisticRegression()
model.fit(X_train, y_train.ravel())

# In hệ số
print("\n=== THƯ VIỆN SKLEARN - HỒI QUY LOGISTIC ===")
print("Intercept (w0):", model.intercept_[0])
print("Coefficient (w1):", model.coef_[0][0])

# Dự đoán
y_lib_pred = model.predict(X_test)
y_lib_prob = model.predict_proba(X_test)[:, 1]

# In một số kết quả test
print("\nDự đoán trên tập test (Thư viện):")
for i in range(min(5, len(y_test))):
    print(f"Thực tế: {y_test[i][0]}, Xác suất: {y_lib_prob[i]:.2f}, Dự đoán: {y_lib_pred[i]}")

# So sánh độ chính xác
print("\nĐộ chính xác:")
print("Thủ công:", accuracy_score(y_test, y_test_pred))
print("Thư viện:", accuracy_score(y_test, y_lib_pred))

# =========================== VẼ ĐỒ THỊ ===========================

# Đường hồi quy từ mô hình thủ công
x_vals = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
x_vals_b = np.hstack([np.ones((x_vals.shape[0], 1)), x_vals])
y_probs = sigmoid(x_vals_b @ theta)

plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, label="Train", color='blue', alpha=0.6)
plt.scatter(X_test, y_test, label="Test", color='orange', alpha=0.6)
plt.plot(x_vals, y_probs, color="red", label="Thủ công")
plt.plot(x_vals, model.predict_proba(x_vals)[:,1], color="green", linestyle="--", label="Thư viện")
plt.xlabel("Kinh nghiệm (năm)")
plt.ylabel("Kết quả tuyển dụng (1:Được tuyển, 0:Không tuyển)")
plt.title("So sánh Logistic Regression (thủ công vs sklearn)")
plt.grid(True)
plt.legend()
plt.show()

# =========================== DỰ ĐOÁN ===========================

# Dự đoán xác suất với 20 nam kinh nghiem 
x_test_case = np.array([[1, 20]])
prob_4 = sigmoid(x_test_case @ theta)[0][0]
print(f"\n[Thủ công] Xác suất Kinh nghiệm (năm) nếu có 20 nam kinh nghiem la : {prob_4:.2f}")

# Dự đoán với người dùng nhập
n = float(input("Nhập số Kinh nghiệm (năm) trước đó: "))
x_input = np.array([[1, n]])
prob_user = sigmoid(x_input @ theta)[0][0]
print(f"[Thủ công] Xác suất Kinh nghiệm (năm) với {n} nam Kết quả tuyển dụng (1:Được tuyển, 0:Không tuyển) n: {prob_user:.2f}")


# =========================== GIẢI THÍCH TỰ LUẬN ===========================

# ☑ PHƯƠNG TRÌNH HỒI QUY LOGISTIC KHI CÓ 1 BIẾN ĐỘC LẬP:
# P(y = 1 | x) = 1 / (1 + exp(-(w0 + w1 * x)))
# Trong đó:
# - P(y = 1 | x): xác suất đối tượng sẽ mua bảo hiểm nếu có x vụ tai nạn
# - w0: hệ số chặn (intercept)
# - w1: hệ số góc (coefficient)
# - exp: hàm mũ (e ≈ 2.71828), giúp giới hạn đầu ra từ 0 đến 1

# ☑ CÓ NÊN CHUẨN HÓA DỮ LIỆU KHÔNG?
# - Với dữ liệu chỉ có 1 đặc trưng (như bài này), chuẩn hóa không cần thiết.
# - Tuy nhiên, nếu có nhiều đặc trưng và mỗi đặc trưng có thang đo khác nhau thì:
#   => NÊN chuẩn hóa để giúp mô hình hội tụ nhanh và tránh ưu tiên sai lệch.
#   => Chuẩn hóa là một phần trong tiền xử lý giúp tăng hiệu quả mô hình.

