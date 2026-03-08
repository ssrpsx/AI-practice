import numpy as np

# ข้อมูลสมมติ: [มีเมฆ(1/0), ลมแรง(1/0)] -> ฝนตก(1/0)
data = np.array([
    [1, 1, 1], # เมฆมี ลมแรง -> ตก
    [1, 0, 1], # เมฆมี ลมนิ่ง -> ตก
    [0, 1, 0], # เมฆไม่มี ลมแรง -> ไม่ตก
    [0, 0, 0]  # เมฆไม่มี ลมนิ่ง -> ไม่ตก
])

def calculate_bayes(feature_idx, feature_val, target_val):
    # P(Target) - Prior
    p_target = np.mean(data[:, 2] == target_val)
    
    # P(Feature | Target) - Likelihood
    subset = data[data[:, 2] == target_val]
    p_feature_given_target = np.mean(subset[:, feature_idx] == feature_val)
    
    # P(Feature) - Evidence
    p_feature = np.mean(data[:, feature_idx] == feature_val)
    
    # Bayes' Theorem
    posterior = (p_feature_given_target * p_target) / p_feature
    return posterior

# ลองคำนวณ: ความน่าจะเป็นที่ "ฝนจะตก" เมื่อเห็นว่า "มีเมฆ (feature 0 = 1)"
prob = calculate_bayes(0, 1, 1)
print(f"ความน่าจะเป็นที่ฝนจะตกเมื่อมีเมฆ: {prob * 100:.2f}%")