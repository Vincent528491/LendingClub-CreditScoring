import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. 数据加载与特征筛选 ====================
df = pd.read_csv('../过程数据/loan_cleaned.csv')

# 删除无信息量/共线性特征
drop_cols = ['tot_coll_amt', 'installment', 'total_acc', 'total_rev_hi_lim']
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

# 删除ID类特征
df = df.drop(columns=[c for c in ['id', 'member_id'] if c in df.columns], errors='ignore')

# 删除高基数日期类特征（one-hot后全是噪声）
df = df.drop(columns=[c for c in df.columns if c.startswith('earliest_cr_line')], errors='ignore')
df = df.drop(columns=[c for c in df.columns if c.startswith('issue_d')], errors='ignore')

# 删除高基数/冗余类别特征
df = df.drop(columns=[c for c in df.columns if c.startswith('addr_state')], errors='ignore')
df = df.drop(columns=[c for c in df.columns if c.startswith('emp_length')], errors='ignore')
df = df.drop(columns=['sub_grade'], errors='ignore')

# 删除loan_status（标签泄露）
df = df.drop(columns=[c for c in df.columns if c.startswith('loan_status')], errors='ignore')

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 类别特征编码
categorical_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()

# grade 有序编码
if 'grade' in categorical_cols:
    grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    X['grade'] = pd.Categorical(X['grade'], categories=grade_order, ordered=True).codes
    categorical_cols.remove('grade')

# 其他类别特征 one-hot（低基数，不会爆炸）
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

print(f"最终特征数: {X.shape[1]}")
print(f"特征列表: {X.columns.tolist()}")

# ==================== 2. 数据划分 ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n训练集: {X_train.shape}, 坏样本率: {y_train.mean():.2%}")
print(f"测试集: {X_test.shape}, 坏样本率: {y_test.mean():.2%}")

# ==================== 3. 标准化（仅逻辑回归需要） ====================
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numeric_cols = X_train.select_dtypes(include=[np.number]).columns
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ==================== 4. 逻辑回归 ====================
print("\n" + "=" * 60)
print("【模型1：逻辑回归】")
print("=" * 60)

lr = LogisticRegression(
    class_weight='balanced',
    max_iter=1000,
    random_state=42,
    solver='lbfgs',
    C=1.0
)
lr.fit(X_train_scaled, y_train)

y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_pred_lr = lr.predict(X_test_scaled)

auc_lr = roc_auc_score(y_test, y_prob_lr)
fpr, tpr, thresholds = roc_curve(y_test, y_prob_lr)
ks_lr = max(tpr - fpr)

print(f"\nAUC: {auc_lr:.4f}")
print(f"KS:  {ks_lr:.4f}")
print(f"KS位置: 阈值={thresholds[np.argmax(tpr - fpr)]:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# 逻辑回归系数
lr_coef = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr.coef_[0],
    'Abs_Coefficient': np.abs(lr.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\nTop 15 重要特征（按|系数|排序）:")
print(lr_coef.head(15).to_string(index=False))

# ==================== 5. LightGBM ====================
print("\n" + "=" * 60)
print("【模型2：LightGBM】")
print("=" * 60)

lgb_model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=20,
    min_child_samples=200,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    is_unbalance=True,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)

lgb_model.fit(
    X_train, y_train,
    eval_X=X_test,
    eval_y=y_test,
    eval_metric='auc'
)

y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]

# --- 阈值调优：找到最优KS对应的阈值 ---
fpr_lgb, tpr_lgb, thresholds_lgb = roc_curve(y_test, y_prob_lgb)
ks_lgb = max(tpr_lgb - fpr_lgb)
best_threshold_lgb = thresholds_lgb[np.argmax(tpr_lgb - fpr_lgb)]

# 用最优阈值重新预测
y_pred_lgb = (y_prob_lgb >= best_threshold_lgb).astype(int)

auc_lgb = roc_auc_score(y_test, y_prob_lgb)

print(f"\nAUC: {auc_lgb:.4f}")
print(f"KS:  {ks_lgb:.4f}")
print(f"KS最优阈值: {best_threshold_lgb:.4f}")
print(f"概率分布: min={y_prob_lgb.min():.4f}, max={y_prob_lgb.max():.4f}, "
      f"mean={y_prob_lgb.mean():.4f}, median={np.median(y_prob_lgb):.4f}")
print(f"\n分类报告 (阈值={best_threshold_lgb:.4f}):")
print(classification_report(y_test, y_pred_lgb, zero_division=0))

# ==================== 6. 可视化对比 ====================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ROC曲线
axes[0].plot(fpr, tpr, label=f'Logistic Regression (AUC={auc_lr:.4f})', color='blue', lw=2)
axes[0].plot(fpr_lgb, tpr_lgb, label=f'LightGBM (AUC={auc_lgb:.4f})', color='red', lw=2)
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_title('ROC Curve Comparison', fontsize=14)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()
axes[0].grid(True)

# 特征重要性
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': lgb_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

axes[1].barh(importance['Feature'][::-1], importance['Importance'][::-1], color='coral')
axes[1].set_title(f'LightGBM Feature Importance (best_iter={lgb_model.best_iteration_})', fontsize=14)
axes[1].set_xlabel('Importance')
axes[1].grid(True, axis='x', alpha=0.3)

# KS曲线（修复长度不匹配）
n = len(thresholds_lgb)
axes[2].plot(thresholds_lgb, tpr_lgb[:n], label='TPR (Recall)', color='green', lw=2)
axes[2].plot(thresholds_lgb, fpr_lgb[:n], label='FPR', color='red', lw=2)
axes[2].plot(thresholds_lgb, tpr_lgb[:n] - fpr_lgb[:n], label='KS', color='blue', lw=2, linestyle='--')
axes[2].axvline(x=best_threshold_lgb, color='black', linestyle=':', label=f'Best threshold={best_threshold_lgb:.4f}')
axes[2].set_title('KS Curve', fontsize=14)
axes[2].set_xlabel('Threshold')
axes[2].set_ylabel('Rate')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('../分析图表/Model_Comparison.png', dpi=150)
plt.close()

# ==================== 7. 对比总结 ====================
print("\n" + "=" * 60)
print("【模型对比总结】")
print("=" * 60)
print(f"{'Metric':<16} {'Logistic Regression':<22} {'LightGBM':<22}")
print("-" * 60)
print(f"{'AUC':<16} {auc_lr:<22.4f} {auc_lgb:<22.4f}")
print(f"{'KS':<16} {ks_lr:<22.4f} {ks_lgb:<22.4f}")
print(f"{'最优阈值':<16} {'0.5000 (默认)':<22} {best_threshold_lgb:<22.4f}")
print(f"\nLightGBM 最优轮次: {lgb_model.best_iteration_}")
print("\n所有图表已保存至: ../分析图表/")
