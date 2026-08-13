import pandas as pd
import numpy as np

# ============================================================
# 第一步：读取数据
# ============================================================
df = pd.read_csv("./原始数据/loan.csv", encoding="utf-8", low_memory=False)
print(f"原始数据: {df.shape[0]} 行, {df.shape[1]} 列")

# ============================================================
# 第二步：删除高缺失率列（缺失率 > 50%）
# ============================================================
missing_ratio = df.isnull().sum() / len(df)
cols_to_drop_by_missing = missing_ratio[missing_ratio > 0.5].index.tolist()
df.drop(columns=cols_to_drop_by_missing, inplace=True)
print(f"\n因缺失率>50%删除的列({len(cols_to_drop_by_missing)}列): {cols_to_drop_by_missing}")
print(f"删除后: {df.shape[0]} 行, {df.shape[1]} 列")

# ============================================================
# 第三步：删除冗余/无建模价值的列
# ============================================================
cols_to_drop_manual = [
    'id', 'member_id',            # 唯一标识，无建模意义
    'url',                         # 链接，无意义
    'desc',                        # 大段文本，基础模型不用
    'title',                       # 与purpose重复
    'zip_code',                    # 邮编太细，用addr_state即可
    'emp_title',                   # 公司名称太杂，难以编码
    'pymnt_plan',                  # 几乎全是n
    'policy_code',                 # 只有一个值1
    'application_type',            # 几乎全是Individual
    'initial_list_status',         # 对违约预测价值低
    # ---- 以下是贷后衍生变量，会导致信息泄露 ----
    'out_prncp', 'out_prncp_inv',
    'total_pymnt', 'total_pymnt_inv',
    'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
    'recoveries', 'collection_recovery_fee',
    'last_pymnt_d', 'last_pymnt_amnt',
    'next_pymnt_d',
    'last_credit_pull_d',
    'funded_amnt', 'funded_amnt_inv',
]
# 只删除实际存在的列
cols_to_drop_manual = [c for c in cols_to_drop_manual if c in df.columns]
df.drop(columns=cols_to_drop_manual, inplace=True)
print(f"\n手动删除的冗余/泄露列({len(cols_to_drop_manual)}列): {cols_to_drop_manual}")
print(f"删除后: {df.shape[0]} 行, {df.shape[1]} 列")

# ============================================================
# 第四步：处理日期列 & 筛选表现期
# ============================================================
# issue_d 格式为 "MMM-YYYY"，如 "Dec-2011"
df['issue_d'] = pd.to_datetime(df['issue_d'], format='%b-%Y')

# 找到数据中最新的放款日期
max_issue_date = df['issue_d'].max()
print(f"\n放款日期范围: {df['issue_d'].min()} ~ {max_issue_date}")

# 设定表现期为12个月：只保留放款日期在截止日期之前的样本
# 即确保每笔贷款至少有12个月的表现期来观察是否违约
cutoff_date = max_issue_date - pd.DateOffset(months=12)
print(f"表现期截止日(放款日+12个月): {cutoff_date}")

df = df[df['issue_d'] <= cutoff_date].copy()
print(f"筛选表现期后: {df.shape[0]} 行")

# ============================================================
# 第五步：定义标签（Y值）
# ============================================================
# 好客户
good_status = ['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid']
# 坏客户
bad_status = ['Charged Off', 'Default', 'Late (31-120 days)',
              'Does not meet the credit policy. Status:Charged Off']

# 只保留好客户和坏客户，剔除仍处于贷中/模糊状态的样本
df = df[df['loan_status'].isin(good_status + bad_status)].copy()

# 生成二分类标签
df['label'] = df['loan_status'].apply(lambda x: 1 if x in bad_status else 0)

print(f"\n标签定义后: {df.shape[0]} 行")
print(f"好客户(0): {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
print(f"坏客户(1): {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")

# ============================================================
# 第六步：处理缺失值
# ============================================================
# 数值型：用中位数填充
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'label']  # 排除标签列

for col in num_cols:
    if df[col].isnull().sum() > 0:
        median_val = df[col].median()
        # 【修改点1】用重新赋值的方式替代 inplace=True
        df[col] = df[col].fillna(median_val)

# 类别型：用 "Missing" 填充
# 【修改点2】明确包含 'str' 类型以消除警告
cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        # 【修改点1】同样，用重新赋值的方式替代 inplace=True
        df[col] = df[col].fillna('Missing')

print(f"\n缺失值填充完成，剩余缺失数: {df.isnull().sum().sum()}")

# ============================================================
# 第七步：处理异常值
# ============================================================
# 年收入为0的样本剔除（逻辑错误）
df = df[df['annual_inc'] > 0].copy()
print(f"剔除年收入<=0后: {df.shape[0]} 行")

# 对年收入做截断处理（Winsorization），限制在1%~99%分位数
lower = df['annual_inc'].quantile(0.01)
upper = df['annual_inc'].quantile(0.99)
df['annual_inc'] = df['annual_inc'].clip(lower, upper)
print(f"年收入截断范围: {lower:.0f} ~ {upper:.0f}")

# ============================================================
# 第八步：最终检查
# ============================================================
print("\n" + "=" * 50)
print(f"最终数据: {df.shape[0]} 行, {df.shape[1]} 列")
print(f"特征列数: {df.shape[1] - 1} (不含标签)")
print(f"\n剩余列名:")
print(df.columns.tolist())
print(f"\n各列缺失值:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# 保存清洗后的数据
df.to_csv("./过程数据/loan_cleaned.csv", index=False, encoding="utf-8-sig")
print("\n清洗后数据已保存至: ./过程数据/loan_cleaned.csv")
