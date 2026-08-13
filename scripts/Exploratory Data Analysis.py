import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')

# 读取清洗后的数据
df = pd.read_csv("../过程数据/loan_cleaned.csv", encoding="utf-8-sig")
print(f"Data shape: {df.shape}")

# ============================================================
# Fig 1: Target Variable Distribution
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

label_counts = df['label'].value_counts()
colors = ['#2ecc71', '#e74c3c']

axes[0].pie(label_counts, labels=['Good (0)', 'Bad (1)'],
            autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 13})
axes[0].set_title('Good / Bad Customer Ratio', fontsize=15, fontweight='bold')

bars = axes[1].bar(['Good (0)', 'Bad (1)'], label_counts.values, color=colors, width=0.5)
for bar in bars:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000,
                 f'{int(bar.get_height()):,}', ha='center', fontsize=12)
axes[1].set_title('Good / Bad Customer Count', fontsize=15, fontweight='bold')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('../分析图表/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nBad rate: {(df['label']==1).mean()*100:.2f}%")

# ============================================================
# Fig 2: Correlation Heatmap
# ============================================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'label']

corr_with_label = df[num_cols + ['label']].corr()['label'].drop('label').sort_values(key=abs, ascending=False)

print("\n[Correlation of Each Numeric Feature with Default (sorted by |r|)]")
print(corr_with_label.to_string())

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = df[num_cols + ['label']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 8})
ax.set_title('Numeric Features Correlation Heatmap', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('../分析图表/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Fig 3: Top 10 Correlated Features
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
top10 = corr_with_label.head(10)
colors_bar = ['#e74c3c' if x > 0 else '#3498db' for x in top10.values]
top10.plot(kind='barh', ax=ax, color=colors_bar)
ax.set_title('Top 10 Numeric Features Correlated with Default', fontsize=15, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Feature')
ax.axvline(x=0, color='black', linewidth=0.8)
for i, v in enumerate(top10.values):
    ax.text(v + 0.002 if v >= 0 else v - 0.002, i, f'{v:.3f}',
            va='center', ha='left' if v >= 0 else 'right', fontsize=10)
plt.tight_layout()
plt.savefig('../分析图表/03_top10_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Fig 4: Boxplot - Good vs Bad on Key Features
# ============================================================
top_features = corr_with_label.head(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(top_features):
    sns.boxplot(data=df, x='label', y=col, ax=axes[i],
                hue='label', palette={0: '#2ecc71', 1: '#e74c3c'}, legend=False)
    axes[i].set_title(f'{col} by Good / Bad', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_xticks([0,1])
    axes[i].set_xticklabels(['Good', 'Bad'])

plt.suptitle('Key Features Distribution: Good vs Bad (Boxplot)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../分析图表/04_boxplot_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Fig 5: Default Rate by Categorical Features
# ============================================================
cat_cols = ['grade', 'sub_grade', 'term', 'home_ownership',
            'verification_status', 'purpose', 'addr_state', 'emp_length']
cat_cols = [c for c in cat_cols if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
axes = axes.flatten()

plot_cats = ['grade', 'term', 'home_ownership', 'purpose']
plot_cats = [c for c in plot_cats if c in df.columns]

for i, col in enumerate(plot_cats):
    rate = df.groupby(col)['label'].agg(['mean', 'count']).reset_index()
    rate.columns = [col, 'default_rate', 'count']
    rate = rate.sort_values('default_rate', ascending=False)
    rate = rate[rate['count'] >= 100]

    bars = axes[i].barh(rate[col].astype(str), rate['default_rate'] * 100, color='#3498db')
    axes[i].set_title(f'Default Rate by {col}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Default Rate (%)')
    axes[i].set_ylabel(col)

    for bar, (_, row) in zip(bars, rate.iterrows()):
        axes[i].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                     f"{row['default_rate']*100:.1f}% (n={int(row['count']):,})",
                     va='center', fontsize=9)

plt.suptitle('Categorical Features vs Default Rate', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../分析图表/05_categorical_default_rate.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Fig 6: Violin Plot - Key Continuous Features
# ============================================================
key_continuous = ['loan_amnt', 'int_rate', 'annual_inc', 'dti']
key_continuous = [c for c in key_continuous if c in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(key_continuous):
    sns.violinplot(data=df, x='label', y=col, ax=axes[i],
                   hue='label', palette={0: '#2ecc71', 1: '#e74c3c'},
                   inner='box', legend=False)
    axes[i].set_title(f'{col} by Good / Bad (Violin)', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_xticks([0,1])
    axes[i].set_xticklabels(['Good', 'Bad'])

plt.suptitle('Key Continuous Features: Good vs Bad (Violin)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('../分析图表/06_violin_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("[EDA Summary]")
print("=" * 60)

print("\n1. Top 10 Numeric Features Correlated with Default:")
for rank, (feat, corr) in enumerate(corr_with_label.head(10).items(), 1):
    direction = "Positive (higher = riskier)" if corr > 0 else "Negative (higher = safer)"
    print(f"   {rank}. {feat}: {corr:.4f} ({direction})")

print("\n2. Default Rate by Categorical Features:")
for col in plot_cats:
    if col in df.columns:
        rate = df.groupby(col)['label'].mean().sort_values(ascending=False)
        print(f"\n   {col}:")
        for val, r in rate.items():
            count = (df[col] == val).sum()
            if count >= 100:
                print(f"      {val}: {r*100:.1f}% (n={count:,})")

print("\n3. All charts saved to: ../分析图表/")
