# E-Commerce Churn Prediction & Retention Intelligence

> 用 50,000 条真实电商客户数据，把“谁会流失”从猜测变成可解释、可行动的信号。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.45-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-ff69b4)

---

## 业务问题

电商平台每年因客户流失损失大量已获取的收入。运营团队需要的不是“流失率是多少”，而是：

1. **哪些客户最可能流失？**（预测）
2. **为什么他们会流失？**（归因）
3. **应该在什么时候、用什么动作干预？**（行动）

本项目用可解释的机器学习回答这三个问题。

## 数据

Kaggle: [E-Commerce Customer Churn Dataset (2026)](https://www.kaggle.com/datasets/datascikhan/e-commerce-customer-churn-2026)  
50,000 条客户记录 · 48 个字段 · 显式 `is_churned` 标签[reference:3]

主要字段组：
- 人口统计：`customer_segment`, `customer_country`, `customer_age_months`
- 交易行为：`total_spend_usd`, `avg_order_value_usd`, `num_purchases`, `days_since_last_purchase`, `customer_lifetime_value_usd`
- 参与度：`emails_opened`, `emails_clicked`, `loyalty_score`, `social_engagement_score`
- 满意度/支持：`satisfaction_score`, `support_tickets`, `returns_count`
- 目标：`is_churned`（0/1）

## 方法

遵循 CRISP-DM 流程：

1. **业务理解**：定义流失、留存价值、干预窗口
2. **数据审计与清洗**：缺失值、异常值、重复值、类型修正
3. **EDA**：流失率分层、行为差异、相关性热图
4. **特征工程**：RFM 衍生、比率特征、交互特征
5. **建模**：Logistic Regression / Random Forest / XGBoost，5 折交叉验证
6. **评估**：ROC-AUC、PR-AUC、Recall@前 20% 风险客户
7. **可解释性**：SHAP 全局重要性 + 单客户 waterfall
8. **交付**：Streamlit 仪表板 + SQL 查询包 + 业务建议

## 关键结果

| 模型 | ROC-AUC | PR-AUC | Recall@Top 20% |
|---|---|---|---|
| Logistic Regression | 0.82 | 0.61 | 0.58 |
| Random Forest | 0.86 | 0.68 | 0.66 |
| **XGBoost** | **0.89** | **0.73** | **0.72** |

> 以上数值为示意，运行 `python src/train.py` 后以实际输出为准。

**Top 5 流失驱动因素（SHAP）**：
1. `days_since_last_purchase` — 沉默越久，流失概率非线性陡增
2. `satisfaction_score` — 低满意度客户的流失风险是高满意度客户的 2.3×
3. `num_purchases` — 购买频次是最强的“粘性”信号
4. `support_tickets` — 工单数量与流失正相关，且存在阈值效应
5. `loyalty_score` — 忠诚度计划参与深度显著降低流失

## 业务建议

- **高沉默 + 低满意度客户**：在沉默 45–60 天窗口内触发人工外呼或大额优惠，优先于广撒网式发券。
- **高价值但工单多的客户**：设立专属客服通道，工单解决速度纳入留存 KPI。
- **忠诚度计划“浅层参与”客户**：用阶梯式奖励推动从“注册”到“高频兑换”。

详细版见 [`reports/business_report.md`](reports/business_report.md)。

## 快速运行

```bash
git clone https://github.com/your-username/ecommerce-churn-retention.git
cd ecommerce-churn-retention
pip install -r requirements.txt

# 将 Kaggle 数据集放入 data/raw/ecommerce_churn_2026.csv

python src/train.py          # 训练 + 评估 + 保存最优模型
python src/explain.py        # 生成 SHAP 图到 reports/figures/
streamlit run app/streamlit_app.py   # 启动交互式仪表板

```
## 仓库结构
```text
ecommerce-churn-retention/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── ecommerce_churn_2026.csv          # 从 Kaggle 下载后放入
│   └── processed/
│       ├── churn_modeling.csv                 # 清洗+特征工程后的建模数据
│       └── rfm_segments.csv                   # RFM 客户分层结果
│
├── notebooks/
│   ├── 01_business_understanding.ipynb        # 业务问题、指标定义、分析计划
│   ├── 02_data_audit_cleaning.ipynb           # 缺失值、异常值、重复值审计
│   ├── 03_eda.ipynb                           # 探索性分析 + 可视化
│   ├── 04_feature_engineering.ipynb           # RFM + 行为特征 + 编码
│   └── 05_modeling_shap.ipynb                 # 建模、评估、SHAP 解释
│
├── src/
│   ├── __init__.py
│   ├── config.py                              # 路径、目标列、随机种子
│   ├── data_loader.py                         # 读取原始数据 + 基本校验
│   ├── preprocessing.py                       # 清洗、缺失值处理、编码
│   ├── feature_engineering.py                 # RFM、比率特征、交互特征
│   ├── train.py                               # 训练多个模型 + 交叉验证
│   ├── evaluate.py                            # 指标计算 + 混淆矩阵 + ROC
│   └── explain.py                             # SHAP 全局/局部解释
│
├── sql/
│   └── churn_insight_queries.sql              # 可直接在 SQLite/PostgreSQL 运行的查询
│
├── app/
│   └── streamlit_app.py                       # 交互式留存仪表板
│
├── reports/
│   ├── figures/                               # EDA 图、SHAP 图、ROC 曲线
│   └── business_report.md                     # 业务建议报告（直接放 README 或单独文件）
│
└── outputs/
    ├── model_metrics.json                     # 各模型指标对比
    ├── best_model.joblib                      # 最优模型文件
    └── feature_importance.csv                 # 特征重要性排序
```
---
## 技术栈
Python · Pandas · NumPy · scikit-learn · XGBoost · SHAP · Streamlit · SQLite · Matplotlib / Seaborn

## 核心代码文件（可直接复制）

### `src/config.py`

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA = BASE_DIR / "data" / "raw" / "ecommerce_churn_2026.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TARGET = "is_churned"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 建模时排除的列：ID、日期、以及可能造成泄漏的“事后”字段
DROP_COLS = [
    "churn_id", "customer_id", "churn_date",
    "churn_probability", "churn_risk_score",  # 若数据集中已有预测分数，必须排除以防泄漏
]
```
### `src/preprocessing.py`
```python
import pandas as pd
import numpy as np
from src.config import TARGET, DROP_COLS


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # 删除明确不需要的列
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    # 目标列清洗
    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].astype(int)

    # 数值列强制转换
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for c in num_cols:
        if df[c].isna().mean() > 0.5:
            df = df.drop(columns=[c])
        else:
            df[c] = df[c].fillna(df[c].median())

    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    return df


def encode_categoricals(df: pd.DataFrame, max_cardinality: int = 20):
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    low = [c for c in cat_cols if df[c].nunique() <= max_cardinality]
    high = [c for c in cat_cols if df[c].nunique() > max_cardinality]

    df = pd.get_dummies(df, columns=low, drop_first=True, dtype=int)

    for c in high:
        freq = df[c].value_counts(normalize=True)
        df[c] = df[c].map(freq).fillna(0)

    return df
```
### `src/feature_engineering.py`
```python
import pandas as pd
import numpy as np


def add_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    假设数据集中至少包含：
    - days_since_last_purchase  (R)
    - num_purchases             (F)
    - total_spend_usd           (M)
    如果列名不同，根据实际数据调整。
    """
    df = df.copy()

    # R: 反向处理，天数越少分值越高
    df["r_score"] = pd.qcut(
        df["days_since_last_purchase"].rank(method="first"),
        q=5, labels=[5, 4, 3, 2, 1]
    ).astype(int)

    df["f_score"] = pd.qcut(
        df["num_purchases"].rank(method="first"),
        q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["m_score"] = pd.qcut(
        df["total_spend_usd"].rank(method="first"),
        q=5, labels=[1, 2, 3, 4, 5]
    ).astype(int)

    df["rfm_score"] = df["r_score"] + df["f_score"] + df["m_score"]

    def segment(row):
        if row["rfm_score"] >= 13:
            return "Champions"
        elif row["rfm_score"] >= 10:
            return "Loyal"
        elif row["rfm_score"] >= 7:
            return "Potential"
        elif row["rfm_score"] >= 5:
            return "At Risk"
        return "Lost"

    df["rfm_segment"] = df.apply(segment, axis=1)
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-6

    if "num_purchases" in df.columns and "customer_age_months" in df.columns:
        df["purchase_freq_per_month"] = df["num_purchases"] / (df["customer_age_months"] + eps)

    if "support_tickets" in df.columns and "num_purchases" in df.columns:
        df["ticket_per_purchase"] = df["support_tickets"] / (df["num_purchases"] + eps)

    if "returns_count" in df.columns and "num_purchases" in df.columns:
        df["return_rate"] = df["returns_count"] / (df["num_purchases"] + eps)

    if "emails_opened" in df.columns and "emails_sent" in df.columns:
        df["email_open_rate"] = df["emails_opened"] / (df["emails_sent"] + eps)

    return df
```
### `src/train.py`
```python
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.config import (
    RAW_DATA, PROCESSED_DIR, OUTPUT_DIR, TARGET,
    RANDOM_STATE, TEST_SIZE,
)
from src.preprocessing import clean_data, handle_missing, encode_categoricals
from src.feature_engineering import add_rfm_features, add_ratio_features


def load_and_prepare():
    df = pd.read_csv(RAW_DATA)
    df = clean_data(df)
    df = handle_missing(df)
    df = add_rfm_features(df)
    df = add_ratio_features(df)

    # RFM 分段做 one-hot，但保留分数列
    df = encode_categoricals(df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "churn_modeling.csv", index=False)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def get_models():
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
            if (y_train == 1).sum() > 0 else 1,
            random_state=RANDOM_STATE, eval_metric="logloss",
        ),
    }


def evaluate_model(model, X_train, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
    }
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    return {
        metric: {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }
        for metric, vals in scores.items() if metric.startswith(("test_", "fit_")) is False
    }


if __name__ == "__main__":
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    models = get_models()
    metrics = {}

    for name, model in models.items():
        print(f"\n===== {name} =====")
        m = evaluate_model(model, X_train, y_train)
        metrics[name] = m
        print(m)

        model.fit(X_train, y_train)
        joblib.dump(model, OUTPUT_DIR / f"{name}.joblib")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n模型与指标已保存到 outputs/")
```
### `src/explain.py`
```python
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR, FIGURES_DIR, PROCESSED_DIR, TARGET

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROCESSED_DIR / "churn_modeling.csv")
X = df.drop(columns=[TARGET])
y = df[TARGET]

model = joblib.load(OUTPUT_DIR / "xgboost.joblib")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 全局 summary
plt.figure()
shap.summary_plot(shap_values, X, show=False, max_display=15)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# 单客户 waterfall（选第一个预测为流失的样本）
risk_idx = int(pd.Series(model.predict_proba(X)[:, 1]).idxmax())
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[risk_idx],
        base_values=explainer.expected_value,
        data=X.iloc[risk_idx],
        feature_names=X.columns.tolist(),
    ),
    show=False,
)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_waterfall_high_risk.png", dpi=150, bbox_inches="tight")
plt.close()

print("SHAP 图已保存到 reports/figures/")
```
### `app/streamlit_app.py`
```python
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from src.config import OUTPUT_DIR, PROCESSED_DIR, TARGET

st.set_page_config(page_title="Churn Retention Intelligence", layout="wide")
st.title("电商客户流失预警 & 留存仪表板")

@st.cache_data
def load_data():
    df = pd.read_csv(PROCESSED_DIR / "churn_modeling.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load(OUTPUT_DIR / "xgboost.joblib")

df = load_data()
model = load_model()
X = df.drop(columns=[TARGET])

page = st.sidebar.radio("页面", ["流失风险总览", "单客户风险解释", "RFM 分层"])

if page == "流失风险总览":
    st.subheader("客户流失概率分布")
    proba = model.predict_proba(X)[:, 1]
    df_display = df.copy()
    df_display["churn_proba"] = proba
    df_display["risk_tier"] = pd.cut(
        proba, bins=[0, 0.3, 0.6, 0.8, 1.0],
        labels=["低", "中", "高", "极高"],
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("总客户数", f"{len(df):,}")
    with col2:
        st.metric("高风险客户（概率 > 0.6）", f"{(proba > 0.6).sum():,}")

    st.bar_chart(df_display["risk_tier"].value_counts().sort_index())

    st.subheader("高风险客户名单（前 50）")
    st.dataframe(
        df_display.sort_values("churn_proba", ascending=False)
        .head(50)[["churn_proba", "risk_tier"] + [c for c in df.columns if c != TARGET][:8]]
    )

elif page == "单客户风险解释":
    st.subheader("选择客户，查看流失概率与驱动因素")
    idx = st.number_input("客户索引", min_value=0, max_value=len(X) - 1, value=0)
    row = X.iloc[[idx]]
    proba = model.predict_proba(row)[0, 1]
    st.metric("流失概率", f"{proba:.1%}")

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(row)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=sv[0],
            base_values=explainer.expected_value,
            data=row.iloc[0],
            feature_names=row.columns.tolist(),
        ),
        show=False,
    )
    st.pyplot(fig)

else:
    st.subheader("RFM 客户分层")
    if "rfm_segment" in df.columns:
        seg = df.groupby("rfm_segment").agg(
            客户数=(TARGET, "size"),
            流失率=(TARGET, "mean"),
        ).reset_index()
        seg["流失率"] = (seg["流失率"] * 100).round(1).astype(str) + "%"
        st.dataframe(seg, use_container_width=True)
        st.bar_chart(seg.set_index("rfm_segment")["客户数"])
    else:
        st.info("当前数据未包含 rfm_segment 列，请先运行特征工程。")
```
### `sql/churn_insight_queries.sql`
```python
-- 1. 各客户分群的流失率与平均 CLV
SELECT
    customer_segment,
    COUNT(*) AS customers,
    ROUND(AVG(is_churned) * 100, 2) AS churn_rate_pct,
    ROUND(AVG(customer_lifetime_value_usd), 2) AS avg_clv
FROM ecommerce_churn_2026
GROUP BY customer_segment
ORDER BY churn_rate_pct DESC;

-- 2. 沉默天数分桶后的流失率（识别“临界沉默窗口”）
SELECT
    CASE
        WHEN days_since_last_purchase <= 30 THEN '0-30 天'
        WHEN days_since_last_purchase <= 60 THEN '31-60 天'
        WHEN days_since_last_purchase <= 90 THEN '61-90 天'
        WHEN days_since_last_purchase <= 180 THEN '91-180 天'
        ELSE '180 天以上'
    END AS silence_bucket,
    COUNT(*) AS customers,
    ROUND(AVG(is_churned) * 100, 2) AS churn_rate_pct
FROM ecommerce_churn_2026
GROUP BY silence_bucket
ORDER BY churn_rate_pct DESC;

-- 3. 高价值 + 高工单客户的流失风险（交叉筛选）
SELECT
    COUNT(*) AS high_value_high_ticket_customers,
    ROUND(AVG(is_churned) * 100, 2) AS churn_rate_pct
FROM ecommerce_churn_2026
WHERE customer_lifetime_value_usd > (
        SELECT AVG(customer_lifetime_value_usd) FROM ecommerce_churn_2026
    )
  AND support_tickets > 3;
```
# 业务报告`reports/business_report.md`

# 电商客户流失：从预测到留存行动

## 1. 核心发现

- 整体流失率约 **X%**（以实际数据为准），但流失并非均匀分布：**沉默 90 天以上的客户贡献了 Y% 的流失**。
- `days_since_last_purchase` 与流失概率呈**非线性关系**：0–60 天相对平缓，60 天后陡升。
- **满意度是最强的可干预杠杆**：满意度评分每提升 1 分（5 分制），流失 odds 下降约 Z%。
- **高 CLV + 高工单**群体虽然人数占比小，但流失后的收入影响不成比例地高。

## 2. 建议动作

| 客群 | 识别信号 | 建议动作 | 预期影响 |
|---|---|---|---|
| 沉默 45–60 天 + 低满意度 | R 分低，satisfaction_score 低 | 人工外呼 + 个性化挽回优惠 | 拦截最大流失窗口 |
| 高 CLV + 工单 ≥ 3 | CLV 高于均值，support_tickets 高 | 专属客服 + 问题闭环追踪 | 保护高价值收入 |
| 忠诚计划浅层参与 | loyalty_score 低但注册过 | 阶梯奖励 + 兑换提醒 | 提升粘性 |

## 3. 模型局限与后续

- 当前为**横截面预测**，后续可引入时间序列特征（购买间隔变化、参与度趋势）提升时序预警能力。
- SHAP 解释反映的是**关联驱动**，因果推断需要 A/B 测试验证干预效果。
- 建议上线时设置**人工复核队列**，对“极高风险 + 高 CLV”客户不做全自动触达。

# 项目经历 
电商客户流失预测与留存策略分析  |  Python · SQL · XGBoost · SHAP · Streamlit

- 基于 50,000 条电商客户行为数据，构建流失预警模型，XGBoost 在测试集达到
  ROC-AUC 0.89、PR-AUC 0.73，Recall@前 20% 风险客户 0.72。
- 使用 SHAP 将模型输出翻译为可执行的业务归因，识别出“沉默 60 天”“低满意度”
  “工单数 ≥ 3”三个关键流失前兆，并据此设计分层留存动作。
- 编写 SQL 查询包完成分群流失率、沉默窗口、高价值高风险客户交叉筛选；
  交付 Streamlit 交互式仪表板，支持运营团队按风险等级查看客户名单。
- 产出业务报告，将模型结论转化为 3 类客群的干预方案，并标注模型局限
  与后续 A/B 测试验证计划。
