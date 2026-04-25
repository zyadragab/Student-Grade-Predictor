import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
 
# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Grade Predictor",
    page_icon="🎓",
    layout="wide",
)
 
# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #555;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .grade-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .winner-tag {
        background: #d4edda;
        color: #155724;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    div[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
MODEL_PATH = "models/"
 
def grade_label(g):
    if g >= 17: return ("Excellent", "#155724", "#d4edda")
    if g >= 14: return ("Good",      "#004085", "#cce5ff")
    if g >= 10: return ("Pass",      "#856404", "#fff3cd")
    return              ("Fail",      "#721c24", "#f8d7da")
 
@st.cache_resource
def load_or_train_models(df):
    os.makedirs(MODEL_PATH, exist_ok=True)
 
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
 
    X = df.drop(columns=['G3'])
    y = df['G3']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
 
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
 
    # ── Linear Regression ──
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
 
    # ── Random Forest ──
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
 
    # ── SVR ──
    svr = SVR(kernel='rbf', C=10, epsilon=0.5)
    svr.fit(X_train_sc, y_train)
    svr_pred = svr.predict(X_test_sc)
 
    # ── XGBoost ──
    xgb = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
 
    # Save models & scaler
    joblib.dump(lr,     MODEL_PATH + "lr.pkl")
    joblib.dump(rf,     MODEL_PATH + "rf.pkl")
    joblib.dump(svr,    MODEL_PATH + "svr.pkl")
    joblib.dump(xgb,    MODEL_PATH + "xgb.pkl")
    joblib.dump(scaler, MODEL_PATH + "scaler.pkl")
 
    metrics = {
        'Linear Regression': {
            'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
            'MAE':  mean_absolute_error(y_test, lr_pred),
            'R²':   r2_score(y_test, lr_pred),
            'pred': lr_pred, 'color': '#4C72B0'
        },
        'Random Forest': {
            'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'MAE':  mean_absolute_error(y_test, rf_pred),
            'R²':   r2_score(y_test, rf_pred),
            'pred': rf_pred, 'color': '#DD8452'
        },
        'SVR': {
            'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
            'MAE':  mean_absolute_error(y_test, svr_pred),
            'R²':   r2_score(y_test, svr_pred),
            'pred': svr_pred, 'color': '#55A868'
        },
        'XGBoost': {
            'RMSE': np.sqrt(mean_squared_error(y_test, xgb_pred)),
            'MAE':  mean_absolute_error(y_test, xgb_pred),
            'R²':   r2_score(y_test, xgb_pred),
            'pred': xgb_pred, 'color': '#C44E52'
        },
    }
 
    feat_imp = {
        'Random Forest': pd.Series(rf.feature_importances_,  index=X.columns),
        'XGBoost':       pd.Series(xgb.feature_importances_, index=X.columns),
    }
 
    return (lr, rf, svr, xgb, scaler, X, y_test, metrics, feat_imp)
 
 
# ─────────────────────────────────────────────
# Sidebar — upload or use sample
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
    st.title("🎓 Grade Predictor")
    st.markdown("---")
 
    uploaded = st.file_uploader("Upload student_data.csv", type="csv")
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_raw)} records")
    else:
        st.info("Upload your CSV or use the sample data below.")
        # Generate a small synthetic sample so the app works without a file
        np.random.seed(42)
        n = 300
        sample = pd.DataFrame({
            'school':     np.random.choice(['GP','MS'], n),
            'sex':        np.random.choice(['F','M'], n),
            'age':        np.random.randint(15, 22, n),
            'address':    np.random.choice(['U','R'], n),
            'famsize':    np.random.choice(['LE3','GT3'], n),
            'Pstatus':    np.random.choice(['T','A'], n),
            'Medu':       np.random.randint(0, 5, n),
            'Fedu':       np.random.randint(0, 5, n),
            'Mjob':       np.random.choice(['teacher','health','services','at_home','other'], n),
            'Fjob':       np.random.choice(['teacher','health','services','at_home','other'], n),
            'reason':     np.random.choice(['home','reputation','course','other'], n),
            'guardian':   np.random.choice(['mother','father','other'], n),
            'traveltime': np.random.randint(1, 5, n),
            'studytime':  np.random.randint(1, 5, n),
            'failures':   np.random.choice([0,0,0,1,2,3], n),
            'schoolsup':  np.random.choice(['yes','no'], n),
            'famsup':     np.random.choice(['yes','no'], n),
            'paid':       np.random.choice(['yes','no'], n),
            'activities': np.random.choice(['yes','no'], n),
            'nursery':    np.random.choice(['yes','no'], n),
            'higher':     np.random.choice(['yes','no'], n),
            'internet':   np.random.choice(['yes','no'], n),
            'romantic':   np.random.choice(['yes','no'], n),
            'famrel':     np.random.randint(1, 6, n),
            'freetime':   np.random.randint(1, 6, n),
            'goout':      np.random.randint(1, 6, n),
            'Dalc':       np.random.randint(1, 6, n),
            'Walc':       np.random.randint(1, 6, n),
            'health':     np.random.randint(1, 6, n),
            'absences':   np.random.randint(0, 40, n),
        })
        g1 = np.clip(np.round(
            10 + sample['studytime']*0.8 - sample['failures']*2
            + np.random.normal(0, 1.5, n)), 0, 20).astype(int)
        g2 = np.clip(g1 + np.random.randint(-1, 2, n), 0, 20)
        g3 = np.clip(g2 + np.random.randint(-1, 2, n), 0, 20)
        sample['G1'] = g1
        sample['G2'] = g2
        sample['G3'] = g3
        df_raw = sample
        st.caption("Using synthetic sample data (300 students)")
 
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 Model Comparison", "🔮 Predict Grade", "📈 Feature Importance"],
    )
 
# ─────────────────────────────────────────────
# Train / load
# ─────────────────────────────────────────────
with st.spinner("Training models... this takes a few seconds"):
    lr, rf, svr, xgb, scaler, X, y_test, metrics, feat_imp = load_or_train_models(df_raw.copy())
 
# ═══════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🎓 Student Grade Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">4 ML models trained to predict final grade G3 — Compare, analyze, and test.</p>', unsafe_allow_html=True)
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Dataset size", f"{len(df_raw)} students")
    with col2:
        st.metric("Features", f"{X.shape[1]}")
    with col3:
        st.metric("Test set", f"{len(y_test)} students")
    with col4:
        best_model = min(metrics, key=lambda m: metrics[m]['RMSE'])
        st.metric("Best model (RMSE)", best_model)
 
    st.markdown("---")
 
    st.subheader("Models trained")
    desc = {
        'Linear Regression': "Baseline model. Fast & interpretable. Works well when features are linearly related to G3.",
        'Random Forest':     "Ensemble of 200 decision trees. Handles non-linearity and is robust to outliers.",
        'SVR':               "Support Vector Regressor with RBF kernel. Good at capturing complex patterns in scaled data.",
        'XGBoost':           "Gradient boosting — focuses on correcting residual errors. Usually the strongest performer.",
    }
    colors = {'Linear Regression':'#4C72B0','Random Forest':'#DD8452','SVR':'#55A868','XGBoost':'#C44E52'}
    c1, c2 = st.columns(2)
    for i, (name, text) in enumerate(desc.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            rmse = metrics[name]['RMSE']
            r2   = metrics[name]['R²']
            is_best = name == best_model
            with st.container(border=True):
                if is_best:
                    st.success(f"⭐ {name} — Best Model")
                else:
                    st.markdown(f"**{name}**")
                st.caption(text)
                st.markdown(f"RMSE: `{rmse:.3f}` &nbsp;|&nbsp; R²: `{r2:.3f}`")
 
    st.markdown("---")
    st.subheader("Target distribution — G3")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].hist(df_raw['G3'], bins=20, color='#4C72B0', edgecolor='white', linewidth=0.8)
    axes[0].axvline(df_raw['G3'].mean(),   color='red',    linestyle='--', label=f"Mean: {df_raw['G3'].mean():.1f}")
    axes[0].axvline(df_raw['G3'].median(), color='orange', linestyle='--', label=f"Median: {df_raw['G3'].median():.1f}")
    axes[0].set_title('G3 Distribution', fontweight='bold')
    axes[0].set_xlabel('Final Grade (G3)')
    axes[0].legend()
    axes[0].spines[['top','right']].set_visible(False)
 
    axes[1].boxplot(df_raw['G3'], patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.6),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('G3 Boxplot', fontweight='bold')
    axes[1].set_ylabel('Final Grade (G3)')
    axes[1].spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ═══════════════════════════════════════════════
# PAGE: Model Comparison
# ═══════════════════════════════════════════════
elif page == "📊 Model Comparison":
    st.markdown('<p class="main-header">📊 Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Side-by-side performance metrics for all 4 models.</p>', unsafe_allow_html=True)
 
    # Metrics table
    comp_df = pd.DataFrame({
        name: {'RMSE': f"{v['RMSE']:.3f}", 'MAE': f"{v['MAE']:.3f}", 'R²': f"{v['R²']:.3f}"}
        for name, v in metrics.items()
    }).T
    st.dataframe(comp_df.style.highlight_min(subset=['RMSE','MAE'], color='#d4edda')
                              .highlight_max(subset=['R²'],          color='#d4edda'),
                 use_container_width=True)
 
    st.markdown("---")
 
    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names  = list(metrics.keys())
    colors = [metrics[m]['color'] for m in names]
 
    for ax, metric in zip(axes, ['RMSE', 'MAE', 'R²']):
        vals = [metrics[m][metric] for m in names]
        bars = ax.bar(names, vals, color=colors, edgecolor='white', linewidth=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.02,
                    f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(vals) * 1.2)
        ax.tick_params(axis='x', rotation=15)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle('Model Comparison — All 4 Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
    st.markdown("---")
    st.subheader("Actual vs Predicted — per model")
 
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (name, v) in zip(axes, metrics.items()):
        ax.scatter(y_test, v['pred'], alpha=0.5, color=v['color'], edgecolors='white', s=35)
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5)
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.set_xlabel('Actual G3')
        ax.set_ylabel('Predicted G3')
        ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
 
# ═══════════════════════════════════════════════
# PAGE: Predict Grade
# ═══════════════════════════════════════════════
elif page == "🔮 Predict Grade":
    st.markdown('<p class="main-header">🔮 Predict a Student\'s Grade</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Fill in the student details and get predictions from all 4 models instantly.</p>', unsafe_allow_html=True)
 
    with st.form("predict_form"):
        st.subheader("📚 Grades history")
        c1, c2 = st.columns(2)
        g1 = c1.slider("G1 — First period grade",  0, 20, 12)
        g2 = c2.slider("G2 — Second period grade", 0, 20, 12)
 
        st.subheader("👤 Personal factors")
        c1, c2, c3 = st.columns(3)
        studytime = c1.slider("Study time (1–4)", 1, 4, 2,
                              help="1 = <2h/week  |  2 = 2–5h  |  3 = 5–10h  |  4 = >10h")
        failures  = c2.slider("Past failures",    0, 4, 0)
        absences  = c3.slider("Absences",         0, 93, 4)
        age       = c1.slider("Age",              15, 22, 17)
        health    = c2.slider("Health (1–5)",     1, 5, 3)
        traveltime= c3.slider("Travel time (1–4)", 1, 4, 1)
 
        st.subheader("🏠 Family & social")
        c1, c2, c3 = st.columns(3)
        medu    = c1.slider("Mother education (0–4)", 0, 4, 2)
        fedu    = c2.slider("Father education (0–4)", 0, 4, 2)
        famrel  = c3.slider("Family relationship (1–5)", 1, 5, 4)
        freetime= c1.slider("Free time (1–5)", 1, 5, 3)
        goout   = c2.slider("Go out (1–5)",   1, 5, 3)
        dalc    = c3.slider("Weekday alcohol (1–5)", 1, 5, 1)
        walc    = c1.slider("Weekend alcohol (1–5)", 1, 5, 1)
 
        st.subheader("✅ Yes / No features")
        c1, c2, c3, c4 = st.columns(4)
        sex        = c1.selectbox("Sex",               ["Female (0)", "Male (1)"])
        address    = c2.selectbox("Address",            ["Rural (0)", "Urban (1)"])
        internet   = c3.selectbox("Internet access",   ["No (0)", "Yes (1)"])
        higher     = c4.selectbox("Wants higher edu",  ["No (0)", "Yes (1)"])
        schoolsup  = c1.selectbox("School support",    ["No (0)", "Yes (1)"])
        famsup     = c2.selectbox("Family support",    ["No (0)", "Yes (1)"])
        paid       = c3.selectbox("Paid classes",      ["No (0)", "Yes (1)"])
        activities = c4.selectbox("Activities",        ["No (0)", "Yes (1)"])
        nursery    = c1.selectbox("Nursery",           ["No (0)", "Yes (1)"])
        romantic   = c2.selectbox("Romantic relation", ["No (0)", "Yes (1)"])
 
        submitted = st.form_submit_button("🔮 Predict now", use_container_width=True)
 
    if submitted:
        def bin_val(s): return int(s.split("(")[1].replace(")", ""))
 
        input_dict = {
            'school': 0, 'sex': bin_val(sex), 'age': age,
            'address': bin_val(address), 'famsize': 1, 'Pstatus': 1,
            'Medu': medu, 'Fedu': fedu,
            'Mjob': 2, 'Fjob': 2, 'reason': 0, 'guardian': 0,
            'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
            'schoolsup': bin_val(schoolsup), 'famsup': bin_val(famsup),
            'paid': bin_val(paid), 'activities': bin_val(activities),
            'nursery': bin_val(nursery), 'higher': bin_val(higher),
            'internet': bin_val(internet), 'romantic': bin_val(romantic),
            'famrel': famrel, 'freetime': freetime, 'goout': goout,
            'Dalc': dalc, 'Walc': walc, 'health': health, 'absences': absences,
            'G1': g1, 'G2': g2,
        }
 
        input_df = pd.DataFrame([input_dict])[X.columns]
        input_sc = scaler.transform(input_df)
 
        preds = {
            'Linear Regression': float(np.clip(lr.predict(input_sc)[0], 0, 20)),
            'Random Forest':     float(np.clip(rf.predict(input_df)[0], 0, 20)),
            'SVR':               float(np.clip(svr.predict(input_sc)[0], 0, 20)),
            'XGBoost':           float(np.clip(xgb.predict(input_df)[0], 0, 20)),
        }
 
        colors_map = {'Linear Regression':'#4C72B0','Random Forest':'#DD8452','SVR':'#55A868','XGBoost':'#C44E52'}
 
        st.markdown("---")
        st.subheader("Predictions")
        cols = st.columns(4)
        for col, (name, pred) in zip(cols, preds.items()):
            label, tc, bc = grade_label(pred)
            with col:
                with st.container(border=True):
                    st.caption(name)
                    st.metric(label="Grade", value=f"{pred:.1f} / 20")
                    if label == "Excellent":
                        st.success(label)
                    elif label == "Good":
                        st.info(label)
                    elif label == "Pass":
                        st.warning(label)
                    else:
                        st.error(label)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        # Prediction bar chart
        fig, ax = plt.subplots(figsize=(8, 3))
        names_p = list(preds.keys())
        vals_p  = list(preds.values())
        clrs_p  = [colors_map[m] for m in names_p]
        bars = ax.barh(names_p, vals_p, color=clrs_p, edgecolor='white', height=0.5)
        ax.axvline(10, color='gray', linestyle='--', lw=1, alpha=0.5, label='Pass threshold (10)')
        for bar, v in zip(bars, vals_p):
            ax.text(v + 0.15, bar.get_y() + bar.get_height()/2,
                    f'{v:.1f}', va='center', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 22)
        ax.set_xlabel('Predicted G3')
        ax.set_title('Model Predictions Comparison', fontweight='bold')
        ax.legend()
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
        best_pred = max(preds, key=preds.get)
        avg_pred  = np.mean(list(preds.values()))
        st.info(f"📊 **Average prediction across all models: {avg_pred:.1f} / 20** — Highest: {best_pred} ({preds[best_pred]:.1f})")
 
# ═══════════════════════════════════════════════
# PAGE: Feature Importance
# ═══════════════════════════════════════════════
elif page == "📈 Feature Importance":
    st.markdown('<p class="main-header">📈 Feature Importance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Which features matter most for predicting G3?</p>', unsafe_allow_html=True)
 
    tab1, tab2 = st.tabs(["🌲 Random Forest", "⚡ XGBoost"])
 
    for tab, model_name in zip([tab1, tab2], ['Random Forest', 'XGBoost']):
        with tab:
            imp = feat_imp[model_name].sort_values(ascending=False)
            color = '#DD8452' if model_name == 'Random Forest' else '#C44E52'
 
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
            # Top 10 bar
            top10 = imp.head(10)[::-1]
            axes[0].barh(top10.index, top10.values, color=color, edgecolor='white')
            axes[0].set_title(f'{model_name} — Top 10 Features', fontweight='bold')
            axes[0].set_xlabel('Importance')
            axes[0].spines[['top','right']].set_visible(False)
 
            # Full feature heatmap-style bar
            all_imp = imp.reset_index()
            all_imp.columns = ['Feature','Importance']
            cmap = plt.cm.get_cmap('RdYlGn')
            norm_vals = (all_imp['Importance'] - all_imp['Importance'].min()) / \
                        (all_imp['Importance'].max() - all_imp['Importance'].min() + 1e-9)
            bar_colors = [cmap(v) for v in norm_vals[::-1]]
            axes[1].barh(all_imp['Feature'][::-1], all_imp['Importance'][::-1],
                         color=bar_colors, edgecolor='white', height=0.7)
            axes[1].set_title('All Features', fontweight='bold')
            axes[1].set_xlabel('Importance')
            axes[1].spines[['top','right']].set_visible(False)
            axes[1].tick_params(axis='y', labelsize=8)
 
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
            st.caption("Top 3 features: " + ", ".join(imp.head(3).index.tolist()))
 
    st.markdown("---")
    st.subheader("💡 Key insight")
    st.info("""
    **G1 and G2** (previous period grades) are by far the strongest predictors of G3 — 
    which makes sense since students who perform well early tend to maintain their performance.
 
    After grades, **study time** and **past failures** are the most impactful behavioral features.
    """)