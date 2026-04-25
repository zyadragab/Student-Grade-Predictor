# 🎓 Student Grade Predictor — Streamlit App

## Files
```
app.py              ← الـ app الرئيسي
requirements.txt    ← الـ libraries المطلوبة
student_data.csv    ← داتاك (حطها في نفس الفولدر)
```

---

## 🚀 تشغيل محلياً (على جهازك)

```bash
# 1. install المكتبات
pip install -r requirements.txt

# 2. شغّل الـ app
streamlit run app.py
```

سيفتح تلقائياً على: `http://localhost:8501`

---

## 🌐 Deploy على Streamlit Cloud (لينك مجاني دائم)

### الخطوات:

**1. ارفع الملفات على GitHub**
- اعمل repo جديد على github.com
- ارفع: `app.py` + `requirements.txt` + `student_data.csv`

**2. اتصل بـ Streamlit Cloud**
- روح على: https://share.streamlit.io
- سجّل دخول بحسابك على GitHub
- اضغط **"New app"**

**3. اختار الـ repo**
- Repository: اسم الـ repo بتاعك
- Branch: `main`
- Main file path: `app.py`
- اضغط **"Deploy!"**

✅ بعد دقيقتين هيديك لينك زي:
`https://your-name-grade-predictor.streamlit.app`

---

## 📄 صفحات الـ App

| الصفحة | المحتوى |
|--------|---------|
| 🏠 Overview | نظرة عامة على الداتا والموديلز |
| 📊 Model Comparison | مقارنة RMSE / MAE / R² للـ 4 موديلز |
| 🔮 Predict Grade | ادخل بيانات طالب واتوقع الدرجة |
| 📈 Feature Importance | أهم الـ features في Random Forest و XGBoost |

---

## ⚠️ ملاحظة على الـ Free Tier

الـ app بيتـ"sleep" لو مفيش حد بيستخدمه لفترة،
بس بيصحى تاني لما حد يفتحه (بياخد ~30 ثانية).
