import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

@st.cache_resource
def load_bundle():
    return joblib.load("bundle.joblib")

bundle = load_bundle()
COLUMNS = bundle["columns"]

st.title("🎓 Student Performance Predictor")
st.caption("Predicts the final grade G3 (Lasso regression) and pass/fail, G3 ≥ 10 (linear SVM).")

with st.form("student"):
    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Grades")
        g1 = st.slider("G1 (first period grade)", 0, 20, 10)
        g2 = st.slider("G2 (second period grade)", 0, 20, 10)
        absences = st.slider("Absences", 0, 75, 4)
        failures = st.selectbox("Past class failures", [0, 1, 2, 3])
        studytime = st.selectbox("Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)", [1, 2, 3, 4], index=1)
        higher = st.selectbox("Wants higher education", ["yes", "no"])

    with c2:
        st.subheader("Student")
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home address", ["U (urban)", "R (rural)"])[0]
        famsize = st.selectbox("Family size", ["GT3 (more than 3)", "LE3 (3 or less)"])[:3]
        Pstatus = st.selectbox("Parents living", ["T (together)", "A (apart)"])[0]
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.selectbox("Travel time (1: <15m, 2: 15-30m, 3: 30-60m, 4: >1h)", [1, 2, 3, 4])
        health = st.slider("Health (1 very bad - 5 very good)", 1, 5, 3)

    with c3:
        st.subheader("Family and social")
        Medu = st.selectbox("Mother's education (0-4)", [0, 1, 2, 3, 4], index=2)
        Fedu = st.selectbox("Father's education (0-4)", [0, 1, 2, 3, 4], index=2)
        Mjob = st.selectbox("Mother's job", ["at_home", "health", "other", "services", "teacher"], index=2)
        Fjob = st.selectbox("Father's job", ["at_home", "health", "other", "services", "teacher"], index=2)
        reason = st.selectbox("Reason for choosing school", ["course", "home", "reputation", "other"])
        famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
        freetime = st.slider("Free time (1-5)", 1, 5, 3)
        goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
        Dalc = st.slider("Workday alcohol use (1-5)", 1, 5, 1)
        Walc = st.slider("Weekend alcohol use (1-5)", 1, 5, 2)

    st.subheader("Support and activities")
    d1, d2, d3, d4, d5, d6 = st.columns(6)
    schoolsup = d1.selectbox("School support", ["no", "yes"])
    famsup = d2.selectbox("Family support", ["yes", "no"])
    paid = d3.selectbox("Paid classes", ["no", "yes"])
    activities = d4.selectbox("Activities", ["no", "yes"])
    nursery = d5.selectbox("Attended nursery", ["yes", "no"])
    internet = d6.selectbox("Internet at home", ["yes", "no"])
    romantic = st.selectbox("In a relationship", ["no", "yes"])

    submitted = st.form_submit_button("Predict", type="primary")


def yn(v):
    return 1 if v == "yes" else 0


def build_row():
    row = {
        "age": age, "Medu": Medu, "Fedu": Fedu, "traveltime": traveltime,
        "studytime": studytime, "failures": failures, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "G1": g1, "G2": g2,
        # same outlier rule as the notebook
        "absences": min(absences, bundle["absences_upper"]),
        # one-hot columns (dropped dummies: GP, F, R, GT3, A, at_home, course, other guardian)
        "school_MS": int(school == "MS"),
        "sex_M": int(sex == "M"),
        "address_U": int(address == "U"),
        "famsize_LE3": int(famsize == "LE3"),
        "Pstatus_T": int(Pstatus == "T"),
        "guardian_father": int(guardian == "father"),
        "guardian_mother": int(guardian == "mother"),
        "reason_home": int(reason == "home"),
        "reason_other": int(reason == "other"),
        "reason_reputation": int(reason == "reputation"),
        # label-encoded yes/no columns
        "en_schoolsup": yn(schoolsup), "en_famsup": yn(famsup), "en_paid": yn(paid),
        "en_activities": yn(activities), "en_nursery": yn(nursery),
        "en_higher": yn(higher), "en_internet": yn(internet), "en_romantic": yn(romantic),
    }
    for job, val in (("Mjob", Mjob), ("Fjob", Fjob)):
        for cat in ("health", "other", "services", "teacher"):
            row[f"{job}_{cat}"] = int(val == cat)
    return row


if submitted:
    row = build_row()
    missing = [c for c in COLUMNS if c not in row]
    extra = [c for c in row if c not in COLUMNS]
    if missing or extra:
        st.error(f"Feature mismatch between the app and the saved model. "
                 f"Missing in app: {missing}. Not in model: {extra}.")
        st.stop()

    X = pd.DataFrame([row])[COLUMNS]
    g3 = float(bundle["lasso"].predict(X)[0])
    g3 = max(0.0, min(20.0, g3))
    passed = int(bundle["svm"].predict(X)[0]) == 1
    margin = float(bundle["svm"].decision_function(X)[0])

    st.divider()
    a, b, c = st.columns(3)
    a.metric("Predicted final grade (G3)", f"{g3:.1f} / 20")
    b.metric("Pass / Fail (G3 ≥ 10)", "Pass ✅" if passed else "Fail ❌")
    c.metric("SVM margin", f"{margin:+.2f}", help="Distance from the decision boundary. Closer to 0 means less certain.")
    if (g3 >= 10) != passed:
        st.warning("The two models disagree near the pass mark (10). Treat this student as borderline.")
    if abs(margin) < 0.5:
        st.info("This is close to the decision boundary, so the pass/fail result is less certain.")

with st.expander("About the models"):
    m = bundle.get("metrics", {})
    st.write(f"- Regression: Lasso, cross-validated R² ≈ {m.get('lasso_cv_r2', '?')}%")
    st.write(f"- Classification: linear SVM, cross-validated accuracy ≈ {m.get('svm_cv_acc', '?')}%")
    st.write("These models rely heavily on G1 and G2, and predictions are estimates, not guarantees. "
             "Trained on the Portuguese-class math dataset (395 students).")
