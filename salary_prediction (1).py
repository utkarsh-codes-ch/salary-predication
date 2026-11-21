# Employee Salary Prediction - Streamlit app
# Name: Utkarsh Choudhary | Roll No: 2026m01410024 | utkarsh.26m01410024@abes.ac.in
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

GENDER = {'Female': 0, 'Male': 1}
EDUCATION = {'Diploma': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
DEPARTMENT = {'HR': 0, 'IT': 1, 'Marketing': 2, 'Operations': 3, 'Sales': 4}
JOB_LEVEL = {'Junior': 0, 'Mid': 1, 'Senior': 2, 'Lead': 3, 'Manager': 4}
REMOTE = {'No': 0, 'Yes': 1}
CITY = {'Chennai': 0, 'Delhi': 1, 'Hyderabad': 2, 'Mumbai': 3}

FEATURES = ['Age', 'Gender', 'Education', 'Experience_Years', 'Department', 'Job_Level',
            'Performance_Rating', 'Certifications', 'Overtime_Hours', 'Remote_Work', 'City',
            'Company_Tenure', 'Projects_Completed', 'Skill_Score']


@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("salary_model.json")
    return model


def main():
    st.markdown("<h1>Employee Salary Prediction</h1>", unsafe_allow_html=True)
    st.markdown("This app predicts an employee's **annual salary (in LPA)** from their details.")

    model = load_model()

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", 18, 65, 30, step=1)
        gender = st.selectbox("Gender", list(GENDER.keys()))
        education = st.selectbox("Education", list(EDUCATION.keys()), index=1)
        experience = st.number_input("Experience (years)", 0, 47, 5, step=1)
        department = st.selectbox("Department", list(DEPARTMENT.keys()))
        job_level = st.selectbox("Job level", list(JOB_LEVEL.keys()))
        rating = st.slider("Performance rating", 1, 5, 3)
    with col2:
        certs = st.slider("Certifications", 0, 10, 3)
        overtime = st.slider("Overtime hours", 0, 60, 20)
        remote = st.selectbox("Remote work", list(REMOTE.keys()))
        city = st.selectbox("City", list(CITY.keys()))
        tenure = st.number_input("Years in this company", 0, 15, 2, step=1)
        projects = st.slider("Projects completed", 1, 30, 10)
        skill = st.slider("Skill score", 50, 100, 75)

    data_new = pd.DataFrame({
        'Age': age,
        'Gender': GENDER[gender],
        'Education': EDUCATION[education],
        'Experience_Years': experience,
        'Department': DEPARTMENT[department],
        'Job_Level': JOB_LEVEL[job_level],
        'Performance_Rating': rating,
        'Certifications': certs,
        'Overtime_Hours': overtime,
        'Remote_Work': REMOTE[remote],
        'City': CITY[city],
        'Company_Tenure': tenure,
        'Projects_Completed': projects,
        'Skill_Score': skill,
    }, index=[0])[FEATURES]

    if st.button("Predict"):
        if experience > age - 18:
            st.warning("Experience cannot be more than (age - 18). Please correct the values.")
        elif tenure > experience:
            st.warning("Years in this company cannot be more than total experience.")
        else:
            pred = model.predict(data_new)[0]
            st.success("Predicted annual salary: {:.2f} LPA".format(pred))

    st.caption("Developed by Utkarsh Choudhary (2026m01410024), MCA, ABES College")


if __name__ == '__main__':
    main()
