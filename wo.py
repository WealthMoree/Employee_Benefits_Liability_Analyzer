import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import plotly.express as px
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ------- Page config & style -------
st.set_page_config(page_title="Actuarial Benefits Valuation Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.header {font-size:28px; font-weight:600;}
.card {background-color: #f8f9fa; padding: 12px; border-radius: 10px}
.small {font-size:12px; color: #6c757d}
</style>
""", unsafe_allow_html=True)

# ------- Helper functions -------
def generate_sample_data(n=200, seed=42):
    np.random.seed(seed)
    ages = np.random.randint(22, 62, n)
    service = np.clip(np.random.randint(0, 35, n), 0, ages-18)
    salary = np.round(np.random.normal(800000, 300000, n)).astype(int)
    salary = np.clip(salary, 120000, None)
    dept = np.random.choice(["Sales","HR","Finance","Ops","IT","R&D"], n)
    emp_id = [f"E{1000+i}" for i in range(n)]
    df = pd.DataFrame({
        'EmployeeID': emp_id,
        'Age': ages,
        'ServiceYears': service,
        'LastDrawnSalary': salary,
        'Department': dept,
        'Gender': np.random.choice(['M','F','O'], n, p=[0.6,0.38,0.02])
    })
    return df

def validate_data(df):
    issues = []
    required = ['EmployeeID','Age','ServiceYears','LastDrawnSalary']
    for col in required:
        if col not in df.columns:
            issues.append(("missing_col", f"Required column '{col}' is missing."))
    if issues:
        return issues
    for col in required:
        nulls = df[col].isnull().sum()
        if nulls>0:
            issues.append(("missing_values", f"Column '{col}' has {nulls} missing values."))
    bad_age = df[(df['Age']<18) | (df['Age']>100)]
    if not bad_age.empty:
        issues.append(("age_range", f"{len(bad_age)} rows with unrealistic ages (<18 or >100)."))
    bad_service = df[(df['ServiceYears']<0) | (df['ServiceYears']> (df['Age'] - 18).clip(lower=0))]
    if not bad_service.empty:
        issues.append(("service_issue", f"{len(bad_service)} rows with inconsistent service years."))
    bad_salary = df[df['LastDrawnSalary']<=0]
    if not bad_salary.empty:
        issues.append(("salary_issue", f"{len(bad_salary)} rows with non-positive salary."))
    return issues

def compute_liabilities(df, assumptions):
    r_age = assumptions['retirement_age']
    disc = assumptions['discount_rate']
    esc = assumptions['salary_escalation']
    accrual_rate = assumptions['accrual_rate']
    gratuity_rate = assumptions['gratuity_rate']
    rows = []
    today = date.today()
    for idx, row in df.iterrows():
        age = float(row['Age'])
        years_to_ret = max(0, r_age - age)
        proj_salary = row['LastDrawnSalary'] * ((1+esc) ** years_to_ret)
        expected_payable_years = assumptions['expected_payable_years']
        pension_nominal = accrual_rate * proj_salary * row['ServiceYears']
        gratuity_nominal = gratuity_rate * proj_salary * row['ServiceYears']
        pv_factor = (1 + disc) ** (-years_to_ret) if years_to_ret>0 else 1.0
        pension_pv = pension_nominal * pv_factor
        gratuity_pv = gratuity_nominal * pv_factor
        total_liability = pension_pv + gratuity_pv
        csc = accrual_rate * row['LastDrawnSalary'] * (1 + esc) ** 0
        interest_cost = disc * total_liability
        rows.append({
            'EmployeeID': row['EmployeeID'],
            'Age': age,
            'ServiceYears': row['ServiceYears'],
            'LastDrawnSalary': row['LastDrawnSalary'],
            'YearsToRetirement': years_to_ret,
            'ProjectedSalaryAtRetirement': proj_salary,
            'PensionPV': pension_pv,
            'GratuityPV': gratuity_pv,
            'TotalLiability': total_liability,
            'CurrentServiceCost': csc,
            'InterestCost': interest_cost
        })
    res = pd.DataFrame(rows)
    return res

def to_excel_bytes(df_dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        # FIX APPLIED: Removed writer.save() line here
    processed_data = output.getvalue()
    return processed_data

def create_pdf_report(summary_text, filename='actuarial_report.pdf'):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    textobj = c.beginText(40, height - 60)
    textobj.setFont('Helvetica', 11)
    for line in summary_text.split('\n'):
        textobj.textLine(line)
    c.drawText(textobj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ------- Sidebar (Inputs & Assumptions) -------
with st.sidebar:
    st.header("Assumptions & Controls")
    sample_btn = st.button("Download Sample Dataset")
    if sample_btn:
        sample_df = generate_sample_data(250)
        towrite = io.BytesIO()
        sample_df.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f"data:file/csv;base64,{b64}"
        st.markdown(f"[Download sample CSV]({href})")
    st.markdown("---")
    retirement_age = st.number_input("Retirement Age", min_value=55, max_value=75, value=60)
    discount_rate = st.slider("Discount rate (annual)", min_value=0.01, max_value=0.12, value=0.06, step=0.005, format="%.3f")
    salary_escalation = st.slider("Salary escalation (annual)", min_value=0.00, max_value=0.12, value=0.05, step=0.005, format="%.3f")
    accrual_rate = st.slider("Accrual rate (pension accrual per year)", min_value=0.01, max_value=0.20, value=0.015, step=0.001, format="%.3f")
    gratuity_rate = st.slider("Gratuity accrual factor (demo)", min_value=0.01, max_value=0.5, value=0.1875, step=0.005, format="%.3f")
    expected_payable_years = st.number_input("Expected payable years (annuity simplification)", min_value=1, max_value=30, value=12)
    st.markdown("---")
    st.subheader("Sensitivity")
    sens_enable = st.checkbox("Enable sensitivity analysis", value=True)
    sens_range = st.slider("Sensitivity range for discount rate (±%)", min_value=0.5, max_value=5.0, value=1.0, step=0.5)
    st.markdown("---")
    st.caption("Built for demo & learning. Replace formulas with certified actuarial models for production.")

assumptions = {
    'retirement_age': int(retirement_age),
    'discount_rate': float(discount_rate),
    'salary_escalation': float(salary_escalation),
    'accrual_rate': float(accrual_rate),
    'gratuity_rate': float(gratuity_rate),
    'expected_payable_years': int(expected_payable_years)
}

# ------- Main layout -------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="header">Actuarial Benefits Valuation Dashboard</div>', unsafe_allow_html=True)
    st.write("Upload employee dataset (CSV or Excel). Must contain: EmployeeID, Age, ServiceYears, LastDrawnSalary")
with col2:
    st.metric("Model date", date.today().isoformat())

uploaded = st.file_uploader("Upload employee data (CSV/XLSX)", type=['csv','xlsx'])
if uploaded is None:
    st.info("No file uploaded — using sample data. You can upload your own CSV/XLSX with required columns.")
    df = generate_sample_data(200)
else:
    try:
        if uploaded.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        df = generate_sample_data(200)

with st.expander("Preview employee data", expanded=False):
    st.dataframe(df.head(100))

issues = validate_data(df)
if issues:
    st.warning("Data validation flagged issues. Expand to see details.")
    with st.expander("Validation issues", expanded=False):
        for code, msg in issues:
            st.write(f"- **{code}**: {msg}")
else:
    st.success("Data passed basic validation checks.")

val_results = compute_liabilities(df, assumptions)
total_liability = val_results['TotalLiability'].sum()
avg_per_employee = val_results['TotalLiability'].mean()
total_csc = val_results['CurrentServiceCost'].sum()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Liability (₹)", f"{total_liability:,.0f}")
k2.metric("Avg Liability per Employee (₹)", f"{avg_per_employee:,.0f}")
k3.metric("Total Current Service Cost (₹)", f"{total_csc:,.0f}")
k4.metric("Employees", f"{len(val_results)}")
st.markdown("---")

viz_col1, viz_col2 = st.columns(2)
with viz_col1:
    st.subheader("Liability by Age Group")
    val_results['AgeGroup'] = pd.cut(val_results['Age'], bins=[18,30,40,50,60,80], labels=['18-30','31-40','41-50','51-60','60+'])
    grp = val_results.groupby('AgeGroup')['TotalLiability'].sum().reset_index()
    fig1 = px.bar(grp, x='AgeGroup', y='TotalLiability', title='Total Liability by Age Group', labels={'TotalLiability':'Liability (₹)'})
    st.plotly_chart(fig1, use_container_width=True)
with viz_col2:
    st.subheader("Top 10 Liability Employees")
    top10 = val_results.sort_values('TotalLiability', ascending=False).head(10)
    fig2 = px.bar(top10, x='EmployeeID', y='TotalLiability', title='Top 10 Employees by Liability')
    st.plotly_chart(fig2, use_container_width=True)
st.markdown("---")

if sens_enable:
    st.subheader("Sensitivity Analysis")
    base_disc = assumptions['discount_rate']
    del_pct = sens_range / 100.0
    disc_vals = np.linspace(base_disc*(1-del_pct), base_disc*(1+del_pct), 11)
    sens_rows = []
    for d in disc_vals:
        a2 = assumptions.copy()
        a2['discount_rate'] = float(d)
        res = compute_liabilities(df, a2)
        sens_rows.append({'DiscountRate': d, 'TotalLiability': res['TotalLiability'].sum()})
    sens_df = pd.DataFrame(sens_rows)
    fig3 = px.line(sens_df, x='DiscountRate', y='TotalLiability', title='Sensitivity of Total Liability to Discount Rate')
    st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(sens_df.style.format({'TotalLiability':'{:.0f}','DiscountRate':'{:.3f}'}))
st.markdown("---")

rec_col1, rec_col2 = st.columns([2,1])
with rec_col1:
    st.subheader("Governance & Funding Recommendations")
    st.write("**Funding options**:\n- Pay-As-You-Go: lower asset volatility but higher employer cashflow variability.\n- Pre-funding: invest in a fund to match liabilities; needs governance and trustee oversight.")
    st.write("**Recommended controls**:\n- Quarterly reconciliation of payroll & census data.\n- Documented assumption governance; board approval for key rates.\n- Regular experience studies for turnover and mortality.")
    st.write("**Next steps for client**:\n1. Agree assumptions.\n2. Run full projection with certified actuarial model.\n3. Prepare client report and trustee presentation.")
with rec_col2:
    st.subheader("Quick checklist")
    st.checkbox("Census data validated", value=True)
    st.checkbox("Assumptions reviewed by senior", value=False)
    st.checkbox("Funding strategy discussed", value=False)
    st.checkbox("Trustee reporting up to date", value=False)
st.markdown("---")

st.subheader("Export & Client Deliverables")
export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    st.write("Download detailed results (Excel)")
    excel_bytes = to_excel_bytes({'ValuationResults': val_results, 'Inputs': df})
    st.download_button("Download Excel", data=excel_bytes, file_name='valuation_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
with export_col2:
    st.write("Download simple PDF summary")
    summary_text = f"Actuarial Valuation Summary\nDate: {date.today().isoformat()}\nTotal liability: {total_liability:,.0f}\nEmployees: {len(val_results)}\nAssumptions:\n - Retirement age: {assumptions['retirement_age']}\n - Discount rate: {assumptions['discount_rate']:.3f}\n - Salary escalation: {assumptions['salary_escalation']:.3f}\n"
    pdf_bytes = create_pdf_report(summary_text)
    st.download_button("Download PDF summary", data=pdf_bytes, file_name='actuarial_summary.pdf', mime='application/pdf')
with export_col3:
    st.write("Download CSV of results")
    csv_bytes = val_results.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv_bytes, file_name='valuation_results.csv', mime='text/csv')
st.markdown("---")

with st.expander("Draft: Client report text (editable)"):
    report_title = st.text_input("Report title", value=f"Actuarial Valuation Report - {date.today().year}")
    draft = f"{report_title}\n\nPrepared on: {date.today().isoformat()}\n\nSummary of results:\nTotal projected liability (PV): ₹{total_liability:,.0f}\nNumber of employees: {len(val_results)}\n\nAssumptions considered:\n - Retirement age: {assumptions['retirement_age']}\n - Discount rate: {assumptions['discount_rate']:.3f}\n - Salary escalation: {assumptions['salary_escalation']:.3f}\n\nNotes:\n- This valuation is a simplified educational example. For statutory valuations, please use certified actuarial methods and signed reports from qualified personnel.\n\nRecommendations:\n- Review assumptions annually.\n- Consider funding strategy options and trustee governance.\n"
    user_edit = st.text_area("Editable report draft", value=draft, height=300)
    st.download_button("Download report (TXT)", data=user_edit.encode('utf-8'), file_name='client_report.txt', mime='text/plain')
st.markdown("---")
st.caption("Made with ❤️ — demo Streamlit app for actuarial valuation workflows. Replace model internals with certified actuarial engine before production use.")
