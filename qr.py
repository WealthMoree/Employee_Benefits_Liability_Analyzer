import streamlit as st

st.title("Internship Certificate Verification")

# Form to take input
with st.form("student_form"):
    name = st.text_input("Student Name")
    email = st.text_input("Email")
    domain = st.text_input("Training Domain")
    remark = st.text_area("Remark")
    submitted = st.form_submit_button("Verify Details")

if submitted:
    if name and email and domain:
        st.success("Details Verified ✅")
        st.markdown(f"""
        ### Certificate Issued
        This certificate is issued to **{name}** and has successfully completed internship in **{domain}**.
        """)
        if remark:
            st.info(f"Remark: {remark}")
    else:
        st.warning("Please fill in all required fields!")
