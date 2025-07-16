# streamlit_app.py
import os
import json
import streamlit as st
import asyncio
from utils.scraper import fetch_profile, evaluate_job_descriptions
from utils.llm_chain import get_langgraph_app

st.set_page_config(page_title="LinkedIn Optimizer", layout="centered")
st.title("🤖 LinkedIn Profile Optimizer")

st.write("Paste your LinkedIn profile URL below to get a full analysis:")
email = st.text_input("📧 Enter your Email",value="aryan070@icloud.com", placeholder="you@example.com")
PROFILE_URL = st.text_input("LinkedIn URL", value="https://www.linkedin.com/in/aryan-shah070/", placeholder="https://www.linkedin.com/in/...")
target_role = st.text_input("Target Job Role", value="Java Developer", placeholder="e.g., Product Manager,Data Scientist, etc.")
exp_level = st.selectbox("Experience Level", ["internship", "entry_level", "associate","mid_senior_level","director"], index=2)
def get_profile_data(linkedin_id):
    filename = f"profile.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    else:
        return fetch_profile(linkedin_id)

async def fetch_all_data(linkedin_id, target_role):
    loop = asyncio.get_event_loop()
    profile_task = loop.run_in_executor(None, get_profile_data, linkedin_id)
    jobs_task = loop.run_in_executor(None, evaluate_job_descriptions, target_role, exp_level)
    profile_data, job_data = await asyncio.gather(profile_task, jobs_task)
    return profile_data, job_data

def get_saved_result(email_id):
    file_path = "linkedin_optimizer_results.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            all_results = json.load(f)
        return all_results.get(email_id)
    return None

def save_result(email_id, result):
    file_path = "linkedin_optimizer_results.json"
    all_results = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            all_results = json.load(f)
    all_results[email_id] = result
    with open(file_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
if st.button("Load Previous"):
    if not email:
        st.error("Please enter your email.")
    else:
        compiled = get_langgraph_app()

        # 1️⃣ Get the latest snapshot (most recent checkpoint for this thread_id)
        snapshot = compiled.get_state(
            config={"configurable": {"thread_id": email}}
        )

        if snapshot is None:
            st.warning("No previous data found.")
        else:
            st.success("🧠 Loaded latest result")

            # Display the content of each graph node
            for section in ["analysis", "fit", "rewrite", "counseling"]:
                st.markdown(f"### {section.capitalize()}")
                st.write(snapshot.values[section])

            cp = snapshot.config["configurable"].get("checkpoint_id")
            ts = snapshot.created_at
            st.caption(f"(Checkpoint ID: {cp}, created at {ts})")
            
if st.button("Analyze Profile") and PROFILE_URL and target_role:
    if not (email and PROFILE_URL and target_role):
        st.error("Fill all fields first!")
    with st.spinner("Fetching profile and jobs..."):
        profile_data, job_data = asyncio.run(fetch_all_data(PROFILE_URL, target_role))
    st.subheader("✅ Profile Data Retrieved")
    st.write(profile_data)
    st.subheader("🔍 Job Description Analysis")
    st.write(job_data)

    with st.spinner("Running LinkedIn Optimizer..."):
        compiled = get_langgraph_app()
        result = compiled.invoke(
        {
        "profile": profile_data,
        "job_desc": job_data
        },
    config={"configurable": {"thread_id": email}}
)

    st.subheader("🧠 Full LinkedIn Optimization Results")

    st.markdown("### 1. Profile Analysis 📝")
    st.write(result["analysis"])

    st.markdown("### 2. Job Fit Analysis 🤝")
    st.write(result["fit"])

    st.markdown("### 3. Rewritten LinkedIn Sections ✍️")
    st.write(result["rewrite"])

    st.markdown("### 4. Career Counseling & Skill Gap Advice 🎯")
    st.write(result["counseling"])