# streamlit_app.py
import os
import json
import re
import streamlit as st
import asyncio
from utils.scraper import fetch_profile, evaluate_job_descriptions
from utils.llm_chain import get_langgraph_app

# --- Streamlit Page Config ---
st.set_page_config(page_title="LinkedIn Optimizer", layout="centered")
st.title("🤖 LinkedIn Profile Optimizer")
st.write("Paste your LinkedIn profile URL below to get a full analysis:")

# --- Input Fields ---
user_email = st.text_input("📧 Enter your Email", placeholder="you@example.com")
linkedin_profile_url = st.text_input("LinkedIn URL", placeholder="https://www.linkedin.com/in/...")
target_job_role = st.text_input("Target Job Role", placeholder="e.g., AI Developer, Product Manager, Data Scientist, etc.")
experience_level = st.selectbox(
    "Experience Level",
    ["internship", "entry_level", "associate", "mid_senior_level", "director"],
    index=2
)

# --- Validators ---
def validate_email_format(email: str) -> bool:
    """
    Validates the email address format using regex.
    Args:
        email (str): The email address to validate.
    Returns:
        bool: True if valid, False otherwise.
    """
    return bool(re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", email))

def validate_linkedin_url_format(url: str) -> bool:
    """
    Validates the LinkedIn profile URL format using regex.
    Args:
        url (str): The LinkedIn profile URL to validate.
    Returns:
        bool: True if valid, False otherwise.
    """
    return bool(re.match(r"^https:\/\/(www\.)?linkedin\.com\/in\/[A-Za-z0-9\-\_\.]+\/?$", url))

# --- Data Handling Functions ---
def load_profile_data_from_cache_or_scrape(profile_url: str) -> dict:
    """
    Loads LinkedIn profile data from cache if available, otherwise scrapes it.
    Args:
        profile_url (str): The LinkedIn profile URL.
    Returns:
        dict: The profile data.
    """
    cache_filename = "profile.json"
    if os.path.exists(cache_filename):
        with open(cache_filename, "r") as file:
            return json.load(file)
    return fetch_profile(profile_url)

async def fetch_profile_and_job_data(profile_url: str, job_role: str) -> tuple:
    """
    Asynchronously fetches both LinkedIn profile data and job description analysis.
    Args:
        profile_url (str): The LinkedIn profile URL.
        job_role (str): The target job role.
    Returns:
        tuple: (profile_data, job_data)
    """
    loop = asyncio.get_event_loop()
    profile_task = loop.run_in_executor(None, load_profile_data_from_cache_or_scrape, profile_url)
    jobs_task = loop.run_in_executor(None, evaluate_job_descriptions, job_role, experience_level)
    profile_data, job_data = await asyncio.gather(profile_task, jobs_task)
    return profile_data, job_data

def load_saved_optimization_result(email_id: str) -> dict:
    """
    Loads previously saved optimization results for a given email.
    Args:
        email_id (str): The user's email address.
    Returns:
        dict: The saved result if exists, else None.
    """
    file_path = "linkedin_optimizer_results.json"
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            all_results = json.load(file)
        return all_results.get(email_id)
    return None

def save_optimization_result(email_id: str, result: dict) -> None:
    """
    Saves the optimization result for a given email.
    Args:
        email_id (str): The user's email address.
        result (dict): The result data to save.
    """
    file_path = "linkedin_optimizer_results.json"
    all_results = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            all_results = json.load(file)
    all_results[email_id] = result
    with open(file_path, "w") as file:
        json.dump(all_results, file, indent=4)

# --- Load Previous Results ---
if st.button("Load Previous"):
    # Only allow valid email input
    if not user_email:
        st.error("Please enter your email.")
    elif not validate_email_format(user_email):
        st.error("Please enter a valid email address.")
    else:
        compiled_app = get_langgraph_app()
        snapshot = compiled_app.get_state(config={"configurable": {"thread_id": user_email}})
        if snapshot is None:
            st.warning("No previous data found.")
        else:
            st.success("🧠 Loaded latest result")
            for section in ["analysis", "fit", "rewrite", "counseling"]:
                st.markdown(f"### {section.capitalize()}")
                st.write(snapshot.values[section])
            checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")
            created_at = snapshot.created_at
            st.caption(f"(Checkpoint ID: {checkpoint_id}, created at {created_at})")

# --- Analyze Profile ---
if st.button("Analyze Profile"):
    # --- Input Validation: Only allow valid email and LinkedIn URL ---
    if not user_email or not linkedin_profile_url or not target_job_role:
        st.error("Fill all fields first!")
    elif not validate_email_format(user_email):
        st.error("Please enter a valid email address.")
    elif not validate_linkedin_url_format(linkedin_profile_url):
        st.error("Please enter a valid LinkedIn profile URL (e.g., https://www.linkedin.com/in/username/).")
    else:
        with st.spinner("Fetching profile and jobs..."):
            profile_data, job_data = asyncio.run(
                fetch_profile_and_job_data(linkedin_profile_url, target_job_role)
            )
        st.subheader("✅ Profile Data Retrieved")
        st.write(profile_data)
        st.subheader("🔍 Job Description Analysis")
        st.write(job_data)

        with st.spinner("Running LinkedIn Optimizer..."):
            compiled_app = get_langgraph_app()
            result = compiled_app.invoke(
                {
                    "profile": profile_data,
                    "job_desc": job_data
                },
                config={"configurable": {"thread_id": user_email}}
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