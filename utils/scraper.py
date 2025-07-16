import requests
import json
import re
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("SCRAPPING_API_KEY")

def fetch_linkedin_profile(linkedin_id: str) -> dict:
    """
    Fetches a LinkedIn profile using the ScrapingDog API and saves the result to a local JSON file.
    Args:
        linkedin_id (str): The LinkedIn profile identifier.
    Returns:
        dict: The profile data if successful, None otherwise.
    """
    url = "https://api.scrapingdog.com/linkedin"
    params = {
        "api_key": api_key,
        "type": "profile",
        "linkId": linkedin_id,
        "premium": "false"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        with open("profile.json", "w") as f:
            json.dump(data, f, indent=4)
        return data
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None

def fetch_linkedin_job_listings(field: str, exp_level: str = "associate") -> list:
    """
    Fetches job listings from LinkedIn for a given field and experience level using the ScrapingDog API.
    Args:
        field (str): The job field or keyword.
        exp_level (str): The experience level (default: "associate").
    Returns:
        list: List of job listings as dictionaries.
    """
    url = "https://api.scrapingdog.com/linkedinjobs"
    params = {
        "api_key": api_key,
        "field": field,
        "geoid": "106300413",
        "page": 1,
        "sort_by": "day",
        "job_type": "full_time",
        "exp_level": exp_level,
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return []

def fetch_linkedin_job_overview(job_id: str) -> dict:
    """
    Fetches a detailed overview for a specific LinkedIn job using the ScrapingDog API.
    Args:
        job_id (str): The LinkedIn job identifier.
    Returns:
        dict: The job overview data if successful, empty dict otherwise.
    """
    url = "https://api.scrapingdog.com/linkedinjobs"
    params = {
        "api_key": api_key,
        "job_id": job_id
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Request failed with status code: {response.status_code}")
        return {}

def fetch_top_linkedin_job_overviews(field: str, exp_level: str, top_n: int = 5) -> list:
    """
    Fetches and aggregates overviews for the top N LinkedIn jobs in a given field and experience level.
    Args:
        field (str): The job field or keyword.
        exp_level (str): The experience level.
        top_n (int): Number of top jobs to fetch (default: 5).
    Returns:
        list: List of job overview dictionaries.
    """
    jobs = fetch_linkedin_job_listings(field, exp_level)
    if not jobs or not isinstance(jobs, list):
        print("No jobs found or invalid response format.")
        return []

    job_ids = [job["job_id"] for job in jobs[:top_n] if isinstance(job, dict) and "job_id" in job]
    overviews = []
    for job_id in job_ids:
        overview = fetch_linkedin_job_overview(job_id)
        # overview can be a list or dict, handle both
        if isinstance(overview, list) and len(overview) > 0:
            overview = overview[0]
        if isinstance(overview, dict) and "job_description" in overview:
            overviews.append({
                "job_id": job_id,
                "job_position": overview.get("job_position", ""),
                "job_description": overview.get("job_description", "")
            })
        else:
            print(f"Failed to fetch overview or missing fields for job ID: {job_id}")
    return overviews

class JobSummary(BaseModel):
    """
    Enterprise-grade schema for summarizing a job description.
    """
    position: str
    skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]
    industry_practices: List[str]
    highlights: List[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_json_from_llm_response(response_text: str) -> str:
    """
    Extracts a JSON object from a string that may be wrapped in markdown code block.
    Args:
        response_text (str): The LLM response text.
    Returns:
        str: The extracted JSON string.
    """
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0)
    return response_text

def evaluate_linkedin_job_descriptions(field: str, exp_level: str, top_n: int = 5) -> dict:
    """
    Fetches top job overviews, combines all job descriptions, and returns a single summarized evaluation object.
    Deduplicates and merges overlapping skills, responsibilities, qualifications, industry practices, and highlights.
    Args:
        field (str): The job field or keyword.
        exp_level (str): The experience level.
        top_n (int): Number of top jobs to evaluate (default: 5).
    Returns:
        dict: Summarized job description data.
    """
    overviews = fetch_top_linkedin_job_overviews(field, exp_level, top_n)
    all_descriptions = [overview.get("job_description", "") for overview in overviews if overview.get("job_description")]
    combined_descriptions = "\n\n".join(all_descriptions)

    prompt = f"""
        You are an expert assistant. Analyze the following job descriptions and extract:
        - skills (deduplicate and merge similar/overlapping skills, use standard names)
        - responsibilities (deduplicate and merge similar/overlapping responsibilities)
        - qualifications (deduplicate and merge similar/overlapping qualifications)
        - industry_practices (deduplicate and merge similar/overlapping practices)
        - highlights (deduplicate and merge similar/overlapping highlights)

        Return output as a single JSON object with these fields only (do NOT include company, position, or job_id).

        Job Descriptions:
        \"\"\"
        {combined_descriptions}
        \"\"\"
        """
    resp = llm([HumanMessage(content=prompt)])
    json_str = extract_json_from_llm_response(resp.content)
    try:
        data = json.loads(json_str)
        # Remove unwanted fields if present
        for key in ["company", "position", "job_id"]:
            data.pop(key, None)
        # Deduplicate lists if LLM missed it
        for k, v in data.items():
            if isinstance(v, list):
                seen = set()
                deduped = []
                for item in v:
                    norm = item.strip().lower()
                    if norm not in seen:
                        deduped.append(item)
                        seen.add(norm)
                data[k] = deduped
        return data
    except Exception as e:
        print(f"Error parsing evaluation response: {e}\nRaw response:\n{resp.content}")
        return {}

# Aliases for compatibility with the rest of the codebase
fetch_profile = fetch_linkedin_profile
