import requests
import json
import re
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv
# ...existing code...

load_dotenv()
api_key = os.getenv("SCRAPPING_API_KEY")

def fetch_profile(linkedin_id):
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
        with open(f"profile.json", "w") as f:
            json.dump(data, f, indent=4)
        return data
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None

def fetch_job_listings(field,exp_level="associate"):
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

def fetch_job_overview(job_id):
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

def fetch_top_job_overviews(field, exp_level,top_n=5):
    jobs = fetch_job_listings(field,exp_level)
    if not jobs or not isinstance(jobs, list):
        print("No jobs found or invalid response format.")
        return []

    job_ids = [job["job_id"] for job in jobs[:top_n] if isinstance(job, dict) and "job_id" in job]
    overviews = []
    for job_id in job_ids:
        overview = fetch_job_overview(job_id)
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

# Output schema for job summary
class JobSummary(BaseModel):
    position: str
    skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]
    industry_practices: List[str]
    highlights: List[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def extract_json_from_response(response_text):
    """
    Extracts JSON object from a string that may be wrapped in markdown code block.
    """
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        return match.group(0)
    return response_text


def evaluate_job_descriptions(field, exp_level,top_n=5):
    """
    Fetches top job overviews, combines all job descriptions, and returns a single summarised evaluation object.
    Removes duplicate or overlapping skills, responsibilities, qualifications, industry practices, and highlights.
    """
    overviews = fetch_top_job_overviews(field, exp_level,top_n)
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
    json_str = extract_json_from_response(resp.content)
    try:
        data = json.loads(json_str)
        # Remove unwanted fields if present
        for key in ["company", "position", "job_id"]:
            data.pop(key, None)
        # Deduplicate lists if LLM missed it
        for k, v in data.items():
            if isinstance(v, list):
                # Lowercase and strip for better deduplication
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
