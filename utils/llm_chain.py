from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from typing import TypedDict, List
import streamlit as st

# Shared LLM
llm = OpenAI(temperature=0.1, model="gpt-4o-mini")

# 1. Define job description schema
class JobDesc(TypedDict):
    position: str
    skills: List[str]
    responsibilities: List[str]
    qualifications: List[str]
    industry_practices: List[str]
    highlights: List[str]

# 2. Define state schema for the graph
class LinkedInState(TypedDict):
    profile: dict
    job_desc: JobDesc
    analysis: str
    fit: str
    rewrite: str
    counseling: str

# Agent functions now accept and return updates, no side-effects
def analyze_profile(state: LinkedInState) -> dict:
    """
    Performs a comprehensive analysis of the user's LinkedIn profile.
    Identifies gaps, inconsistencies, and areas for improvement across all major sections.
    Provides actionable, structured suggestions to enhance professional presentation and keyword optimization.
    Returns a dictionary containing the analysis results.
    """
    prompt = PromptTemplate(
        input_variables=["profile"],
        template="""
        You are a highly experienced LinkedIn profile optimization expert and career coach.
        Your task is to thoroughly analyze the provided LinkedIn profile data.
        Identify and articulate specific gaps, inconsistencies, redundant information, and missing keywords
        across sections like 'About', 'Experience', 'Skills', 'Headline', and 'Summary'.

        For each identified area, provide actionable, concise, and professional suggestions for improvement.
        Highlight opportunities to:
        - Incorporate powerful action verbs.
        - Quantify achievements with metrics (e.g., percentages, numbers, monetary values).
        - Integrate relevant industry-specific keywords for enhanced searchability.
        - Streamline language for clarity and impact.
        - Ensure a compelling narrative that showcases value.

        Present your analysis in a structured, bullet-point format, focusing on direct improvements.

        ---
        Profile Data:
        {profile}

        ---
        Professional Profile Analysis and Improvement Suggestions:
        """
    )
    result = llm.invoke(prompt.format(profile=state["profile"]))
    return {"analysis": result}

def job_fit_analysis(state: LinkedInState) -> dict:
    """
    Compares the user's LinkedIn profile against the target job description.
    Generates a quantitative match score, highlights missing qualifications or skills, and recommends improvements for better alignment.
    Returns a dictionary with the fit analysis.
    """
    prompt = PromptTemplate(
        input_variables=["profile", "job_desc"],
        template="""
        You are an expert in precise job fit analysis, specializing in LinkedIn profile alignment.
        Your task is to meticulously compare the provided LinkedIn profile data against the target job description.

        Perform the following analysis:
        1.  **Generate a Match Score (70-100):** Provide a quantitative assessment of how well the profile aligns with the job requirements.
        2.  **Identify Missing Qualifications/Skills:** Detail specific skills, experiences, or qualifications present in the job description but absent or insufficiently highlighted in the profile.
        3.  **Suggest Improvements for Better Alignment:** Offer concrete, actionable advice on how the profile could be enhanced to better match the job description. This includes:
            - Recommending specific keywords from the job description to integrate.
            - Highlighting transferable skills that can be emphasized.
            - Suggesting areas where achievements could be rephrased to fit the role's needs.
        4.  **Headline Keyword Recommendation:** Propose concise, impactful keywords or phrases for the LinkedIn headline that would immediately signal relevance for this target role.

        Maintain a professional, direct, and actionable tone throughout your analysis.

        ---
        LinkedIn Profile Data:
        {profile}

        ---
        Target Job Description:
        {job_desc}

        ---
        Job Fit Analysis:
        """
    )
    result = llm.invoke(prompt.format(profile=state["profile"], job_desc=state["job_desc"]))
    return {"fit": result}

def rewrite_sections(state: LinkedInState) -> dict:
    """
    Rewrites key sections of the LinkedIn profile (About, Experience, Skills, Headline) for clarity, impact, and alignment with industry standards and the target job description.
    Ensures integration of relevant keywords, quantifiable achievements, and compelling narrative.
    Returns a dictionary with the rewritten sections.
    """
    prompt = PromptTemplate(
        input_variables=["profile", "job_desc"],
        template="""
        You are a top-tier LinkedIn content optimization specialist, adept at crafting compelling narratives.
        Your objective is to rewrite the designated sections of the provided LinkedIn profile to be highly
        concise, impactful, and precisely aligned with both industry best practices and the target job description.

        Focus on the following sections: 'About', 'Experience' (for relevant roles), 'Skills', and 'Headline'.

        Ensure the rewrite incorporates:
        - **Strategic Keyword Integration:** Naturally weave in all relevant keywords from the job description.
        - **Quantifiable Achievements:** Transform responsibilities into measurable accomplishments (e.g., "Increased X by Y%", "Managed Z projects achieving A").
        - **Powerful Action Verbs:** Start sentences and bullet points with strong, dynamic verbs.
        - **Concise & Scannable Language:** Optimize for readability by recruiters (mix of short paragraphs and bullet points).
        - **Tailored Content:** Adjust the tone and emphasis to directly appeal to the hiring manager for the target role.
        - **Industry-Specific Resonance:** Ensure the language reflects current industry trends and norms.
        - **Compelling Narrative:** The 'About' section should clearly articulate value proposition and career goals.

        Provide the rewritten sections clearly demarcated.

        ---
        Current LinkedIn Profile Data:
        {profile}

        ---
        Target Job Description (for alignment):
        {job_desc}

        ---
        Rewritten LinkedIn Profile Sections:
        """
    )
    result = llm.invoke(prompt.format(profile=state["profile"], job_desc=state["job_desc"]))
    return {"rewrite": result}

def career_counseling(state: LinkedInState) -> dict:
    """
    Delivers strategic career counseling based on the user's LinkedIn profile and the target job description.
    Identifies skill gaps, recommends learning resources, suggests career paths, and provides actionable advice for professional growth and personal branding.
    Returns a dictionary with the counseling output.
    """
    prompt = PromptTemplate(
        input_variables=["profile", "job_desc", "analysis", "fit", "rewrite"],
        template="""
        You are a seasoned and insightful career counselor with extensive knowledge of current industry trends and learning resources.
        Your primary role is to provide actionable and strategic career guidance based on the candidate's LinkedIn profile and the target job description.

        Leverage the provided profile analysis, job fit assessment, and rewritten profile sections to offer comprehensive advice.

        Specifically:
        1.  **Identify Critical Skill Gaps:** Pinpoint specific technical, soft, or domain-specific skills that are essential for the target role but appear to be missing or underdeveloped in the profile.
        2.  **Suggest Learning Resources & Certifications:** Recommend reputable online courses (e.g., Coursera, edX, LinkedIn Learning), certifications, bootcamps, or platforms to acquire identified missing skills. Be specific where possible.
        3.  **Advise on Strategic Career Paths:** Beyond the immediate job, suggest potential next steps or alternative career trajectories that align with the candidate's existing strengths and the demands of the target industry.
        4.  **Guidance for Excelling in the Role:** Provide practical advice on how to effectively prepare for and succeed in the target role, including interview tips, networking strategies, and ways to demonstrate ongoing value.
        5.  **Personal Branding Advice:** Offer insights on maintaining and growing a strong professional brand on LinkedIn and beyond.

        Deliver your advice in a professional, encouraging, and highly actionable manner.

        ---
        LinkedIn Profile Data (for full context):
        {profile}

        ---
        Target Job Description:
        {job_desc}

        ---
        Previous Profile Analysis:
        {analysis}

        ---
        Job Fit Assessment:
        {fit}

        ---
        Rewritten Profile Sections:
        {rewrite}

        ---
        Comprehensive Career Counseling:
        """
    )
    result = llm.invoke(prompt.format(profile=state["profile"], job_desc=state["job_desc"],analysis=state["analysis"], fit=state["fit"], rewrite=state["rewrite"]))
    return {"counseling": result}

# 3. Build the StateGraph

@st.cache_resource
def get_langgraph_app():
    """
    Constructs and compiles the LinkedIn optimization workflow as a stateful graph.
    Initializes persistent storage for checkpoints, defines the sequence of analysis nodes, and sets entry and exit points.
    Returns the compiled graph application for use in the Streamlit interface.
    """
    conn = sqlite3.connect("linkedin_memory.db", check_same_thread=False)
    saver = SqliteSaver(conn)
    graph = StateGraph(LinkedInState)
    graph.add_node("analysis", analyze_profile)
    graph.add_node("fit", job_fit_analysis)
    graph.add_node("rewrite", rewrite_sections)
    graph.add_node("counseling", career_counseling)
    graph.set_entry_point("analysis")
    graph.set_finish_point("counseling")
    graph.add_edge("analysis", "fit")
    graph.add_edge("fit", "rewrite")
    graph.add_edge("rewrite", "counseling")
    return graph.compile(checkpointer=saver)
