# Streamlit Chat LinkedIn Project

This project is a Streamlit application that analyzes LinkedIn profiles, provides actionable feedback, suggests job recommendations, and offers career counseling using AI and real job market data.

---

## Features

- **Chat Interface**: Engages users to collect their LinkedIn URL and target job role.
- **Profile Analysis**: Evaluates LinkedIn profiles for strengths, gaps, and improvements.
- **Job Recommendations**: Fetches and summarizes real job descriptions for the chosen role.
- **AI-Powered Optimization**: Uses advanced LLMs to analyze, rewrite, and optimize LinkedIn sections.
- **Career Counseling**: Offers personalized advice, skill gap analysis, and learning resources.

---

## Project Structure

```
streamlit-chat-linkedin/
├── app.py
├── requirements.txt
├── .env
├── README.md
└── utils/
    ├── scraper.py
    └── llm_chain.py
```

---

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Aryan-070/linkedin_optmiser.git
   cd streamlit-chat-linkedin
   ```

2. **Create a `.env` file in the project root:**
   ```
   SCRAPPING_API_KEY=your_scrapingdog_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```sh
   streamlit run app.py
   ```

---

## Usage

- Enter your email, LinkedIn profile URL, and target job role in the app.
- Click **Analyze Profile** to receive:
  - Profile analysis and improvement suggestions
  - Job fit analysis and keyword recommendations
  - Rewritten LinkedIn sections
  - Career counseling and skill gap advice
- Use **Load Previous** to retrieve your last analysis by email.

---

## Environment Variables

- The application requires a `.env` file with your [ScrapingDog](https://www.scrapingdog.com/) API key