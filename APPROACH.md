# Project Approach, Challenges, Solutions, and Architecture

---

## Approach

This project delivers actionable, AI-powered feedback on LinkedIn profiles and recommends job opportunities based on real market data. The system combines Streamlit for the UI, ScrapingDog for LinkedIn/job data extraction, and OpenAI-powered LLMs (via LangChain and LangGraph) for analysis, rewriting, and career counseling.

**Key Steps:**
1. **User Input:** Collects user email (serving as a unique session/thread ID), LinkedIn profile URL, and target job role.
2. **Data Extraction:** Uses ScrapingDog API to fetch LinkedIn profile and job data.
3. **Job Market Analysis:** Aggregates and summarizes job descriptions for the target role.
4. **AI Analysis:** Employs LLMs to analyze the profile, compare it to job requirements, rewrite sections, and provide career counseling.
5. **Persistence:** Saves and retrieves user results for repeat visits, using robust checkpointing.

---

## Challenges & Solutions

### 1. **Session Identity & Personalization**
- **Challenge:** Users need to revisit and continue previous analyses.
- **Solution:** Email is used as a unique `thread_id`, enabling personalized, history-linked sessions. Users can load or continue previous sessions simply by entering their email.

### 2. **Reliable State Persistence**
- **Challenge:** Streamlit reruns and concurrency can disrupt in-memory state.
- **Solution:** Integrated `SqliteSaver` with `check_same_thread=False` for thread-safe, persistent storage in `linkedin_memory.db`. All workflow checkpoints are automatically saved and retrievable.

### 3. **Checkpoint Retrieval & History**
- **Challenge:** Users may want to see their latest or previous results.
- **Solution:** `get_state(thread_id)` fetches the most recent snapshot for a user. Optional `get_state_history()` supports inspection of prior runs.

### 4. **Composable, Modular Workflow**
- **Challenge:** The analysis pipeline must be modular, extensible, and maintainable.
- **Solution:** Used LangGraph's `StateGraph` to define a clear, four-node workflow:  
  `analysis → fit → rewrite → counseling`.  
  The graph is compiled once and cached for efficient reuse.

### 5. **Snapshot Access & Metadata**
- **Challenge:** Need to display detailed results and checkpoint metadata in the UI.
- **Solution:** Node outputs are accessed via `snapshot.values[...]`. Metadata such as `checkpoint_id` and `created_at` are extracted and shown in the UI for transparency.

### 6. **Thread-Safe Streamlit Integration**
- **Challenge:** Streamlit's rerun model and async requirements can cause race conditions.
- **Solution:** Maintains a single, persistent graph using `@st.cache_resource`. Asynchronous scraping tasks are dispatched via `asyncio.run` for efficiency and thread safety.

### 7. **User Experience**
- **Challenge:** Users need a seamless way to analyze or reload results.
- **Solution:**  
  - **Analyze Profile:** Runs a fresh workflow and displays results.
  - **Load Previous:** Retrieves and displays the latest user-specific data, with optional history view.

---

## Architecture Explanation

### 1. **Frontend (Streamlit)**
- Collects user input (email, LinkedIn URL, target role, experience level).
- Provides "Analyze Profile" and "Load Previous" buttons.
- Displays results in organized sections, including checkpoint metadata.

### 2. **Backend Logic**
- **`utils/scraper.py`:** Handles all ScrapingDog API calls for profile and job data, aggregates job descriptions.
- **`utils/llm_chain.py`:** Contains all LLM-based analysis, rewriting, and counseling logic, using LangChain and LangGraph for modular, stateful workflows.

### 3. **LangGraph Workflow**
- **Nodes:**  
  - `analysis`: Profile analysis and improvement suggestions  
  - `fit`: Job fit scoring and gap analysis  
  - `rewrite`: Rewriting LinkedIn sections  
  - `counseling`: Career counseling and skill gap advice
- **Persistence:**  
  - Uses `SqliteSaver` for checkpointing in `linkedin_memory.db`
  - Each user session is keyed by email (`thread_id`)
- **Snapshot Access:**  
  - Node outputs accessed via `snapshot.values[...]`
  - Metadata (checkpoint ID, timestamp) shown in UI

### 4. **LLM Integration**
- Uses OpenAI's GPT-4o-mini via LangChain for all analysis and rewriting tasks.

### 5. **Security & Environment**
- Sensitive API keys are loaded from `.env` using `python-dotenv`.
- No keys are hardcoded.

---

## Data Flow Diagram

```
User Input (Streamlit, email as thread_id)
      │
      ▼
Profile & Job Data Fetch (scraper.py, async)
      │
      ▼
LangGraph Workflow (llm_chain.py: analysis → fit → rewrite → counseling)
      │
      ▼
Checkpoint Persistence (SqliteSaver, linkedin_memory.db)
      │
      ▼
Results Display & Metadata (Streamlit UI)
```

---

## Extensibility

- **LLM Model:** Easily switch to more advanced models or providers.
- **Job Sources:** Add more job boards or APIs for broader market analysis.
- **Workflow:** Add or modify nodes in the LangGraph pipeline for new features.