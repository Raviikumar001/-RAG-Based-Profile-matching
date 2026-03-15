# AI-Powered Resume Matching System (RAG-Based)

I built this project to streamline the process of matching candidate resumes with specific job descriptions using a Retrieval-Augmented Generation (RAG) architecture. By leveraging Google's Gemini models for both embeddings and intelligent analysis, this system can parse complex resumes, store them in a vector database, and perform high-precision semantic searches to rank the best candidates for any given JD.

## 🚀 Key Features

- **Intelligent Resume Parsing:** Automatically splits PDF resumes into logical sections (Experience, Education, Skills, etc.) while maintaining context.
- **RAG-Driven Matching:** Uses a hybrid approach of semantic similarity and LLM-based verification to find the most relevant candidates.
- **Gemini Integration:** 
  - `gemini-embedding-2-preview` for high-dimensional vector representations.
  - `gemini-3-flash-preview` for intelligent tool-calling and final candidate scoring.
- **Vector Storage:** Powered by ChromaDB for persistent, efficient similarity searches.
- **Detailed Analytics:** Includes a professional Jupyter Notebook for latency benchmarking, chunking statistics, and system performance evaluation.

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **LLM/Embeddings:** Google GenAI (Gemini)
- **Vector DB:** ChromaDB
- **PDF Processing:** `pdfplumber`
- **Data Analysis:** `pandas`, `matplotlib`, `numpy`
- **Environment:** VS Code / Jupyter

## 📂 Project Structure

- `resume_rag.py`: The core ingestion pipeline. It handles PDF text extraction, section-based chunking, and embedding generation.
- `job_matcher.py`: The matching engine. It takes a Job Description, queries the vector store, and uses Gemini's tool-calling capabilities to score candidates.
- `rag_analysis.ipynb`: A diagnostic notebook I used to verify data ingestion, measure retrieval latency, and visualize performance metrics.
- `diverse_resume/`: Directory containing the target PDF resumes.
- `jd/`: Contains `job-description.json` which serves as the input for matching.
- `chroma_db/`: Persistent storage for resume embeddings.

## ⚙️ Setup & Installation

1. **Clone the repository**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment:**
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

## 📈 Performance Summary

Based on my analysis in the [rag_analysis.ipynb](rag_analysis.ipynb):
- **Total Resumes Processed:** 31
- **Total Chunks Created:** 135
- **Avg. Retrieval Latency:** ~650ms - 950ms
- **Top Metrics:** The system successfully identifies core competencies and maps them to JD requirements with a high degree of semantic accuracy.

## 📝 How it Works

1. **Ingestion:** Run `resume_rag.py` to process the resumes in `diverse_resume/`. This populates the ChromaDB collection.
2. **Matching:** Run `job_matcher.py` to evaluate the resumes against the JD provided in `jd/job-description.json`.
3. **Evaluation:** Open `rag_analysis.ipynb` to see a detailed breakdown of how the system is performing.
