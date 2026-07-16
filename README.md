# 🏥 Hospital Multilingual RAG System

A Retrieval-Augmented Generation (RAG) system built to serve as an intelligent, multilingual hospital assistant. This project allows users to query hospital data—such as doctor availability, department details, and emergency services—in **English, සිංහල (Sinhala), and தமிழ் (Tamil)**.

## 🚀 Key Features
* **100% Local Execution:** Utilizes Ollama to run Large Language Models locally, ensuring complete data privacy without requiring an internet connection.
* **Multilingual Semantic Search:** Uses Sentence-Transformers and FAISS to understand and retrieve information across three different languages.
* **Structured Knowledge Base:** Built on a lightweight SQLite database containing realistic hospital data (doctors, departments, and services).
* **Interactive UI:** Features a sleek web interface built with Streamlit for seamless user interaction.

## 🛠️ Tools & Technologies Used
* **LLM Engine:** [Ollama](https://ollama.com/) (Default Model: `llama3.2:1b` for fast, CPU-friendly inference)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Relational Database:** SQLite
* **RAG Orchestration:** LangChain
* **Embeddings:** `sentence-transformers`
* **Frontend:** Streamlit
* **Data Manipulation:** Pandas, NumPy

## 📂 Project Structure
* `app.py` - The main Streamlit web application.
* `hospital_rag.ipynb` - The complete step-by-step pipeline, from database creation to RAG implementation.
* `hospital.db` - The SQLite database serving as the core knowledge base.
* `requirements.txt` - Project dependencies.

## ⚙️ Setup & Installation

**1. Install Ollama & Pull the Model**
Download Ollama from their official website. Open your terminal and pull the required model:
```bash
ollama pull llama3.2:1b
ollama serve
```

2. Set Up the Environment
Clone the repository and install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Run the Application
Launch the Streamlit web interface:
```bash
streamlit run app.py
```