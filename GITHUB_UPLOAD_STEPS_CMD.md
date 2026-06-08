# Command Prompt Steps to Upload to GitHub

## 1. Unzip the project
Download the ZIP and extract it into Downloads.

## 2. Open Command Prompt and enter the folder

```cmd
cd /d "%USERPROFILE%\Downloads\insightops-ai-agentic-rag-mlops"
```

## 3. Create GitHub repo
Create a new GitHub repository named:

```text
insightops-ai-agentic-rag-mlops
```

Do not initialize with README if this local folder already has files.

## 4. Push code

```cmd
git init
git branch -M main
git add .
git commit -m "Initial commit - InsightOps AI Agentic RAG MLOps platform"
git remote add origin https://github.com/YOUR_USERNAME/insightops-ai-agentic-rag-mlops.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

## 5. If remote origin already exists

```cmd
git remote set-url origin https://github.com/YOUR_USERNAME/insightops-ai-agentic-rag-mlops.git
git push -u origin main
```

## 6. If push rejected / fetch first

```cmd
git pull origin main --rebase
git push -u origin main
```

## 7. Run locally

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.data_generation.generate_synthetic_data
python -m src.pipelines.feature_engineering
python -m src.ml.train_ticket_severity_model
python -m src.ml.train_churn_model
python -m src.ml.anomaly_detection
python -m src.rag.vector_store
python -m src.rag.evaluate_rag
uvicorn src.api.main:app --reload
```

Open another CMD:

```cmd
cd /d "%USERPROFILE%\Downloads\insightops-ai-agentic-rag-mlops"
.venv\Scripts\activate
streamlit run dashboard\app.py
```
