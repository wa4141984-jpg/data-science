# 📊 AI Data Science Assistant

A beginner-friendly web app built with **Python**, **Streamlit** and the **Google Gemini API**.
It works as an AI tutor, a dataset analyzer, a chart builder, a code generator, a machine learning
guide and a study-notes maker, all in one dashboard.

> Made for Data Science students who want to learn Python, Data Analysis, Machine Learning and AI by using a real project.

---

## ✨ Main Features

| Page | What it does |
|------|--------------|
| 🏠 **Home** | Dashboard overview, quick-start guide and API key status |
| 🎓 **AI Data Science Tutor** | Chat with an AI tutor about Python, Statistics, Math, ML, Deep Learning and Analytics |
| 📂 **Dataset Analyzer** | Upload a CSV, see rows/columns, missing values, duplicates, statistics, quality insights, and generate an AI analysis report |
| 📈 **Data Visualization** | Interactive bar, line, pie, histogram and scatter charts with column selection |
| 💻 **Python Code Generator** | Generates code (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn) with a line-by-line explanation |
| 🤖 **Machine Learning Assistant** | Explains ML concepts, recommends algorithms, creates beginner examples |
| 📝 **Study Notes Generator** | Short or detailed notes, exam questions and MCQs |

A **sample dataset** can be loaded with one click, so you can try everything without your own CSV.

## 🛠️ Technologies Used

- **Python 3.9+**: backend logic
- **Streamlit**: user interface
- **Pandas / NumPy**: data handling
- **Plotly**: interactive charts
- **Google GenAI SDK (`google-genai`)** + **Gemini API**: AI features
- **Google AI Studio**: API key generation
- **GitHub** and **Streamlit Community Cloud**: source control and deployment

## 📁 Project Structure

```
your-project/
├── app.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml     <- you create this (never upload it to GitHub)
```

---

## 🚀 Installation

### 1. Check Python
```bash
python --version
```
Python 3.9 or newer is recommended.

### 2. Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the requirements
```bash
pip install -r requirements.txt
```

## 🔑 How to Generate a Gemini API Key

1. Open **Google AI Studio**: <https://aistudio.google.com>
2. Sign in with your Google account.
3. Click **Get API key** (left menu), then **Create API key**.
4. Copy the key and keep it private. Treat it like a password.

Google's free tier has usage limits, so if you see a "rate limit" message, wait a minute and try again.

## 🔐 How to Configure the API Key (Secrets)

The app **never** stores the key in code. It looks for the key in this order:

1. The sidebar field **⚙️ Settings → Paste Gemini API key** (only for the current session)
2. Streamlit secrets (`GEMINI_API_KEY`)
3. Environment variables `GEMINI_API_KEY` or `GOOGLE_API_KEY`

### Option A: Streamlit secrets (recommended)

Create a folder named `.streamlit` next to `app.py`, then create a file `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "paste-your-key-here"
```

### Option B: Environment variable

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="paste-your-key-here"

# macOS / Linux
export GEMINI_API_KEY="paste-your-key-here"
```

### Choosing the model

The sidebar has a **Gemini model** selector (default: `gemini-2.5-flash`). Google renames and retires
models from time to time. If you see *"Model not found"*, pick another model in the sidebar, or set the
`GEMINI_MODEL` environment variable to a current model name from Google AI Studio.

## ▶️ Run the App Locally

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>.

---

## 🐙 Upload the Project to GitHub

1. Create a free account at <https://github.com> and click **New repository** (for example `ai-data-science-assistant`).
2. In your project folder, create a file named `.gitignore` with this content so your key is **not** uploaded:
   ```
   .streamlit/secrets.toml
   venv/
   __pycache__/
   *.pyc
   ```
3. Run these commands in the project folder:
   ```bash
   git init
   git add app.py requirements.txt README.md .gitignore
   git commit -m "Initial commit: AI Data Science Assistant"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/ai-data-science-assistant.git
   git push -u origin main
   ```
4. Refresh your GitHub page and check that `secrets.toml` is **not** listed.

## ☁️ Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. Click **Create app** (or **New app**) and choose your repository.
3. Set the **branch** to `main` and **Main file path** to `app.py`.
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```
5. Click **Deploy**. After a few minutes you get a public link you can share.

To update the live app later, just `git push` new changes. It redeploys automatically.

---

## 🧯 Troubleshooting

| Problem | Solution |
|---------|----------|
| "API key not found" | Check `.streamlit/secrets.toml` spelling (`GEMINI_API_KEY`) or paste the key in the sidebar |
| "Invalid API key" | Create a new key in Google AI Studio and copy it again without spaces |
| "Model not found" | Choose a different model in the sidebar |
| "Rate limit" | Wait a minute; free tier has limits |
| CSV will not load | Save it as UTF-8 CSV with a header row |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your virtual environment |

## 🔒 Privacy Notes

- For AI analysis, only a **compact summary** of your dataset (column types, statistics and the first rows) is sent to Gemini, not the full file. Do not upload confidential or personal data if you are not allowed to share it.
- AI answers can contain mistakes. Always verify important results and test generated code.

## 🎯 Ideas to Extend the Project

- Add a model-training page using Scikit-learn on the uploaded dataset
- Add Excel file support
- Save chat history to a file
- Add more chart types (box plot, heatmap)

Happy learning! 🚀# data-science
