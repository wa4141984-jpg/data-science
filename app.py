"""
AI Data Science Assistant
=========================
A beginner-friendly Streamlit application that uses the Google Gemini API
(through the official `google-genai` SDK) to help students learn Data Science.

Run locally with:
    streamlit run app.py
"""

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION AND STYLING
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Science Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #06b6d4 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #ffffff; margin: 0 0 0.4rem 0; font-size: 2.1rem; }
    .hero p  { color: #eef2ff; margin: 0; font-size: 1.05rem; }

    .page-title {
        background: linear-gradient(90deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2rem;
        margin-bottom: 0.2rem;
    }
    .feature-card {
        background: rgba(124, 58, 237, 0.07);
        border: 1px solid rgba(124, 58, 237, 0.25);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        height: 100%;
        min-height: 150px;
    }
    .feature-card h4 { margin: 0 0 0.4rem 0; }
    .feature-card p  { margin: 0; font-size: 0.95rem; opacity: 0.9; }

    div[data-testid="stMetric"] {
        background: rgba(79, 70, 229, 0.07);
        border: 1px solid rgba(79, 70, 229, 0.2);
        padding: 0.8rem 1rem;
        border-radius: 12px;
    }
    .footer {
        text-align: center;
        opacity: 0.65;
        font-size: 0.85rem;
        margin-top: 2.5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 2. CONSTANTS
# ---------------------------------------------------------------------------
# The model can be changed from the sidebar, because Google updates model
# names from time to time. You can also set GEMINI_MODEL in secrets/env.
DEFAULT_MODEL = "gemini-2.5-flash"
MODEL_CHOICES = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]

TUTOR_SYSTEM_PROMPT = (
    "You are a friendly, patient Data Science tutor for beginners. "
    "Explain concepts in simple English, avoid unnecessary jargon, and define any "
    "technical term you use. Use short paragraphs, small examples, analogies, and "
    "Python code snippets (Pandas, NumPy, Scikit-learn) when helpful. "
    "Format answers with Markdown. Be accurate, and say so if you are unsure."
)

PAGES = [
    "🏠 Home",
    "🎓 AI Data Science Tutor",
    "📂 Dataset Analyzer",
    "📈 Data Visualization",
    "💻 Python Code Generator",
    "🤖 Machine Learning Assistant",
    "📝 Study Notes Generator",
]


# ---------------------------------------------------------------------------
# 3. SESSION STATE
# ---------------------------------------------------------------------------
def init_session_state():
    """Create all the variables we want to remember between reruns."""
    defaults = {
        "chat_history": [],      # list of {"role": "user"/"assistant", "content": str}
        "df": None,              # the currently loaded dataset
        "df_name": None,         # file name of the dataset
        "ai_report": "",         # last AI analysis text
        "model_name": os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        "pasted_key": "",        # optional API key typed in the sidebar (session only)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# 4. GEMINI API HELPERS
# ---------------------------------------------------------------------------
def get_api_key() -> str:
    """
    Read the API key safely. Order of priority:
      1. Key pasted in the sidebar (kept only in this browser session)
      2. Streamlit secrets  (.streamlit/secrets.toml or Cloud 'Secrets')
      3. Environment variables GEMINI_API_KEY or GOOGLE_API_KEY
    The key is NEVER hardcoded in this file.
    """
    pasted = st.session_state.get("pasted_key", "").strip()
    if pasted:
        return pasted

    key = ""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        # No secrets file exists - that is fine, we try environment variables next.
        key = ""

    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    return str(key).strip()


@st.cache_resource(show_spinner=False)
def build_client(api_key: str):
    """Create (and cache) one Gemini client per API key."""
    return genai.Client(api_key=api_key)


def ask_gemini(prompt: str, system_instruction: str = None, history: list = None,
               temperature: float = 0.7):
    """
    Send a prompt to Gemini and return the answer text.
    Returns None (after showing a friendly message) if something goes wrong.

    `history` is an optional list of {"role": "user"/"assistant", "content": str}
    used for multi-turn chat.
    """
    api_key = get_api_key()
    if not api_key:
        st.error(
            "🔑 **Gemini API key not found.**\n\n"
            "Add `GEMINI_API_KEY` to `.streamlit/secrets.toml`, set it as an "
            "environment variable, or paste it in the sidebar. "
            "See the README for step-by-step instructions."
        )
        return None

    # Convert our chat history into the format expected by the SDK.
    contents = []
    for message in history or []:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    try:
        client = build_client(api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
        )
        response = client.models.generate_content(
            model=st.session_state.model_name,
            contents=contents,
            config=config,
        )
        text = response.text
        if not text:
            st.warning(
                "⚠️ Gemini returned an empty answer (the response may have been "
                "blocked by safety filters). Try rephrasing your request."
            )
            return None
        return text

    except genai_errors.ClientError as err:
        code = getattr(err, "code", None)
        details = str(err)
        lowered = details.lower()
        if "api key" in lowered and ("invalid" in lowered or "not valid" in lowered):
            st.error("🔑 **Invalid API key.** Please check your key in Google AI Studio and try again.")
        elif code in (401, 403):
            st.error("🚫 **Access denied.** Your API key may be wrong, restricted, or not allowed to use this model.")
        elif code == 404:
            st.error(
                f"❓ **Model not found:** `{st.session_state.model_name}`. "
                "Choose another model in the sidebar (Google renames models over time)."
            )
        elif code == 429:
            st.error("⏳ **Rate limit or free quota reached.** Wait a minute and try again.")
        else:
            st.error(f"❌ **Request error ({code}).** {details}")
    except genai_errors.ServerError:
        st.error("🛠️ **Gemini server problem.** This is usually temporary - please try again shortly.")
    except genai_errors.APIError as err:
        st.error(f"❌ **Gemini API error:** {err}")
    except Exception as err:  # network problems, unexpected issues, etc.
        st.error(f"❌ **Unexpected error:** {err}")
    return None


def run_ai_task(prompt: str, system_instruction: str = None, spinner_text: str = "Gemini is thinking..."):
    """Run a prompt with a loading spinner and return the text (or None)."""
    with st.spinner(spinner_text):
        return ask_gemini(prompt, system_instruction=system_instruction)


# ---------------------------------------------------------------------------
# 5. DATASET HELPERS
# ---------------------------------------------------------------------------
def load_csv(uploaded_file):
    """Read an uploaded CSV file, trying common text encodings."""
    for encoding in ("utf-8", "latin-1"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            if df.empty or len(df.columns) == 0:
                raise ValueError("The CSV file has no data.")
            return df
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The CSV file is empty.")
        except pd.errors.ParserError as err:
            raise ValueError(f"The file could not be parsed as CSV: {err}")
    raise ValueError("Could not read the file. Please save it as a UTF-8 CSV and try again.")


def make_sample_dataset() -> pd.DataFrame:
    """Create a small practice dataset (with some missing values and duplicates)."""
    rng = np.random.default_rng(42)
    n = 200
    study_hours = rng.normal(5, 2, n).clip(0.5, 12).round(1)
    attendance = rng.normal(80, 12, n).clip(30, 100).round(0)
    math = (35 + study_hours * 5 + attendance * 0.15 + rng.normal(0, 8, n)).clip(0, 100).round(0)
    science = (30 + study_hours * 4.5 + attendance * 0.2 + rng.normal(0, 9, n)).clip(0, 100).round(0)
    df = pd.DataFrame({
        "student_id": np.arange(1, n + 1),
        "age": rng.integers(15, 20, n),
        "gender": rng.choice(["Male", "Female"], n),
        "city": rng.choice(["Peshawar", "Lahore", "Karachi", "Islamabad"], n),
        "study_hours": study_hours,
        "attendance": attendance,
        "math_score": math,
        "science_score": science,
    })
    df["passed"] = np.where((df["math_score"] + df["science_score"]) / 2 >= 50, "Yes", "No")
    # Add realistic problems for learning: missing values and duplicates.
    for column in ("attendance", "science_score", "city"):
        df.loc[rng.choice(n, 12, replace=False), column] = np.nan
    df = pd.concat([df, df.sample(6, random_state=1)], ignore_index=True)
    return df


def column_info(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table describing every column."""
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(t) for t in df.dtypes],
        "Non-Null": df.notna().sum().values,
        "Missing": df.isna().sum().values,
        "Missing %": (df.isna().mean() * 100).round(2).values,
        "Unique Values": df.nunique().values,
    })
    return info


def quality_insights(df: pd.DataFrame) -> list:
    """Create simple, rule-based data quality messages."""
    insights = []
    rows = len(df)
    missing_total = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    if missing_total == 0:
        insights.append("✅ No missing values were found.")
    else:
        insights.append(f"⚠️ The dataset has {missing_total:,} missing cells.")
        for column, pct in (df.isna().mean() * 100).items():
            if pct >= 30:
                insights.append(f"🔴 Column `{column}` is missing {pct:.1f}% of its values - consider dropping it or filling it carefully.")
            elif pct > 0:
                insights.append(f"🟡 Column `{column}` is missing {pct:.1f}% of its values - consider filling (mean/median/mode) or removing those rows.")

    if duplicates == 0:
        insights.append("✅ No duplicate rows were found.")
    else:
        insights.append(f"⚠️ {duplicates:,} duplicate rows were found ({duplicates / rows * 100:.1f}%). Consider removing them.")

    for column in df.columns:
        unique = df[column].nunique(dropna=True)
        if unique <= 1:
            insights.append(f"⚠️ Column `{column}` has only one unique value, so it adds no information.")
        elif not pd.api.types.is_numeric_dtype(df[column]) and rows > 20 and unique / rows > 0.9:
            insights.append(f"ℹ️ Column `{column}` looks like an ID or free text (almost every value is unique).")

    # Outlier check using the IQR rule on numeric columns.
    for column in df.select_dtypes(include="number").columns:
        series = df[column].dropna()
        if len(series) < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outliers = int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())
        if outliers / len(series) > 0.05:
            insights.append(f"📌 Column `{column}` has {outliers} possible outliers (IQR rule).")
    return insights


def quality_score(df: pd.DataFrame) -> int:
    """A simple 0-100 score. This is a rough teaching heuristic, not a standard."""
    missing_pct = df.isna().mean().mean() * 100
    duplicate_pct = df.duplicated().mean() * 100
    return int(max(0, min(100, round(100 - missing_pct * 1.5 - duplicate_pct * 2))))


def dataset_summary_text(df: pd.DataFrame, max_rows: int = 10) -> str:
    """Create a compact text description of a dataset to send to Gemini."""
    parts = [
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
        "Column types and missing values:\n" + column_info(df).to_string(index=False),
        f"Duplicate rows: {int(df.duplicated().sum())}",
        f"First {max_rows} rows (CSV):\n" + df.head(max_rows).to_csv(index=False),
    ]
    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        parts.append("Numeric summary:\n" + numeric.describe().round(3).to_string())
        if numeric.shape[1] >= 2:
            parts.append("Correlation matrix:\n" + numeric.corr().round(2).to_string())
    categorical = df.select_dtypes(exclude="number")
    for column in list(categorical.columns)[:8]:
        top = df[column].value_counts().head(5).to_dict()
        parts.append(f"Top values of '{column}': {top}")
    text = "\n\n".join(parts)
    return text[:12000]  # keep the prompt a reasonable size


def require_dataset() -> bool:
    """Show a hint and return False if no dataset is loaded."""
    if st.session_state.df is None:
        st.info("📂 No dataset loaded yet. Go to **Dataset Analyzer** to upload a CSV file or load the sample dataset.")
        return False
    return True


def page_header(title: str, subtitle: str):
    """Show a consistent page title."""
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.caption(subtitle)
    st.divider()


def extract_python_code(markdown_text: str):
    """Return the first Python code block from a Markdown answer (or None)."""
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", markdown_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# 6. SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    """Draw the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown("## 📊 AI Data Science Assistant")
        st.caption("Learn Python, Data Analysis, ML and AI")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()

        # API key status
        if get_api_key():
            st.success("API key detected ✅")
        else:
            st.warning("API key missing ⚠️")

        with st.expander("⚙️ Settings"):
            st.text_input(
                "Paste Gemini API key (optional)",
                type="password",
                key="pasted_key",
                help="Kept only in this browser session. Prefer secrets.toml for permanent setup.",
            )
            options = list(MODEL_CHOICES)
            if st.session_state.model_name not in options:
                options.insert(0, st.session_state.model_name)
            st.selectbox("Gemini model", options, key="model_name")

        if st.session_state.df is not None:
            st.divider()
            df = st.session_state.df
            st.caption("Loaded dataset")
            st.write(f"**{st.session_state.df_name}**")
            st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    return page


# ---------------------------------------------------------------------------
# 7. PAGES
# ---------------------------------------------------------------------------
def page_home():
    st.markdown(
        """
        <div class="hero">
            <h1>📊 AI Data Science Assistant</h1>
            <p>Your personal AI tutor and toolkit for learning Python, Data Analysis,
            Machine Learning and Artificial Intelligence - powered by Google Gemini.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Features", "6 tools")
    c2.metric("AI Engine", "Google Gemini")
    c3.metric("API key", "Ready ✅" if get_api_key() else "Missing ⚠️")

    st.markdown("### What can you do here?")
    cards = [
        ("🎓 AI Tutor", "Ask any Data Science question and get simple explanations with examples."),
        ("📂 Dataset Analyzer", "Upload a CSV, check data quality and get AI-generated insights and a report."),
        ("📈 Data Visualization", "Build interactive bar, line, pie, histogram and scatter charts."),
        ("💻 Code Generator", "Describe a task and get Python code with a line-by-line explanation."),
        ("🤖 ML Assistant", "Learn ML concepts, get algorithm recommendations and starter examples."),
        ("📝 Study Notes", "Create short or detailed notes, exam questions and MCQs on any topic."),
    ]
    for start in (0, 3):
        columns = st.columns(3)
        for column, (title, text) in zip(columns, cards[start:start + 3]):
            column.markdown(
                f'<div class="feature-card"><h4>{title}</h4><p>{text}</p></div>',
                unsafe_allow_html=True,
            )
        st.write("")

    st.markdown("### 🚀 Quick start")
    st.markdown(
        "1. Make sure your **Gemini API key** is set (see the sidebar status).\n"
        "2. Open **Dataset Analyzer** and upload a CSV - or load the sample dataset.\n"
        "3. Explore the data, create charts, and ask the AI for insights.\n"
        "4. Use the Tutor, Code Generator and Notes pages to keep learning."
    )

    if not get_api_key():
        st.info("🔑 No API key found yet. Follow the README to create a free key in Google AI Studio.")


def page_tutor():
    page_header("🎓 AI Data Science Tutor", "Ask questions about Python, Statistics, Math, Machine Learning, Deep Learning and Data Analytics.")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        topic = st.selectbox(
            "Pick a topic for a quick start (optional)",
            ["-- Choose a topic --", "Python", "Statistics", "Mathematics for ML",
             "Machine Learning", "Deep Learning", "Data Analytics"],
        )
    with col_b:
        level = st.selectbox("Explanation level", ["Absolute beginner", "Beginner", "Intermediate"])

    if topic != "-- Choose a topic --":
        if st.button(f"Explain the basics of {topic}"):
            st.session_state.chat_history.append(
                {"role": "user", "content": f"Explain the basics of {topic} for a Data Science student, with a simple example."}
            )
            st.session_state["_pending_reply"] = True

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.pop("_pending_reply", None)
        st.rerun()

    # Show the conversation so far.
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_text = st.chat_input("Ask a Data Science question...")
    if user_text:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)
        st.session_state["_pending_reply"] = True

    # Answer the latest user message if one is waiting.
    if st.session_state.get("_pending_reply") and st.session_state.chat_history:
        st.session_state.pop("_pending_reply", None)
        history = st.session_state.chat_history[:-1][-12:]  # last few turns for context
        latest = st.session_state.chat_history[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_gemini(
                    latest,
                    system_instruction=f"{TUTOR_SYSTEM_PROMPT} The student's level is: {level}.",
                    history=history,
                )
            if answer:
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            else:
                # Remove the unanswered question so history stays clean.
                st.session_state.chat_history.pop()


def page_dataset_analyzer():
    page_header("📂 Dataset Analyzer", "Upload a CSV file, check its quality, and let Gemini find insights.")

    up_col, sample_col = st.columns([3, 1])
    with up_col:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    with sample_col:
        st.write("")
        st.write("")
        if st.button("Load sample dataset"):
            st.session_state.df = make_sample_dataset()
            st.session_state.df_name = "sample_students.csv"
            st.session_state.ai_report = ""
        if st.session_state.df is not None and st.button("Clear dataset"):
            st.session_state.df = None
            st.session_state.df_name = None
            st.session_state.ai_report = ""
            st.rerun()

    if uploaded is not None and uploaded.name != st.session_state.df_name:
        try:
            st.session_state.df = load_csv(uploaded)
            st.session_state.df_name = uploaded.name
            st.session_state.ai_report = ""
        except ValueError as err:
            st.error(f"❌ {err}")
        except Exception as err:
            st.error(f"❌ Could not read the file: {err}")

    if not require_dataset():
        return

    df = st.session_state.df
    tab_overview, tab_quality, tab_stats, tab_ai = st.tabs(
        ["📋 Overview", "🧹 Missing & Duplicates", "📊 Statistics", "🧠 AI Analysis"]
    )

    # ---- Overview tab ----
    with tab_overview:
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Rows", f"{df.shape[0]:,}")
        m2.metric("Columns", df.shape[1])
        m3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        m4.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
        m5.metric("Quality score", f"{quality_score(df)}/100")

        st.markdown("#### Data preview")
        max_rows = max(6, min(100, len(df)))
        rows_to_show = st.slider("Rows to display", 5, max_rows, min(10, max_rows))
        st.dataframe(df.head(rows_to_show))

        st.markdown("#### Column information")
        st.dataframe(column_info(df))
        st.caption(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # ---- Missing & duplicates tab ----
    with tab_quality:
        missing = column_info(df)
        missing = missing[missing["Missing"] > 0]
        st.markdown("#### Missing values")
        if missing.empty:
            st.success("No missing values 🎉")
        else:
            fig = px.bar(missing, x="Column", y="Missing %", text="Missing",
                         title="Missing values per column (%)", color="Missing %",
                         color_continuous_scale="Purples")
            st.plotly_chart(fig)
            st.dataframe(missing)

        st.markdown("#### Duplicate records")
        duplicate_count = int(df.duplicated().sum())
        if duplicate_count == 0:
            st.success("No duplicate rows 🎉")
        else:
            st.warning(f"{duplicate_count:,} duplicate rows found.")
            with st.expander("Show duplicate rows"):
                st.dataframe(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)[:1]))
            if st.button("Remove duplicate rows"):
                st.session_state.df = df.drop_duplicates().reset_index(drop=True)
                st.session_state.ai_report = ""
                st.rerun()

        st.markdown("#### Data quality insights")
        for line in quality_insights(df):
            st.markdown(f"- {line}")

        st.download_button(
            "⬇️ Download current dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

    # ---- Statistics tab ----
    with tab_stats:
        numeric = df.select_dtypes(include="number")
        other = df.select_dtypes(exclude="number")
        st.markdown("#### Numeric columns")
        if numeric.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(numeric.describe().T.round(3))
            if numeric.shape[1] >= 2:
                corr = numeric.corr().round(2)
                fig = px.imshow(corr, text_auto=True, aspect="auto",
                                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                title="Correlation heatmap")
                st.plotly_chart(fig)
        st.markdown("#### Text / category columns")
        if other.empty:
            st.info("No text or category columns found.")
        else:
            st.dataframe(other.describe().T)

    # ---- AI analysis tab ----
    with tab_ai:
        st.markdown("Gemini will read a compact summary of your dataset (column types, statistics and the first rows - not the whole file).")
        goal = st.text_input("What would you like to know? (optional)", placeholder="e.g. What affects student scores the most?")
        if st.button("✨ Generate AI analysis"):
            prompt = (
                "You are a senior data analyst helping a beginner. Analyze the dataset summary below.\n\n"
                "Write a clear Markdown report with these sections:\n"
                "1. Dataset overview (what the data seems to be about)\n"
                "2. Key insights (specific numbers where possible)\n"
                "3. Patterns, trends and relationships\n"
                "4. Data quality problems and how to fix them\n"
                "5. Suggested improvements and next steps (including useful charts or ML ideas)\n"
                "6. Short conclusion\n\n"
                "Only use facts supported by the summary, and say when something is a guess.\n"
                f"User's question: {goal or 'None'}\n\n"
                f"DATASET SUMMARY:\n{dataset_summary_text(df)}"
            )
            result = run_ai_task(prompt, spinner_text="Analyzing your dataset...")
            if result:
                st.session_state.ai_report = result

        if st.session_state.ai_report:
            st.markdown(st.session_state.ai_report)
            local_report = (
                f"# Data Analysis Report\n\n"
                f"- File: {st.session_state.df_name}\n"
                f"- Generated: {datetime.now():%Y-%m-%d %H:%M}\n"
                f"- Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
                f"- Missing cells: {int(df.isna().sum().sum())}\n"
                f"- Duplicate rows: {int(df.duplicated().sum())}\n"
                f"- Quality score (rough heuristic): {quality_score(df)}/100\n\n"
                f"## Data quality checks\n" + "\n".join(f"- {x}" for x in quality_insights(df)) +
                f"\n\n## AI analysis\n\n{st.session_state.ai_report}\n"
            )
            st.download_button("⬇️ Download report (Markdown)", data=local_report,
                               file_name="analysis_report.md", mime="text/markdown")


def parse_dates_if_possible(series: pd.Series) -> pd.Series:
    """Try to convert a text column to dates. Return the original if it is not date-like."""
    # Works for both old pandas ("object") and new pandas ("str") text columns.
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        parsed = pd.to_datetime(series, errors="coerce")
    except Exception:
        return series
    return parsed if parsed.notna().mean() > 0.8 else series


def page_visualization():
    page_header("📈 Data Visualization", "Create interactive charts from your dataset. Hover, zoom and download them.")
    if not require_dataset():
        return

    df = st.session_state.df
    all_cols = list(df.columns)
    num_cols = list(df.select_dtypes(include="number").columns)

    chart_type = st.selectbox("Chart type", ["Bar chart", "Line chart", "Pie chart", "Histogram", "Scatter plot"])
    fig = None

    try:
        if chart_type == "Bar chart":
            c1, c2, c3, c4 = st.columns(4)
            x = c1.selectbox("Category (X axis)", all_cols)
            y_options = ["(count rows)"] + [c for c in num_cols if c != x]
            y = c2.selectbox("Value (Y axis)", y_options)
            agg = c3.selectbox("Aggregation", ["sum", "mean", "median", "max", "min"], disabled=(y == "(count rows)"))
            top_n = c4.slider("Top N categories", 3, 50, 15)
            if y == "(count rows)":
                counts = df[x].value_counts().head(top_n)
                data = pd.DataFrame({x: counts.index.astype(str), "Count": counts.values})
                fig = px.bar(data, x=x, y="Count", color="Count", color_continuous_scale="Viridis",
                             title=f"Number of rows per {x}")
            else:
                data = (df.groupby(x)[y].agg(agg).sort_values(ascending=False).head(top_n).reset_index())
                data[x] = data[x].astype(str)
                fig = px.bar(data, x=x, y=y, color=y, color_continuous_scale="Viridis",
                             title=f"{agg.title()} of {y} by {x}")

        elif chart_type == "Line chart":
            if not num_cols:
                st.warning("A line chart needs at least one numeric column.")
            else:
                c1, c2, c3 = st.columns(3)
                x = c1.selectbox("X axis", all_cols)
                y = c2.selectbox("Y axis (numeric)", [c for c in num_cols if c != x] or num_cols)
                color = c3.selectbox("Color by (optional)", ["(none)"] + [c for c in all_cols if df[c].nunique() <= 15 and c not in (x, y)])
                plot_df = df[[x, y] + ([] if color == "(none)" else [color])].dropna().copy()
                plot_df[x] = parse_dates_if_possible(plot_df[x])
                plot_df = plot_df.sort_values(x)
                fig = px.line(plot_df, x=x, y=y, color=None if color == "(none)" else color,
                              markers=len(plot_df) <= 100, title=f"{y} over {x}")

        elif chart_type == "Pie chart":
            c1, c2, c3 = st.columns(3)
            names = c1.selectbox("Slices (category)", all_cols)
            values = c2.selectbox("Slice size", ["(count rows)"] + [c for c in num_cols if c != names])
            top_n = c3.slider("Show top N slices", 2, 15, 6)
            if values == "(count rows)":
                series = df[names].value_counts()
            else:
                series = df.groupby(names)[values].sum().sort_values(ascending=False)
            series = series[series > 0]
            if len(series) > top_n:  # group the small slices together
                other_total = series.iloc[top_n:].sum()
                series = series.iloc[:top_n]
                series.loc["Other"] = other_total
            data = pd.DataFrame({"Category": series.index.astype(str), "Value": series.values})
            fig = px.pie(data, names="Category", values="Value", hole=0.35, title=f"Share by {names}")

        elif chart_type == "Histogram":
            if not num_cols:
                st.warning("A histogram needs at least one numeric column.")
            else:
                c1, c2, c3 = st.columns(3)
                col = c1.selectbox("Numeric column", num_cols)
                bins = c2.slider("Number of bins", 5, 100, 20)
                color = c3.selectbox("Color by (optional)", ["(none)"] + [c for c in all_cols if df[c].nunique() <= 10 and c != col])
                fig = px.histogram(df, x=col, nbins=bins, color=None if color == "(none)" else color,
                                   marginal="box", title=f"Distribution of {col}")

        elif chart_type == "Scatter plot":
            if len(num_cols) < 2:
                st.warning("A scatter plot needs at least two numeric columns.")
            else:
                c1, c2, c3 = st.columns(3)
                x = c1.selectbox("X axis", num_cols)
                y = c2.selectbox("Y axis", [c for c in num_cols if c != x])
                color = c3.selectbox("Color by (optional)", ["(none)"] + [c for c in all_cols if df[c].nunique() <= 15 and c not in (x, y)])
                fig = px.scatter(df, x=x, y=y, color=None if color == "(none)" else color,
                                 opacity=0.75, title=f"{y} vs {x}")
    except Exception as err:
        st.error(f"❌ Could not create this chart: {err}")
        fig = None

    if fig is not None:
        fig.update_layout(template="plotly_white", title_font_size=18)
        st.plotly_chart(fig)
        st.caption("Tip: use the camera icon on the chart toolbar to download it as an image.")

        if st.button("💡 Get AI tips for this chart"):
            prompt = (
                f"A student is creating a '{chart_type}' from a dataset. Chart title: '{fig.layout.title.text}'.\n"
                f"Dataset summary:\n{dataset_summary_text(df, max_rows=5)}\n\n"
                "In 4-6 bullet points, explain what to look for in this kind of chart for these columns, "
                "and which other chart might reveal more. Keep it beginner-friendly."
            )
            answer = run_ai_task(prompt)
            if answer:
                st.markdown(answer)


def page_code_generator():
    page_header("💻 Python Code Generator", "Describe a Data Science task and get working code with a line-by-line explanation.")

    task = st.text_area(
        "What should the code do?",
        height=120,
        placeholder="e.g. Load a CSV, fill missing values with the median, and plot a correlation heatmap.",
    )
    c1, c2 = st.columns(2)
    libraries = c1.multiselect(
        "Libraries to use",
        ["Pandas", "NumPy", "Matplotlib", "Seaborn", "Scikit-learn"],
        default=["Pandas", "NumPy"],
    )
    level = c2.selectbox("Explanation level", ["Absolute beginner", "Beginner", "Intermediate"])

    use_columns = False
    if st.session_state.df is not None:
        use_columns = st.checkbox(f"Use the columns of the loaded dataset ({st.session_state.df_name})", value=True)

    if st.button("⚡ Generate code"):
        if not task.strip():
            st.warning("Please describe the task first.")
            return
        if not libraries:
            st.warning("Please choose at least one library.")
            return
        column_text = ""
        if use_columns and st.session_state.df is not None:
            df = st.session_state.df
            column_text = "\nThe dataset has these columns and types: " + ", ".join(f"{c} ({t})" for c, t in df.dtypes.astype(str).items()) + "."
        prompt = (
            f"Task: {task}\nLibraries: {', '.join(libraries)}{column_text}\n\n"
            "Respond in Markdown with exactly these parts:\n"
            "## Code\nOne complete, runnable Python code block (include all imports and comments).\n"
            "## Line-by-line explanation\nExplain the code line by line (or small groups of lines) in a numbered list, "
            f"for a {level.lower()} learner.\n"
            "## How to run it\nShort instructions and the expected output.\n"
            "## Practice ideas\n2-3 small exercises to extend the code."
        )
        answer = run_ai_task(prompt, system_instruction="You are an expert Python and Data Science teacher. Write correct, clean, beginner-friendly code.", spinner_text="Writing your code...")
        if answer:
            st.session_state["generated_code_answer"] = answer

    answer = st.session_state.get("generated_code_answer")
    if answer:
        st.markdown(answer)
        code = extract_python_code(answer)
        if code:
            st.download_button("⬇️ Download code (.py)", data=code, file_name="generated_code.py", mime="text/x-python")
        st.caption("⚠️ Always read and test AI-generated code before using it on important data.")


def page_ml_assistant():
    page_header("🤖 Machine Learning Assistant", "Learn ML concepts, choose an algorithm, and get starter examples.")
    tab_learn, tab_recommend, tab_example = st.tabs(["📚 Learn concepts", "🧭 Recommend an algorithm", "🧪 Example generator"])

    with tab_learn:
        concept = st.selectbox("Choose a concept", [
            "Supervised learning", "Unsupervised learning", "Classification",
            "Regression", "Clustering", "Overfitting and underfitting",
            "Train/test split and cross-validation", "Model evaluation metrics",
        ])
        custom = st.text_input("...or type your own ML topic (optional)")
        if st.button("Explain it", key="ml_explain"):
            topic = custom.strip() or concept
            prompt = (
                f"Explain '{topic}' in machine learning to a beginner. Include: a simple definition, "
                "a real-life analogy, when to use it, 2-3 common algorithms, and a tiny Scikit-learn example."
            )
            result = run_ai_task(prompt, system_instruction=TUTOR_SYSTEM_PROMPT)
            if result:
                st.session_state["ml_explain_result"] = result
        if st.session_state.get("ml_explain_result"):
            st.markdown(st.session_state["ml_explain_result"])

    with tab_recommend:
        c1, c2 = st.columns(2)
        problem = c1.selectbox("What do you want to do?", [
            "Predict a category (classification)",
            "Predict a number (regression)",
            "Group similar items (clustering)",
            "Reduce the number of features",
            "Detect unusual data points (anomaly detection)",
        ])
        size = c2.selectbox("Dataset size", ["Small (< 1,000 rows)", "Medium (1,000 - 100,000 rows)", "Large (> 100,000 rows)"])
        c3, c4 = st.columns(2)
        priority = c3.selectbox("Top priority", ["Easy to understand", "Highest accuracy", "Fast training", "Handles messy data"])
        feature_types = c4.multiselect("Feature types", ["Numeric", "Categorical", "Text", "Images", "Time series"], default=["Numeric"])
        extra = st.text_area("Anything else? (optional)", placeholder="e.g. I want to predict house prices from size, rooms and location.")

        dataset_text = ""
        if st.session_state.df is not None and st.checkbox(f"Use my loaded dataset ({st.session_state.df_name}) as context"):
            dataset_text = "\n\nDataset summary:\n" + dataset_summary_text(st.session_state.df, max_rows=5)

        if st.button("Recommend algorithms", key="ml_recommend"):
            prompt = (
                f"Goal: {problem}\nDataset size: {size}\nPriority: {priority}\n"
                f"Feature types: {', '.join(feature_types) or 'not specified'}\nExtra details: {extra or 'none'}{dataset_text}\n\n"
                "Recommend the 3 most suitable machine learning algorithms for a beginner. For each: "
                "why it fits, pros, cons, and the Scikit-learn class name. Then give a recommended step-by-step workflow "
                "(data cleaning, split, training, evaluation metric to use) and one common beginner mistake to avoid."
            )
            result = run_ai_task(prompt, system_instruction=TUTOR_SYSTEM_PROMPT)
            if result:
                st.session_state["ml_recommend_result"] = result
        if st.session_state.get("ml_recommend_result"):
            st.markdown(st.session_state["ml_recommend_result"])

    with tab_example:
        algorithm = st.selectbox("Algorithm", [
            "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest",
            "K-Nearest Neighbors", "Support Vector Machine", "K-Means Clustering", "PCA",
        ])
        dataset_choice = st.selectbox("Practice data", ["A built-in Scikit-learn dataset", "Generated (synthetic) data"])
        if st.button("Generate example", key="ml_example"):
            prompt = (
                f"Write a complete beginner-friendly Python example using {algorithm} with Scikit-learn on "
                f"{dataset_choice.lower()}. Include imports, data loading, train/test split (if supervised), training, "
                "evaluation with an appropriate metric, and comments. Then explain each step in simple English and "
                "suggest 2 ways to improve the model. Format as Markdown with the code in one Python block."
            )
            result = run_ai_task(prompt, system_instruction=TUTOR_SYSTEM_PROMPT, spinner_text="Preparing your example...")
            if result:
                st.session_state["ml_example_result"] = result
        if st.session_state.get("ml_example_result"):
            st.markdown(st.session_state["ml_example_result"])
            code = extract_python_code(st.session_state["ml_example_result"])
            if code:
                st.download_button("⬇️ Download example (.py)", data=code, file_name="ml_example.py", mime="text/x-python")


def page_notes():
    page_header("📝 Study Notes Generator", "Turn any topic into clear notes, exam questions and MCQs.")

    topic = st.text_input("Topic", placeholder="e.g. Bias-variance tradeoff, Normal distribution, Pandas GroupBy")
    c1, c2 = st.columns(2)
    style = c1.selectbox("Notes style", ["Short notes (revision)", "Detailed notes", "Explain like I'm a beginner (easy language)"])
    include_exam = c2.checkbox("Include important exam questions", value=True)
    c3, c4 = st.columns(2)
    num_questions = c3.slider("Number of exam questions", 3, 15, 5, disabled=not include_exam)
    num_mcqs = c4.slider("Number of MCQs (0 = none)", 0, 15, 5)

    if st.button("📚 Generate study material"):
        if not topic.strip():
            st.warning("Please enter a topic.")
            return
        parts = [f"Create study material about: {topic}\nNotes style: {style}.\n"]
        parts.append("Start with a title, then the notes using headings and bullet points, key terms, and one simple example.")
        if include_exam:
            parts.append(f"Then add a section '## Important Exam Questions' with {num_questions} questions (include short model answers).")
        if num_mcqs > 0:
            parts.append(
                f"Then add a section '## Multiple-Choice Questions' with {num_mcqs} MCQs. Each has options A-D. "
                "Put an '**Answer:**' line with a one-sentence explanation after each question."
            )
        parts.append("Use simple English and be accurate.")
        result = run_ai_task("\n".join(parts), system_instruction=TUTOR_SYSTEM_PROMPT, spinner_text="Creating your notes...")
        if result:
            st.session_state["notes_result"] = result
            st.session_state["notes_topic"] = topic.strip()

    if st.session_state.get("notes_result"):
        st.markdown(st.session_state["notes_result"])
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "_", st.session_state.get("notes_topic", "notes")).strip("_").lower() or "notes"
        st.download_button("⬇️ Download notes (Markdown)", data=st.session_state["notes_result"],
                           file_name=f"{safe_name}_notes.md", mime="text/markdown")


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------
def main():
    init_session_state()
    page = render_sidebar()

    routes = {
        PAGES[0]: page_home,
        PAGES[1]: page_tutor,
        PAGES[2]: page_dataset_analyzer,
        PAGES[3]: page_visualization,
        PAGES[4]: page_code_generator,
        PAGES[5]: page_ml_assistant,
        PAGES[6]: page_notes,
    }
    routes[page]()

    st.markdown(
        '<div class="footer">Built with Streamlit, Python and Google Gemini · '
        'AI answers can contain mistakes - always double-check important information.</div>',
        unsafe_allow_html=True,
    )


main()
