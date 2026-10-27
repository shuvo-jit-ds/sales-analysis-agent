"""
Sales Analysis Agent  -  upload Excel/CSV -> analysis, charts, prediction, Ask-AI chat

Run:   python -m streamlit run sales_agent_app.py
Needs: pip install streamlit pandas numpy matplotlib scikit-learn openpyxl
Optional (real Claude AI answers): pip install anthropic
"""

import html
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from sklearn.linear_model import LinearRegression

CLAUDE_MODEL = "claude-sonnet-5-5"

# ---------- colors used by the charts ----------
BG = "#161B26"
FG = "#E6EAF2"
MUTED = "#9AA4B8"
GRID = "#2A3142"
PALETTE = ["#5B8DEF", "#F59E0B", "#34D399", "#F472B6", "#A78BFA", "#22D3EE", "#F87171", "#FACC15"]


# =====================================================================
#  PART 1: READ THE FILE AND FIND THE COLUMNS
# =====================================================================
def load_file(uploaded):
    """Read an Excel or CSV file into a DataFrame."""
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def category_columns(df, date_col=None, sales_col=None):
    """Text-like columns with a small number of different values (Product, Region...)."""
    cols = []
    for c in df.columns:
        if c in (date_col, sales_col):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if 1 < df[c].nunique() <= 50:
            cols.append(c)
    return cols


def detect_columns(df):
    """Guess (date_col, product_col, sales_col). Any of them can be None."""
    # date column
    date_col = None
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_col = c
            break
    if date_col is None:
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                continue
            if any(w in str(c).lower() for w in ["date", "month", "day", "time"]):
                parsed = pd.to_datetime(df[c], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    date_col = c
                    break

    # sales column
    numeric = [c for c in df.select_dtypes("number").columns if c != date_col]
    sales_col = None
    for hint in ["sales", "revenue", "amount", "total", "income", "profit", "price", "qty", "quantity"]:
        for c in numeric:
            if hint in str(c).lower():
                sales_col = c
                break
        if sales_col:
            break
    if sales_col is None and numeric:
        sales_col = numeric[0]

    # product column
    cats = category_columns(df, date_col, sales_col)
    product_col = None
    for hint in ["product", "item", "category", "name", "region"]:
        for c in cats:
            if hint in str(c).lower():
                product_col = c
                break
        if product_col:
            break
    if product_col is None and cats:
        product_col = cats[0]

    return date_col, product_col, sales_col


# =====================================================================
#  PART 2: ANALYSIS
# =====================================================================
def by_group(df, col, sales_col):
    return df.groupby(col)[sales_col].sum().sort_values(ascending=False)


def by_month(df, date_col, sales_col):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    return d.groupby(d[date_col].dt.to_period("M"))[sales_col].sum().sort_index()


def forecast(monthly, steps=3):
    """Fit a straight line to the monthly totals and predict the next months."""
    X = np.arange(len(monthly)).reshape(-1, 1)
    model = LinearRegression().fit(X, monthly.values)
    future_X = np.arange(len(monthly), len(monthly) + steps).reshape(-1, 1)
    future_y = np.clip(model.predict(future_X), 0, None)
    future_index = pd.period_range(monthly.index[-1] + 1, periods=steps, freq="M")
    return pd.Series(future_y, index=future_index), float(model.coef_[0])


def growth_pct(monthly):
    if len(monthly) < 2 or monthly.iloc[0] == 0:
        return None
    return (monthly.iloc[-1] - monthly.iloc[0]) / monthly.iloc[0] * 100


def make_insights(df, date_col, product_col, sales_col):
    """List of (finding, how_i_found_it)."""
    out = []
    total = df[sales_col].sum()
    out.append((f"Total {sales_col} is {total:,.0f}.",
                f"I added up every value in the '{sales_col}' column."))

    if product_col:
        prod = by_group(df, product_col, sales_col)
        share = prod.iloc[0] / total * 100 if total else 0
        out.append((f"Best {product_col}: {prod.index[0]} with {prod.iloc[0]:,.0f} ({share:.0f}% of all sales).",
                    f"I grouped the rows by '{product_col}', added up each group and picked the biggest."))
        if len(prod) > 1:
            out.append((f"Weakest {product_col}: {prod.index[-1]} with {prod.iloc[-1]:,.0f}.",
                        "It has the smallest total after grouping."))

    if date_col:
        monthly = by_month(df, date_col, sales_col)
        if len(monthly) >= 2:
            out.append((f"Best month: {monthly.idxmax()} ({monthly.max():,.0f}). "
                        f"Weakest month: {monthly.idxmin()} ({monthly.min():,.0f}).",
                        "I added up all sales inside each month."))
            g = growth_pct(monthly)
            if g is not None:
                word = "grew" if g > 0 else "fell"
                out.append((f"Monthly sales {word} by {abs(g):.0f}% from {monthly.index[0]} to {monthly.index[-1]}.",
                            "I compared the first month with the last month."))
    return out


# =====================================================================
#  PART 3: THE AI CHAT (works offline, no key needed)
# =====================================================================
def example_questions(df, date_col, product_col, sales_col):
    qs = []
    if product_col:
        top = by_group(df, product_col, sales_col)
        qs.append(f"Which {str(product_col).lower()} is best?")
        if len(top) >= 2:
            qs.append(f"Compare {top.index[0]} and {top.index[1]}")
    if date_col:
        qs += ["What is the best month?", "Predict next month", "Is the trend going up?"]
    qs.append("Give me a summary")
    return qs[:6]


def mentioned_values(q, df, cats):
    found = []
    for c in cats:
        for v in df[c].dropna().unique():
            s = str(v).lower()
            if len(s) >= 2 and re.search(r"\b" + re.escape(s) + r"\b", q):
                found.append((c, v))
    return found


def ask_offline(question, df, date_col, product_col, sales_col):
    """Answer a question about the data. Returns (text, table_or_None)."""
    q = " " + question.lower().strip() + " "
    cats = category_columns(df, date_col, sales_col)
    vals = mentioned_values(q, df, cats)
    cols = [c for c in cats if str(c).lower() in q]
    examples = example_questions(df, date_col, product_col, sales_col)

    def has(*words):
        return any(w in q for w in words)

    monthly_all = by_month(df, date_col, sales_col) if date_col else None
    sub = df
    sub_name = ""
    if vals:
        sub = df[df[vals[0][0]] == vals[0][1]]
        sub_name = " for " + str(vals[0][1])
    monthly = by_month(sub, date_col, sales_col) if date_col else None
    have_months = monthly is not None and len(monthly) > 0

    # ---- help ----
    if q.strip() in ("hi", "hello", "hey", "help") or has("what can you do", "what can i ask"):
        return ("Hi! I can answer questions about your data and explain how I got each answer. Try:\n\n"
                + "\n".join(f"- {e}" for e in examples)), None

    # ---- prediction ----
    if has("predict", "forecast", "next month", "future", "expect"):
        if monthly_all is None or len(monthly_all) < 4:
            return "I need a date column with at least 4 months of data to predict.", None
        future, slope = forecast(monthly_all, 3)
        table = pd.DataFrame({"Month": [str(p) for p in future.index],
                              f"Predicted {sales_col}": future.round(0).astype(int).values})
        direction = "rises" if slope > 0 else "falls"
        return (f"I predict **{future.iloc[0]:,.0f}** {sales_col} for **{future.index[0]}**.\n\n"
                f"**Why:** I drew a straight trend line through your {len(monthly_all)} monthly totals. "
                f"The line {direction} by about {abs(slope):,.0f} per month, so I extended it forward. "
                f"This is a rough estimate, not a promise."), table

    # ---- compare two values ----
    if has("compare", " vs ", "versus", "better", "difference") and len(vals) >= 2:
        col = vals[0][0]
        picked = [v for c, v in vals if c == col]
        if len(picked) >= 2:
            totals = df[df[col].isin(picked)].groupby(col)[sales_col].sum().sort_values(ascending=False)
            top, second = totals.index[0], totals.index[1]
            diff = totals.iloc[0] - totals.iloc[1]
            pct = diff / totals.iloc[1] * 100 if totals.iloc[1] else 0
            table = totals.rename(sales_col).reset_index()
            return (f"**{top}** is ahead of **{second}** by {diff:,.0f} ({pct:.0f}% more).\n\n"
                    f"**Why:** I added up {sales_col} for each {col}: "
                    f"{top} = {totals.iloc[0]:,.0f} and {second} = {totals.iloc[1]:,.0f}."), table

    # ---- average ----
    if has("average", "avg", "mean"):
        text = f"Average {sales_col} per row{sub_name}: **{sub[sales_col].mean():,.1f}**."
        if have_months:
            text += f"\n\nAverage per month: **{monthly.mean():,.1f}**."
        return text + "\n\n**Why:** I added the values up and divided by how many there are.", None

    # ---- month by month table ----
    if has("month") and has(" by ", "each", "every", " per ", "breakdown", "list", "all months"):
        if not have_months:
            return "I could not find a date column, so I cannot split by month.", None
        table = pd.DataFrame({"Month": [str(p) for p in monthly.index], sales_col: monthly.values})
        return f"Here are the monthly totals{sub_name}.", table

    # ---- breakdown by a category ----
    if has(" by ", "each", "every", " per ", "breakdown", "split", "share", "distribution"):
        col = cols[0] if cols else product_col
        if col:
            totals = by_group(df, col, sales_col)
            table = totals.rename(sales_col).reset_index()
            table["Share %"] = (totals.values / totals.sum() * 100).round(1)
            return (f"Here is {sales_col} split by {col}. **{totals.index[0]}** is the biggest.\n\n"
                    f"**Why:** I grouped the rows by {col} and added up each group."), table

    # ---- best / worst ----
    worst_words = ("worst", "lowest", "least", "weak", "minimum", "bottom", "poor")
    best_words = ("best", "top", "highest", "most", "maximum", "strong", "biggest")
    if has(*worst_words, *best_words):
        worst = has(*worst_words)
        word = "weakest" if worst else "best"
        if has("month") and date_col:
            if not have_months:
                return "I could not read the dates, so I cannot compare months.", None
            m = monthly.idxmin() if worst else monthly.idxmax()
            v = monthly.min() if worst else monthly.max()
            table = pd.DataFrame({"Month": [str(p) for p in monthly.index], sales_col: monthly.values})
            return (f"The {word} month{sub_name} is **{m}** with {v:,.0f}.\n\n"
                    f"**Why:** I added up the sales inside each month and compared them."), table
        col = cols[0] if cols else (product_col or (cats[0] if cats else None))
        if col:
            totals = by_group(df, col, sales_col)
            pick = totals.index[-1] if worst else totals.index[0]
            share = totals[pick] / totals.sum() * 100 if totals.sum() else 0
            table = totals.rename(sales_col).reset_index()
            return (f"The {word} {col} is **{pick}** with {totals[pick]:,.0f} ({share:.0f}% of the total).\n\n"
                    f"**Why:** I grouped the rows by {col}, added up each group and compared them."), table
        row = df[sales_col].idxmin() if worst else df[sales_col].idxmax()
        return f"The {word} single value of {sales_col} is {df.loc[row, sales_col]:,.0f}.", None

    # ---- trend ----
    if has("trend", "growth", "growing", "grow", "increase", "decrease", "falling", "going up", "going down"):
        if not have_months or len(monthly) < 2:
            return "I need a date column with a few months of data to see a trend.", None
        g = growth_pct(monthly)
        text = ""
        if g is not None:
            word = "grew" if g > 0 else "fell"
            text += f"Monthly sales{sub_name} {word} by **{abs(g):.0f}%** from {monthly.index[0]} to {monthly.index[-1]}."
        if len(monthly) >= 4:
            _, slope = forecast(monthly, 1)
            text += f"\n\nOn average the trend line moves {slope:+,.0f} per month."
        table = pd.DataFrame({"Month": [str(p) for p in monthly.index], sales_col: monthly.values})
        return text + "\n\n**Why:** I compared the first and last month and fitted a straight line through all months.", table

    # ---- one specific value (e.g. 'how is Laptop doing') ----
    if vals:
        col, v = vals[0]
        total = df[sales_col].sum()
        s = sub[sales_col].sum()
        text = (f"**{v}** has {s:,.0f} {sales_col} in total, which is {s / total * 100:.0f}% of everything. "
                f"Average per row: {sub[sales_col].mean():,.1f}.")
        if have_months and len(monthly) > 1:
            text += f"\n\nIts best month is **{monthly.idxmax()}** ({monthly.max():,.0f})."
        return text + f"\n\n**Why:** I filtered the rows where {col} is {v} and added them up.", None

    # ---- summary ----
    if has("summary", "overview", "insight", "tell me", "analy", "report", "everything"):
        items = make_insights(df, date_col, product_col, sales_col)
        return "Here is the big picture:\n\n" + "\n".join(f"- {a}" for a, _ in items), None

    # ---- total ----
    if has("total", " sum ", "how much", "overall", "altogether", "sold"):
        text = f"Total {sales_col}: **{df[sales_col].sum():,.0f}** across {len(df)} rows."
        return text + f"\n\n**Why:** I added up every value in the '{sales_col}' column.", None

    return ("I am not sure how to answer that yet. Try one of these:\n\n"
            + "\n".join(f"- {e}" for e in examples)), None


def build_context(df, date_col, product_col, sales_col):
    """A short text summary of the data to give to Claude (not the whole file)."""
    lines = [f"Rows: {len(df)}",
             f"Columns: {', '.join(map(str, df.columns))}",
             f"Sales column: {sales_col}. Total {df[sales_col].sum():,.0f}. Average per row {df[sales_col].mean():,.1f}."]
    for c in category_columns(df, date_col, sales_col)[:3]:
        lines.append(f"Totals by {c}:\n{by_group(df, c, sales_col).to_string()}")
    if date_col:
        monthly = by_month(df, date_col, sales_col)
        if len(monthly):
            lines.append(f"Monthly totals:\n{monthly.to_string()}")
            if len(monthly) >= 4:
                future, slope = forecast(monthly, 3)
                lines.append(f"Linear trend forecast (next 3 months):\n{future.round(0).to_string()}\n"
                             f"Trend slope per month: {slope:,.1f}")
    lines.append("First 15 rows:\n" + df.head(15).to_string(index=False))
    return "\n\n".join(lines)


def ask_claude(api_key, history, df, date_col, product_col, sales_col):
    """Ask the real Claude model. history = [{'role','content'}, ...] ending with the user question."""
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=api_key)
    system = ("You are a friendly sales analyst helping a beginner. Answer using ONLY the data summary below. "
              "Explain simply, mention how you got the answer, and keep it short. If the data cannot answer, say so.\n\n"
              + build_context(df, date_col, product_col, sales_col))
    resp = client.messages.create(model=CLAUDE_MODEL, max_tokens=700, system=system, messages=history)
    return resp.content[0].text


# =====================================================================
#  PART 4: CHARTS (dark, clean style)
# =====================================================================
def fmt_axis(x, _pos):
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


def new_fig(title, size=(7, 4.2)):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def sparse_ticks(ax, labels):
    step = max(1, int(np.ceil(len(labels) / 12)))
    idx = list(range(0, len(labels), step))
    ax.set_xticks(idx)
    ax.set_xticklabels([labels[i] for i in idx], rotation=45, ha="right", color=MUTED)


def chart_bars(series, title, top=10):
    s = series.head(top)[::-1]
    fig, ax = new_fig(title)
    colors = [PALETTE[0]] * len(s)
    colors[-1] = PALETTE[1]
    bars = ax.barh([str(i) for i in s.index], s.values, color=colors, height=0.6)
    for b, v in zip(bars, s.values):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2, f"  {v:,.0f}",
                va="center", color=FG, fontsize=9)
    ax.set_xlim(0, s.max() * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_axis))
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", colors=FG, labelsize=10)
    fig.tight_layout()
    return fig


def chart_donut(series, title):
    fig, ax = new_fig(title, (6, 4.2))
    total = series.sum()
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(series))]
    wedges, _ = ax.pie(series.values, colors=colors, startangle=90,
                       wedgeprops=dict(width=0.38, edgecolor=BG, linewidth=2))
    labels = [f"{i}  ({v / total:.0%})" for i, v in zip(series.index, series.values)]
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, labelcolor=FG)
    ax.text(0, 0, f"{total:,.0f}", ha="center", va="center", color=FG, fontsize=18, fontweight="bold")
    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(title, color=FG, fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    return fig


def chart_monthly(monthly, title):
    labels = [str(p) for p in monthly.index]
    x = np.arange(len(monthly))
    y = monthly.values
    fig, ax = new_fig(title)
    ymin = max(0, y.min() * 0.85)
    ax.plot(x, y, color=PALETTE[1], marker="o", linewidth=2.4, markersize=5)
    ax.fill_between(x, ymin, y, color=PALETTE[1], alpha=0.14)
    ax.set_ylim(ymin, y.max() * 1.12)
    best = int(np.argmax(y))
    ax.scatter([best], [y[best]], s=90, color=PALETTE[2], zorder=5)
    ax.annotate(f"Best: {labels[best]}  {y[best]:,.0f}", (best, y[best]), textcoords="offset points",
                xytext=(0, 12), ha="center", color=FG, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_axis))
    sparse_ticks(ax, labels)
    fig.tight_layout()
    return fig


def chart_forecast(monthly, future, title):
    n = len(monthly)
    labels = [str(p) for p in monthly.index] + [str(p) for p in future.index]
    fig, ax = new_fig(title)
    ax.plot(np.arange(n), monthly.values, color=PALETTE[0], marker="o", linewidth=2.4, markersize=5, label="Actual")
    fx = np.arange(n - 1, n + len(future))
    ax.plot(fx, [monthly.values[-1]] + list(future.values), color=PALETTE[3], linestyle="--",
            marker="o", linewidth=2.2, markersize=5, label="Predicted")
    ax.axvspan(n - 1, n - 1 + len(future), color=PALETTE[3], alpha=0.08)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_axis))
    ax.legend(frameon=False, labelcolor=FG, loc="upper left")
    sparse_ticks(ax, labels)
    fig.tight_layout()
    return fig


# =====================================================================
#  PART 5: THE WEB PAGE
# =====================================================================
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg, #4F46E5 0%, #0EA5E9 100%); padding: 28px 34px;
       border-radius: 20px; color: #fff; margin-bottom: 22px;}
.hero h1 {margin: 0; font-size: 2.1rem; color: #fff; padding: 0;}
.hero p {margin: 8px 0 0 0; font-size: 1.02rem; opacity: .92;}
.kpi {background: linear-gradient(145deg, #1E2433, #161B26); border: 1px solid #2A3142;
      border-radius: 16px; padding: 18px 20px; height: 118px;}
.kpi .label {color: #9AA4B8; font-size: .75rem; text-transform: uppercase; letter-spacing: .08em;}
.kpi .value {color: #FFFFFF; font-size: 1.85rem; font-weight: 700; margin-top: 4px; line-height: 1.2;}
.kpi .sub {font-size: .85rem; margin-top: 4px;}
.kpi .up {color: #34D399;} .kpi .down {color: #F87171;} .kpi .flat {color: #9AA4B8;}
.insight {background: #1B2130; border-left: 4px solid #5B8DEF; border-radius: 10px;
          padding: 12px 18px; margin-bottom: 10px; color: #E6EAF2;}
.insight small {color: #9AA4B8; display: block; margin-top: 3px;}
.step {background: #1B2130; border: 1px solid #2A3142; border-radius: 16px; padding: 20px; height: 150px; color: #E6EAF2;}
.step b {font-size: 1.05rem;}
.step span {color: #9AA4B8; font-size: .9rem;}
.stTabs [data-baseweb="tab"] {font-size: 1rem; padding: 10px 18px;}
</style>
"""


def kpi(label, value, sub="", tone="flat"):
    return (f'<div class="kpi"><div class="label">{html.escape(str(label))}</div>'
            f'<div class="value">{html.escape(str(value))}</div>'
            f'<div class="sub {tone}">{html.escape(str(sub))}</div></div>')


def pick_index(options, value):
    return options.index(value) if value in options else 0


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


def render_ai_tab(df, date_col, product_col, sales_col, use_claude, api_key):
    st.markdown("Ask me anything about your sales data. I explain how I found each answer.")
    if use_claude and api_key:
        st.caption("Claude AI mode is ON. Only a short summary of your data is sent, not the whole file.")
    elif use_claude:
        st.caption("Claude AI mode needs an API key in the sidebar. Using the built-in assistant for now.")

    chat_box = st.container()
    clicked = None
    suggestions = example_questions(df, date_col, product_col, sales_col)[:4]
    cols = st.columns(len(suggestions))
    for i, (col, s) in enumerate(zip(cols, suggestions)):
        if col.button(s, key=f"suggest_{i}"):
            clicked = s

    with st.form("ask_form", clear_on_submit=True):
        typed = st.text_input("Your question", placeholder="Example: which product is best?")
        sent = st.form_submit_button("Ask")

    question = clicked or (typed.strip() if sent and typed.strip() else None)
    if question:
        st.session_state.chat.append({"role": "user", "content": question, "table": None})
        answer, table = None, None
        if use_claude and api_key:
            try:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
                answer = ask_claude(api_key, history, df, date_col, product_col, sales_col)
            except Exception as e:
                st.warning(f"Claude AI could not answer ({e}). Using the built-in assistant instead.")
        if answer is None:
            answer, table = ask_offline(question, df, date_col, product_col, sales_col)
        st.session_state.chat.append({"role": "assistant", "content": answer, "table": table})

    with chat_box:
        if not st.session_state.chat:
            st.info("No questions yet. Click a suggestion below or type your own.")
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("table") is not None:
                    st.dataframe(m["table"], hide_index=True)

    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat = []
        st.rerun()


def main():
    st.set_page_config(page_title="Sales Analysis Agent", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>📊 Sales Analysis Agent</h1>'
                '<p>Upload your sales file. I find the columns, draw the charts, predict the future '
                'and answer your questions.</p></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])
    if uploaded is None:
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="step"><b>1. Upload</b><br><span>Drop an Excel or CSV file with a Sales '
                    'column. Date and Product columns make it even better.</span></div>', unsafe_allow_html=True)
        c2.markdown('<div class="step"><b>2. Analyze</b><br><span>I find the best products and months, '
                    'draw charts and predict the next months.</span></div>', unsafe_allow_html=True)
        c3.markdown('<div class="step"><b>3. Ask AI</b><br><span>Chat with your data: "Which product is '
                    'best?", "Predict next month".</span></div>', unsafe_allow_html=True)
        return

    try:
        df = load_file(uploaded)
    except Exception as e:
        st.error(f"I could not read this file: {e}")
        return
    df = df.dropna(how="all")
    if df.empty:
        st.error("The file is empty.")
        return

    if st.session_state.get("file_name") != uploaded.name:
        st.session_state.file_name = uploaded.name
        st.session_state.chat = []
    if "chat" not in st.session_state:
        st.session_state.chat = []

    date_guess, product_guess, sales_guess = detect_columns(df)
    num_cols = list(df.select_dtypes("number").columns)
    if sales_guess is None or not num_cols:
        st.error("I could not find a number column to analyze. Please check your file.")
        return

    # ---- sidebar ----
    none = "(none)"
    st.sidebar.header("⚙️ Columns")
    st.sidebar.caption("I guessed these. Change them if I was wrong.")
    sales_col = st.sidebar.selectbox("Sales (number) column", num_cols, index=pick_index(num_cols, sales_guess))
    date_options = [none] + [c for c in df.columns if c != sales_col]
    date_col = st.sidebar.selectbox("Date column", date_options, index=pick_index(date_options, date_guess))
    date_col = None if date_col == none else date_col
    prod_options = [none] + [c for c in df.columns if c not in (sales_col, date_col)]
    product_col = st.sidebar.selectbox("Product / category column", prod_options,
                                       index=pick_index(prod_options, product_guess))
    product_col = None if product_col == none else product_col

    st.sidebar.header("🤖 AI mode")
    use_claude = st.sidebar.checkbox("Use Claude AI for answers")
    api_key = ""
    if use_claude:
        api_key = st.sidebar.text_input("Anthropic API key", type="password",
                                        help="Needs: pip install anthropic")

    monthly = None
    if date_col:
        monthly = by_month(df, date_col, sales_col)
        if len(monthly) == 0:
            monthly = None

    tabs = st.tabs(["📊 Overview", "📈 Charts", "🔮 Prediction", "🤖 Ask AI", "🗂 Data"])

    # ---- Overview ----
    with tabs[0]:
        total = df[sales_col].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(kpi(f"Total {sales_col}", f"{total:,.0f}", f"{len(df):,} rows"), unsafe_allow_html=True)
        k2.markdown(kpi("Average per row", f"{df[sales_col].mean():,.1f}", f"highest: {df[sales_col].max():,.0f}"),
                    unsafe_allow_html=True)
        if product_col:
            prod = by_group(df, product_col, sales_col)
            k3.markdown(kpi(f"Best {product_col}", prod.index[0], f"{prod.iloc[0] / total * 100:.0f}% of all sales",
                            "up"), unsafe_allow_html=True)
        else:
            k3.markdown(kpi("Best category", "n/a", "no category column"), unsafe_allow_html=True)
        g = growth_pct(monthly) if monthly is not None else None
        if g is not None:
            k4.markdown(kpi("Growth", f"{g:+.0f}%", f"{monthly.index[0]} to {monthly.index[-1]}",
                            "up" if g > 0 else "down"), unsafe_allow_html=True)
        else:
            k4.markdown(kpi("Growth", "n/a", "needs a date column"), unsafe_allow_html=True)

        st.markdown("### What I found")
        for finding, why in make_insights(df, date_col, product_col, sales_col):
            st.markdown(f'<div class="insight">{html.escape(finding)}<small>How: {html.escape(why)}</small></div>',
                        unsafe_allow_html=True)

    # ---- Charts ----
    with tabs[1]:
        if product_col:
            prod = by_group(df, product_col, sales_col)
            left, right = st.columns(2)
            with left:
                show_fig(chart_bars(prod, f"{sales_col} by {product_col}"))
            with right:
                if len(prod) <= 8:
                    show_fig(chart_donut(prod, f"Share of total {sales_col}"))
        if monthly is not None and len(monthly) >= 2:
            show_fig(chart_monthly(monthly, f"{sales_col} by month"))
        extra = [c for c in category_columns(df, date_col, sales_col) if c != product_col]
        extra = [c for c in extra if df[c].nunique() <= 12]
        if extra:
            ex = by_group(df, extra[0], sales_col)
            left, right = st.columns(2)
            with left:
                show_fig(chart_bars(ex, f"{sales_col} by {extra[0]}"))
            with right:
                if len(ex) <= 8:
                    show_fig(chart_donut(ex, f"Share by {extra[0]}"))
        if not product_col and monthly is None:
            st.info("I need a category column or a date column to draw charts.")

    # ---- Prediction ----
    with tabs[2]:
        if monthly is not None and len(monthly) >= 4:
            steps = st.slider("How many months to predict?", 1, 12, 3)
            future, slope = forecast(monthly, steps)
            show_fig(chart_forecast(monthly, future, f"{steps}-month forecast"))
            word = "going up" if slope > 0 else "going down"
            st.markdown(f"**Next month:** {future.iloc[0]:,.0f}  \n"
                        f"**Why:** I drew a straight line through your monthly totals. "
                        f"It changes by about {slope:,.0f} per month, so sales are {word}.")
            st.caption("This is a simple trend line, so treat it as a rough estimate.")
            table = pd.DataFrame({"Month": [str(p) for p in future.index],
                                  f"Predicted {sales_col}": future.round(0).astype(int).values})
            st.dataframe(table, hide_index=True)
            st.download_button("Download prediction (CSV)", table.to_csv(index=False).encode("utf-8"),
                               file_name="prediction.csv", mime="text/csv")
        else:
            st.info("I need a date column with at least 4 months of data to make a prediction.")

    # ---- Ask AI ----
    with tabs[3]:
        render_ai_tab(df, date_col, product_col, sales_col, use_claude, api_key)

    # ---- Data ----
    with tabs[4]:
        st.caption(f"{len(df):,} rows, {len(df.columns)} columns")
        st.dataframe(df, hide_index=True)
        st.markdown("**Quick statistics**")
        st.dataframe(df.describe())


if __name__ == "__main__":
    main()