import streamlit as st
import yfinance as yf
import pandas as pd
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import concurrent.futures
import plotly.graph_objects as go
import io
import re
import json
import os
import requests

# Safely import autorefresh & transformers
try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st.error("Please run: pip install streamlit-autorefresh")

try:
    from transformers import pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    st.error("Missing FinBERT dependencies! Please run: pip install transformers torch")

st.set_page_config(page_title="Stock Market Tracker", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# SESSION STATE & ML MODELS
# -----------------------------
if "last_prices" not in st.session_state:
    st.session_state.last_prices = {}

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

PORTFOLIO_FILE = "portfolio.json"
DIARY_FILE = "diary.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    return {}

def save_portfolio(portfolio_dict):
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(portfolio_dict, f)

def load_diary():
    if os.path.exists(DIARY_FILE):
        with open(DIARY_FILE, "r") as f:
            return json.load(f)
    return []

def save_diary(diary_list):
    with open(DIARY_FILE, "w") as f:
        json.dump(diary_list, f)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

if "diary" not in st.session_state:
    st.session_state.diary = load_diary()

@st.cache_resource
def load_finbert_model():
    if FINBERT_AVAILABLE:
        return pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return None

finbert = load_finbert_model()

# -----------------------------
# UI SIDEBAR
# -----------------------------
st.sidebar.title("🎛️ Control Panel")

st.sidebar.markdown("### ⚙️ Scan Filters")
min_price_option = st.sidebar.selectbox("Minimum Price:", ["No Filter", "> $1", "> $2", "> $5", "> $10", "> $20"])
min_price = float(min_price_option.replace("> $", "").replace("No Filter", "0"))

min_vol_option = st.sidebar.selectbox("Minimum Daily Volume:", ["No Filter", "> 100k", "> 500k", "> 1M"])
vol_map = {"No Filter": 0, "> 100k": 100000, "> 500k": 500000, "> 1M": 1000000}
min_vol = vol_map[min_vol_option]

min_yield = st.sidebar.slider("Minimum Dividend Yield (%)", 0.0, 10.0, 0.0, step=0.5)

st.sidebar.divider()

st.sidebar.markdown("### 🌍 Target Markets")
options = [
    "Manual", "FTSE 100 (UK)", "FTSE 250 (UK)", "FTSE SmallCap (UK)", "DAX 40 (Germany)", "CAC 40 (France)", 
    "IBEX 35 (Spain)", "WIG 20 (Poland)", "FTSE MIB (Italy)", "S&P 500 (US)", 
    "S&P 400 MidCap (US)", "S&P 600 SmallCap (US)", "Nasdaq 100 (US)"
]
selected_markets = st.sidebar.multiselect("Select Indices:", options, default=["FTSE 100 (UK)"])

ticker_input = ""
if "Manual" in selected_markets:
    ticker_input = st.sidebar.text_area("Manual Tickers (comma separated):", "AAPL, MSFT")

st.sidebar.divider()

st.sidebar.markdown("### ⏱️ App Settings")
chart_preference = st.sidebar.radio("Preferred Chart:", ["Candlestick", "Line"], horizontal=True)
refresh_interval = st.sidebar.selectbox("Auto-Refresh:", ["Off", "1 min", "5 mins", "10 mins", "15 mins", "30 mins"])
auto_run_scan = st.sidebar.toggle("Auto-Run Scan on Refresh", value=False)
color_coding = st.sidebar.toggle("Color Code Dataframe", value=True)

if refresh_interval != "Off":
    interval_map = {"1 min": 60, "5 mins": 300, "10 mins": 600, "15 mins": 900, "30 mins": 1800}
    st_autorefresh(interval=interval_map[refresh_interval] * 1000, key="data_refresh")
    st.sidebar.success(f"Auto-refresh active: {refresh_interval}")

if st.sidebar.button("🔄 Clear Cache & Restart"):
    st.cache_data.clear()
    st.session_state.last_prices = {}
    st.session_state.scan_results = None
    st.rerun()

# -----------------------------
# DYNAMIC INDEX SCRAPER
# -----------------------------
@st.cache_data(ttl=86400)
def get_index_constituents(index_name):
    # 1. Local File Scraper for WIG 20
    if index_name == "WIG 20 (Poland)":
        if os.path.exists("wig20.txt"):
            valid_pairs = []
            with open("wig20.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if ',' in line:
                        ticker, name = line.strip().split(',', 1)
                        valid_pairs.append((ticker.strip(), name.strip()))
            if valid_pairs: return valid_pairs
        return [("AAPL", "Apple - Missing wig20.txt")]
        
    # 2. Local File Scraper for FTSE SmallCap
    if index_name == "FTSE SmallCap (UK)":
        if os.path.exists("ftse_smallcap.txt"):
            valid_pairs = []
            with open("ftse_smallcap.txt", "r", encoding="utf-8") as f:
                for line in f:
                    if ',' in line:
                        ticker, name = line.strip().split(',', 1)
                        valid_pairs.append((ticker.strip(), name.strip()))
            if valid_pairs: return valid_pairs
        return [("AAPL", "Apple - Missing ftse_smallcap.txt")]

    # 3. Standard Wikipedia Scraper for remaining indices
    index_map = {
        "FTSE 100 (UK)": ('https://en.wikipedia.org/wiki/FTSE_100_Index', '.L'),
        "FTSE 250 (UK)": ('https://en.wikipedia.org/wiki/FTSE_250_Index', '.L'),
        "DAX 40 (Germany)": ('https://en.wikipedia.org/wiki/DAX', '.DE'),
        "CAC 40 (France)": ('https://en.wikipedia.org/wiki/CAC_40', '.PA'),
        "IBEX 35 (Spain)": ('https://en.wikipedia.org/wiki/IBEX_35', '.MC'),
        "FTSE MIB (Italy)": ('https://en.wikipedia.org/wiki/FTSE_MIB', '.MI'),
        "S&P 500 (US)": ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', ''),
        "S&P 400 MidCap (US)": ('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies', ''),
        "S&P 600 SmallCap (US)": ('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies', ''),
        "Nasdaq 100 (US)": ('https://en.wikipedia.org/wiki/Nasdaq-100', '')
    }
    
    url, suffix = index_map.get(index_name, ('', ''))
    if not url: return [("AAPL", "Apple")]

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=10).read()
        tables = pd.read_html(io.StringIO(html.decode('utf-8')))
        
        for table in tables:
            if isinstance(table.columns, pd.MultiIndex): table.columns = table.columns.get_level_values(0)
            
            ticker_col = next((col for col in table.columns if any(x in str(col).lower() for x in ['ticker', 'symbol', 'code', 'epic'])), None)
            
            if ticker_col and len(table) > 10:
                raw_tickers = table[ticker_col].astype(str).str.strip()
                raw_tickers = raw_tickers.apply(lambda x: re.sub(r'[^A-Za-z0-9.-]', '', x))
                
                name_col = next((col for col in table.columns if any(x in str(col).lower() for x in ['company', 'security', 'name'])), None)
                names = table[name_col].astype(str) if name_col else raw_tickers
                
                valid_pairs = []
                for t, n in zip(raw_tickers, names):
                    if not t or t.lower() == 'nan': continue
                    if suffix == '.L': t = t.replace('.', '-')
                    if suffix and not t.endswith(suffix): t = f"{t}{suffix}"
                    valid_pairs.append((t, n))
                    
                if valid_pairs: return valid_pairs
    except Exception:
        pass
        
    return [("AAPL", "Apple")]

# -----------------------------
# INDICATORS & FETCHING
# -----------------------------
@st.cache_data(ttl=60) 
def get_price_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")
        if df.empty: return df
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
        if not df.empty and 'Close' in df.columns and 'Open' in df.columns: 
            return df.dropna(subset=['Close', 'Open'])
    except:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_stage2_data(ticker, company_name):
    clean_name = re.sub(r'\b(Inc\.|Corp\.|plc|S\.A\.|SE|AG)\b', '', company_name, flags=re.IGNORECASE).strip()
    if clean_name == "Unknown": clean_name = ticker.split('.')[0]
    url = f"https://news.google.com/rss/search?q={urllib.parse.quote(f'{clean_name} stock')}&hl=en-US&gl=US&ceid=US:en"
    articles = []
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        root = ET.fromstring(urllib.request.urlopen(req, timeout=3).read())
        for item in root.findall('.//item')[:5]:
            articles.append({"title": item.find('title').text, "published": parsedate_to_datetime(item.find('pubDate').text)})
    except:
        pass

    short_pct = 0.0
    try:
        info = yf.Ticker(ticker).info
        short_pct = info.get('shortPercentOfFloat', 0) or 0.0
    except:
        pass
        
    return articles, short_pct * 100 

def calculate_rsi(price_series, period=14):
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(price_series):
    ema_12 = price_series.ewm(span=12, adjust=False).mean()
    ema_26 = price_series.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

def calculate_bollinger(price_series, window=20):
    sma = price_series.rolling(window).mean()
    std = price_series.rolling(window).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, lower

# -----------------------------
# MASTER ALGORITHM ENGINES
# -----------------------------
def analyze_technical_metrics(df):
    if df.empty or len(df) < 200: 
        return 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, []
    
    receipt = []
    c, o, v = df["Close"].squeeze(), df["Open"].squeeze(), df["Volume"].squeeze()
    y_c, t_o, t_v, t_c = float(c.iloc[-2]), float(o.iloc[-1]), float(v.iloc[-1]), float(c.iloc[-1])
    avg_v = float(v.iloc[-11:-1].mean())
    sma_50 = float(c.rolling(50).mean().iloc[-2])
    sma_200 = float(c.rolling(200).mean().iloc[-2])

    rsi = calculate_rsi(c).iloc[-1]
    macd_line, macd_signal, macd_hist = calculate_macd(c)
    bb_upper, bb_lower = calculate_bollinger(c)
    
    gap_pct = (t_o - y_c) / y_c if y_c > 0 else 0
    gap_score = 5 if gap_pct > 0.08 else 4 if gap_pct > 0.05 else 2 if gap_pct > 0.02 else -5 if gap_pct < -0.08 else -4 if gap_pct < -0.05 else -2 if gap_pct < -0.02 else 0
    if gap_score != 0: receipt.append(f"**{'+' if gap_score > 0 else ''}{gap_score} pts**: Gap Size ({gap_pct*100:.2f}%)")

    vol_spike = t_v / avg_v if avg_v > 0 else 1.0
    vol_base = 6 if vol_spike > 5.0 else 4 if vol_spike > 3.0 else 2 if vol_spike > 1.5 else 0
    vol_score = vol_base if gap_pct >= 0 else -vol_base
    if vol_score != 0: receipt.append(f"**{'+' if vol_score > 0 else ''}{vol_score} pts**: Volume Spike ({vol_spike:.1f}x)")

    trend_score = 0
    brk_status = "None"
    if y_c < sma_50 and t_o > sma_50:
        trend_score += 3
        brk_status = "Bull 50"
        receipt.append("**+3 pts**: Bullish 50 SMA Breakout")
    elif y_c > sma_50 and t_o < sma_50:
        trend_score -= 3
        brk_status = "Bear 50"
        receipt.append("**-3 pts**: Bearish 50 SMA Breakdown")
    
    if t_c > sma_200: 
        trend_score += 2
        receipt.append("**+2 pts**: Above 200 SMA Trend")
    else: 
        trend_score -= 2
        receipt.append("**-2 pts**: Below 200 SMA Trend")

    core_score = gap_score + vol_score + trend_score

    macd_score = 0
    macd_status = "Neutral"
    if macd_line.iloc[-1] > macd_signal.iloc[-1] and macd_line.iloc[-2] <= macd_signal.iloc[-2]: 
        macd_score += 2; macd_status = "Bull Cross"
    elif macd_line.iloc[-1] < macd_signal.iloc[-1] and macd_line.iloc[-2] >= macd_signal.iloc[-2]: 
        macd_score -= 2; macd_status = "Bear Cross"
    
    if macd_score != 0: receipt.append(f"**{'+' if macd_score > 0 else ''}{macd_score} pts**: MACD ({macd_status})")

    if macd_hist.iloc[-1] > macd_hist.iloc[-2] > 0: 
        macd_score += 2
        receipt.append("**+2 pts**: MACD Bullish Histogram Growth")
    elif macd_hist.iloc[-1] < macd_hist.iloc[-2] < 0: 
        macd_score -= 2
        receipt.append("**-2 pts**: MACD Bearish Histogram Drop")

    bb_score = 0
    bb_status = "Inside Bands"
    if t_c > bb_upper.iloc[-1]: 
        bb_score += 4; bb_status = "Upper Breakout"
        receipt.append("**+4 pts**: Bollinger Upper Breakout")
    elif t_c < bb_lower.iloc[-1]: 
        bb_score -= 4; bb_status = "Lower Breakdown"
        receipt.append("**-4 pts**: Bollinger Lower Breakdown")

    rsi_score = 0
    if vol_spike > 3.0 and bb_status == "Upper Breakout":
        if rsi > 70: 
            rsi_score = 3
            receipt.append("**+3 pts**: RSI Overbought (Ignored due to Squeeze Regime)")
    elif vol_spike > 3.0 and bb_status == "Lower Breakdown":
        if rsi < 30: 
            rsi_score = -3
            receipt.append("**-3 pts**: RSI Oversold (Ignored due to Panic Regime)")
    else:
        if rsi < 30: 
            rsi_score = 3
            receipt.append(f"**+3 pts**: RSI Oversold ({rsi:.1f})")
        elif rsi < 40: 
            rsi_score = 1
            receipt.append(f"**+1 pt**: RSI Cooling ({rsi:.1f})")
        elif rsi > 70: 
            rsi_score = -3
            receipt.append(f"**-3 pts**: RSI Overbought ({rsi:.1f})")
        elif rsi > 60: 
            rsi_score = -1
            receipt.append(f"**-1 pt**: RSI Heating Up ({rsi:.1f})")

    osc_score = macd_score + bb_score + rsi_score

    wk = (t_c - float(c.iloc[-6])) / float(c.iloc[-6])
    mo = (t_c - float(c.iloc[-22])) / float(c.iloc[-22])
    mom_score = 0
    if wk > 0.05: 
        mom_score += 2
        receipt.append("**+2 pts**: 1-Week Momentum (>5%)")
    elif wk < -0.05: 
        mom_score -= 2
        receipt.append("**-2 pts**: 1-Week Momentum (< -5%)")
    
    if mo > 0.10: 
        mom_score += 2
        receipt.append("**+2 pts**: 1-Month Momentum (>10%)")
    elif mo < -0.10: 
        mom_score -= 2
        receipt.append("**-2 pts**: 1-Month Momentum (< -10%)")

    return core_score, osc_score, mom_score, sma_50, rsi, vol_spike, gap_pct, brk_status, macd_status, bb_status, receipt

def process_ticker(ticker, company_name, p_min, v_min, min_yield_filter, last_price_memory):
    df = get_price_data(ticker)
    if df.empty or len(df) < 200: return None
    
    latest_close = float(df["Close"].squeeze().iloc[-1])
    avg_vol = float(df["Volume"].squeeze().iloc[-11:-1].mean())
    if latest_close < p_min or avg_vol < v_min: return None

    try:
        if 'Dividends' in df.columns:
            annual_dividend = float(df['Dividends'].sum())
            yield_pct = (annual_dividend / latest_close) * 100 if latest_close > 0 else 0.0
        else: yield_pct = 0.0
    except: yield_pct = 0.0
        
    if min_yield_filter > 0 and yield_pct < min_yield_filter: return None

    if last_price_memory == 0.0: last_price_memory = latest_close

    core_score, osc_score, mom_score, sma_50, rsi, vol, gap, brk, macd_st, bb_st, receipt = analyze_technical_metrics(df)
    
    cat_score = mom_score
    short_val = 0.0
    sent_label = "Neutral"
    upc = "None"
    news = []

    if abs(core_score + osc_score) >= 4:
        news, short_val = fetch_stage2_data(ticker, company_name)
        
        if short_val > 10.0:
            if latest_close > sma_50: 
                cat_score += 4
                receipt.append(f"**+4 pts**: Squeeze Setup (>10% Short + Above 50 SMA)")
            elif latest_close < sma_50: 
                cat_score -= 4
                receipt.append(f"**-4 pts**: Short Breakdown (>10% Short + Below 50 SMA)")

        if news and finbert:
            pos_c, neg_c = 0, 0
            for a in news:
                try:
                    res = finbert(a["title"])[0]
                    if res['label'] == 'positive': pos_c += 1
                    elif res['label'] == 'negative': neg_c += 1
                except: pass
            
            sent_s = 3 if pos_c > neg_c else -3 if neg_c > pos_c else 0
            sent_label = "Positive" if sent_s > 0 else "Negative" if sent_s < 0 else "Neutral"
            cat_score += sent_s
            if sent_s != 0: receipt.append(f"**{'+' if sent_s > 0 else ''}{sent_s} pts**: AI News Sentiment ({sent_label})")

            now = datetime.now(timezone.utc)
            for a in news:
                t, d = a["title"].lower(), (now - a["published"]).days if a["published"] else 10
                if any(k in t for k in ["results","earnings","update"]) and d <= 5: 
                    bonus = 2 if sent_s >= 0 else -2
                    cat_score += bonus
                    receipt.append(f"**{'+' if bonus > 0 else ''}{bonus} pts**: Recent Earnings/Update Catalyst")
                    break
                    
                if any(e in t for e in ["results", "earnings"]) and any(f in t for f in ["upcoming", "expected", "tomorrow"]): 
                    upc = "Upcoming Event"

    total_score = core_score + osc_score + cat_score
    
    if total_score >= 25: label = "🔥 PRIME BULL"
    elif total_score >= 15: label = "🟢 SWING BULL"
    elif total_score >= 8:  label = "🟡 TREND BULL"
    elif total_score <= -25: label = "🩸 PRIME BEAR"
    elif total_score <= -15: label = "🔴 SWING BEAR"
    elif total_score <= -8:  label = "🟠 TREND BEAR"
    else: label = "⚪ NEUTRAL"

    return {
        "Signal": label, "Ticker": ticker, "Company": company_name, "Total Score": total_score,
        "Core Tech Score": core_score, "Oscillator Score": osc_score, "Catalyst Score": cat_score,
        "Price ($)": latest_close, "Gap %": gap * 100, "Vol Spike (x)": vol, "RSI": rsi, 
        "Short Int %": short_val, "Yield %": yield_pct, "MACD Status": macd_st, "BB Status": bb_st,
        "Breakout": brk, "AI Sentiment": sent_label, "Upcoming Event": upc,
        "Score Receipt": receipt,
        "Headlines": [n["title"] for n in news] if news else ["Skipped AI: Insufficient technical movement"]
    }

def generate_mini_chart(df, ticker, company_name, chart_type):
    df_chart = df.tail(30)
    fig = go.Figure()
    if chart_type == "Candlestick": fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'].squeeze(), high=df_chart['High'].squeeze(), low=df_chart['Low'].squeeze(), close=df_chart['Close'].squeeze()))
    else: fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'].squeeze(), mode='lines', line=dict(color='#00ff88', width=2)))
    fig.update_layout(title=f"{company_name} ({ticker})", margin=dict(l=0, r=0, t=30, b=0), height=300, xaxis_rangeslider_visible=False, template="plotly_dark")
    return fig

# -----------------------------
# SAFE DATAFRAME STYLING
# -----------------------------
def apply_dataframe_styling(df, active):
    if not active: return df
    
    def color_gen(val):
        if isinstance(val, (int, float)):
            if val > 0: return 'color: #00FF00'
            elif val < 0: return 'color: #FF4B4B'
        return ''
        
    def color_rsi(val):
        if pd.isna(val): return ''
        if val < 30: return 'color: #00FF00; font-weight: bold'
        elif val > 70: return 'color: #FF4B4B; font-weight: bold'
        return ''
        
    def color_vol(val):
        if pd.isna(val): return ''
        if val >= 1.5: return 'color: #00FF00'
        elif val <= 0.5: return 'color: #FF4B4B'
        return ''

    target_gen_cols = ["Gap %", "Total Score", "Core Tech Score", "Oscillator Score", "Catalyst Score"]
    existing_gen_cols = [col for col in target_gen_cols if col in df.columns]
    
    styler = df.style
    if existing_gen_cols:
        styler = styler.map(color_gen, subset=existing_gen_cols)
        
    if "RSI" in df.columns:
        styler = styler.map(color_rsi, subset=["RSI"])
        
    if "Vol Spike (x)" in df.columns:
        styler = styler.map(color_vol, subset=["Vol Spike (x)"])
        
    return styler

col_config_settings = {
    "Total Score": st.column_config.NumberColumn(format="%d"),
    "Core Tech Score": st.column_config.NumberColumn(format="%d"),
    "Oscillator Score": st.column_config.NumberColumn(format="%d"),
    "Catalyst Score": st.column_config.NumberColumn(format="%d"),
    "Price ($)": st.column_config.NumberColumn(format="%.2f"),
    "Gap %": st.column_config.NumberColumn(format="%.2f"),
    "Vol Spike (x)": st.column_config.NumberColumn(format="%.2f"),
    "RSI": st.column_config.NumberColumn(format="%.1f"), 
    "Short Int %": st.column_config.NumberColumn(format="%.1f"),
    "Yield %": st.column_config.NumberColumn(format="%.2f")
}

display_cols_main = ["Track", "Signal", "Ticker", "Company", "Total Score", "Core Tech Score", "Oscillator Score", "Catalyst Score", "Price ($)", "Gap %", "Yield %"]
display_cols_detailed = ["Vol Spike (x)", "RSI", "MACD Status", "BB Status", "Short Int %", "Breakout", "AI Sentiment", "Upcoming Event"]

# -----------------------------
# MAIN APP BUILDER
# -----------------------------
st.title("🌍 Stock Market Tracker")
st.markdown("40-Point Master Algorithm | Regime Filtering | Double-Edged Short Squeeze Engine")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Live Deep Scanner", "📰 Global Sentiment", "📈 Yahoo Movers", "💼 Paper Portfolio", "📓 My Diary"])

target_list = []
if "Manual" in selected_markets and ticker_input:
    target_list.extend([(t.strip().upper(), "Unknown") for t in ticker_input.split(",") if t.strip()])
for market in selected_markets:
    if market != "Manual": target_list.extend(get_index_constituents(market))
final_target_list = list(dict(target_list).items())

# ==========================================
# TAB 1: GAP SCANNER
# ==========================================
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1: st.write(f"Ready to scan **{len(final_target_list)}** unique stocks.")
    with col2: run_scan = st.button("🚀 Run Deep Scan", type="primary", use_container_width=True)

    if (run_scan or auto_run_scan) and len(final_target_list) > 0:
        results, progress_bar, status_text = [], st.progress(0), st.empty()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(process_ticker, t[0], t[1], min_price, min_vol, min_yield, st.session_state.last_prices.get(t[0], 0.0)): t[0] for t in final_target_list}
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                ticker = futures[future]
                try:
                    res = future.result(timeout=15) 
                    if res: 
                        st.session_state.last_prices[res["Ticker"]] = res["Price ($)"]
                        results.append(res)
                except concurrent.futures.TimeoutError:
                    st.toast(f"⚠️ Skipped {ticker} (Yahoo Finance timed out)")
                except Exception: pass
                progress_bar.progress((i + 1) / len(final_target_list))
                status_text.text(f"Processed {i+1} / {len(final_target_list)}...")
                
        status_text.empty()
        progress_bar.empty()
        
        if results:
            df_results = pd.DataFrame(results)
            df_results = df_results.drop_duplicates(subset=['Ticker'])
            df_results['Abs_Score'] = df_results['Total Score'].abs()
            df_results = df_results.sort_values(by="Abs_Score", ascending=False).drop(columns=['Abs_Score'])
            st.session_state.scan_results = df_results
        else:
            st.session_state.scan_results = pd.DataFrame()
            st.warning("No stocks passed your minimum price, volume, and yield filters.")

    if st.session_state.scan_results is not None and not st.session_state.scan_results.empty:
        df_results = st.session_state.scan_results.copy()
        
        p_bull = len(df_results[df_results['Signal'] == "🔥 PRIME BULL"])
        s_bull = len(df_results[df_results['Signal'] == "🟢 SWING BULL"])
        t_bull = len(df_results[df_results['Signal'] == "🟡 TREND BULL"])
        
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Passing Filters", len(df_results))
        mc2.metric("🔥 PRIME BULLS", p_bull)
        mc3.metric("🟢 SWING BULLS", s_bull)
        mc4.metric("🟡 TREND BULLS", t_bull)
        
        df_results.insert(0, 'Track', df_results['Ticker'].apply(lambda t: t in st.session_state.portfolio))
        
        edited_df = st.data_editor(
            apply_dataframe_styling(df_results[display_cols_main], color_coding),
            use_container_width=True,
            hide_index=True,
            column_config={
                **col_config_settings,
                "Track": st.column_config.CheckboxColumn("Track", help="Select to add to Paper Portfolio")
            },
            disabled=[col for col in display_cols_main if col != "Track"],
            key="main_table_editor"
        )
        
        portfolio_changed = False
        for _, row in edited_df.iterrows():
            ticker = row['Ticker']
            is_tracked = row['Track']
            was_tracked = ticker in st.session_state.portfolio
            
            if is_tracked and not was_tracked:
                st.session_state.portfolio[ticker] = {
                    "Company": row['Company'],
                    "Entry Price": row['Price ($)'],
                    "Signal": row['Signal'],
                    "Date Added": datetime.now().strftime("%Y-%m-%d"),
                    "Amount": 25.0
                }
                portfolio_changed = True
            elif not is_tracked and was_tracked:
                del st.session_state.portfolio[ticker]
                portfolio_changed = True
                
        if portfolio_changed:
            save_portfolio(st.session_state.portfolio)
        
        st.divider()
        st.subheader("📰 Deep Dive: Actionable Setups")
        
        for index, row in df_results[df_results["Total Score"].abs() >= 8].iterrows():
            with st.expander(f"{row['Signal']} | {row['Company']} ({row['Ticker']}) - Score: {row['Total Score']}"):
                
                df_single = pd.DataFrame([row])[display_cols_detailed]
                st.dataframe(apply_dataframe_styling(df_single, color_coding), hide_index=True, use_container_width=True, column_config=col_config_settings)

                st.markdown("##### 🧮 Score Receipt (Transparency Engine)")
                c_rec1, c_rec2 = st.columns(2)
                half_list = len(row["Score Receipt"]) // 2 + 1
                with c_rec1:
                    for item in row["Score Receipt"][:half_list]: st.markdown(f"- {item}")
                with c_rec2:
                    for item in row["Score Receipt"][half_list:]: st.markdown(f"- {item}")
                
                st.divider()

                c1, c2 = st.columns([1, 1.5])
                with c1:
                    st.markdown("### Latest News")
                    for h in row["Headlines"][:4]: st.markdown(f"- {h}")
                with c2:
                    if not (df_chart := get_price_data(row['Ticker'])).empty:
                        st.plotly_chart(generate_mini_chart(df_chart, row['Ticker'], row['Company'], chart_preference), use_container_width=True)

# ==========================================
# TAB 2 & 3: GLOBAL SENTIMENT & YAHOO MOVERS 
# ==========================================
with tab2:
    st.subheader("📰 Global FinBERT Sentiment")
    st.write("Processing top global headlines...")
    if st.button("🔄 Generate Report") and FINBERT_AVAILABLE:
        st.info("Run scan to trigger sentiment mapping pipeline.")

with tab3:
    st.subheader("📈 Analyze Yahoo Finance Top Movers")
    mover_category = st.selectbox("Select Category:", ["gainers", "losers", "active"])
    if st.button("🔄 Fetch & Analyze Movers"):
        st.info("Run scan to trigger movers pipeline.")

# ==========================================
# TAB 4: PAPER PORTFOLIO
# ==========================================
with tab4:
    st.subheader("💼 Active Paper Trades & Watchlist")
    
    if not st.session_state.portfolio:
        st.info("Your portfolio is empty. Check the 'Track' box in the Live Deep Scanner table to add a stock here.")
    else:
        # --- 1. INTERCEPT EDITS BEFORE RENDERING THE UI ---
        if "portfolio_editor" in st.session_state:
            edits = st.session_state["portfolio_editor"].get("edited_rows", {})
            if edits and "port_row_map" in st.session_state:
                port_changed = False
                for row_idx_str, changes in edits.items():
                    row_idx = int(row_idx_str)
                    # Match the row you edited to the correct Ticker symbol
                    if row_idx < len(st.session_state.port_row_map):
                        ticker = st.session_state.port_row_map[row_idx]
                        port_item = st.session_state.portfolio[ticker]
                        
                        if "Status" in changes:
                            port_item["Status"] = changes["Status"]
                            port_changed = True
                        if "Entry Price" in changes:
                            port_item["Entry Price"] = float(changes["Entry Price"])
                            port_changed = True
                        if "Invested (£)" in changes:
                            port_item["Amount"] = float(changes["Invested (£)"])
                            port_changed = True
                            
                # Lock it directly into your JSON file
                if port_changed:
                    import json
                    try:
                        with open("portfolio.json", "w") as f:
                            json.dump(st.session_state.portfolio, f)
                    except Exception as e:
                        st.error(f"Save failed: {e}")
        # --------------------------------------------------

        # 2. Clean up old data structures seamlessly
        for tick, data in st.session_state.portfolio.items():
            if "Status" not in data: data["Status"] = "Watching"
            if "Amount" not in data: data["Amount"] = 0.0
            if "Entry Price" not in data: data["Entry Price"] = 0.0

        port_row_map = []
        portfolio_data = []
        total_invested = 0.0
        current_value = 0.0
        
        # 3. Build the live dataframe
        for tick, data in st.session_state.portfolio.items():
            port_row_map.append(tick) # Map the row index for the editor interceptor
            
            company_name = data.get("Company", tick)
            
            try:
                # Call your master function directly! 
                # We pass 0 for all filters so we don't accidentally hide stocks you own
                result = process_ticker(tick, company_name, p_min=0, v_min=0, min_yield_filter=0, last_price_memory=0.0)
                
                if result is not None:
                    live_signal = result["Signal"]
                    live_price = float(result["Price ($)"])
                else:
                    # Failsafe if the stock has < 200 days of data
                    live_price = float(data.get('Entry Price', 0.0))
                    live_signal = "⚪ NEUTRAL (No Data)"
                    
            except Exception as e:
                live_price = float(data.get('Entry Price', 0.0))
                live_signal = "Error"
                
            entry_price = float(data.get('Entry Price', 0.0))
            pnl_pct = ((live_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
            invested_amount = float(data.get('Amount', 0.0))
            
            # Calculate live values only if you actually own it
            if data["Status"] == "Owned":
                current_pos_val = invested_amount * (1 + (pnl_pct / 100))
                total_invested += invested_amount
                current_value += current_pos_val
            else:
                current_pos_val = 0.0
            
            portfolio_data.append({
                "Ticker": tick,
                "Status": data["Status"],
                "Date": data.get("Date Added", ""),
                "Signal": live_signal,  # <-- Now pulls "🔥 PRIME BULL", etc.
                "Company": company_name,
                "Entry Price": entry_price,
                "Live Price": live_price,
                "Invested (£)": invested_amount,
                "Current Val (£)": current_pos_val,
                "P&L %": pnl_pct
            })
            
        # --- CONVERT TO DATAFRAME & DEFINE COLORS ---
        
        # Save the row map to session state so the editor interceptor works perfectly
        st.session_state["port_row_map"] = port_row_map 
        
        # Convert list to a Pandas DataFrame
        df_port = pd.DataFrame(portfolio_data)
        
        # Define the coloring function
        def color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: #00FF00'  # Bright green
                elif val < 0:
                    return 'color: #FF4B4B'  # Streamlit red
            return ''
        # --------------------------------------------
            
        # 4. Render the interactive table
        st.data_editor(
            df_port.style.map(color_pnl, subset=["P&L %", "Current Val (£)"]), 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Status": st.column_config.SelectboxColumn("Status", options=["Watching", "Owned"], required=True),
                "Entry Price": st.column_config.NumberColumn("Entry Price", format="%.2f"),
                "Live Price": st.column_config.NumberColumn("Live Price", format="%.2f"),
                "Invested (£)": st.column_config.NumberColumn("Invested (£)", format="£%.2f", min_value=0.0),
                "Current Val (£)": st.column_config.NumberColumn("Current Val (£)", format="£%.2f"),
                "P&L %": st.column_config.NumberColumn("P&L %", format="%.2f%%")
            },
            disabled=["Ticker", "Company", "Live Price", "Current Val (£)", "P&L %", "Signal", "Date"],
            key="portfolio_editor"
        )

# ==========================================
# TAB 5: MY DIARY
# ==========================================
with tab5:
    st.subheader("📓 My Trading Diary")
    
    c_diary1, c_diary2 = st.columns([1, 2.5])
    
    with c_diary1:
        st.markdown("##### Log a Closed Trade")
        with st.form("diary_form", clear_on_submit=True):
            d_date = st.date_input("Date Closed", value=datetime.today())
            d_ticker = st.text_input("Ticker (e.g., AAPL, MTRO.L)").upper()
            d_pnl = st.number_input("Realized Profit/Loss (£)", value=0.0, step=10.0)
            d_notes = st.text_input("Notes (Optional)")
            
            submitted = st.form_submit_button("💾 Save to Diary")
            
            if submitted and d_ticker:
                with st.spinner("Fetching company details..."):
                    try:
                        c_name = yf.Ticker(d_ticker).info.get('shortName', d_ticker)
                    except:
                        c_name = d_ticker
                        
                    entry = {
                        "Date": str(d_date),
                        "Ticker": d_ticker,
                        "Company": c_name,
                        "P&L (£)": d_pnl,
                        "Notes": d_notes
                    }
                    st.session_state.diary.append(entry)
                    save_diary(st.session_state.diary)
                    st.rerun()

    with c_diary2:
        st.markdown("##### Performance Graph")
        if st.session_state.diary:
            df_diary = pd.DataFrame(st.session_state.diary)
            
            df_grouped = df_diary.groupby("Date")["P&L (£)"].sum().reset_index()
            df_grouped = df_grouped.sort_values("Date")
            df_grouped["Cumulative P&L (£)"] = df_grouped["P&L (£)"].cumsum()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_grouped["Date"], 
                y=df_grouped["P&L (£)"], 
                name='Daily P&L', 
                marker_color=df_grouped['P&L (£)'].apply(lambda x: '#00FF00' if x >= 0 else '#FF4B4B')
            ))
            fig.add_trace(go.Scatter(
                x=df_grouped["Date"], 
                y=df_grouped["Cumulative P&L (£)"], 
                mode='lines+markers', 
                name='Cumulative Growth', 
                line=dict(color='#00BFFF', width=3)
            ))
            
            fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), height=350, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Log your first trade to generate your performance graph!")

    st.divider()
    st.markdown("##### Trade History")
    
    if st.session_state.diary:
        df_hist = pd.DataFrame(st.session_state.diary)
        df_hist.insert(0, "Delete", False)
        
        def color_diary_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0: return 'color: #00FF00'
                elif val < 0: return 'color: #FF4B4B'
            return ''

        edited_hist = st.data_editor(
            df_hist.style.map(color_diary_pnl, subset=["P&L (£)"]),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Delete": st.column_config.CheckboxColumn("🗑️ Remove", help="Tick to delete this trade"),
                "P&L (£)": st.column_config.NumberColumn(format="£%.2f")
            },
            disabled=["Date", "Ticker", "Company", "P&L (£)", "Notes"],
            key="diary_editor"
        )
        
        if edited_hist["Delete"].any():
            keep_indices = edited_hist[~edited_hist["Delete"]].index.tolist()
            st.session_state.diary = [st.session_state.diary[i] for i in keep_indices]
            save_diary(st.session_state.diary)
            st.rerun()
    else:
        st.write("No closed trades logged yet.")
