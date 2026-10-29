import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
from io import BytesIO

st.set_page_config(page_title="Nexora Backtest & Journal", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.1rem; max-width:1450px}
[data-testid="stMetric"] {background:rgba(120,120,120,.08);padding:13px;border-radius:12px}
</style>
""", unsafe_allow_html=True)

st.title("📊 Nexora Backtest & Journal")
st.caption("Focused on strategy testing and trade journaling • Tablet-friendly • No broker execution")

if "journal" not in st.session_state:
    st.session_state.journal = pd.DataFrame(columns=[
        "Date","Market","Strategy","Side","Entry","Exit","Stop Loss","Take Profit",
        "Size","Fees","P&L","R","Setup Grade","Emotion","Followed Rules","Screenshot/Chart Link","Notes"
    ])
if "bt_result" not in st.session_state:
    st.session_state.bt_result = None

def clean_ohlc(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {"datetime":"date","timestamp":"date","time":"date","open_time":"date",
              "o":"open","h":"high","l":"low","c":"close"}
    df = df.rename(columns=rename)
    required = ["open","high","low","close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing OHLC columns: " + ", ".join(missing))
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df = df.dropna(subset=["date"]).sort_values("date")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 40:
        raise ValueError("Need at least 40 valid candles.")
    return df

def run_ema_backtest(df, start_time, rr, one_per_day, fee_per_unit, slippage, ema_fast, ema_slow):
    x = clean_ohlc(df)
    x["ema_fast"] = x["close"].ewm(span=ema_fast, adjust=False).mean()
    x["ema_slow"] = x["close"].ewm(span=ema_slow, adjust=False).mean()
    x["signal"] = 0
    x.loc[(x["ema_fast"] > x["ema_slow"]) & (x["ema_fast"].shift(1) <= x["ema_slow"].shift(1)), "signal"] = 1
    x.loc[(x["ema_fast"] < x["ema_slow"]) & (x["ema_fast"].shift(1) >= x["ema_slow"].shift(1)), "signal"] = -1
    trades = []
    days = set()
    i = max(ema_slow + 1, 2)
    while i < len(x)-1:
        sig = int(x.loc[i,"signal"])
        if sig == 0:
            i += 1
            continue
        ts = x.loc[i,"date"] if "date" in x.columns else None
        if ts is not None and pd.notna(ts):
            # Input timestamps parsed UTC; session is converted to India time.
            local = ts.tz_convert("Asia/Kolkata")
            day_key = str(local.date())
            if local.strftime("%H:%M") < start_time:
                i += 1
                continue
        else:
            day_key = str(i)
        if one_per_day and day_key in days:
            i += 1
            continue
        entry = float(x.loc[i+1,"open"])
        prev = x.loc[i-1]
        sl = float(prev["low"] if sig == 1 else prev["high"])
        risk = (entry-sl)*sig
        if risk <= 0:
            i += 1
            continue
        tp = entry + sig*rr*risk
        exit_price = float(x.iloc[-1]["close"])
        outcome = "OPEN"
        exit_i = len(x)-1
        exit_time = x.iloc[-1]["date"] if "date" in x.columns else None
        for j in range(i+1,len(x)):
            bar = x.iloc[j]
            hit_sl = bar["low"] <= sl if sig == 1 else bar["high"] >= sl
            hit_tp = bar["high"] >= tp if sig == 1 else bar["low"] <= tp
            # Conservative fill if both targets hit within same candle.
            if hit_sl:
                outcome, exit_price, exit_i = "LOSS", sl, j
                exit_time = bar["date"] if "date" in x.columns else None
                break
            if hit_tp:
                outcome, exit_price, exit_i = "WIN", tp, j
                exit_time = bar["date"] if "date" in x.columns else None
                break
        gross_r = (exit_price-entry)*sig/risk
        # simple per-unit costs; user must calibrate for the symbol/broker
        cost_r = (fee_per_unit + slippage) / risk
        net_r = gross_r - cost_r
        trades.append({
            "Signal time": str(ts) if ts is not None else str(i),
            "Entry time": str(x.loc[i+1,"date"]) if "date" in x.columns else str(i+1),
            "Exit time": str(exit_time),
            "Market side": "BUY" if sig == 1 else "SELL",
            "Entry": entry, "Stop Loss": sl, "Take Profit": tp, "Exit": exit_price,
            "Result": outcome, "Gross R": round(gross_r,3), "Net R": round(net_r,3),
            "Net P&L per unit": round((exit_price-entry)*sig-fee_per_unit-slippage,6),
            "Bars held": exit_i-(i+1)
        })
        if one_per_day:
            days.add(day_key)
        i = exit_i+1
    return x, pd.DataFrame(trades)

def metrics(trades):
    closed = trades[trades["Result"].isin(["WIN","LOSS"])].copy()
    if closed.empty:
        return closed, {"trades":0,"winrate":0.0,"net_r":0.0,"pf":0.0,"maxdd":0.0,"avg_r":0.0}
    vals = closed["Net R"].astype(float)
    wins = vals[vals > 0].sum()
    losses = abs(vals[vals < 0].sum())
    equity = vals.cumsum()
    dd = equity - equity.cummax()
    return closed, {
        "trades":len(closed),
        "winrate":float((vals>0).mean()*100),
        "net_r":float(vals.sum()),
        "pf":float(wins/losses) if losses else (float("inf") if wins else 0.0),
        "maxdd":float(dd.min()),
        "avg_r":float(vals.mean())
    }

def journal_normalize(df):
    df = df.copy()
    aliases={"pnl":"P&L","profit":"P&L","r_multiple":"R","result_r":"R","symbol":"Market",
             "date_time":"Date","direction":"Side","sl":"Stop Loss","tp":"Take Profit",
             "quantity":"Size","commission":"Fees","rule_followed":"Followed Rules"}
    df.columns=[aliases.get(str(c).strip().lower().replace(" ","_"),str(c).strip()) for c in df.columns]
    cols=["Date","Market","Strategy","Side","Entry","Exit","Stop Loss","Take Profit","Size","Fees","P&L","R","Setup Grade","Emotion","Followed Rules","Screenshot/Chart Link","Notes"]
    defaults={"Date":str(date.today()),"Market":"XAUUSD","Strategy":"8/30 EMA crossover","Side":"BUY",
              "Entry":0,"Exit":0,"Stop Loss":0,"Take Profit":0,"Size":1,"Fees":0,"P&L":0,"R":0,
              "Setup Grade":"","Emotion":"Neutral","Followed Rules":"Yes","Screenshot/Chart Link":"","Notes":""}
    for c in cols:
        if c not in df.columns: df[c]=defaults[c]
    for c in ["Entry","Exit","Stop Loss","Take Profit","Size","Fees","P&L","R"]:
        df[c]=pd.to_numeric(df[c],errors="coerce").fillna(0)
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce").dt.strftime("%Y-%m-%d").fillna(str(date.today()))
    df["Side"]=df["Side"].astype(str).str.upper().replace({"LONG":"BUY","SHORT":"SELL"})
    calc=np.where(df["Side"]=="SELL",(df["Entry"]-df["Exit"])*df["Size"],(df["Exit"]-df["Entry"])*df["Size"])-df["Fees"]
    df["P&L"]=np.where(df["P&L"]!=0,df["P&L"],calc)
    risk=(df["Entry"]-df["Stop Loss"]).abs()*df["Size"]
    df["R"]=np.where(df["R"]!=0,df["R"],np.where(risk>0,df["P&L"]/risk,0))
    return df[cols]

with st.sidebar:
    st.header("Nexora TradeLab")
    page=st.radio("Go to",["Backtesting Lab","Trade Journal","Journal Analytics","Strategy Settings","Help & Limits"])
    st.divider()
    st.caption("Historical testing only. No broker connection or order execution.")

if page=="Backtesting Lab":
    st.subheader("Strategy backtesting")
    st.write("Upload OHLC candle CSV for your market/timeframe. Required columns: `Open, High, Low, Close`; a timestamp column named `Date`, `Datetime` or `Timestamp` is strongly recommended.")
    upload=st.file_uploader("Upload historical candle CSV",type=["csv"])
    with st.expander("Backtest rules",expanded=True):
        a,b,c=st.columns(3)
        fast=a.number_input("Fast EMA",min_value=1,max_value=100,value=8)
        slow=b.number_input("Slow EMA",min_value=2,max_value=300,value=30)
        rr=c.selectbox("Reward:risk", [1.0,1.5,2.0,2.5,3.0],index=2)
        a,b,c=st.columns(3)
        session=a.selectbox("Entry allowed after (India time)",["00:00","06:30","12:00","15:00","18:30","19:00","20:00"],index=4)
        one_day=b.checkbox("One trade per day",value=True)
        costs=c.checkbox("Include estimated costs",value=False)
        a,b=st.columns(2)
        fee=a.number_input("Fees per unit (price units)",min_value=0.0,value=0.0,help="Simple approximation, not a broker commission model.")
        slip=b.number_input("Slippage per unit (price units)",min_value=0.0,value=0.0)
    if upload:
        try:
            raw=pd.read_csv(upload)
            st.success(f"Loaded {len(raw):,} raw rows.")
            st.dataframe(raw.head(5),use_container_width=True,hide_index=True)
            if st.button("Run backtest",type="primary",use_container_width=True):
                use_fee=fee if costs else 0.0
                use_slip=slip if costs else 0.0
                candles,result=run_ema_backtest(raw,session,rr,one_day,use_fee,use_slip,int(fast),int(slow))
                st.session_state.bt_result=result
                st.session_state.bt_candles=candles
        except Exception as e:
            st.error(f"CSV issue: {e}")
    if st.session_state.bt_result is not None:
        result=st.session_state.bt_result
        st.markdown("### Results")
        closed,m=metrics(result)
        c=st.columns(5)
        c[0].metric("Closed trades",m["trades"])
        c[1].metric("Win rate",f'{m["winrate"]:.1f}%')
        c[2].metric("Net R",f'{m["net_r"]:.2f}R')
        c[3].metric("Profit factor","∞" if np.isinf(m["pf"]) else f'{m["pf"]:.2f}')
        c[4].metric("Max drawdown",f'{m["maxdd"]:.2f}R')
        if not result.empty:
            st.line_chart(closed["Net R"].cumsum().reset_index(drop=True).rename("Equity in R"))
            st.dataframe(result,use_container_width=True,hide_index=True)
            st.download_button("Download trades CSV",result.to_csv(index=False).encode(),"backtest_trades.csv","text/csv")
            if st.button("Copy backtest trades into journal"):
                imported=result.copy()
                imported["Date"]=pd.to_datetime(imported["Entry time"],errors="coerce",utc=True).dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d")
                imported["Market"]="Imported backtest"
                imported["Strategy"]=f"EMA {fast}/{slow} crossover"
                imported["Side"]=imported["Market side"]
                imported["Size"]=1
                imported["Fees"]=0
                imported["P&L"]=imported["Net P&L per unit"]
                imported["R"]=imported["Net R"]
                imported["Setup Grade"]=""
                imported["Emotion"]="Backtest"
                imported["Followed Rules"]="Yes"
                imported["Screenshot/Chart Link"]=""
                imported["Notes"]="Imported from backtest; price-unit P&L, not account currency."
                st.session_state.journal=journal_normalize(pd.concat([st.session_state.journal,imported],ignore_index=True))
                st.success("Backtest trades copied into the journal.")
        st.warning("Limitations: signal is taken on the next candle open; stop is previous candle's low/high before the signal. If SL and TP hit in the same candle, SL is assumed first. Costs are approximate. This is not tick-level execution.")
elif page=="Trade Journal":
    st.subheader("Trade journal")
    with st.form("journal_add",clear_on_submit=True):
        a,b,c=st.columns(3)
        d=a.date_input("Trade date",value=date.today())
        market=b.selectbox("Market",["XAUUSD","EURUSD","GBPUSD","USDJPY","BTCUSD","ETHUSD","USOIL","XAGUSD","Other"])
        side=c.selectbox("Side",["BUY","SELL"])
        a,b,c=st.columns(3)
        strategy=a.selectbox("Strategy",["8/30 EMA crossover","Breakout","FVG / Order block","Support / Resistance","Other"])
        entry=b.number_input("Entry",value=0.0,format="%.6f")
        exitp=c.number_input("Exit",value=0.0,format="%.6f")
        a,b,c=st.columns(3)
        sl=a.number_input("Stop loss",value=0.0,format="%.6f")
        tp=b.number_input("Take profit",value=0.0,format="%.6f")
        size=c.number_input("Size / units",min_value=0.0,value=1.0)
        a,b,c=st.columns(3)
        fees=a.number_input("Fees in account currency",min_value=0.0,value=0.0)
        emotion=b.selectbox("Emotion",["Neutral","Calm","Fear","Greed","Revenge","FOMO","Overconfident"])
        rules=c.selectbox("Followed rules?",["Yes","Partly","No"])
        grade=st.selectbox("Setup grade",["A","B","C","D","Unrated"])
        link=st.text_input("Screenshot / chart link (optional)")
        notes=st.text_area("Notes: reason, execution, lesson")
        save=st.form_submit_button("Save trade",type="primary",use_container_width=True)
        if save:
            new=pd.DataFrame([{"Date":str(d),"Market":market,"Strategy":strategy,"Side":side,"Entry":entry,"Exit":exitp,"Stop Loss":sl,"Take Profit":tp,"Size":size,"Fees":fees,"Emotion":emotion,"Followed Rules":rules,"Setup Grade":grade,"Screenshot/Chart Link":link,"Notes":notes}])
            st.session_state.journal=journal_normalize(pd.concat([st.session_state.journal,new],ignore_index=True))
            st.success("Trade added to journal.")
            st.rerun()
    st.divider()
    st.markdown("### Journal records")
    if st.session_state.journal.empty:
        st.info("No journal entries yet. Add a trade above or import your existing journal in Strategy Settings.")
    else:
        st.dataframe(st.session_state.journal,use_container_width=True,hide_index=True)
        edited=st.data_editor(st.session_state.journal,num_rows="dynamic",use_container_width=True,key="journal_editor")
        if st.button("Save table edits"):
            st.session_state.journal=journal_normalize(edited)
            st.success("Edits saved.")
            st.rerun()
elif page=="Journal Analytics":
    st.subheader("Journal analytics")
    j=st.session_state.journal.copy()
    if j.empty:
        st.info("Add trades or import a journal CSV to see analytics.")
    else:
        pnl=j["P&L"]
        win=pnl[pnl>0].sum(); loss=abs(pnl[pnl<0].sum())
        a,b,c,d,e=st.columns(5)
        a.metric("Net P&L",f"{pnl.sum():,.2f}")
        b.metric("Win rate",f"{(pnl>0).mean()*100:.1f}%")
        c.metric("Trades",len(j))
        d.metric("Profit factor","∞" if loss==0 and win>0 else (f"{win/loss:.2f}" if loss else "0.00"))
        dts=pd.to_datetime(j["Date"],errors="coerce")
        daily=j.assign(_date=dts).groupby("_date")["P&L"].sum().sort_index()
        eq=daily.cumsum()
        dd=(eq-eq.cummax()).min() if len(eq) else 0
        e.metric("Max drawdown",f"{dd:,.2f}")
        left,right=st.columns(2)
        with left:
            st.markdown("#### Cumulative P&L")
            st.line_chart(eq.rename("Cumulative P&L"))
        with right:
            st.markdown("#### P&L by strategy")
            st.bar_chart(j.groupby("Strategy")["P&L"].sum())
        st.markdown("#### Performance by market")
        st.dataframe(j.groupby("Market").agg(Trades=("P&L","size"),Net_PnL=("P&L","sum"),Avg_PnL=("P&L","mean"),Win_Rate=("P&L",lambda x: (x>0).mean()*100)),use_container_width=True)
        st.markdown("#### Psychology / rule discipline")
        st.dataframe(j.groupby(["Emotion","Followed Rules"]).agg(Trades=("P&L","size"),Net_PnL=("P&L","sum"),Avg_PnL=("P&L","mean")),use_container_width=True)
elif page=="Strategy Settings":
    st.subheader("Import / export journal")
    j=st.session_state.journal
    if not j.empty:
        st.download_button("Download journal backup CSV",j.to_csv(index=False).encode(),"nexora_journal_backup.csv","text/csv",use_container_width=True)
    up=st.file_uploader("Import journal CSV",type=["csv"])
    if up:
        try:
            imp=journal_normalize(pd.read_csv(up))
            st.write(f"Found {len(imp)} rows.")
            if st.button("Replace current journal with imported CSV"):
                st.session_state.journal=imp
                st.success("Journal imported.")
                st.rerun()
        except Exception as e:
            st.error(f"Import error: {e}")
    st.divider()
    st.markdown("#### Strategy rules")
    st.markdown("""
    Default backtest logic:
    - EMA 8/30 crossover on uploaded candle data.
    - Entry at the next candle open after a crossover.
    - Stop-loss at the previous candle's low (BUY) or high (SELL).
    - Take-profit uses selected reward:risk.
    - Optional one trade per India-time day and time-of-day filter.
    """)
elif page=="Help & Limits":
    st.subheader("How to use the app")
    st.markdown("""
    **Backtesting**
    1. Download/export candle history as CSV from a data provider.
    2. CSV must include `Open, High, Low, Close` and preferably `Date` or `Timestamp`.
    3. Upload it in Backtesting Lab, choose settings and tap Run backtest.
    4. Download results or copy trades into the journal.

    **Journal**
    1. Enter each trade's market, direction, entry/exit, SL/TP, size, fees and notes.
    2. Record emotion and whether you followed your rules.
    3. Use Journal Analytics to compare performance by market, setup and psychology.

    **Data and limitations**
    - This prototype runs in a Streamlit session. Download journal backups regularly; data may reset.
    - It does not fetch historical market data automatically.
    - It does not connect to a broker or place demo/live orders.
    - Backtest is candle-based, not tick-based; it does not perfectly model spread, slippage, swaps, execution latency or broker-specific contract sizing.
    - Session filter assumes timestamps in the uploaded file represent UTC; if your CSV timestamps use another timezone, convert them to UTC first.
    - R and P&L are only meaningful if your size, fees and price units are consistent.
    - No strategy guarantees profit. Test out-of-sample and demo before live trading.
    """)
    st.code("Date,Open,High,Low,Close",language="text")

st.divider()
st.caption("Nexora Backtest & Journal • Educational tool • Not financial advice")
