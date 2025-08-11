# -*- coding: utf-8 -*-
import re
import requests
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

# =========================================
# (요청) 대시보드 코드 "위"에 위치할 웹훅 코드
# =========================================
# 웹훅 URL
url = "https://soyso2sohee.app.n8n.cloud/webhook-test/mail"

# 전송할 데이터 (긴 텍스트)
data = {
    "message":
    """여기에 아주 긴 텍스트를 넣으시면 됩니다.
줄바꿈도 가능하고,
문단도 여러 개 작성할 수 있습니다.
예: 첫 문단

두 번째 문단

세 번째 문단...
"""
}

# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 출력
print("응답 코드:", response.status_code)
print("응답 내용:", response.text)
# =========================================

st.set_page_config(page_title="Note Analytics Dashboard", layout="wide")

# (요청) 메인 컨텐츠 시작 시 webhook 버튼 추가
st.title("Note Analytics Dashboard")

if st.button('webhook'):
    data = {
        "message":
        """여기에 아주 긴 텍스트를 넣으시면 됩니다.
    줄바꿈도 가능하고,
    문단도 여러 개 작성할 수 있습니다.
    예: 첫 문단

    두 번째 문단

    세 번째 문단...
    """
    }
    response = requests.post(url, json=data)
    st.write("응답 코드:", response.status_code)
    st.write("응답 내용:", response.text)

# (요청) placeholder 변수
all_file_data = {}

# =========================
# 스타일 (섹션 카드/인사이트 박스)
# =========================
st.markdown("""
<style>
/* 전체 배경 톤 다운 */
.reportview-container, .main, .block-container { background: #fafafa !important; }
/* 섹션 카드 */
.section-card {
  background: #ffffff;
  border: 1px solid #eee;
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
  margin-bottom: 18px;
}
/* 인사이트 박스 */
.insight-card {
  border-left: 8px solid #7c3aed; /* 보라 */
  background: linear-gradient(90deg, rgba(124,58,237,0.10), rgba(124,58,237,0.04));
  border-radius: 14px;
  padding: 14px 16px;
  margin: 6px 0;
}
/* 보조 인사이트 색상 */
.insight-green { border-left-color:#10b981; background: linear-gradient(90deg, rgba(16,185,129,0.10), rgba(16,185,129,0.04)); }
.insight-blue  { border-left-color:#2563eb; background: linear-gradient(90deg, rgba(37,99,235,0.10), rgba(37,99,235,0.04)); }
.insight-amber { border-left-color:#f59e0b; background: linear-gradient(90deg, rgba(245,158,11,0.10), rgba(245,158,11,0.04)); }
/* 소제목 */
h3, h4 { margin-top: 0.4rem; }
</style>
""",
            unsafe_allow_html=True)

# =========================
# Altair 전역 스타일
# =========================
alt.themes.enable('default')
BASE_SCHEME = "tableau10"  # 선명한 분류 색
HEAT_SCHEME = "tealblues"  # 히트맵
AREA_SCHEME = "set2"  # 면적
CAT_SCHEME = "category20"  # 범주 다수

# -------------------------
# 유틸: 문자열에서 월/일/주차 파싱
# -------------------------
re_day = re.compile(r"(\d{1,2})월\s*(\d{1,2})일")
re_week = re.compile(r"(\d{1,2})월?\s*(\d)주차")
re_month = re.compile(r"(\d{1,2})월")


def _find_date_cols_by_regex(df, row_idx=1, pattern=re_day):
    cols = []
    for c in range(df.shape[1]):
        val = str(df.iat[row_idx, c]) if row_idx < len(df.index) and c < len(
            df.columns) else ""
        if pattern.search(val):
            cols.append(c)
    return cols


def _extract_month_day(label):
    m = re_day.search(str(label))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_week(label):
    m = re_week.search(str(label))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_month(label):
    m = re_month.search(str(label))
    if m:
        return int(m.group(1))
    return None


# -------------------------
# 포맷/인사이트/이상치 유틸
# -------------------------
def is_percentage_series(s: pd.Series) -> bool:
    name = (getattr(s, "name", "") or "").lower()
    if any(k in name for k in ["비중", "ratio", "rate", "%"]):
        return True
    vals = s.dropna()
    if len(vals) == 0:
        return False
    share_01 = (vals.between(0, 1)).mean()
    return share_01 >= 0.8


def y_axis_encoding(field, series_like, title):
    if is_percentage_series(series_like):
        return alt.Y(f"{field}:Q", title=title, axis=alt.Axis(format=".0%"))
    else:
        return alt.Y(f"{field}:Q", title=title, axis=alt.Axis(format=",.0f"))


def moving_avg(s: pd.Series, window=7):
    return s.rolling(window=window, min_periods=max(1, window // 2)).mean()


def detect_anomalies_zscore(s: pd.Series, z=2.5) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    mu = vals.mean()
    sd = vals.std(ddof=0)
    if pd.isna(sd) or sd == 0:
        return pd.Series(False, index=vals.index)
    return (np.abs((vals - mu) / sd) > z)


def chart_by_type(df,
                  x,
                  y,
                  color=None,
                  tooltip=None,
                  chart_type="Line",
                  scheme=BASE_SCHEME,
                  height=320,
                  y_series=None,
                  title_y=None):
    """
    버튼(라디오)로 선택된 chart_type에 맞게 Altair 차트 생성
    """
    y_enc = y_axis_encoding(y, y_series if y_series is not None else df[y],
                            title_y if title_y else y)
    enc = {}
    enc["x"] = alt.X(f"{x}:O", title=x)
    enc["y"] = y_enc
    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=alt.Scale(scheme=scheme))
    if tooltip:
        enc["tooltip"] = tooltip

    base = alt.Chart(df)
    if chart_type == "Line":
        ch = base.mark_line(point=True).encode(**enc)
    elif chart_type == "Area":
        ch = base.mark_area(opacity=0.7).encode(**enc)
    elif chart_type == "Column":
        ch = base.mark_bar().encode(**enc)
    elif chart_type == "Box":
        # Box는 x에 그룹, y에 연속값 필요
        ch = base.mark_boxplot().encode(
            x=alt.X(f"{color or 'group'}:N", title=color or "group")
            if color else alt.X(f"{x}:N", title=x),
            y=y_enc,
            color=alt.Color(f"{color}:N",
                            scale=alt.Scale(scheme=scheme),
                            legend=None) if color else alt.value("#666"))
    else:
        ch = base.mark_line(point=True).encode(**enc)

    return ch.properties(height=height)


# -------------------------
# 시트 파서들 (A열/B열 그룹 정보 포함)
# -------------------------
def parse_daily(df_raw, file_label):
    df = df_raw.copy()
    date_cols = _find_date_cols_by_regex(df, row_idx=1, pattern=re_day)
    if not date_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dau_row_idx = 2
    labels = [str(df.iat[1, c]) for c in date_cols]
    months, days = zip(*[_extract_month_day(x) for x in labels])

    # DAU
    dau_values = [
        pd.to_numeric(df.iat[dau_row_idx, c], errors="coerce")
        for c in date_cols
    ]
    daily_dau = pd.DataFrame({
        "file": file_label,
        "month": months,
        "day": days,
        "value": dau_values
    }).dropna(subset=["value"])

    # 지표 (B~C열): col=2에 지표명, A/B열은 그룹 태그로 활용
    metric_rows = []
    for r in range(dau_row_idx + 1, df.shape[0]):
        metric_name = str(df.iat[r, 2]) if 2 < df.shape[1] else ""
        if metric_name and metric_name != "nan":
            group_a = str(df.iat[r, 0]) if 0 < df.shape[1] else ""
            group_b = str(df.iat[r, 1]) if 1 < df.shape[1] else ""
            values = [
                pd.to_numeric(df.iat[r, c], errors="coerce") for c in date_cols
            ]
            tmp = pd.DataFrame({
                "file": file_label,
                "group_a": group_a,
                "group_b": group_b,
                "metric": metric_name,
                "month": months,
                "day": days,
                "value": values
            })
            metric_rows.append(tmp)
    daily_metrics = pd.concat(
        metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()

    # daily의 D열(MAU) → 통합 MAU 용으로만 사용
    daily_mau = pd.DataFrame()
    try:
        val = pd.to_numeric(df.iat[dau_row_idx, 3], errors="coerce")
        if pd.notna(val):
            rep_month = pd.Series(months).mode().iloc[0]
            daily_mau = pd.DataFrame([{
                "file": file_label,
                "month": rep_month,
                "MAU": val
            }])
    except Exception:
        pass

    return daily_dau, daily_metrics, daily_mau


def parse_weekly(df_raw, file_label):
    df = df_raw.copy()
    week_cols, month_list, week_list = [], [], []
    for c in range(df.shape[1]):
        label = str(df.iat[1, c]) if 1 < len(df.index) else ""
        m, w = _extract_week(label)
        if m and w:
            week_cols.append(c)
            month_list.append(m)
            week_list.append(w)

    if not week_cols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    wau_row_idx = 2
    wau_values = [
        pd.to_numeric(df.iat[wau_row_idx, c], errors="coerce")
        for c in week_cols
    ]
    weekly_wau = pd.DataFrame({
        "file": file_label,
        "month": month_list,
        "week": week_list,
        "value": wau_values
    }).dropna(subset=["value"])

    metric_rows = []
    for r in range(wau_row_idx + 1, df.shape[0]):
        metric_name = str(df.iat[r, 2]) if 2 < df.shape[1] else ""
        if metric_name and metric_name != "nan":
            group_a = str(df.iat[r, 0]) if 0 < df.shape[1] else ""
            group_b = str(df.iat[r, 1]) if 1 < df.shape[1] else ""
            values = [
                pd.to_numeric(df.iat[r, c], errors="coerce") for c in week_cols
            ]
            tmp = pd.DataFrame({
                "file": file_label,
                "group_a": group_a,
                "group_b": group_b,
                "metric": metric_name,
                "month": month_list,
                "week": week_list,
                "value": values
            })
            metric_rows.append(tmp)
    weekly_metrics = pd.concat(
        metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()

    weekly_mau = pd.DataFrame()
    try:
        val = pd.to_numeric(df.iat[wau_row_idx, 3], errors="coerce")
        if pd.notna(val):
            rep_month = pd.Series(month_list).mode().iloc[0]
            weekly_mau = pd.DataFrame([{
                "file": file_label,
                "month": rep_month,
                "MAU": val
            }])
    except Exception:
        pass

    return weekly_wau, weekly_metrics, weekly_mau


def parse_monthly(df_raw, file_label):
    df = df_raw.copy()
    month_cols, months = [], []
    for c in range(df.shape[1]):
        label = str(df.iat[1, c]) if 1 < len(df.index) else ""
        m = _extract_month(label)
        if m:
            month_cols.append(c)
            months.append(m)
    if not month_cols:
        return pd.DataFrame(), pd.DataFrame()

    # "MAU" 혹은 유사 키워드 찾기
    target_row_idx = None
    for r in range(df.shape[0]):
        name = str(df.iat[r, 2]) if 2 < df.shape[1] else ""
        if any(k in name for k in ["MAU", "mau", "월 누적", "Unique"]):
            target_row_idx = r
            break
    if target_row_idx is None:
        # 백업: 숫자합 최대 행
        candidates = []
        for r in range(2, df.shape[0]):
            vals = [
                pd.to_numeric(df.iat[r, c], errors="coerce")
                for c in month_cols
            ]
            candidates.append((r, np.nansum(vals)))
        if candidates:
            target_row_idx = sorted(candidates,
                                    key=lambda x: x[1],
                                    reverse=True)[0][0]

    mau_vals = [
        pd.to_numeric(df.iat[target_row_idx, c], errors="coerce")
        for c in month_cols
    ]
    monthly_mau = pd.DataFrame({
        "file": file_label,
        "month": months,
        "MAU": mau_vals
    }).dropna(subset=["MAU"])

    # 지표(B~C열) + 그룹 A/B
    metric_rows = []
    for r in range(2, df.shape[0]):
        if r == target_row_idx:
            continue
        metric_name = str(df.iat[r, 2]) if 2 < df.shape[1] else ""
        if metric_name and metric_name != "nan":
            group_a = str(df.iat[r, 0]) if 0 < df.shape[1] else ""
            group_b = str(df.iat[r, 1]) if 1 < df.shape[1] else ""
            values = [
                pd.to_numeric(df.iat[r, c], errors="coerce")
                for c in month_cols
            ]
            tmp = pd.DataFrame({
                "file": file_label,
                "group_a": group_a,
                "group_b": group_b,
                "metric": metric_name,
                "month": months,
                "value": values
            })
            metric_rows.append(tmp)
    monthly_metrics = pd.concat(
        metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()

    return monthly_mau, monthly_metrics


def parse_duration(df_raw, file_label):
    df = df_raw.copy()
    headers = []
    for c in range(1, 5):
        headers.append(
            str(df.iat[1, c]) if (
                1 < df.shape[0] and c < df.shape[1]) else f"col{c}")
    rows = []
    for r in range(2, 9):
        if r < df.shape[0]:
            bucket = str(df.iat[r, 0])
            if bucket and bucket != "nan":
                row = {"file": file_label, "bucket": bucket}
                for i, c in enumerate(range(1, 5)):
                    row[headers[i]] = pd.to_numeric(df.iat[r, c],
                                                    errors="coerce")
                rows.append(row)
    return pd.DataFrame(rows)


# -------------------------
# 파일 업로더
# -------------------------
uploaded_files = st.file_uploader("엑셀 파일 업로드 (여러 개 가능)",
                                  type=["xlsx"],
                                  accept_multiple_files=True)

if uploaded_files:
    all_daily_dau, all_daily_metrics, all_daily_mau = [], [], []
    all_weekly_wau, all_weekly_metrics, all_weekly_mau = [], [], []
    all_monthly_mau, all_monthly_metrics = [], []
    all_duration = []

    for file in uploaded_files:
        file_label = getattr(file, "name", "uploaded")
        xls = pd.ExcelFile(file)
        sheets = {s.lower(): s for s in xls.sheet_names}

        if "daily" in sheets:
            df_daily = pd.read_excel(file,
                                     sheet_name=sheets["daily"],
                                     header=None)
            d_dau, d_metrics, d_mau = parse_daily(df_daily, file_label)
            if not d_dau.empty: all_daily_dau.append(d_dau)
            if not d_metrics.empty: all_daily_metrics.append(d_metrics)
            if not d_mau.empty: all_daily_mau.append(d_mau)

        if "weekly" in sheets:
            df_weekly = pd.read_excel(file,
                                      sheet_name=sheets["weekly"],
                                      header=None)
            w_wau, w_metrics, w_mau = parse_weekly(df_weekly, file_label)
            if not w_wau.empty: all_weekly_wau.append(w_wau)
            if not w_metrics.empty: all_weekly_metrics.append(w_metrics)
            if not w_mau.empty: all_weekly_mau.append(w_mau)

        if "monthly" in sheets:
            df_monthly = pd.read_excel(file,
                                       sheet_name=sheets["monthly"],
                                       header=None)
            m_mau, m_metrics = parse_monthly(df_monthly, file_label)
            if not m_mau.empty: all_monthly_mau.append(m_mau)
            if not m_metrics.empty: all_monthly_metrics.append(m_metrics)

        if "녹음시간별" in sheets:
            df_dur = pd.read_excel(file,
                                   sheet_name=sheets["녹음시간별"],
                                   header=None)
            dur_df = parse_duration(df_dur, file_label)
            if not dur_df.empty: all_duration.append(dur_df)

    # 병합
    daily_dau = pd.concat(
        all_daily_dau, ignore_index=True) if all_daily_dau else pd.DataFrame()
    daily_metrics = pd.concat(
        all_daily_metrics,
        ignore_index=True) if all_daily_metrics else pd.DataFrame()
    weekly_wau = pd.concat(
        all_weekly_wau,
        ignore_index=True) if all_weekly_wau else pd.DataFrame()
    weekly_metrics = pd.concat(
        all_weekly_metrics,
        ignore_index=True) if all_weekly_metrics else pd.DataFrame()

    monthly_mau = pd.concat(
        all_monthly_mau,
        ignore_index=True) if all_monthly_mau else pd.DataFrame()
    aux_mau = pd.concat(
        [*all_daily_mau, *all_weekly_mau], ignore_index=True) if (
            all_daily_mau or all_weekly_mau) else pd.DataFrame()
    mau_all = pd.concat([monthly_mau, aux_mau], ignore_index=True) if (
        not monthly_mau.empty or not aux_mau.empty) else pd.DataFrame()

    # ======================
    # 중요 인사이트 (색 박스 + 이모지) — 최상단
    # ======================
    insight_msgs = []
    if not daily_dau.empty:
        dd = daily_dau.copy()
        peak_idx = dd["value"].idxmax()
        low_idx = dd["value"].idxmin()
        if pd.notna(peak_idx):
            insight_msgs.append((
                "insight-blue",
                f"🚀 **DAU 피크**: {int(dd.loc[peak_idx,'month'])}월 {int(dd.loc[peak_idx,'day'])}일 — {int(dd.loc[peak_idx,'value']):,}"
            ))
        if pd.notna(low_idx):
            insight_msgs.append((
                "insight-amber",
                f"📉 **DAU 저점**: {int(dd.loc[low_idx,'month'])}월 {int(dd.loc[low_idx,'day'])}일 — {int(dd.loc[low_idx,'value']):,}"
            ))
        dd["is_anom"] = dd.groupby("month")["value"].transform(
            lambda s: detect_anomalies_zscore(s, z=2.5))
        anom_cnt = int(dd["is_anom"].sum())
        if anom_cnt > 0:
            insight_msgs.append(
                ("insight-card",
                 f"🔍 **이상치(의심)** 감지 {anom_cnt}건 — 노출 영역/프로모션/서버 이벤트 확인 권장"))

    if not mau_all.empty:
        mv = mau_all.groupby([
            "file", "month"
        ])["MAU"].mean().reset_index().sort_values(["file", "month"])
        mv["pct"] = mv.groupby("file")["MAU"].pct_change() * 100
        last_avg = mv.dropna(
            subset=["pct"]).groupby("file")["pct"].last().mean()
        if pd.notna(last_avg):
            arrow = "⬆️" if last_avg >= 0 else "⬇️"
            insight_msgs.append(
                ("insight-green",
                 f"{arrow} **MAU 전월 변화 평균**: {last_avg:+.1f}%"))

    if not weekly_wau.empty:
        ww = weekly_wau
        peak = ww.loc[ww["value"].idxmax()]
        insight_msgs.append((
            "insight-blue",
            f"🏁 **WAU 최고 주차**: {int(peak['month'])}월 {int(peak['week'])}주차 — {int(peak['value']):,}"
        ))

    if all_duration:
        dur = pd.concat(all_duration, ignore_index=True)
        share_col = None
        for c in dur.columns:
            if ("비중" in c) or ("ratio"
                               in c.lower()) or ("share" in c.lower()) or (
                                   "rate" in c.lower()) or (c.endswith("%")):
                share_col = c
                break
        if share_col and dur[share_col].notna().any():
            s = dur[share_col]
            if s.max() > 1.0: s = s / 100.0
            idx = s.idxmax()
            if pd.notna(idx):
                insight_msgs.append(
                    ("insight-amber",
                     f"⏱️ **이용 비중 최다 구간**: {dur.loc[idx,'bucket']}"))

    # 상단 인사이트 섹션 렌더 (최대 10개)
    if insight_msgs:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 🔎 Top Insights")
        for cls, msg in insight_msgs[:10]:
            st.markdown(f'<div class="insight-card {cls}">{msg}</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # [섹션] 월별 DAU 분석 (히트맵 + 박스플롯)  — 첫 번째 중복 그래프 제거한 버전
    # =========================================================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📆 월별 DAU 분석 (히트맵 + 박스플롯)")
    if not daily_dau.empty:
        dd = daily_dau.copy().sort_values(["month", "day"])
        months_avail = sorted(
            pd.Series(dd["month"].unique()).dropna().astype(int).tolist())
        month_sel = st.multiselect("월 선택 (DAU)",
                                   months_avail,
                                   default=months_avail,
                                   key="dau_month_sel")
        dd = dd[dd["month"].isin(month_sel)]
        if not dd.empty:
            col1, col2 = st.columns([2, 1.4])
            with col1:
                heat = alt.Chart(dd).mark_rect().encode(
                    x=alt.X("day:O", title="일(Day)"),
                    y=alt.Y("month:O", title="월(Month)"),
                    color=alt.Color("value:Q",
                                    title="DAU",
                                    scale=alt.Scale(scheme=HEAT_SCHEME)),
                    tooltip=[
                        "file", "month", "day",
                        alt.Tooltip("value:Q", format=",.0f")
                    ]).properties(height=320)
                st.altair_chart(heat, use_container_width=True)
            with col2:
                box = alt.Chart(dd).mark_boxplot().encode(
                    x=alt.X("month:O", title="월(Month)"),
                    y=y_axis_encoding("value", dd["value"], "DAU"),
                    color=alt.Color("month:N",
                                    scale=alt.Scale(scheme=BASE_SCHEME),
                                    legend=None),
                    tooltip=[alt.Tooltip("value:Q", format=",.0f")
                             ]).properties(height=320)
                st.altair_chart(box, use_container_width=True)
        else:
            st.info("선택 조건에 해당하는 DAU가 없습니다.")
    else:
        st.info("daily 시트에서 DAU 데이터를 찾지 못했습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # [섹션] Daily 지표 (B~C열) — 차트 타입 선택 버튼
    # =========================================================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Daily 지표 (B~C열) — 월별 비교")
    if not daily_metrics.empty:
        metrics_list = sorted(
            daily_metrics["metric"].dropna().unique().tolist())
        sel_metric = st.selectbox("지표 선택",
                                  metrics_list,
                                  key="daily_metric_sel2")
        dm = daily_metrics[daily_metrics["metric"] == sel_metric].copy()
        months_avail = sorted(
            pd.Series(dm["month"].unique()).dropna().astype(int).tolist())
        month_sel = st.multiselect("월 선택",
                                   months_avail,
                                   default=months_avail,
                                   key="dm_months2")
        dm = dm[dm["month"].isin(month_sel)]
        if not dm.empty:
            chart_type = st.radio("차트 유형", ["Area", "Column", "Line", "Box"],
                                  horizontal=True,
                                  key="dm_chart_type")
            dm = dm.sort_values(["month", "day"])
            chart = chart_by_type(dm,
                                  x="day",
                                  y="value",
                                  color="month",
                                  tooltip=[
                                      "file", "metric", "month", "day",
                                      alt.Tooltip("value:Q", format=",.2f")
                                  ],
                                  chart_type=chart_type,
                                  scheme=CAT_SCHEME,
                                  height=320,
                                  y_series=dm["value"],
                                  title_y=sel_metric)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("선택한 조건에 데이터가 없습니다.")
    else:
        st.info("daily 지표(B~C열) 데이터가 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # [섹션] Weekly: WAU (월별 비교) — 차트 타입 선택 버튼
    # =========================================================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Weekly: WAU (월별 비교)")
    if not weekly_wau.empty:
        ww = weekly_wau.copy().sort_values(["month", "week"])
        months_avail = sorted(
            pd.Series(ww["month"].unique()).dropna().astype(int).tolist())
        month_sel = st.multiselect("월 선택 (WAU)",
                                   months_avail,
                                   default=months_avail,
                                   key="wau_months2")
        ww = ww[ww["month"].isin(month_sel)]
        if not ww.empty:
            chart_type = st.radio("차트 유형", ["Column", "Line", "Area", "Box"],
                                  horizontal=True,
                                  key="wau_chart_type")
            chart = chart_by_type(ww,
                                  x="week",
                                  y="value",
                                  color="month",
                                  tooltip=[
                                      "file", "month", "week",
                                      alt.Tooltip("value:Q", format=",.0f")
                                  ],
                                  chart_type=chart_type,
                                  scheme=BASE_SCHEME,
                                  height=340,
                                  y_series=ww["value"],
                                  title_y="WAU")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("선택 조건에 해당하는 WAU가 없습니다.")
    else:
        st.info("weekly 시트에서 WAU 데이터를 찾지 못했습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # [섹션] MAU (월별) — 통합 차트 (monthly 포함)  ← (요청 3) monthly 차트 유지/복구
    # =========================================================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧩 MAU (월별) — 통합 (daily/weekly의 MAU 포함 + monthly)")
    if not mau_all.empty:
        mv = mau_all.groupby([
            "file", "month"
        ])["MAU"].mean().reset_index().sort_values(["file", "month"])
        months_avail = sorted(
            pd.Series(mv["month"].unique()).dropna().astype(int).tolist())
        month_sel = st.multiselect("월 선택 (MAU)",
                                   months_avail,
                                   default=months_avail,
                                   key="mau_months2")
        mv = mv[mv["month"].isin(month_sel)]
        if not mv.empty:
            chart_type = st.radio("차트 유형", ["Area", "Line", "Column", "Box"],
                                  horizontal=True,
                                  key="mau_chart_type")
            chart = chart_by_type(
                mv,
                x="month",
                y="MAU",
                color="file",
                tooltip=["file", "month",
                         alt.Tooltip("MAU:Q", format=",.0f")],
                chart_type=chart_type,
                scheme=AREA_SCHEME,
                height=330,
                y_series=mv["MAU"],
                title_y="MAU")
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("선택 조건에 해당하는 MAU가 없습니다.")
    else:
        st.info("monthly 기반의 MAU 데이터를 찾지 못했습니다. monthly 시트를 확인해주세요.")
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # [섹션] 녹음시간별 — 4개 지표 모두 차트  ← (요청 4)
    # =========================================================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### ⏱️ 녹음시간별 분포 — 4개 지표 모두 보기")
    if all_duration:
        dur = pd.concat(all_duration, ignore_index=True)
        st.caption("원시 테이블")
        st.dataframe(dur)

        # 컬럼 후보 식별
        unique_col = next((c for c in dur.columns if ("Unique" in c) or (
            "unique" in c.lower()) or ("user" in c.lower())), None)
        share_col = next(
            (c for c in dur.columns
             if ("비중" in c) or ("ratio" in c.lower()) or ("share" in c.lower())
             or ("rate" in c.lower()) or (str(c).endswith("%"))), None)
        note_col = next((c for c in dur.columns if "인당 평균 노트 수" in c), None)
        min_col = next((c for c in dur.columns if "인당 평균 분 수" in c), None)

        # 4개 카드 레이아웃
        cols = st.columns(4)
        # 1) Unique user 수
        with cols[0]:
            st.markdown("**Unique user 수**")
            if unique_col:
                chart = alt.Chart(dur).mark_bar().encode(
                    x=alt.X("bucket:N", title="구간"),
                    y=y_axis_encoding(unique_col, dur[unique_col], unique_col),
                    color=alt.Color("file:N",
                                    scale=alt.Scale(scheme=BASE_SCHEME)),
                    tooltip=[
                        "file", "bucket",
                        alt.Tooltip(f"{unique_col}:Q", format=",.0f")
                    ]).properties(height=250)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("컬럼을 찾지 못했습니다.")
        # 2) 비중(%) — 도넛
        with cols[1]:
            st.markdown("**비중(%)**")
            if share_col:
                files_opts = dur["file"].unique().tolist()
                fsel = st.selectbox("파일", files_opts, key="dur_share_file")
                pie_df = dur[dur["file"] == fsel][[
                    "bucket", share_col
                ]].dropna().rename(columns={share_col: "value"})
                if not pie_df.empty:
                    if pie_df["value"].max() > 1.0:
                        pie_df["value"] = pie_df["value"] / 100.0
                    pie = alt.Chart(pie_df).mark_arc(innerRadius=40).encode(
                        theta="value:Q",
                        color=alt.Color("bucket:N",
                                        scale=alt.Scale(scheme=AREA_SCHEME)),
                        tooltip=[
                            "bucket",
                            alt.Tooltip("value:Q", format=".1%")
                        ]).properties(height=250)
                    st.altair_chart(pie, use_container_width=True)
                else:
                    st.info("해당 파일에 비중 데이터가 없습니다.")
            else:
                st.info("비중 컬럼을 찾지 못했습니다.")
        # 3) 인당 평균 노트 수
        with cols[2]:
            st.markdown("**인당 평균 노트 수**")
            if note_col:
                chart = alt.Chart(dur).mark_bar().encode(
                    x=alt.X("bucket:N"),
                    y=y_axis_encoding(note_col, dur[note_col], note_col),
                    color=alt.Color("file:N",
                                    scale=alt.Scale(scheme=CAT_SCHEME)),
                    tooltip=[
                        "file", "bucket",
                        alt.Tooltip(f"{note_col}:Q", format=",.2f")
                    ]).properties(height=250)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("컬럼을 찾지 못했습니다.")
        # 4) 인당 평균 분 수
        with cols[3]:
            st.markdown("**인당 평균 분 수**")
            if min_col:
                chart = alt.Chart(dur).mark_bar().encode(
                    x=alt.X("bucket:N"),
                    y=y_axis_encoding(min_col, dur[min_col], min_col),
                    color=alt.Color("file:N",
                                    scale=alt.Scale(scheme=CAT_SCHEME)),
                    tooltip=[
                        "file", "bucket",
                        alt.Tooltip(f"{min_col}:Q", format=",.2f")
                    ]).properties(height=250)
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("컬럼을 찾지 못했습니다.")
    else:
        st.info("녹음시간별 시트 데이터를 찾지 못했습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

    # =========================================================
    # [섹션] A열/B열 기반 '큰 지표' 그룹 차트 (예: OS별 aos/ios)  ← (요청 5)
    # =========================================================
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("### 🧭 그룹(큰 지표) 탐색 — A열/B열 기준 (예: OS별)")

    # 사용자가 그룹/서브그룹 선택 → 같은 이름의 metric들을 월/주/일 단위로 그려 비교
    def group_ui_and_plot(title,
                          df,
                          x_col,
                          extra_cols,
                          key_prefix,
                          default_chart="Column"):
        if df.empty:
            st.info(f"{title} 데이터가 없습니다.")
            return
        ga_list = sorted(
            [g for g in df["group_a"].dropna().unique() if g and g != "nan"])
        gb_list = sorted(
            [g for g in df["group_b"].dropna().unique() if g and g != "nan"])
        cols = st.columns([1.2, 1.2, 2])
        with cols[0]:
            ga_sel = st.selectbox("A열 그룹 선택", ["(전체)"] + ga_list,
                                  key=f"{key_prefix}_ga")
        with cols[1]:
            gb_sel = st.selectbox("B열 그룹 선택", ["(전체)"] + gb_list,
                                  key=f"{key_prefix}_gb")
        with cols[2]:
            chart_type = st.radio("차트 유형", ["Column", "Line", "Area", "Box"],
                                  horizontal=True,
                                  key=f"{key_prefix}_ctype",
                                  index=["Column", "Line", "Area",
                                         "Box"].index(default_chart))

        sub = df.copy()
        if ga_sel != "(전체)":
            sub = sub[sub["group_a"] == ga_sel]
        if gb_sel != "(전체)":
            sub = sub[sub["group_b"] == gb_sel]
        if sub.empty:
            st.info("선택한 그룹 조합에 데이터가 없습니다.")
            return

        # metric 선택
        metrics = sorted(sub["metric"].dropna().unique().tolist())
        msel = st.selectbox("지표 선택", metrics, key=f"{key_prefix}_metric")
        sub = sub[sub["metric"] == msel]
        if sub.empty:
            st.info("선택한 지표 데이터가 없습니다.")
            return

        # 색상은 월/파일 중 택1 (월별 비교가 목적에 부합)
        color_dim = "month" if "month" in sub.columns else extra_cols[0]
        tooltip = ["file", "group_a", "group_b", "metric"] + [
            c for c in [x_col, color_dim, "value"] if c in sub.columns
        ]
        chart = chart_by_type(sub.sort_values([color_dim, x_col]),
                              x=x_col,
                              y="value",
                              color=color_dim,
                              tooltip=tooltip,
                              chart_type=chart_type,
                              scheme=CAT_SCHEME,
                              height=320,
                              y_series=sub["value"],
                              title_y=msel)
        st.altair_chart(chart, use_container_width=True)

    # Daily 그룹 차트
    st.markdown("#### Daily 그룹 차트")
    if not daily_metrics.empty:
        group_ui_and_plot("Daily 그룹",
                          daily_metrics,
                          x_col="day",
                          extra_cols=["month"],
                          key_prefix="grp_daily",
                          default_chart="Column")
    else:
        st.info("Daily 그룹 데이터가 없습니다.")

    # Weekly 그룹 차트
    st.markdown("#### Weekly 그룹 차트")
    if not weekly_metrics.empty:
        group_ui_and_plot("Weekly 그룹",
                          weekly_metrics,
                          x_col="week",
                          extra_cols=["month"],
                          key_prefix="grp_weekly",
                          default_chart="Column")
    else:
        st.info("Weekly 그룹 데이터가 없습니다.")

    # Monthly 그룹 차트
    st.markdown("#### Monthly 그룹 차트")
    if not all_monthly_metrics:
        # monthly_metrics 자체는 상단에서 concat 하지 않았으니 재조합
        monthly_metrics = pd.concat(
            all_monthly_metrics,
            ignore_index=True) if all_monthly_metrics else pd.DataFrame()
    if 'monthly_metrics' in locals() and not monthly_metrics.empty:
        group_ui_and_plot("Monthly 그룹",
                          monthly_metrics,
                          x_col="month",
                          extra_cols=[],
                          key_prefix="grp_monthly",
                          default_chart="Line")
    else:
        st.info("Monthly 그룹 데이터가 없습니다.")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("엑셀 파일을 업로드해주세요. (여러 개 업로드 가능)")

###############sc############
import streamlit as st
import requests
from PIL import Image
import io

# 웹훅 URL
WEBHOOK_URL = 'https://soyso2sohee.app.n8n.cloud/webhook-test/mail'


# 스크린샷 기능 (웹 페이지 캡처)
def take_screenshot():
    # 웹 페이지 캡처를 위한 간단한 예시로, 이미지 파일을 생성하여 반환하는 형태로 대체
    img = Image.new('RGB', (800, 600), color=(73, 109, 137))
    return img


# 웹훅으로 스크린샷 발송
def send_screenshot_to_webhook(image):
    # 이미지를 웹훅으로 전송하려면 이미지를 파일 형식으로 변환해야 함
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    response = requests.post(WEBHOOK_URL, files={'file': img_byte_arr})
    return response.status_code


# Streamlit 인터페이스
st.title("Dashboard with Screenshot Functionality")

# 'sc' 버튼 클릭 시 스크린샷 캡처 및 웹훅 발송
if st.button("Capture Screenshot"):
    screenshot = take_screenshot()  # 스크린샷 캡처
    status_code = send_screenshot_to_webhook(screenshot)  # 웹훅으로 전송
    if status_code == 200:
        st.success("Screenshot sent successfully!")
    else:
        st.error(f"Failed to send screenshot. Status code: {status_code}")
