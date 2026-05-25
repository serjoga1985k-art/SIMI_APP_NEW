import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Для великих файлів не ставимо 2**31 - 1: це може різко збільшити памʼять під Styler.
pd.set_option("styler.render.max_elements", 300_000)
try:
    pd.options.mode.copy_on_write = True  # pandas 2.x: менше зайвих копій при фільтрації
except Exception:
    pass

# ── Constants ────────────────────────────────────────────────────────────────
MONTH_MAP = {
    "Січ": 1, "Лют": 2, "Бер": 3, "Кві": 4,
    "Тра": 5, "Чер": 6, "Лип": 7, "Сер": 8,
    "Вер": 9, "Жов": 10, "Лис": 11, "Гру": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
MONTH_LABELS = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr",
    5: "may", 6: "jun", 7: "jul", 8: "aug",
    9: "sep", 10: "oct", 11: "nov", 12: "dec",
}
MONTHS_LIST = [MONTH_LABELS[m] for m in range(1, 13)]

UA_MONTH_NAMES = {
    1: "січень", 2: "лютий", 3: "березень", 4: "квітень",
    5: "травень", 6: "червень", 7: "липень", 8: "серпень",
    9: "вересень", 10: "жовтень", 11: "листопад", 12: "грудень",
}

def _single_value_or_none(values):
    vals = []
    try:
        for v in values:
            if pd.notna(v):
                vals.append(v)
    except Exception:
        return None
    uniq = pd.Series(vals).dropna().unique().tolist() if vals else []
    return uniq[0] if len(uniq) == 1 else None

def _format_ua_month_year(month_value=None, year_value=None):
    m_num = None
    if month_value is not None and pd.notna(month_value):
        try:
            m_num = int(get_month_num(pd.Series([month_value])).iloc[0])
        except Exception:
            m_num = None
    m_txt = UA_MONTH_NAMES.get(m_num, str(month_value).strip() if month_value is not None and pd.notna(month_value) else "")

    y_txt = ""
    if year_value is not None and pd.notna(year_value):
        try:
            y_float = float(year_value)
            y_txt = str(int(y_float)) if y_float.is_integer() else str(year_value)
        except Exception:
            y_txt = str(year_value).strip()

    return " ".join([x for x in [m_txt, y_txt] if x]).strip()

def _build_article_period_title(article=None, articles=None, df_context=None, col_article=None, col_month=None, col_year=None, suffix=None):
    """Формує заголовок: Стаття "Водопостачання" за січень 2026."""
    article_val = article
    if article_val is None and articles is not None:
        article_val = _single_value_or_none(articles)
    if article_val is None and df_context is not None and col_article and col_article in df_context.columns:
        article_val = _single_value_or_none(df_context[col_article].dropna().unique())

    if article_val is None:
        base = "Стаття всі статті"
    else:
        base = f'Стаття "{article_val}"'

    month_val = None
    year_val = None
    if df_context is not None and not df_context.empty:
        if col_month and col_month in df_context.columns:
            month_val = _single_value_or_none(df_context[col_month].dropna().unique())
        if col_year and col_year in df_context.columns:
            year_val = _single_value_or_none(df_context[col_year].dropna().unique())

    period = _format_ua_month_year(month_val, year_val)
    if period:
        base = f"{base} за {period}"

    if suffix:
        base = f"{base} · {suffix}"
    return base


PURPLE    = "#5b2d8e"
GREY      = "#c0c0c0"
RED_LINE  = "#c0392b"
YELLOW    = "#f0c000"
GREEN_HDR = "#2e7d32"
TEAL      = "#0d7377"
TEAL_HDR  = "#085f63"
ORANGE    = "#e67e22"


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_month_num(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype("int16")
    month_map_lower = {str(k).lower(): v for k, v in MONTH_MAP.items()}
    return series.astype(str).str.strip().str.lower().map(month_map_lower).fillna(0).astype("int16")


@st.cache_data(show_spinner=False, ttl=3600)
def load_excel(file_bytes, file_name, sheet_name):
    """
    Швидке читання Excel для великих файлів.
    Кешується на 1 годину: при перемиканні фільтрів Excel не перечитується заново.
    """
    import os

    buf = io.BytesIO(file_bytes)
    ext = os.path.splitext(file_name)[1].lower()
    engine = "pyxlsb" if ext == ".xlsb" else None

    read_kwargs = {"sheet_name": sheet_name}
    if engine:
        read_kwargs["engine"] = engine

    df = pd.read_excel(buf, **read_kwargs)
    df.columns = df.columns.astype(str).str.strip()
    df = _optimize_df_types(df)
    return df


def _optimize_df_types(df):
    """
    Оптимізація памʼяті без агресивного переведення тексту в category.
    Category не застосовуємо автоматично, бо в Streamlit/фільтрах часто виникає
    помилка Cannot setitem on a Categorical with a new category.
    """
    if df is None or df.empty:
        return df

    out = df.copy()

    float_cols = out.select_dtypes(include=["float64"]).columns
    for c in float_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce", downcast="float")

    int_cols = out.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce", downcast="integer")

    return out


def _ensure_numeric_inplace(df, cols):
    """Безпечно приводить потрібні колонки до чисел один раз."""
    if df is None or df.empty:
        return df
    for c in cols:
        if c and c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            if str(df[c].dtype) == "float64":
                df[c] = pd.to_numeric(df[c], downcast="float")
    return df




def _norm_col_name(name):
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def _find_col_by_keywords(cols, exact=None, contains=None, all_contains=None):
    """Find best matching column: exact first, then contains, then all words."""
    exact = exact or []
    contains = contains or []
    all_contains = all_contains or []
    norm_map = {c: _norm_col_name(c) for c in cols}

    for wanted in exact:
        wanted_norm = _norm_col_name(wanted)
        for c, n in norm_map.items():
            if n == wanted_norm:
                return c

    for key in contains:
        key_norm = _norm_col_name(key)
        for c, n in norm_map.items():
            if key_norm in n:
                return c

    for keys in all_contains:
        keys_norm = [_norm_col_name(k) for k in keys]
        for c, n in norm_map.items():
            if all(k in n for k in keys_norm):
                return c

    return None


def auto_map_columns(df):
    """Automatic default column mapping. User can still change every selectbox manually."""
    cols = df.columns.tolist()

    return {
        "col_tt": _find_col_by_keywords(
            cols,
            exact=["TT", "ТТ", "Магазин", "TT (Магазин)"],
            contains=["тт", "магазин", "store", "shop"]
        ),
        "col_month": _find_col_by_keywords(
            cols,
            exact=["Місяць", "Месяц", "Month"],
            contains=["місяц", "месяц", "month"]
        ),
        "col_plf": _find_col_by_keywords(
            cols,
            exact=["PL/F", "PL / F", "PLF", "План/Факт"],
            contains=["pl/f", "pl / f", "plf", "план", "факт"]
        ),
        "col_level0": _find_col_by_keywords(
            cols,
            exact=["Level_0", "Level 0"],
            contains=["level 0", "level_0", "level"]
        ),
        "col_year": _find_col_by_keywords(
            cols,
            exact=["Рік місяця", "Рік", "Year"],
            contains=["рік місяця", "year", "рік", "год"]
        ),
        "col_value": _find_col_by_keywords(
            cols,
            exact=["Значение", "Значення", "Value", "Amount"],
            contains=["знач", "value", "amount", "сума"]
        ),
        "col_kwh": _find_col_by_keywords(
            cols,
            exact=["ЕЕ_кВт", "ЕЕ кВт", "EE_kWh", "EE kWh", "кВт", "kWh"],
            contains=["ее_квт", "ее квт", "квт", "kwh"]
        ),
        "col_article": _find_col_by_keywords(
            cols,
            exact=["Стаття бюджету", "Статья бюджета", "Article"],
            contains=["стаття бюджету", "статья бюджета", "стаття", "article", "budget"]
        ),
        "col_ratio": _find_col_by_keywords(
            cols,
            exact=["% в ТО без акцизу та без ПДВ", "% в ТО без акциза и без НДС"],
            all_contains=[["%", "то"], ["ratio"]],
            contains=["% в то", "відсот", "процент"]
        ),
        # Назва, яку показуємо в шапці магазину замість ТТ, коли вибраний 1 магазин.
        "col_division": _find_col_by_keywords(
            cols,
            exact=["Підрозділ", "Подразделение", "Division"],
            contains=["підрозділ", "подраздел", "division"]
        ),
        "col_city": _find_col_by_keywords(cols, exact=["Місто", "Город", "City"], contains=["місто", "город", "city"]),
        "col_area": _find_col_by_keywords(cols, exact=["Площа", "Площадь", "Area"], contains=["площа", "площад", "area"]),
        "col_format": _find_col_by_keywords(cols, exact=["Формат ТО", "Формат"], contains=["формат то", "формат", "format"]),
        "col_format2": _find_col_by_keywords(cols, exact=["Формат2", "Формат 2", "Format2", "Format 2"], contains=["формат2", "формат 2", "format2", "format 2"]),
        "col_mega": _find_col_by_keywords(cols, exact=["Мегасегмент"], contains=["мегасегмент", "mega"]),
        "col_rik": _find_col_by_keywords(cols, exact=["Рік відкриття", "Год открытия"], contains=["рік відкрит", "год открыт", "year open"]),
        "col_mis": _find_col_by_keywords(cols, exact=["Місяць відкриття", "Месяц открытия"], contains=["місяць відкрит", "месяц открыт", "month open"]),
    }


def _init_column_state(df, file_name, sheet_name):
    """Initialize defaults once per file/sheet; manual user choice remains available."""
    cols = df.columns.tolist()
    signature = f"{file_name}::{sheet_name}::{len(cols)}::{','.join(map(str, cols))}"
    if st.session_state.get("_column_mapping_signature") != signature:
        st.session_state["_column_mapping_signature"] = signature
        auto_cols = auto_map_columns(df)
        for key, val in auto_cols.items():
            if val in cols:
                st.session_state[key] = val
            elif key not in st.session_state:
                st.session_state[key] = None


def _select_col(label, cols, state_key, allow_empty=False, help=None):
    options = (["—"] + cols) if allow_empty else cols
    current = st.session_state.get(state_key)
    if allow_empty:
        default_value = current if current in cols else "—"
    else:
        default_value = current if current in cols else (cols[0] if cols else None)
    index = options.index(default_value) if default_value in options else 0
    selected = st.selectbox(label, options, index=index, key=f"select_{state_key}", help=help)
    value = None if selected == "—" else selected
    st.session_state[state_key] = value
    return value


def _prep(df, col_month):
    """
    Додає _m лише один раз. Для зрізів робить мінімальну копію тільки тоді,
    коли _m ще немає. Це зменшує повторні копії на великих файлах.
    """
    if df is None or df.empty:
        return df
    if "_m" in df.columns:
        return df
    out = df.copy()
    out["_m"] = get_month_num(out[col_month])
    return out


def _fact_rows(df, col_plf):
    return df[df[col_plf] == "F"] if col_plf and col_plf in df.columns else df


def _plan_rows(df, col_plf):
    if col_plf and col_plf in df.columns:
        return df[df[col_plf] == "PL"]
    return pd.DataFrame(columns=df.columns)


def _make_combo_col(df, factors):
    """Швидке формування тексту комбінації факторів без apply(axis=1)."""
    valid_factors = [f for f in factors if f in df.columns]
    if not valid_factors:
        return pd.Series("", index=df.index), []

    combo = df[valid_factors[0]].where(df[valid_factors[0]].notna(), "").astype(str).str.strip()
    for factor in valid_factors[1:]:
        part = df[factor].where(df[factor].notna(), "").astype(str).str.strip()
        combo = combo.str.cat(part, sep=" | ")

    combo = (
        combo
        .str.replace(r"(^\s*\|\s*)|(\s*\|\s*$)", "", regex=True)
        .str.replace(r"(\s*\|\s*){2,}", " | ", regex=True)
        .str.strip(" |")
    )
    return combo, valid_factors


def _smooth_chart_series(values, drop_ratio=0.72):
    """
    Згладжує тільки графік: прибирає різкі одиночні провали між двома нормальними місяцями.
    Дані в таблицях не змінюються.
    """
    ser = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan)

    if ser.dropna().shape[0] < 3:
        return ser.fillna(0)

    out = ser.copy()

    for i in range(1, len(ser) - 1):
        prev_v = ser.iloc[i - 1]
        cur_v = ser.iloc[i]
        next_v = ser.iloc[i + 1]

        if pd.isna(prev_v) or pd.isna(cur_v) or pd.isna(next_v):
            continue

        neighbor_avg = (prev_v + next_v) / 2
        if neighbor_avg == 0 or pd.isna(neighbor_avg):
            continue

        # якщо точка різко нижча за сусідні — замінюємо її на мʼяке середнє
        if cur_v < neighbor_avg * drop_ratio and cur_v < prev_v and cur_v < next_v:
            out.iloc[i] = neighbor_avg

    # легке згладження, щоб лінія не виглядала ламаною з різкими піками
    out = out.interpolate(limit_direction="both")
    return out.fillna(0)

def _build_global_avg(df_all_fact, col_value, col_article, col_tt,
                      group_factors, agg_fn="mean"):
    """
    Compute global average per (group_factors + article) from the full fact dataset.
    Always uses the FULL dataset (df) so extra filters don't distort the norm.
    """
    if group_factors:
        grp = list(dict.fromkeys(group_factors + [col_article]))
    else:
        grp = [col_tt, col_article, "_m"]

    return (
        df_all_fact
        .groupby(grp, as_index=False, observed=True)[col_value]
        .agg(Average_Calc=agg_fn)
    )


def _merge_avg(tt_table, global_avg, group_factors, col_article, col_tt):
    """Merge global average onto tt_table."""
    if group_factors:
        on = list(dict.fromkeys(group_factors + [col_article]))
    else:
        on = [col_tt, col_article, "_m"]
    return pd.merge(tt_table, global_avg, on=on, how="left")


# ═══════════════════════════════════════════════════════════════════════════════
# FIX: build_article_monthly — Average завжди береться з ПОВНОГО df (df_base),
#      тільки Plan/Fact беруться з df_filtered.
#      tt_table для динамічного Average також будується з df_filtered,
#      але Average_Calc підтягується з global_avg (повний df) через merge.
# ═══════════════════════════════════════════════════════════════════════════════
def build_article_monthly(df, df_filtered, col_tt, col_article, col_month,
                           col_value, col_plf, selected_art, selected_tts, group_factors):
    """Absolute value monthly table: Plan / Fact / Average / Delta."""

    # --- Plan & Fact: з ВІДФІЛЬТРОВАНОГО датасету ---
    art_filt = _prep(df_filtered[df_filtered[col_article] == selected_art], col_month)

    if art_filt.empty:
        return pd.DataFrame(0.0, index=range(1, 13), columns=["Plan", "Fact", "Average", "Delta", "DeltaPlan", "DeltaPlan_%"])

    if not selected_tts:
        selected_tts = art_filt[col_tt].dropna().unique().tolist()

    plan = (_plan_rows(art_filt, col_plf)
            .groupby("_m", observed=True)[col_value].sum()
            .reindex(range(1, 13), fill_value=0).rename("Plan"))
    fact = (_fact_rows(art_filt, col_plf)
            .groupby("_m", observed=True)[col_value].sum()
            .reindex(range(1, 13), fill_value=0).rename("Fact"))

    # --- Average (норматив): ЗАВЖДИ з ПОВНОГО df, щоб фільтри не спотворювали норму ---
    art_all  = _prep(df[df[col_article] == selected_art], col_month)
    all_fact = _fact_rows(art_all, col_plf)
    global_avg = _build_global_avg(all_fact, col_value, col_article, col_tt, group_factors)

    # tt_table будується з df_filtered (які ТТ активні), але Average_Calc — з global_avg (повний df)
    tt_grp = list(dict.fromkeys([col_tt] + group_factors + ["_m", col_article]))
    tt_table = (
        _fact_rows(art_filt, col_plf)
        .groupby(tt_grp, as_index=False, observed=True)[col_value]
        .sum()
        .rename(columns={col_value: "Fact"})
    )
    tt_table = _merge_avg(tt_table, global_avg, group_factors, col_article, col_tt)
    tt_table["Fact"]         = tt_table["Fact"].fillna(0)
    tt_table["Average_Calc"] = tt_table["Average_Calc"].fillna(0)
    tt_table.loc[tt_table["Fact"] == 0, "Average_Calc"] = 0

    dynamic_average = (
        tt_table[tt_table[col_tt].isin(selected_tts)]
        .groupby("_m", observed=True)["Average_Calc"].sum()
        .reindex(range(1, 13), fill_value=0)
        .rename("Average")
    )

    merged = pd.DataFrame(index=range(1, 13)).join(plan).join(fact).join(dynamic_average).fillna(0)
    merged.index.name = "month"
    merged.loc[merged["Fact"] == 0, "Average"] = 0
    merged["Delta"] = merged["Fact"] - merged["Average"]
    # Нова метрика для основної таблиці/графіка: Факт − План
    merged["DeltaPlan"] = merged["Fact"] - merged["Plan"]

    # NEW: відносне відхилення Факт / План у % для першої абсолютної таблиці
    merged["DeltaPlan_%"] = (
        (merged["Fact"] / merged["Plan"].replace(0, np.nan) - 1) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    return merged


# ═══════════════════════════════════════════════════════════════════════════════
# FIX: build_ratio_monthly — Average (норматив %) також береться з ПОВНОГО df.
#      ТО для знаменника у Plan/Fact береться з df_filtered (коректна поведінка),
#      а from-scratch global_avg для норми — з df (без зайвих фільтрів).
# ═══════════════════════════════════════════════════════════════════════════════
def build_ratio_monthly(df_filtered, col_tt, col_article, col_month,
                         col_ratio, col_plf, selected_art, selected_tts,
                         df_all=None, group_factors=None):
    """
    Relative monthly table: Plan / Fact / Average / Delta for "% в ТО".
    Оптимізовано: мінімум копій, один прохід для Plan/Fact, без циклу по factor-combos.
    """
    if group_factors is None:
        group_factors = []

    col_value_abs = "Значение"
    to_article_name = "ТО без ПДВ та без акцизу"
    src_all = df_all if df_all is not None else df_filtered

    art_filt = _prep(df_filtered[df_filtered[col_article].eq(selected_art)], col_month)

    if art_filt.empty:
        return pd.DataFrame(0.0, index=range(1, 13), columns=["Plan", "Fact", "Average", "Delta", "DeltaPlan"])

    if not selected_tts:
        selected_tts = art_filt[col_tt].dropna().unique().tolist()

    selected_tts_clean = [t for t in selected_tts if pd.notna(t)]
    one_tt_selected = len(selected_tts_clean) == 1
    months_used = art_filt["_m"].dropna().unique()

    # ── Plan / Fact %: Значення статті / ТО вибраних ТТ ───────────────────────
    to_mask = (
        src_all[col_article].eq(to_article_name)
        & src_all[col_tt].isin(selected_tts_clean if selected_tts_clean else selected_tts)
    )
    to_base = _prep(src_all[to_mask], col_month)
    if len(months_used):
        to_base = to_base[to_base["_m"].isin(months_used)]

    # numeric без зайвих присвоєнь у великий df
    art_val = pd.to_numeric(art_filt[col_value_abs], errors="coerce")
    to_val = pd.to_numeric(to_base[col_value_abs], errors="coerce")

    art_tmp = art_filt[[col_plf, "_m"]].copy()
    art_tmp["_val"] = art_val.to_numpy()
    to_tmp = to_base[[col_plf, "_m"]].copy()
    to_tmp["_val"] = to_val.to_numpy()

    def calc_ratio_series_fast(data_art, data_to, plf_code):
        a = data_art[data_art[col_plf].eq(plf_code)].groupby("_m", observed=True)["_val"].sum()
        t = data_to[data_to[col_plf].eq(plf_code)].groupby("_m", observed=True)["_val"].sum()
        ratio = (a / t.replace(0, np.nan) * 100).replace([np.inf, -np.inf], np.nan)
        return ratio.fillna(0).reindex(range(1, 13), fill_value=0)

    fact = calc_ratio_series_fast(art_tmp, to_tmp, "F").rename("Fact")
    plan = calc_ratio_series_fast(art_tmp, to_tmp, "PL").rename("Plan")

    # ── Average % ─────────────────────────────────────────────────────────────
    if one_tt_selected and group_factors and col_ratio and col_ratio in src_all.columns:
        selected_tt = selected_tts_clean[0]
        valid_factors = [f for f in group_factors if f in src_all.columns]

        average = pd.Series(0.0, index=range(1, 13), name="Average")

        if valid_factors:
            tt_factor_rows = src_all[
                src_all[col_tt].eq(selected_tt)
                & src_all[col_article].eq(selected_art)
            ]
            tt_factor_rows = _fact_rows(tt_factor_rows, col_plf)

            if not tt_factor_rows.empty:
                selected_factor_combos = (
                    tt_factor_rows[valid_factors]
                    .drop_duplicates()
                    .dropna(how="all")
                )

                src_ratio = _prep(src_all[src_all[col_article].eq(selected_art)], col_month)
                src_ratio = _fact_rows(src_ratio, col_plf)

                # Замість циклу mask по кожній комбінації — один merge по факторах.
                matched = src_ratio.merge(
                    selected_factor_combos,
                    on=valid_factors,
                    how="inner"
                )

                if not matched.empty:
                    ratio_num = pd.to_numeric(matched[col_ratio], errors="coerce")
                    matched = matched[["_m"]].copy()
                    matched["_ratio"] = ratio_num.to_numpy()

                    average = (
                        matched
                        .groupby("_m", observed=True)["_ratio"]
                        .mean()
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(0)
                        .reindex(range(1, 13), fill_value=0)
                        .rename("Average")
                    )
    else:
        # Інші випадки: Average % = Average_abs / Fact_TO вибраних ТТ * 100.
        avg_abs_df = build_article_monthly(
            src_all,
            df_filtered,
            col_tt,
            col_article,
            col_month,
            col_value_abs,
            col_plf,
            selected_art,
            selected_tts_clean if selected_tts_clean else selected_tts,
            group_factors
        )

        avg_abs_by_month = avg_abs_df["Average"].reindex(range(1, 13), fill_value=0)
        fact_to_month = to_tmp[to_tmp[col_plf].eq("F")].groupby("_m", observed=True)["_val"].sum().reindex(range(1, 13), fill_value=0)

        average = (
            (avg_abs_by_month / fact_to_month.replace(0, np.nan)) * 100
        ).replace([np.inf, -np.inf], np.nan).fillna(0).rename("Average")

    merged = (
        pd.DataFrame(index=range(1, 13))
        .join(plan)
        .join(fact)
        .join(average)
        .fillna(0.0)
    )

    merged.index.name = "month"
    merged["Delta"] = merged["Fact"] - merged["Average"]
    # Нова метрика для % таблиці/графіка: Факт % − План %
    merged["DeltaPlan"] = merged["Fact"] - merged["Plan"]

    return merged


def analyze_factor_impact(df, df_filtered, col_tt, col_article, col_month,
                          col_value, col_plf, selected_art, group_factors):
    if not group_factors:
        return None

    # FIX: для аналізу впливу факторів використовуємо ПОВНИЙ df
    art_all  = _prep(df[df[col_article] == selected_art], col_month)
    all_fact = _fact_rows(art_all, col_plf)

    impact_data = []
    for factor in group_factors:
        factor_impact = (
            all_fact.groupby([factor, "_m"], as_index=False, observed=True)[col_value]
            .agg(Average="mean", Count="count", Total="sum", Std="std")
        )
        for _, row in factor_impact.iterrows():
            impact_data.append({
                "Фактор": factor,
                "Значення": row[factor],
                "Місяць": MONTH_LABELS.get(row["_m"], str(row["_m"])),
                "Середнє": row["Average"],
                "Кількість": row["Count"],
                "Сума": row["Total"],
                "Відхилення": row["Std"]
            })

    return pd.DataFrame(impact_data)


def render_factor_impact_analysis(df, df_filtered, col_tt, col_article, col_month,
                                  col_value, col_plf, selected_art, group_factors):
    if not group_factors:
        st.info("Оберіть фактори групування для аналізу впливу.")
        return

    st.markdown(f"""
    <div style="margin-top:20px;margin-bottom:8px;">
      <span style="background:{ORANGE};color:white;font-weight:700;padding:4px 14px;
                   font-size:0.9rem;border-radius:2px;">📊 Аналіз впливу комбінації факторів — {selected_art}</span>
    </div>""", unsafe_allow_html=True)

    selected_factors = st.multiselect(
        "Оберіть фактори для комбінації:",
        options=group_factors,
        default=group_factors[:1],
        key=f"factor_combo_multiselect_{selected_art}"
    )

    if not selected_factors:
        st.info("Оберіть хоча б один фактор.")
        return

    art_all = _prep(df[df[col_article] == selected_art].copy(), col_month)
    all_fact = _fact_rows(art_all, col_plf).copy()
    all_fact[col_value] = pd.to_numeric(all_fact[col_value], errors="coerce")

    combo_col = "Комбінація факторів"

    all_fact[combo_col], valid_selected_factors = _make_combo_col(all_fact, selected_factors)

    if not valid_selected_factors:
        st.warning("Вибрані фактори відсутні в даних.")
        return

    combo_impact = (
        all_fact
        .groupby([combo_col, "_m"], as_index=False, observed=True)[col_value]
        .agg(
            Середнє="mean",
            Кількість="count",
            Сума="sum",
            Відхилення="std"
        )
    )

    combo_impact["Місяць"] = combo_impact["_m"].map(MONTH_LABELS)

    if combo_impact.empty:
        st.warning("Недостатньо даних для аналізу комбінації факторів.")
        return

    pivot_impact = combo_impact.pivot_table(
        index=combo_col,
        columns="Місяць",
        values="Середнє",
        aggfunc="sum"
    )

    # FIX: сортуємо місяці за замовчуванням від jan до dec
    pivot_impact = pivot_impact.reindex(
        columns=[m for m in MONTHS_LIST if m in pivot_impact.columns]
    )

    summary_by_combo = combo_impact.groupby(combo_col, as_index=True, observed=True).agg(
        **{
            "Середнє (загальне)": ("Середнє", "mean"),
            "Всього записів": ("Кількість", "sum"),
            "Сума": ("Сума", "sum")
        }
    )

    pivot_impact = pivot_impact.join(summary_by_combo)

    st.markdown(
        f"**Вплив комбінації факторів: {' + '.join(selected_factors)}**"
    )

    st.dataframe(
        pivot_impact.style
            .background_gradient(
                cmap="RdYlGn_r",
                subset=[c for c in pivot_impact.columns if c in MONTHS_LIST]
            )
            .apply(_style_white_na, axis=None)
            .format(lambda v: f"{v:,.0f}" if pd.notna(v) else "", na_rep=""),
        use_container_width=True
    )

    max_lines = st.slider(
        "Кількість комбінацій на графіку",
        3, 30, 10,
        key=f"factor_combo_top_n_{selected_art}"
    )

    top_combos = (
        combo_impact
        .groupby(combo_col, observed=True)["Сума"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(max_lines)
        .index
    )

    chart_data = combo_impact[combo_impact[combo_col].isin(top_combos)].copy()
    chart_data = chart_data.sort_values([combo_col, "_m"])

    fig = go.Figure()

    # Орієнтир: середнє по всіх показаних комбінаціях за кожен місяць
    avg_line = (
        chart_data
        .groupby("_m", as_index=False, observed=True)["Середнє"]
        .mean()
        .sort_values("_m")
    )
    avg_line["Місяць"] = avg_line["_m"].map(MONTH_LABELS)

    avg_line["Середнє_графік"] = _smooth_chart_series(avg_line["Середнє"])

    fig.add_trace(go.Scatter(
        x=avg_line["Місяць"],
        y=avg_line["Середнє_графік"],
        name="Середнє по вибраних комбінаціях",
        mode="lines",
        line=dict(width=4, dash="dash", shape="spline", smoothing=1.2),
        hovertemplate=(
            "<b>Середнє по комбінаціях</b><br>"
            "Місяць: %{x}<br>"
            "Середнє: %{y:,.0f}<extra></extra>"
        )
    ))

    for combo in chart_data[combo_col].dropna().unique():
        value_data = chart_data[chart_data[combo_col] == combo].copy()
        value_data = value_data.sort_values("_m")

        value_data["Середнє_графік"] = _smooth_chart_series(value_data["Середнє"])
        text_values = [f"{v:,.0f}" for v in value_data["Середнє_графік"]] if max_lines <= 8 else None

        fig.add_trace(go.Scatter(
            x=value_data["Місяць"],
            y=value_data["Середнє_графік"],
            name=str(combo),
            mode="lines+markers+text" if max_lines <= 8 else "lines+markers",
            text=text_values,
            textposition="top center",
            customdata=np.stack([
                value_data["Кількість"].fillna(0),
                value_data["Сума"].fillna(0),
                value_data["Відхилення"].fillna(0),
            ], axis=-1),
            line=dict(width=2, shape="spline", smoothing=1.2),
            marker=dict(
                size=np.clip(value_data["Кількість"].fillna(1).astype(float) + 5, 7, 18),
                line=dict(width=1)
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Місяць: %{x}<br>"
                "Середнє: %{y:,.0f}<br>"
                "Кількість записів: %{customdata[0]:,.0f}<br>"
                "Сума: %{customdata[1]:,.0f}<br>"
                "Std: %{customdata[2]:,.0f}<extra></extra>"
            )
        ))

    # FIX: адаптивна розмірність графіка для вкладки "Вплив комбінації факторів".
    lines_count = int(chart_data[combo_col].nunique())
    dynamic_height = max(460, min(950, 420 + lines_count * 28))

    fig.update_layout(
        title=f"Динаміка впливу комбінації: {' + '.join(selected_factors)}",
        xaxis_title="Місяць",
        yaxis_title="Середнє значення",
        height=dynamic_height,
        hovermode="x unified",
        xaxis=dict(categoryorder="array", categoryarray=MONTHS_LIST, automargin=True),
        yaxis=dict(automargin=True, zeroline=True, rangemode="tozero"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=10),
            itemwidth=30,
        ),
        margin=dict(t=80, b=190, l=60, r=40),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"factor_combo_chart_{selected_art}_{'_'.join(selected_factors)}"
    )


def analyze_ratio_factor_impact(df, df_filtered, col_tt, col_article, col_month,
                                 col_ratio, col_plf, selected_art, group_factors):
    if not group_factors:
        return None

    # FIX: норма % береться з ПОВНОГО df
    art_all  = _prep(df[df[col_article] == selected_art], col_month)
    all_fact = _fact_rows(art_all, col_plf)
    all_fact[col_ratio] = pd.to_numeric(all_fact[col_ratio], errors="coerce")

    impact_data = []
    for factor in group_factors:
        factor_impact = (
            all_fact.groupby([factor, "_m"], as_index=False, observed=True)[col_ratio]
            .agg(Average="mean", Count="count", Total="sum", Std="std")
        )
        for _, row in factor_impact.iterrows():
            impact_data.append({
                "Фактор": factor,
                "Значення": row[factor],
                "Місяць": MONTH_LABELS.get(row["_m"], str(row["_m"])),
                "Середнє": row["Average"],
                "Кількість": row["Count"],
                "Сума": row["Total"],
                "Відхилення": row["Std"]
            })

    return pd.DataFrame(impact_data)


def render_ratio_factor_impact_analysis(df, df_filtered, col_tt, col_article, col_month,
                                         col_ratio, col_plf, selected_art, group_factors):
    if not group_factors:
        st.info("Оберіть фактори групування для аналізу впливу.")
        return

    st.markdown(f"""
    <div style="margin-top:20px;margin-bottom:8px;">
      <span style="background:{ORANGE};color:white;font-weight:700;padding:4px 14px;
                   font-size:0.9rem;border-radius:2px;">📊 Аналіз впливу комбінації факторів (% в ТО) — {selected_art}</span>
    </div>""", unsafe_allow_html=True)

    selected_factors = st.multiselect(
        "Оберіть фактори для комбінації:",
        options=group_factors,
        default=group_factors[:1],
        key=f"ratio_factor_combo_multiselect_{selected_art}"
    )

    if not selected_factors:
        st.info("Оберіть хоча б один фактор.")
        return

    art_all = _prep(df[df[col_article] == selected_art].copy(), col_month)
    all_fact = _fact_rows(art_all, col_plf).copy()
    all_fact[col_ratio] = pd.to_numeric(all_fact[col_ratio], errors="coerce")

    combo_col = "Комбінація факторів"

    all_fact[combo_col], valid_selected_factors = _make_combo_col(all_fact, selected_factors)

    if not valid_selected_factors:
        st.warning("Вибрані фактори відсутні в даних.")
        return

    combo_impact = (
        all_fact
        .groupby([combo_col, "_m"], as_index=False, observed=True)[col_ratio]
        .agg(
            Середнє="mean",
            Кількість="count",
            Сума="sum",
            Відхилення="std"
        )
    )

    combo_impact["Місяць"] = combo_impact["_m"].map(MONTH_LABELS)

    if combo_impact.empty:
        st.warning("Недостатньо даних для аналізу комбінації факторів.")
        return

    pivot_impact = combo_impact.pivot_table(
        index=combo_col,
        columns="Місяць",
        values="Середнє",
        aggfunc="sum"
    )

    # FIX: сортуємо місяці за замовчуванням від jan до dec
    pivot_impact = pivot_impact.reindex(
        columns=[m for m in MONTHS_LIST if m in pivot_impact.columns]
    )


    summary_by_combo = combo_impact.groupby(combo_col, as_index=True, observed=True).agg(
        **{
            "Середнє (загальне)": ("Середнє", "mean"),
            "Всього записів": ("Кількість", "sum"),
            "Сума": ("Сума", "sum")
        }
    )

    pivot_impact = pivot_impact.join(summary_by_combo)

    st.markdown(
        f"**Вплив комбінації факторів на % в ТО: {' + '.join(selected_factors)}**"
    )

    st.dataframe(
        pivot_impact.style
            .background_gradient(
                cmap="RdYlGn_r",
                subset=[c for c in pivot_impact.columns if c in MONTHS_LIST]
            )
            .apply(_style_white_na, axis=None)
            .format(lambda v: f"{v:.2f}%" if pd.notna(v) else "", na_rep=""),
        use_container_width=True
    )

    max_lines = st.slider(
        "Кількість комбінацій на графіку",
        3, 30, 10,
        key=f"ratio_factor_combo_top_n_{selected_art}"
    )

    top_combos = (
        combo_impact
        .groupby(combo_col, observed=True)["Сума"]
        .sum()
        .abs()
        .sort_values(ascending=False)
        .head(max_lines)
        .index
    )

    chart_data = combo_impact[combo_impact[combo_col].isin(top_combos)].copy()
    chart_data = chart_data.sort_values([combo_col, "_m"])

    fig = go.Figure()

    # Орієнтир: середнє по всіх показаних комбінаціях за кожен місяць
    avg_line = (
        chart_data
        .groupby("_m", as_index=False, observed=True)["Середнє"]
        .mean()
        .sort_values("_m")
    )
    avg_line["Місяць"] = avg_line["_m"].map(MONTH_LABELS)

    avg_line["Середнє_графік"] = _smooth_chart_series(avg_line["Середнє"], drop_ratio=0.75)

    fig.add_trace(go.Scatter(
        x=avg_line["Місяць"],
        y=avg_line["Середнє_графік"],
        name="Середнє по вибраних комбінаціях",
        mode="lines",
        line=dict(width=4, dash="dash", shape="spline", smoothing=1.2),
        hovertemplate=(
            "<b>Середнє по комбінаціях</b><br>"
            "Місяць: %{x}<br>"
            "Середнє: %{y:.2f}%<extra></extra>"
        )
    ))

    for combo in chart_data[combo_col].dropna().unique():
        value_data = chart_data[chart_data[combo_col] == combo].copy()
        value_data = value_data.sort_values("_m")

        value_data["Середнє_графік"] = _smooth_chart_series(value_data["Середнє"], drop_ratio=0.75)
        text_values = [f"{v:.2f}%" for v in value_data["Середнє_графік"]] if max_lines <= 8 else None

        fig.add_trace(go.Scatter(
            x=value_data["Місяць"],
            y=value_data["Середнє_графік"],
            name=str(combo),
            mode="lines+markers+text" if max_lines <= 8 else "lines+markers",
            text=text_values,
            textposition="top center",
            customdata=np.stack([
                value_data["Кількість"].fillna(0),
                value_data["Сума"].fillna(0),
                value_data["Відхилення"].fillna(0),
            ], axis=-1),
            line=dict(width=2, shape="spline", smoothing=1.2),
            marker=dict(
                size=np.clip(value_data["Кількість"].fillna(1).astype(float) + 5, 7, 18),
                line=dict(width=1)
            ),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Місяць: %{x}<br>"
                "Середнє: %{y:.2f}%<br>"
                "Кількість записів: %{customdata[0]:,.0f}<br>"
                "Сума: %{customdata[1]:.2f}%<br>"
                "Std: %{customdata[2]:.2f}%<extra></extra>"
            )
        ))

    # FIX: адаптивна розмірність % графіка для вкладки "Вплив комбінації факторів".
    lines_count = int(chart_data[combo_col].nunique())
    dynamic_height = max(460, min(950, 420 + lines_count * 28))

    fig.update_layout(
        title=f"Динаміка впливу комбінації факторів (% в ТО): {' + '.join(selected_factors)}",
        xaxis_title="Місяць",
        yaxis_title="Середнє значення (%)",
        yaxis=dict(tickformat=".2f", ticksuffix="%", zeroline=True, automargin=True, rangemode="tozero"),
        height=dynamic_height,
        hovermode="x unified",
        xaxis=dict(categoryorder="array", categoryarray=MONTHS_LIST, automargin=True),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.28,
            xanchor="left",
            x=0,
            font=dict(size=10),
            itemwidth=30,
        ),
        margin=dict(t=80, b=190, l=60, r=40),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key=f"ratio_factor_combo_chart_{selected_art}_{'_'.join(selected_factors)}"
    )




# ── Local filters for combined factor impact tab ──────────────────────────────
def _apply_combo_factor_filters(df_src, group_factors, key_prefix="combo_tab"):
    """Apply filters only inside 'Аналіз впливу комбінації факторів'."""
    if df_src is None or df_src.empty or not group_factors:
        return df_src

    filtered = df_src.copy()
    valid_factors = [f for f in group_factors if f in filtered.columns]

    if not valid_factors:
        return filtered

    st.markdown("#### 🔎 Фільтри тільки для цього блоку")
    st.caption("Ці фільтри впливають лише на блок аналізу комбінації факторів і не змінюють інші таблиці/графіки.")

    cols = st.columns(min(3, len(valid_factors)))

    for i, factor in enumerate(valid_factors):
        with cols[i % len(cols)]:
            options = (
                filtered[factor]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )

            selected_values = st.multiselect(
                factor,
                options=options,
                default=[],
                key=f"{key_prefix}_filter_{factor}"
            )

        if selected_values:
            filtered = filtered[filtered[factor].astype(str).isin(selected_values)]

    return filtered

# ── Combined factor impact tab ────────────────────────────────────────────────
def render_combined_factor_impact_tab(df, df_filtered, col_tt, col_article, col_month,
                                      col_value, col_ratio, col_plf,
                                      articles_to_show, group_factors):
    st.markdown(f"""
    <div style="margin-top:10px;margin-bottom:8px;">
      <span style="background:{ORANGE};color:white;font-weight:700;padding:5px 14px;
                   font-size:0.95rem;border-radius:3px;">
        📊 Аналіз впливу комбінації факторів
      </span>
    </div>""", unsafe_allow_html=True)

    if not group_factors:
        st.info("Оберіть фактори групування в боковому меню.")
        return

    if not articles_to_show:
        st.info("Оберіть хоча б одну статтю витрат.")
        return

    selected_combo_article = st.selectbox(
        "Стаття витрат для аналізу комбінації факторів",
        options=articles_to_show,
        index=0,
        key="combo_tab_selected_article"
    )

    # FIX: локальні фільтри тільки для цього блоку.
    # Беремо df_filtered як базу, але не змінюємо глобальні фільтри та інші вкладки.
    with st.expander("🔎 Фільтри для аналізу комбінації факторів", expanded=True):
        df_combo_base = df_filtered.copy()
        df_combo_base = df_combo_base[df_combo_base[col_article] == selected_combo_article].copy()

        df_combo_filtered = _apply_combo_factor_filters(
            df_combo_base,
            group_factors,
            key_prefix="combo_tab_only"
        )

    if df_combo_filtered.empty:
        st.warning("Після вибраних фільтрів немає даних для аналізу комбінації факторів.")
        return

    analysis_kind_options = ["Абсолютні значення"]
    if col_ratio:
        analysis_kind_options.extend(["% в ТО", "Абсолютні + % в ТО"])

    analysis_kind = st.radio(
        "Тип аналізу",
        options=analysis_kind_options,
        horizontal=True,
        key="combo_tab_analysis_kind"
    )

    st.caption(
        "Комбінація факторів рахується як один спільний розріз, наприклад: "
        "Місто | Формат ТО | Формат Площа. Локальні фільтри вище працюють тільки в цьому блоці."
    )

    if analysis_kind in ("Абсолютні значення", "Абсолютні + % в ТО"):
        render_factor_impact_analysis(
            df_combo_filtered,
            df_combo_filtered,
            col_tt,
            col_article,
            col_month,
            col_value,
            col_plf,
            selected_combo_article,
            group_factors
        )

    if col_ratio and analysis_kind in ("% в ТО", "Абсолютні + % в ТО"):
        st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
        render_ratio_factor_impact_analysis(
            df_combo_filtered,
            df_combo_filtered,
            col_tt,
            col_article,
            col_month,
            col_ratio,
            col_plf,
            selected_combo_article,
            group_factors
        )


# ── Statistical factor analysis ───────────────────────────────────────────────
def _safe_div(num, den):
    return num / den if den not in (0, None) and pd.notna(den) else 0


def _factor_stat_models(data, factor, value_col):
    """Multi-model statistical impact analysis for one categorical factor."""
    work = data[[factor, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[factor, value_col])
    work[factor] = work[factor].astype(str)

    n = len(work)
    k = work[factor].nunique(dropna=True)

    if n < 5 or k < 2:
        return None, pd.DataFrame()

    overall_mean = work[value_col].mean()
    overall_median = work[value_col].median()

    groups = work.groupby(factor, observed=True)[value_col]
    means = groups.mean()
    medians = groups.median()
    counts = groups.count()
    sums = groups.sum()
    stds = groups.std().fillna(0)

    # Model 1: ANOVA / Eta²
    ss_total = ((work[value_col] - overall_mean) ** 2).sum()
    ss_between = ((means - overall_mean) ** 2 * counts).sum()
    ss_within = groups.apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    eta2 = _safe_div(ss_between, ss_total)

    df_between = k - 1
    df_within = n - k
    f_stat = _safe_div(ss_between / df_between, ss_within / df_within) if df_between > 0 and df_within > 0 else 0

    # Model 2: Correlation ratio η
    correlation_eta = float(np.sqrt(max(eta2, 0)))

    # Model 3: Cramér V after binning numeric target into quantile bins
    cramer_v = 0.0
    try:
        q = min(5, max(2, work[value_col].nunique()))
        target_bins = pd.qcut(work[value_col], q=q, duplicates="drop")
        contingency = pd.crosstab(work[factor], target_bins)
        if contingency.shape[0] > 1 and contingency.shape[1] > 1:
            obs = contingency.to_numpy(dtype=float)
            total = obs.sum()
            expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / total
            chi2 = np.nansum((obs - expected) ** 2 / np.where(expected == 0, np.nan, expected))
            r, c = obs.shape
            cramer_v = float(np.sqrt(_safe_div(chi2, total * (min(r - 1, c - 1)))))
            if not np.isfinite(cramer_v):
                cramer_v = 0.0
    except Exception:
        cramer_v = 0.0

    # Model 4: group-mean prediction vs global mean, measured by MAE reduction
    pred_global = np.full(n, overall_mean)
    pred_group_mean = work[factor].map(means.to_dict()).astype(float)
    mae_global = np.mean(np.abs(work[value_col] - pred_global))
    mae_group = np.mean(np.abs(work[value_col] - pred_group_mean))
    mae_reduction = max(0, _safe_div(mae_global - mae_group, mae_global))

    # Model 5: group-median prediction vs global median, robust to outliers
    pred_group_median = work[factor].map(medians.to_dict()).astype(float)
    mae_global_median = np.mean(np.abs(work[value_col] - overall_median))
    mae_group_median = np.mean(np.abs(work[value_col] - pred_group_median))
    robust_reduction = max(0, _safe_div(mae_global_median - mae_group_median, mae_global_median))

    # Model 6: entropy reduction / information gain after factor split
    entropy_reduction = 0.0
    try:
        bins = np.histogram_bin_edges(work[value_col], bins=min(10, max(2, work[value_col].nunique())))
        hist_total, _ = np.histogram(work[value_col], bins=bins)
        p_total = hist_total / hist_total.sum() if hist_total.sum() else np.array([])
        entropy_total = -np.nansum(p_total * np.log2(p_total + 1e-12)) if len(p_total) else 0
        entropy_cond = 0.0
        for _, vals in groups:
            hist, _ = np.histogram(vals, bins=bins)
            if hist.sum() == 0:
                continue
            p = hist / hist.sum()
            entropy_cond += (len(vals) / n) * (-np.nansum(p * np.log2(p + 1e-12)))
        entropy_reduction = max(0, _safe_div(entropy_total - entropy_cond, entropy_total))
    except Exception:
        entropy_reduction = 0.0

    # Model 7: Top vs Bottom lift — business spread between category averages
    sorted_means = means.sort_values()
    top_bottom_lift = (
        _safe_div(sorted_means.iloc[-1] - sorted_means.iloc[0], abs(overall_mean))
        if len(sorted_means) >= 2 else 0
    )

    # Model 8: weighted deviation of group means from global mean
    std_total = work[value_col].std()
    weighted_mean_deviation = (np.abs(means - overall_mean) * counts).sum() / counts.sum()
    normalized_deviation = _safe_div(weighted_mean_deviation, std_total)

    rating_values = [
        eta2,
        correlation_eta,
        cramer_v,
        mae_reduction,
        robust_reduction,
        entropy_reduction,
        min(abs(top_bottom_lift), 1),
        min(abs(normalized_deviation), 1),
    ]
    integrated_rating = float(np.nanmean([v for v in rating_values if pd.notna(v)])) if rating_values else 0.0

    detail = pd.DataFrame({
        "Значення фактора": means.index.astype(str),
        "Кількість": counts.values,
        "Сума": sums.values,
        "Середнє": means.values,
        "Медіана": medians.values,
        "Std": stds.values,
        "Відхилення від заг. середнього": (means - overall_mean).values,
        "Lift до заг. середнього": [_safe_div(v - overall_mean, abs(overall_mean)) for v in means.values],
    }).sort_values("Сума", ascending=False)

    result = {
        "Фактор": factor,
        "Записів": n,
        "Унікальних значень": k,
        "ANOVA Eta²": eta2,
        "Correlation η": correlation_eta,
        "Cramér V (binned)": cramer_v,
        "F-stat": f_stat,
        "Mean model / MAE покращення": mae_reduction,
        "Median model / MAE покращення": robust_reduction,
        "Entropy ↓": entropy_reduction,
        "Top-Bottom lift": top_bottom_lift,
        "Норм. відхилення середніх": normalized_deviation,
        "Інтегральний рейтинг впливу": integrated_rating,
        "Середнє по статті": overall_mean,
        "Медіана по статті": overall_median,
    }

    return result, detail


def analyze_statistical_factor_models(df, col_article, col_value, col_plf,
                                      selected_art, group_factors):
    if not group_factors:
        return pd.DataFrame(), {}

    art_df = df[df[col_article] == selected_art].copy()
    art_df = _fact_rows(art_df, col_plf).copy()
    art_df[col_value] = pd.to_numeric(art_df[col_value], errors="coerce")
    art_df = art_df.dropna(subset=[col_value])

    rows = []
    details = {}

    for factor in group_factors:
        if factor not in art_df.columns:
            continue

        row, detail = _factor_stat_models(art_df, factor, col_value)
        if row is not None:
            rows.append(row)
            details[factor] = detail

    if not rows:
        return pd.DataFrame(), details

    result = pd.DataFrame(rows)

    score_cols = [
        "ANOVA Eta²",
        "Correlation η",
        "Cramér V (binned)",
        "Mean model / MAE покращення",
        "Median model / MAE покращення",
        "Entropy ↓",
        "Top-Bottom lift",
        "Норм. відхилення середніх",
    ]
    valid_score_cols = [c for c in score_cols if c in result.columns]
    if "Інтегральний рейтинг впливу" not in result.columns:
        result["Інтегральний рейтинг впливу"] = result[valid_score_cols].mean(axis=1) if valid_score_cols else 0.0
    else:
        result["Інтегральний рейтинг впливу"] = result["Інтегральний рейтинг впливу"].fillna(
            result[valid_score_cols].mean(axis=1) if valid_score_cols else 0.0
        )
    result = result.sort_values("Інтегральний рейтинг впливу", ascending=False)

    return result, details


def analyze_combination_statistical_impact(df, col_article, col_value, col_plf,
                                           selected_art, selected_factors):
    if not selected_factors:
        return pd.DataFrame(), pd.DataFrame()

    art_df = df[df[col_article] == selected_art].copy()
    art_df = _fact_rows(art_df, col_plf).copy()
    art_df[col_value] = pd.to_numeric(art_df[col_value], errors="coerce")
    art_df = art_df.dropna(subset=[col_value])

    combo_col = "Комбінація факторів"
    art_df[combo_col], valid_selected_factors = _make_combo_col(art_df, selected_factors)

    if not valid_selected_factors:
        return pd.DataFrame(), pd.DataFrame()

    row, detail = _factor_stat_models(art_df, combo_col, col_value)
    if row is None:
        return pd.DataFrame(), pd.DataFrame()

    score_cols = [
        "ANOVA Eta²",
        "Correlation η",
        "Cramér V (binned)",
        "Mean model / MAE покращення",
        "Median model / MAE покращення",
        "Entropy ↓",
        "Top-Bottom lift",
        "Норм. відхилення середніх",
    ]
    valid_score_cols = [c for c in score_cols if c in row and pd.notna(row.get(c))]
    if "Інтегральний рейтинг впливу" not in row:
        row["Інтегральний рейтинг впливу"] = (
            float(np.mean([row[c] for c in valid_score_cols])) if valid_score_cols else 0.0
        )

    row["Фактор"] = " + ".join(selected_factors)
    row["Тип"] = "Комбінація"

    return pd.DataFrame([row]), detail


def render_statistical_analysis_tab(df, col_article, col_value, col_plf,
                                    articles_to_show, group_factors):
    st.markdown(f"""
    <div style="margin-top:10px;margin-bottom:8px;">
      <span style="background:#2c3e50;color:white;font-weight:700;padding:5px 14px;
                   font-size:0.95rem;border-radius:3px;">
        📊 Статистичний аналіз впливу факторів на статтю витрат
      </span>
    </div>""", unsafe_allow_html=True)

    if not group_factors:
        st.info("Оберіть фактори групування в боковому меню.")
        return

    selected_stat_article = st.selectbox(
        "Стаття витрат для статистичного аналізу",
        options=articles_to_show,
        index=0,
        key="stat_selected_article"
    )

    selected_stat_factors = st.multiselect(
        "Фактори для статистичного аналізу",
        options=group_factors,
        default=group_factors,
        key="stat_selected_factors"
    )

    if not selected_stat_factors:
        st.info("Оберіть хоча б один фактор.")
        return

    stat_df, detail_map = analyze_statistical_factor_models(
        df, col_article, col_value, col_plf,
        selected_stat_article, selected_stat_factors
    )

    if stat_df.empty:
        st.warning("Недостатньо даних для статистичного аналізу.")
        return

    st.markdown("#### 1) Вплив кожного фактора за різними моделями")

    pct_cols = [
        "ANOVA Eta²",
        "Correlation η",
        "Cramér V (binned)",
        "Mean model / MAE покращення",
        "Median model / MAE покращення",
        "Entropy ↓",
        "Інтегральний рейтинг впливу",
    ]

    fmt_map = {
        "ANOVA Eta²": "{:.2%}",
        "Correlation η": "{:.2%}",
        "Cramér V (binned)": "{:.2%}",
        "Mean model / MAE покращення": "{:.2%}",
        "Median model / MAE покращення": "{:.2%}",
        "Entropy ↓": "{:.2%}",
        "Top-Bottom lift": "{:.2%}",
        "Норм. відхилення середніх": "{:.3f}",
        "Інтегральний рейтинг впливу": "{:.2%}",
        "F-stat": "{:,.2f}",
        "Середнє по статті": "{:,.0f}",
        "Медіана по статті": "{:,.0f}",
    }

    st.dataframe(
        stat_df.style
            .background_gradient(cmap="Greens", subset=["Інтегральний рейтинг впливу"])
            .background_gradient(cmap="Blues", subset=["ANOVA Eta²"])
            .background_gradient(cmap="Purples", subset=["Cramér V (binned)"] if "Cramér V (binned)" in stat_df.columns else [])
            .apply(_style_white_na, axis=None)
            .format(fmt_map, na_rep=""),
        use_container_width=True
    )

    chart_cols = [
        c for c in [
            "ANOVA Eta²",
            "Correlation η",
            "Cramér V (binned)",
            "Mean model / MAE покращення",
            "Median model / MAE покращення",
            "Entropy ↓",
            "Інтегральний рейтинг впливу",
        ] if c in stat_df.columns
    ]

    fig = go.Figure()
    for c in chart_cols:
        fig.add_trace(go.Bar(
            x=stat_df["Фактор"],
            y=stat_df[c],
            name=c,
            text=[f"{v:.1%}" for v in stat_df[c]],
            textposition="auto",
        ))

    fig.update_layout(
        title=f"Порівняння сили впливу факторів — {selected_stat_article}",
        xaxis_title="Фактор",
        yaxis_title="Сила впливу",
        yaxis=dict(tickformat=".0%"),
        barmode="group",
        height=430,
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(t=50, b=90, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"stat_models_chart_{selected_stat_article}")

    top_factor = stat_df.iloc[0]
    st.success(
        f"🏆 Найбільш впливовий фактор: **{top_factor['Фактор']}** · "
        f"інтегральний рейтинг: **{top_factor['Інтегральний рейтинг впливу']:.2%}**"
    )

    with st.expander("🔎 Деталізація по значеннях факторів", expanded=False):
        factor_for_detail = st.selectbox(
            "Оберіть фактор для деталізації",
            options=list(detail_map.keys()),
            key=f"stat_detail_factor_{selected_stat_article}"
        )

        detail_df = detail_map.get(factor_for_detail, pd.DataFrame())
        if not detail_df.empty:
            st.dataframe(
                detail_df.style
                    .background_gradient(cmap="RdYlGn_r", subset=["Відхилення від заг. середнього"])
                    .apply(_style_white_na, axis=None)
                    .format({
                        "Кількість": "{:,.0f}",
                        "Сума": "{:,.0f}",
                        "Середнє": "{:,.0f}",
                        "Медіана": "{:,.0f}",
                        "Std": "{:,.0f}",
                        "Відхилення від заг. середнього": "{:,.0f}",
                        "Lift до заг. середнього": "{:.2%}",
                    }),
                use_container_width=True
            )

    st.markdown("#### 2) Статистичний аналіз комбінації факторів")

    combo_factors = st.multiselect(
        "Оберіть фактори для комбінованої статистичної моделі",
        options=selected_stat_factors,
        default=selected_stat_factors[:min(2, len(selected_stat_factors))],
        key=f"stat_combo_factors_{selected_stat_article}"
    )

    if combo_factors:
        combo_stat, combo_detail = analyze_combination_statistical_impact(
            df, col_article, col_value, col_plf,
            selected_stat_article, combo_factors
        )

        if combo_stat.empty:
            st.info("Недостатньо даних для комбінованої статистичної моделі.")
        else:
            if "Інтегральний рейтинг впливу" not in combo_stat.columns:
                score_cols = [
                    "ANOVA Eta²",
                    "Correlation η",
                    "Cramér V (binned)",
                    "Mean model / MAE покращення",
                    "Median model / MAE покращення",
                    "Entropy ↓",
                    "Top-Bottom lift",
                    "Норм. відхилення середніх",
                ]
                valid_score_cols = [c for c in score_cols if c in combo_stat.columns]
                combo_stat["Інтегральний рейтинг впливу"] = (
                    combo_stat[valid_score_cols].mean(axis=1) if valid_score_cols else 0.0
                )

            combo_style = combo_stat.style.apply(_style_white_na, axis=None).format(fmt_map, na_rep="")
            if "Інтегральний рейтинг впливу" in combo_stat.columns:
                combo_style = combo_style.background_gradient(
                    cmap="Purples",
                    subset=["Інтегральний рейтинг впливу"]
                )

            st.dataframe(
                combo_style,
                use_container_width=True
            )

            max_combo_rows = st.slider(
                "Кількість комбінацій у деталізації",
                5, 100, 25,
                key=f"stat_combo_rows_{selected_stat_article}"
            )

            st.dataframe(
                combo_detail.head(max_combo_rows).style
                    .background_gradient(cmap="RdYlGn_r", subset=["Відхилення від заг. середнього"])
                    .apply(_style_white_na, axis=None)
                    .format({
                        "Кількість": "{:,.0f}",
                        "Сума": "{:,.0f}",
                        "Середнє": "{:,.0f}",
                        "Медіана": "{:,.0f}",
                        "Std": "{:,.0f}",
                        "Відхилення від заг. середнього": "{:,.0f}",
                        "Lift до заг. середнього": "{:.2%}",
                    }),
                use_container_width=True
            )

    st.caption(
        "Моделі: ANOVA Eta² показує частку варіації витрат, яку пояснює фактор; "
        "Correlation η та Cramér V показують силу нелінійного/категоріального зв’язку; "
        "Mean/Median model показують зменшення MAE при прогнозі через групові середні/медіани; "
        "Entropy ↓ показує інформаційний виграш; Top-Bottom lift — бізнес-розкид між групами; "
        "інтегральний рейтинг — середня оцінка сили впливу за моделями."
    )


# ── Heatmap builders ─────────────────────────────────────────────────────────

def _clean_heat_group_factors(group_factors, col_month, cols=None):
    """
    Для Heatmap колонка місяця не має йти як окремий текстовий фактор.
    Якщо у факторах вибрано "Місяць", додаємо канонічний числовий ключ _m.

    Важливо:
    - _m використовується саме як фактор для Average / Average %;
    - текстова колонка місяця не дублюється в groupby;
    - фактичні значення все одно групуються по _m для побудови колонок heatmap.
    """
    clean = []
    use_month_factor = False

    for f in (group_factors or []):
        if not f:
            continue

        # Якщо користувач вибрав колонку "Місяць" або вже передано _m,
        # використовуємо _m як фактор у розрахунку нормативу.
        if f == col_month or f == "_m":
            use_month_factor = True
            continue

        if cols is not None and f not in cols:
            continue

        clean.append(f)

    if use_month_factor:
        clean.append("_m")

    return list(dict.fromkeys(clean))


def build_heat_data(df, df_filtered, col_tt, col_article, col_month, col_value,
                    col_plf, group_factors, articles_to_show, mode):
    """
    Heatmap для абсолютних значень.

    FIX:
    - якщо у факторах вибрано "Місяць", Average_Calc / Std рахуються в розрізі _m;
    - якщо "Місяць" НЕ вибрано, Average_Calc рахується як загальний норматив за вибраними факторами;
    - _m завжди лишається в tt_table для побудови місячних колонок heatmap;
    - не дублюємо _m у groupby та merge.
    """
    df_num = df.copy()
    df_num[col_value] = pd.to_numeric(df_num[col_value], errors="coerce")
    df_num = _prep(df_num, col_month)

    heat_group_factors = _clean_heat_group_factors(
        group_factors,
        col_month,
        cols=df_num.columns
    )

    all_fact = _fact_rows(df_num, col_plf)
    all_fact = all_fact[all_fact[col_article].isin(articles_to_show)]

    # Якщо користувач вибрав Місяць як фактор — heat_group_factors містить _m.
    # Якщо не вибрав — _m тут немає, отже Average буде без місячного фактору.
    avg_grp_cols = list(dict.fromkeys(heat_group_factors + [col_article]))

    global_avg_std = (
        all_fact
        .groupby(avg_grp_cols, as_index=False, observed=True)[col_value]
        .agg(Average_Calc="mean", Std="std")
    )

    filt = df_filtered.copy()
    filt[col_value] = pd.to_numeric(filt[col_value], errors="coerce")
    filt = _prep(filt, col_month)

    data_heat = _fact_rows(filt, col_plf)
    data_heat = data_heat[data_heat[col_article].isin(articles_to_show)]

    # _m тут потрібен завжди, бо heatmap має колонки по місяцях.
    # Якщо _m уже є в heat_group_factors, dict.fromkeys прибере дубль.
    tt_grp = list(dict.fromkeys([col_tt] + heat_group_factors + ["_m", col_article]))

    tt_table = (
        data_heat
        .groupby(tt_grp, as_index=False, observed=True)[col_value]
        .sum()
        .rename(columns={col_value: "Fact"})
    )

    merge_cols = list(dict.fromkeys(heat_group_factors + [col_article]))

    tt_table = pd.merge(
        tt_table,
        global_avg_std,
        on=merge_cols,
        how="left"
    )

    tt_table["Fact"] = pd.to_numeric(tt_table["Fact"], errors="coerce").fillna(0)
    tt_table["Average_Calc"] = pd.to_numeric(tt_table["Average_Calc"], errors="coerce").fillna(0)
    tt_table["Std"] = pd.to_numeric(tt_table["Std"], errors="coerce").replace(0, np.nan)

    # Якщо факту в конкретному місяці немає — не показуємо норму як аномалію.
    tt_table.loc[tt_table["Fact"].eq(0), "Average_Calc"] = 0

    tt_table["Delta"] = tt_table["Fact"] - tt_table["Average_Calc"]
    tt_table["Delta_%"] = (
        tt_table["Delta"] / tt_table["Average_Calc"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    tt_table["Z"] = (
        tt_table["Delta"] / tt_table["Std"]
    ).replace([np.inf, -np.inf], np.nan)

    val_col = {
        "Delta": "Delta",
        "Delta %": "Delta_%",
        "Z-score": "Z",
        "Fact": "Fact",
        "Average": "Average_Calc",
    }.get(mode, "Delta")

    heat = tt_table.pivot_table(
        index=col_tt,
        columns="_m",
        values=val_col,
        aggfunc="sum"
    )

    heat = _fill_heat_cols(heat)
    heat["РАЗОМ"] = heat.sum(axis=1, numeric_only=True)

    return heat, tt_table, val_col


def build_ratio_heat_data(df, df_filtered, col_tt, col_article, col_month,
                           col_ratio, col_plf, articles_to_show, mode,
                           group_factors=None):
    """
    Heatmap для % в ТО.

    FIX:
    - якщо у факторах вибрано "Місяць", Average % / Std рахуються в розрізі _m;
    - якщо "Місяць" НЕ вибрано, Average % рахується як загальний норматив за вибраними факторами;
    - _m завжди лишається в tt_table для побудови місячних колонок heatmap;
    - не дублюємо _m у groupby та merge.
    """
    if group_factors is None:
        group_factors = []

    df_num = df_filtered.copy()
    df_num[col_ratio] = pd.to_numeric(df_num[col_ratio], errors="coerce")
    df_num = _prep(df_num, col_month)

    has_plf = col_plf and col_plf in df_num.columns

    data_heat = (
        df_num[(df_num[col_plf] == "F") & df_num[col_article].isin(articles_to_show)].copy()
        if has_plf
        else df_num[df_num[col_article].isin(articles_to_show)].copy()
    )

    ratio_group_factors = _clean_heat_group_factors(
        group_factors,
        col_month,
        cols=df_num.columns
    )

    df_all_num = df.copy()
    df_all_num[col_ratio] = pd.to_numeric(df_all_num[col_ratio], errors="coerce")
    df_all_num = _prep(df_all_num, col_month)

    avg_src = (_fact_rows(df_all_num, col_plf) if has_plf else df_all_num)
    avg_src = avg_src[avg_src[col_article].isin(articles_to_show)]

    # Якщо користувач вибрав Місяць як фактор — ratio_group_factors містить _m.
    # Якщо не вибрав — _m тут немає, отже Average % буде без місячного фактору.
    grp_cols = list(dict.fromkeys(ratio_group_factors + [col_article]))

    global_avg = (
        avg_src
        .groupby(grp_cols, as_index=False, observed=True)[col_ratio]
        .agg(Average_Calc="mean", Std="std")
    )

    # _m тут потрібен завжди, бо heatmap має колонки по місяцях.
    # Якщо _m уже є в ratio_group_factors, dict.fromkeys прибере дубль.
    tt_grp = list(dict.fromkeys([col_tt] + ratio_group_factors + ["_m", col_article]))

    tt_table = (
        data_heat
        .groupby(tt_grp, as_index=False, observed=True)[col_ratio]
        .mean()
        .rename(columns={col_ratio: "Fact"})
    )

    tt_table = pd.merge(
        tt_table,
        global_avg,
        on=grp_cols,
        how="left"
    )

    tt_table["Fact"] = pd.to_numeric(tt_table["Fact"], errors="coerce").fillna(0)
    tt_table["Average_Calc"] = pd.to_numeric(tt_table["Average_Calc"], errors="coerce").fillna(0)
    tt_table["Std"] = pd.to_numeric(tt_table["Std"], errors="coerce").replace(0, np.nan)

    # Якщо фактичного % немає, не показуємо Average % у цьому місяці як аномалію.
    tt_table.loc[tt_table["Fact"].eq(0), "Average_Calc"] = 0

    tt_table["Delta"] = tt_table["Fact"] - tt_table["Average_Calc"]
    tt_table["Delta_%"] = (
        tt_table["Delta"] / tt_table["Average_Calc"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)

    tt_table["Z"] = (
        tt_table["Delta"] / tt_table["Std"]
    ).replace([np.inf, -np.inf], np.nan)

    val_col = {
        "Delta": "Delta",
        "Delta %": "Delta_%",
        "Z-score": "Z",
        "Fact": "Fact",
        "Average": "Average_Calc",
    }.get(mode, "Delta")

    heat = tt_table.pivot_table(
        index=col_tt,
        columns="_m",
        values=val_col,
        aggfunc="mean"
    )

    heat = _fill_heat_cols(heat)
    heat["РАЗОМ"] = heat.mean(axis=1, numeric_only=True)

    return heat, tt_table, val_col


def _fill_heat_cols(heat):
    # Missing months/cells must stay NaN, not None/0.
    # _style_white_na renders empty cells as white.
    for m in range(1, 13):
        if m not in heat.columns:
            heat[m] = np.nan
    heat = heat[sorted(heat.columns)]
    heat.columns = [MONTH_LABELS.get(int(c), str(c)) for c in heat.columns]
    return heat.replace([np.inf, -np.inf], np.nan)


# ── TT Pivot ─────────────────────────────────────────────────────────────────

def build_tt_pivot(df_filtered, col_tt, col_article, col_month, col_value,
                   col_plf, articles_to_show):
    """
    Швидка зведена по ТТ без подвійного циклу ТТ × Стаття.
    Раніше для кожного ТТ і кожної статті окремо запускались groupby.
    Тепер один groupby рахує всі місяці одразу.
    """
    if df_filtered is None or df_filtered.empty or not articles_to_show:
        return pd.DataFrame()

    df_num = _prep(df_filtered, col_month)
    df_num = df_num[df_num[col_article].isin(articles_to_show)].copy()
    if df_num.empty:
        return pd.DataFrame()

    df_num[col_value] = pd.to_numeric(df_num[col_value], errors="coerce").fillna(0)

    grp = (
        df_num
        .groupby([col_tt, col_article, "_m", col_plf], as_index=False, observed=True)[col_value]
        .sum()
    )

    wide = grp.pivot_table(
        index=[col_tt, col_article],
        columns=["_m", col_plf],
        values=col_value,
        aggfunc="sum",
        fill_value=0,
        observed=True,
    )

    rows = wide.reset_index()
    rows = rows.rename(columns={col_tt: "ТТ", col_article: "Стаття"})

    out = pd.DataFrame({
        "ТТ": rows["ТТ"],
        "Стаття": rows["Стаття"],
    })

    for m in range(1, 13):
        ml = MONTH_LABELS[m]

        plan_key = (m, "PL")
        fact_key = (m, "F")

        out[f"plan_{ml}"] = wide[plan_key].to_numpy() if plan_key in wide.columns else 0
        out[f"fact_{ml}"] = wide[fact_key].to_numpy() if fact_key in wide.columns else 0

    plan_cols = [f"plan_{MONTH_LABELS[m]}" for m in range(1, 13)]
    fact_cols = [f"fact_{MONTH_LABELS[m]}" for m in range(1, 13)]
    num_cols = plan_cols + fact_cols

    out["Plan_РАЗОМ"] = out[plan_cols].sum(axis=1)
    out["Fact_РАЗОМ"] = out[fact_cols].sum(axis=1)
    out["Delta_РАЗОМ"] = out["Fact_РАЗОМ"] - out["Plan_РАЗОМ"]
    out["Pct_РАЗОМ"] = np.where(
        out["Plan_РАЗОМ"].ne(0),
        (out["Fact_РАЗОМ"] / out["Plan_РАЗОМ"] - 1) * 100,
        np.nan
    )

    df_agg = out.groupby("ТТ", as_index=False, observed=True)[num_cols + ["Plan_РАЗОМ", "Fact_РАЗОМ", "Delta_РАЗОМ"]].sum()

    df_agg["Pct_РАЗОМ"] = np.where(
        df_agg["Plan_РАЗОМ"].ne(0),
        (df_agg["Fact_РАЗОМ"] / df_agg["Plan_РАЗОМ"] - 1) * 100,
        np.nan
    )

    for m in range(1, 13):
        ml = MONTH_LABELS[m]
        df_agg[f"delta_{ml}"] = df_agg[f"fact_{ml}"] - df_agg[f"plan_{ml}"]
        df_agg[f"pct_{ml}"] = np.where(
            df_agg[f"plan_{ml}"].ne(0),
            (df_agg[f"fact_{ml}"] / df_agg[f"plan_{ml}"] - 1) * 100,
            np.nan
        )

    return df_agg

def _th(bg_color):
    return (f"background:{bg_color};color:white;font-weight:bold;border:1px solid #aaa;"
            "padding:4px 8px;text-align:center;font-size:0.78rem;")

TD  = "border:1px solid #ccc;padding:3px 7px;text-align:right;font-size:0.78rem;"
TL  = "border:1px solid #ccc;padding:3px 7px;font-size:0.78rem;font-weight:600;white-space:nowrap;"


def _style_white_na(data):
    """Make all None/NaN cells white in pandas Styler tables."""
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(
            np.where(pd.isna(data), "background-color: white !important; color: black !important;", ""),
            index=data.index,
            columns=data.columns,
        )
    return ["background-color: white !important; color: black !important;" if pd.isna(v) else "" for v in data]


def _build_tt_display_map(df_src, col_tt, col_division=None):
    """Map technical TT key to display name from Підрозділ. Calculations remain on TT."""
    if (
        df_src is None or df_src.empty
        or not col_tt or col_tt not in df_src.columns
        or not col_division or col_division not in df_src.columns
    ):
        return {}

    work = df_src[[col_tt, col_division]].dropna(subset=[col_tt]).copy()
    work[col_tt] = work[col_tt].astype(str)
    work[col_division] = work[col_division].where(work[col_division].notna(), "")

    display_map = {}
    for tt_key, vals in work.groupby(col_tt, observed=True)[col_division]:
        clean_vals = [str(v).strip() for v in vals.tolist() if str(v).strip() and str(v).strip().lower() != "nan"]
        display_map[tt_key] = clean_vals[0] if clean_vals else tt_key
    return display_map


def _tt_display_label(tt_value, tt_display_map):
    """Return display label for TT using Підрозділ if available."""
    key = str(tt_value)
    return tt_display_map.get(key, tt_value)


def _display_tt_series_index(series, df_src, col_tt, col_division=None):
    """For UI only: replace Series TT index with Підрозділ labels; calculations stay grouped by TT."""
    out = series.copy()
    tt_display_map = _build_tt_display_map(df_src, col_tt, col_division)
    if tt_display_map:
        out.index = [_tt_display_label(v, tt_display_map) for v in out.index]
    return out


def _replace_tt_index_with_division(df_in, df_src, col_tt, col_division=None, index_name="Підрозділ"):
    """For UI display only: replace TT index labels with Підрозділ labels."""
    out = df_in.copy()
    tt_display_map = _build_tt_display_map(df_src, col_tt, col_division)
    if tt_display_map:
        out.index = [_tt_display_label(v, tt_display_map) for v in out.index]
        out.index.name = index_name
    return out


def _add_division_display_column(df_in, df_src, col_tt, col_division=None, display_col="Підрозділ"):
    """For UI display only: add Підрозділ display column and keep calculations grouped by TT."""
    out = df_in.copy()
    tt_display_map = _build_tt_display_map(df_src, col_tt, col_division)
    if col_tt in out.columns:
        out[display_col] = out[col_tt].map(lambda v: _tt_display_label(v, tt_display_map))
    return out




# ── Heatmap TOP/ANTITOP rating + Excel helpers ───────────────────────────────
def _get_heatmap_rating_factor_options(df_src, col_tt=None, col_division=None, group_factors=None):
    """
    Повертає список колонок, за якими можна дивитися рейтинг у Heatmap TOP/ANTITOP.
    Додає стандартні бізнес-розрізи + усі фактори групування, які реально є в df.
    """
    if df_src is None or df_src.empty:
        return []

    cols = df_src.columns.tolist()
    options = []

    def add_col(c):
        if c and c in cols and c not in options:
            options.append(c)

    add_col(col_division)
    add_col(col_tt)

    candidates = [
        _find_col_by_keywords(cols, exact=["Область", "Області", "Region", "Регіон", "Регион"], contains=["област", "регіон", "регион", "region"]),
        _find_col_by_keywords(cols, exact=["Місто", "Город", "City"], contains=["місто", "город", "city"]),
        _find_col_by_keywords(cols, exact=["Формат ТО", "Формат"], contains=["формат то", "формат", "format"]),
        _find_col_by_keywords(cols, exact=["Формат2", "Формат 2", "Format2", "Format 2"], contains=["формат2", "формат 2", "format2", "format 2"]),
        _find_col_by_keywords(cols, exact=["Мегасегмент"], contains=["мегасегмент", "mega"]),
        _find_col_by_keywords(cols, exact=["Площа", "Площадь", "Area"], contains=["площа", "площад", "area"]),
        _find_col_by_keywords(cols, exact=["Рік відкриття", "Год открытия", "Year opened"], contains=["рік відкрит", "год открыт", "year open"]),
    ]
    for c in candidates:
        add_col(c)

    for c in (group_factors or []):
        add_col(c)

    return options

def _prepare_heatmap_rating_table(tt_table, df_src, col_tt, val_col, rating_col, aggfunc="sum"):
    """
    Формує TOP/ANTITOP не тільки по ТТ, а по вибраному фактору:
    область, місто, Формат ТО, Формат 2, Мегасегмент або будь-який group_factor.
    Розрахунок tt_table не змінюється — тільки групування рейтингу.
    """
    if tt_table is None or tt_table.empty or not rating_col or val_col not in tt_table.columns:
        return pd.DataFrame(columns=["Підрозділ", val_col])

    work = tt_table.copy()
    work[val_col] = pd.to_numeric(work[val_col], errors="coerce")

    if rating_col not in work.columns:
        if df_src is not None and rating_col in df_src.columns and col_tt in df_src.columns and col_tt in work.columns:
            rating_map = (
                df_src[[col_tt, rating_col]]
                .dropna(subset=[col_tt])
                .drop_duplicates(subset=[col_tt], keep="first")
            )
            work = work.merge(rating_map, on=col_tt, how="left")
        elif rating_col == col_tt and col_tt in work.columns:
            work[rating_col] = work[col_tt]

    if rating_col not in work.columns:
        return pd.DataFrame(columns=["Підрозділ", val_col])

    work[rating_col] = work[rating_col].where(work[rating_col].notna(), "Без значення").astype(str).str.strip()
    work.loc[work[rating_col].eq("") | work[rating_col].str.lower().eq("nan"), rating_col] = "Без значення"

    if aggfunc == "mean":
        rating = work.groupby(rating_col, as_index=False, observed=True)[val_col].mean()
    else:
        rating = work.groupby(rating_col, as_index=False, observed=True)[val_col].sum()

    rating = rating.rename(columns={rating_col: "Підрозділ"})
    rating[val_col] = pd.to_numeric(rating[val_col], errors="coerce").fillna(0)
    return rating

def _safe_xlsx_filename(name, prefix="block"):
    """Безпечна назва Excel-файлу для окремого блоку."""
    s = str(name) if name is not None else prefix
    for ch in ['\\', '/', '*', '?', ':', '[', ']', '"', "'", '<', '>', '|']:
        s = s.replace(ch, '_')
    s = s.strip().replace(' ', '_')
    return (s[:90] or prefix) + ".xlsx"

def _heatmap_block_to_excel_bytes(title, heat_df, top_df=None, antitop_df=None,
                                  val_col=None, header_color="5B2D8E", percent=False):
    """
    Експорт одного Heatmap-блоку в Excel зі збереженням форматування:
    - окремий лист Heatmap з кольоровою шкалою;
    - окремий лист TOP_ANTITOP, якщо передані top_df / antitop_df;
    - числовий або %-формат залежно від percent.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    def norm_hex(x, default="5B2D8E"):
        x = str(x or default).replace("#", "").strip().upper()
        return x if len(x) == 6 else default

    def safe_sheet_name(name):
        s = str(name or "Sheet")
        for ch in ['\\', '/', '*', '?', ':', '[', ']']:
            s = s.replace(ch, '_')
        return s[:31] or "Sheet"

    header_hex = norm_hex(header_color)
    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    num_fmt = '0.00"%"' if percent else '# ##0;-# ##0;-'

    wb = Workbook()
    ws = wb.active
    ws.title = "Heatmap"

    out = heat_df.copy() if heat_df is not None else pd.DataFrame()
    out = out.replace([np.inf, -np.inf], np.nan)
    if out.index.name is None:
        out.index.name = "Підрозділ"
    out = out.reset_index()

    max_col = max(1, len(out.columns))
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
    title_cell = ws.cell(row=1, column=1, value=str(title))
    title_cell.font = Font(bold=True, color="FFFFFF", size=12)
    title_cell.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
    title_cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24

    for ci, h in enumerate(out.columns, 1):
        cell = ws.cell(row=2, column=ci, value=str(h))
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border

    for ri, row in enumerate(out.itertuples(index=False), 3):
        for ci, v in enumerate(row, 1):
            value = None if pd.isna(v) else v
            if isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.floating,)):
                value = float(value)
            cell = ws.cell(row=ri, column=ci, value=value)
            cell.border = border
            cell.alignment = Alignment(horizontal="right" if ci > 1 else "left", vertical="center")
            if ci > 1 and isinstance(value, (int, float)):
                cell.number_format = num_fmt
                if float(value) < 0:
                    cell.font = Font(color="C0392B")

    # Кольорова шкала як у heatmap: зелений → жовтий → червоний.
    if len(out) > 0 and len(out.columns) > 1:
        start_cell = ws.cell(row=3, column=2).coordinate
        end_cell = ws.cell(row=2 + len(out), column=len(out.columns)).coordinate
        ws.conditional_formatting.add(
            f"{start_cell}:{end_cell}",
            ColorScaleRule(
                start_type="min", start_color="63BE7B",
                mid_type="percentile", mid_value=50, mid_color="FFEB84",
                end_type="max", end_color="F8696B",
            )
        )

    ws.freeze_panes = "B3"
    for ci, col in enumerate(out.columns, 1):
        vals = out.iloc[:100, ci - 1].fillna("").astype(str).tolist() if not out.empty else []
        max_len = max([len(str(col))] + [len(x) for x in vals])
        ws.column_dimensions[get_column_letter(ci)].width = min(max(max_len + 2, 12), 42)

    if top_df is not None or antitop_df is not None:
        ws2 = wb.create_sheet("TOP_ANTITOP")
        ws2.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)
        c = ws2.cell(row=1, column=1, value=f"TOP / ANTITOP — {title}")
        c.font = Font(bold=True, color="FFFFFF", size=12)
        c.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
        c.alignment = Alignment(horizontal="center", vertical="center")

        def write_section(start_row, section_title, data, fill_hex):
            ws2.cell(row=start_row, column=1, value=section_title).font = Font(bold=True, color="FFFFFF")
            ws2.cell(row=start_row, column=1).fill = PatternFill("solid", start_color=fill_hex, end_color=fill_hex)
            ws2.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=2)
            if data is None or data.empty:
                return start_row + 2
            use = data.copy()
            if val_col and val_col in use.columns:
                cols = [c for c in ["Підрозділ", val_col] if c in use.columns]
                use = use[cols]
            for ci, h in enumerate(use.columns, 1):
                cell = ws2.cell(row=start_row + 1, column=ci, value=str(h))
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
                cell.alignment = Alignment(horizontal="center")
                cell.border = border
            for r_i, row in enumerate(use.itertuples(index=False), start_row + 2):
                for c_i, v in enumerate(row, 1):
                    value = None if pd.isna(v) else v
                    if isinstance(value, (np.integer,)):
                        value = int(value)
                    elif isinstance(value, (np.floating,)):
                        value = float(value)
                    cell = ws2.cell(row=r_i, column=c_i, value=value)
                    cell.border = border
                    cell.alignment = Alignment(horizontal="right" if c_i > 1 else "left")
                    if c_i > 1 and isinstance(value, (int, float)):
                        cell.number_format = num_fmt
                        if float(value) < 0:
                            cell.font = Font(color="C0392B")
            return start_row + len(use) + 4

        next_row = write_section(3, "✅ TOP", top_df, "2E7D32")
        final_row = write_section(next_row, "❌ ANTITOP", antitop_df, "C0392B")
        # Градієнтна заливка для значень TOP/ANTITOP.
        if final_row > 5:
            ws2.conditional_formatting.add(
                f"B5:B{max(5, final_row - 1)}",
                ColorScaleRule(
                    start_type="min", start_color="63BE7B",
                    mid_type="percentile", mid_value=50, mid_color="FFEB84",
                    end_type="max", end_color="F8696B",
                )
            )
        ws2.column_dimensions["A"].width = 34
        ws2.column_dimensions["B"].width = 16

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()

def _heatmap_blocks_to_excel_bytes(blocks, workbook_title="Всі Heatmap-блоки"):
    """
    Єдиний Excel-файл для всіх Heatmap-блоків.
    Кожен блок записується на окремі листи:
    - Heatmap / Heatmap_%ТО
    - TOP_ANTITOP / TOP_%ТО

    Логіка градієнта як на екрані:
    мінімум = зелений, середина = жовтий, максимум = червоний.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.formatting.rule import ColorScaleRule

    def norm_hex(x, default="5B2D8E"):
        x = str(x or default).replace("#", "").strip().upper()
        return x if len(x) == 6 else default

    def safe_sheet_name(name, used=None):
        used = used if used is not None else set()
        s = str(name or "Sheet")
        for ch in ['\\', '/', '*', '?', ':', '[', ']']:
            s = s.replace(ch, '_')
        s = s[:31] or "Sheet"
        base = s
        i = 1
        while s in used:
            suffix = f"_{i}"
            s = (base[:31 - len(suffix)] + suffix)[:31]
            i += 1
        used.add(s)
        return s

    thin = Side(style="thin", color="B7B7B7")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    used_sheets = set()

    wb = Workbook()
    ws0 = wb.active
    ws0.title = safe_sheet_name("Зміст", used_sheets)
    ws0.cell(row=1, column=1, value=workbook_title).font = Font(bold=True, size=14)
    ws0.cell(row=3, column=1, value="Блок").font = Font(bold=True)
    ws0.cell(row=3, column=2, value="Лист Heatmap").font = Font(bold=True)
    ws0.cell(row=3, column=3, value="Лист TOP/ANTITOP").font = Font(bold=True)

    def write_df(ws, title, data, header_color="5B2D8E", percent=False, apply_gradient=True):
        header_hex = norm_hex(header_color)
        num_fmt = '0.00"%"' if percent else '# ##0;-# ##0;-'
        out = data.copy() if data is not None else pd.DataFrame()
        out = out.replace([np.inf, -np.inf], np.nan)
        if out.index.name is None:
            out.index.name = "Підрозділ"
        out = out.reset_index()

        max_col = max(1, len(out.columns))
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=max_col)
        c = ws.cell(row=1, column=1, value=str(title))
        c.font = Font(bold=True, color="FFFFFF", size=12)
        c.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
        c.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 24

        for ci, h in enumerate(out.columns, 1):
            cell = ws.cell(row=2, column=ci, value=str(h))
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border

        for ri, row in enumerate(out.itertuples(index=False), 3):
            for ci, v in enumerate(row, 1):
                value = None if pd.isna(v) else v
                if isinstance(value, (np.integer,)):
                    value = int(value)
                elif isinstance(value, (np.floating,)):
                    value = float(value)
                cell = ws.cell(row=ri, column=ci, value=value)
                cell.border = border
                cell.alignment = Alignment(horizontal="right" if ci > 1 else "left", vertical="center")
                if ci > 1 and isinstance(value, (int, float)):
                    cell.number_format = num_fmt
                    if float(value) < 0:
                        cell.font = Font(color="C0392B")

        if apply_gradient and len(out) > 0 and len(out.columns) > 1:
            start_cell = ws.cell(row=3, column=2).coordinate
            end_cell = ws.cell(row=2 + len(out), column=len(out.columns)).coordinate
            ws.conditional_formatting.add(
                f"{start_cell}:{end_cell}",
                ColorScaleRule(
                    start_type="min", start_color="63BE7B",
                    mid_type="percentile", mid_value=50, mid_color="FFEB84",
                    end_type="max", end_color="F8696B",
                )
            )

        ws.freeze_panes = "B3"
        for ci, col in enumerate(out.columns, 1):
            vals = out.iloc[:100, ci - 1].fillna("").astype(str).tolist() if not out.empty else []
            max_len = max([len(str(col))] + [len(x) for x in vals])
            ws.column_dimensions[get_column_letter(ci)].width = min(max(max_len + 2, 12), 42)
        return out

    def write_top_antitop(ws, title, top_df, antitop_df, val_col=None, header_color="5B2D8E", percent=False):
        header_hex = norm_hex(header_color)
        num_fmt = '0.00"%"' if percent else '# ##0;-# ##0;-'
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=4)
        c = ws.cell(row=1, column=1, value=f"TOP / ANTITOP — {title}")
        c.font = Font(bold=True, color="FFFFFF", size=12)
        c.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
        c.alignment = Alignment(horizontal="center", vertical="center")

        def write_section(start_row, section_title, data, fill_hex):
            ws.cell(row=start_row, column=1, value=section_title).font = Font(bold=True, color="FFFFFF")
            ws.cell(row=start_row, column=1).fill = PatternFill("solid", start_color=fill_hex, end_color=fill_hex)
            ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=2)
            if data is None or data.empty:
                return start_row + 2
            use = data.copy()
            if val_col and val_col in use.columns:
                cols = [c for c in ["Підрозділ", val_col] if c in use.columns]
                use = use[cols]
            for ci, h in enumerate(use.columns, 1):
                cell = ws.cell(row=start_row + 1, column=ci, value=str(h))
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", start_color=header_hex, end_color=header_hex)
                cell.alignment = Alignment(horizontal="center")
                cell.border = border
            for r_i, row in enumerate(use.itertuples(index=False), start_row + 2):
                for c_i, v in enumerate(row, 1):
                    value = None if pd.isna(v) else v
                    if isinstance(value, (np.integer,)):
                        value = int(value)
                    elif isinstance(value, (np.floating,)):
                        value = float(value)
                    cell = ws.cell(row=r_i, column=c_i, value=value)
                    cell.border = border
                    cell.alignment = Alignment(horizontal="right" if c_i > 1 else "left")
                    if c_i > 1 and isinstance(value, (int, float)):
                        cell.number_format = num_fmt
                        if float(value) < 0:
                            cell.font = Font(color="C0392B")
            return start_row + len(use) + 4

        next_row = write_section(3, "✅ TOP", top_df, "2E7D32")
        final_row = write_section(next_row, "❌ ANTITOP", antitop_df, "C0392B")
        if final_row > 5:
            ws.conditional_formatting.add(
                f"B5:B{max(5, final_row - 1)}",
                ColorScaleRule(
                    start_type="min", start_color="63BE7B",
                    mid_type="percentile", mid_value=50, mid_color="FFEB84",
                    end_type="max", end_color="F8696B",
                )
            )
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 16

    row_idx = 4
    for i, block in enumerate(blocks or [], 1):
        if not block:
            continue
        title = block.get("title", f"Heatmap {i}")
        prefix = block.get("sheet_prefix", f"Heatmap_{i}")
        header_color = block.get("header_color", "5B2D8E")
        percent = bool(block.get("percent", False))
        val_col = block.get("val_col")

        heat_sheet = safe_sheet_name(prefix, used_sheets)
        ws = wb.create_sheet(heat_sheet)
        write_df(ws, title, block.get("heat_df"), header_color=header_color, percent=percent, apply_gradient=True)

        top_sheet = ""
        if block.get("top_df") is not None or block.get("antitop_df") is not None:
            top_sheet = safe_sheet_name(f"TOP_{prefix}", used_sheets)
            ws2 = wb.create_sheet(top_sheet)
            write_top_antitop(
                ws2, title, block.get("top_df"), block.get("antitop_df"),
                val_col=val_col, header_color=header_color, percent=percent
            )

        ws0.cell(row=row_idx, column=1, value=str(title))
        ws0.cell(row=row_idx, column=2, value=heat_sheet)
        ws0.cell(row=row_idx, column=3, value=top_sheet)
        row_idx += 1

    ws0.column_dimensions["A"].width = 70
    ws0.column_dimensions["B"].width = 24
    ws0.column_dimensions["C"].width = 24

    bio = io.BytesIO()
    wb.save(bio)
    return bio.getvalue()

def _fmt_abs_html(v):
    if pd.isna(v):
        return f'<td style="{TD}background-color:white !important;color:black !important;"></td>'
    return f'<td style="{TD}{"color:#c0392b;" if v < 0 else ""}">{v:,.0f}</td>'


def _fmt_pct_html(v):
    if pd.isna(v):
        return f'<td style="{TD}background-color:white !important;color:black !important;"></td>'
    return f'<td style="{TD}{"color:#c0392b;" if v < 0 else ""}">{v:.2f}%</td>'


def _make_pills(series, color, bg):
    pills = ""
    for tt, val in series.items():
        sign    = "+" if val > 0 else ""
        val_fmt = f"{val:,.0f}".replace(",", " ")
        pills  += (
            f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border-radius:4px;padding:2px 9px;margin:2px 3px;font-size:0.75rem;'
            f'font-weight:600;white-space:nowrap;">'
            f'{tt}&nbsp;<span style="opacity:.7;font-weight:400;">({sign}{val_fmt})</span></span>'
        )
    return pills


def _make_pct_pills(series, color, bg):
    pills = ""
    for tt, val in series.items():
        sign = "+" if val > 0 else ""
        pills += (
            f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border-radius:4px;padding:2px 9px;margin:2px 3px;font-size:0.75rem;'
            f'font-weight:600;white-space:nowrap;">'
            f'{tt}&nbsp;<span style="opacity:.7;font-weight:400;">({sign}{val:.2f}%)</span></span>'
        )
    return pills


def _render_slicer(article_idx, prefix, df_filtered, col_tt, col_article, title, col_division=None):
    skey = f"{prefix}_slicer_tt_{article_idx}"
    if skey not in st.session_state:
        st.session_state[skey] = "__ALL__"
    active_tt = st.session_state[skey]

    available_tts = sorted(
        df_filtered[df_filtered[col_article] == title][col_tt].dropna().unique(), key=str
    )
    if not available_tts:
        return active_tt

    tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)

    with st.expander("🏪 Слайсер по ТТ — клікни для деталізації", expanded=False):
        search_key = f"{prefix}_slicer_search_{article_idx}"
        search_val = st.text_input("🔎 Пошук магазину", value="",
                                   placeholder="Введіть назву...", key=search_key)
        filtered_tts = (
            [
                t for t in available_tts
                if search_val.lower() in str(t).lower()
                or search_val.lower() in str(_tt_display_label(t, tt_display_map)).lower()
            ]
            if search_val else available_tts
        )
        all_options   = ["__ALL__"] + list(filtered_tts)
        VISIBLE_ITEMS = 4 * 6
        show_all_key  = f"{prefix}_slicer_showall_{article_idx}"
        if show_all_key not in st.session_state:
            st.session_state[show_all_key] = False
        items_to_show = all_options if st.session_state[show_all_key] else all_options[:VISIBLE_ITEMS]

        for row_start in range(0, len(items_to_show), 6):
            chunk = items_to_show[row_start:row_start + 6]
            cols  = st.columns(len(chunk))
            for ci, tt_opt in enumerate(chunk):
                label    = "🔁 Всі" if tt_opt == "__ALL__" else str(_tt_display_label(tt_opt, tt_display_map))
                btn_type = "primary" if active_tt == tt_opt else "secondary"
                with cols[ci]:
                    if st.button(label, key=f"{prefix}_btn_{article_idx}_{row_start}_{ci}_{hash(str(tt_opt))}",
                                 type=btn_type, use_container_width=True):
                        st.session_state[skey] = tt_opt
                        st.rerun()

        if len(all_options) > VISIBLE_ITEMS:
            remaining = len(all_options) - VISIBLE_ITEMS
            label = "▲ Згорнути" if st.session_state[show_all_key] else f"▼ Показати ще {remaining}"
            if st.button(label, key=f"{prefix}_toggle_{article_idx}", use_container_width=False):
                st.session_state[show_all_key] = not st.session_state[show_all_key]
                st.rerun()

        if active_tt != "__ALL__":
            st.caption(f"📍 Показано тільки: **{_tt_display_label(active_tt, tt_display_map)}**")
        else:
            st.caption(f"Показано всі ТТ · знайдено: {len(filtered_tts)}")

    return active_tt




def _render_shared_tt_slicer(article_idx, df_filtered, col_tt, col_article, title, col_division=None):
    """Shared compact TT slicer as buttons: one TT = one compact button."""
    skey = f"shared_slicer_tt_{article_idx}"
    search_key = f"shared_slicer_search_{article_idx}"
    show_key = f"shared_slicer_show_more_{article_idx}"

    available_tts = sorted(
        df_filtered[df_filtered[col_article] == title][col_tt].dropna().unique(), key=str
    )
    if not available_tts:
        return "__ALL__"

    tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)

    options_all = ["__ALL__"] + list(available_tts)
    if skey not in st.session_state or st.session_state[skey] not in options_all:
        st.session_state[skey] = "__ALL__"
    if show_key not in st.session_state:
        st.session_state[show_key] = False

    st.markdown(f"""
    <div style="margin:6px 0 4px 0;padding:5px 8px;border:1px solid #ddd;
                border-radius:6px;background:#fafafa;">
      <div style="font-size:0.76rem;color:#555;font-weight:700;line-height:1.1;">
        🏪 Спільний слайсер ТТ — керує обома таблицями та графіками: {title}
      </div>
    </div>
    <style>
      div[data-testid="stButton"] > button {{
          padding: 1px 5px !important;
          font-size: 10.5px !important;
          min-height: 23px !important;
          height: 23px !important;
          border-radius: 5px !important;
          line-height: 1 !important;
          white-space: nowrap !important;
      }}
    </style>
    """, unsafe_allow_html=True)

    with st.expander("🔘 ТТ кнопками", expanded=False):
        top_cols = st.columns([5, 1])
        with top_cols[0]:
            search_val = st.text_input(
                "Пошук ТТ",
                value="",
                placeholder="Пошук...",
                key=search_key,
                label_visibility="collapsed"
            )
        with top_cols[1]:
            if st.button("Скинути", key=f"shared_slicer_reset_{article_idx}", use_container_width=True):
                st.session_state[skey] = "__ALL__"
                st.rerun()

        filtered_tts = (
            [
                t for t in available_tts
                if search_val.lower() in str(t).lower()
                or search_val.lower() in str(_tt_display_label(t, tt_display_map)).lower()
            ]
            if search_val else available_tts
        )
        options = ["__ALL__"] + list(filtered_tts)

        cols_per_row = 10
        visible_limit = 40
        items_to_show = options if st.session_state[show_key] else options[:visible_limit]

        for row_start in range(0, len(items_to_show), cols_per_row):
            chunk = items_to_show[row_start:row_start + cols_per_row]
            cols = st.columns(len(chunk))
            for ci, tt_opt in enumerate(chunk):
                label = "Всі" if tt_opt == "__ALL__" else str(_tt_display_label(tt_opt, tt_display_map))
                if len(label) > 13:
                    label = label[:12] + "…"

                with cols[ci]:
                    if st.button(
                        label,
                        key=f"shared_tt_btn_{article_idx}_{row_start}_{ci}_{hash(str(tt_opt))}",
                        type="primary" if st.session_state[skey] == tt_opt else "secondary",
                        use_container_width=True,
                        help="Всі магазини" if tt_opt == "__ALL__" else str(_tt_display_label(tt_opt, tt_display_map))
                    ):
                        st.session_state[skey] = tt_opt
                        st.rerun()

        if len(options) > visible_limit:
            remaining = len(options) - visible_limit
            toggle_label = "▲ Згорнути" if st.session_state[show_key] else f"▼ Показати ще {remaining}"
            if st.button(toggle_label, key=f"shared_slicer_toggle_{article_idx}"):
                st.session_state[show_key] = not st.session_state[show_key]
                st.rerun()

        if st.session_state[skey] == "__ALL__":
            st.caption(f"Показано всі ТТ · знайдено: {len(filtered_tts)}")
        else:
            st.caption(f"📍 Спільний фільтр для обох блоків: **{_tt_display_label(st.session_state[skey], tt_display_map)}**")

    return st.session_state[skey]


# ── Cost per kWh/m³ block — only utility articles ─────────────────────────────

def _is_utility_cost_article(article):
    """Показуємо блок вартості тільки для Електроенергії та Водопостачання."""
    a = str(article).strip().lower()
    return ("електро" in a) or ("водопост" in a) or ("водоснаб" in a) or ("water" in a)


def build_cost_monthly(df_filtered, col_tt, col_article, col_month,
                       col_value, col_kwh, col_plf, selected_art, selected_tts=None):
    """
    Вартість = Значення / ЕЕ_кВт по місяцях.
    Рахуємо як SUM(Значення) / SUM(ЕЕ_кВт) окремо для PL та F.
    """
    cols = ["Plan", "Fact", "Delta", "Delta_%"]
    if not col_kwh or col_kwh not in df_filtered.columns:
        return pd.DataFrame(0.0, index=range(1, 13), columns=cols)

    art = _prep(df_filtered[df_filtered[col_article].eq(selected_art)], col_month)
    if selected_tts:
        art = art[art[col_tt].isin(selected_tts)]

    if art.empty:
        return pd.DataFrame(0.0, index=range(1, 13), columns=cols)

    tmp = art[[col_plf, "_m", col_value, col_kwh]].copy()
    tmp["_value"] = pd.to_numeric(tmp[col_value], errors="coerce")
    tmp["_kwh"] = pd.to_numeric(tmp[col_kwh], errors="coerce")

    def _calc(plf_code):
        sub = tmp[tmp[col_plf].eq(plf_code)]
        if sub.empty:
            return pd.Series(0.0, index=range(1, 13))
        g = sub.groupby("_m", observed=True)[["_value", "_kwh"]].sum()
        out = (g["_value"] / g["_kwh"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        return out.fillna(0).reindex(range(1, 13), fill_value=0)

    plan = _calc("PL").rename("Plan")
    fact = _calc("F").rename("Fact")

    merged = pd.DataFrame(index=range(1, 13)).join(plan).join(fact).fillna(0.0)
    merged.index.name = "month"
    merged["Delta"] = merged["Fact"] - merged["Plan"]
    merged["Delta_%"] = (
        (merged["Fact"] / merged["Plan"].replace(0, np.nan) - 1) * 100
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    return merged


def render_cost_article_block(title, table_df, df_filtered,
                              col_tt, col_article, col_month, col_value,
                              col_kwh, col_plf, tt_val, article_idx,
                              active_tt=None, col_division=None):
    """Окрема таблиця + графік вартості Значення / ЕЕ_кВт."""
    if not _is_utility_cost_article(title):
        return

    if not col_kwh or col_kwh not in df_filtered.columns:
        st.info("Для таблиці вартості потрібно обрати колонку ЕЕ_кВт у налаштуваннях колонок.")
        return

    if active_tt != "__ALL__" and active_tt is not None:
        display_df = build_cost_monthly(
            df_filtered, col_tt, col_article, col_month, col_value,
            col_kwh, col_plf, title, [active_tt]
        )
    else:
        display_df = table_df

    rows_cfg = [
        ("План вартість", "Plan", "#ffffff", "#333333"),
        ("Факт вартість", "Fact", "#e8f4ff", TEAL_HDR),
        ("Відхилення Fact−Plan", "Delta", "#fff9e0", "#b8860b"),
        ("Відхилення Fact/Plan %", "Delta_%", "#eefaf7", TEAL_HDR),
    ]

    tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)
    active_tt_label = _tt_display_label(active_tt, tt_display_map) if active_tt not in (None, "__ALL__") else active_tt
    badge = (f'<span style="margin-left:10px;background:{TEAL};color:white;font-size:0.78rem;'
             f'padding:2px 10px;border-radius:10px;">📍 {active_tt_label}</span>'
             if active_tt not in (None, "__ALL__") else "")

    st.markdown(f"""
    <div style="margin-top:12px;margin-bottom:4px;">
      <span style="background:{TEAL};color:white;font-weight:700;padding:4px 14px;
                   font-size:0.9rem;border-radius:2px;">💧⚡ Визначення вартості: {title}</span>{badge}
    </div>""", unsafe_allow_html=True)

    th = _th(TEAL_HDR)
    html = (f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;margin-bottom:6px;">'
            f'<thead><tr><th style="{th}">Показник</th>'
            + "".join(f'<th style="{th}">{m}</th>' for m in MONTHS_LIST)
            + f'<th style="{th}">Разом</th></tr></thead><tbody>')

    total_plan_num = total_plan_den = total_fact_num = total_fact_den = 0.0
    # Разом для вартості рахуємо як загальна сума Значення / загальна сума ЕЕ_кВт.
    src = _prep(df_filtered[df_filtered[col_article].eq(title)], col_month)
    if active_tt not in (None, "__ALL__"):
        src = src[src[col_tt].eq(active_tt)]
    elif tt_val:
        src = src[src[col_tt].isin(tt_val)]
    if not src.empty:
        src_val = pd.to_numeric(src[col_value], errors="coerce")
        src_kwh = pd.to_numeric(src[col_kwh], errors="coerce")
        total_plan_num = src_val[src[col_plf].eq("PL")].sum()
        total_plan_den = src_kwh[src[col_plf].eq("PL")].sum()
        total_fact_num = src_val[src[col_plf].eq("F")].sum()
        total_fact_den = src_kwh[src[col_plf].eq("F")].sum()
    total_plan = total_plan_num / total_plan_den if total_plan_den else np.nan
    total_fact = total_fact_num / total_fact_den if total_fact_den else np.nan
    total_delta = total_fact - total_plan if pd.notna(total_fact) and pd.notna(total_plan) else np.nan
    total_delta_pct = ((total_fact / total_plan - 1) * 100) if pd.notna(total_fact) and pd.notna(total_plan) and total_plan != 0 else np.nan
    totals = {"Plan": total_plan, "Fact": total_fact, "Delta": total_delta, "Delta_%": total_delta_pct}

    for label, col, bg, color in rows_cfg:
        html += f'<tr style="background:{bg};"><td style="{TL}color:{color};">{label}</td>'
        for m in range(1, 13):
            v = display_df.loc[m, col]
            html += _fmt_pct_html(v) if col == "Delta_%" else (f'<td style="{TD}{"color:#c0392b;" if pd.notna(v) and v < 0 else ""}">{v:,.2f}</td>' if pd.notna(v) else f'<td style="{TD}background-color:white !important;color:black !important;"></td>')
        total = totals.get(col, np.nan)
        if pd.isna(total):
            total_html = ""
        elif col == "Delta_%":
            total_html = f"{total:.2f}%"
        else:
            total_html = f"{total:,.2f}"
        total_color = "color:#c0392b;" if pd.notna(total) and total < 0 else ""
        html += f'<td style="{TD}font-weight:700;{total_color}">{total_html}</td></tr>'
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    fig = go.Figure([
        go.Bar(x=MONTHS_LIST, y=display_df["Plan"], name="План вартість", marker_color=GREY),
        go.Bar(x=MONTHS_LIST, y=display_df["Fact"], name="Факт вартість", marker_color=TEAL),
        go.Scatter(x=MONTHS_LIST, y=display_df["Delta"], name="Відхилення Fact−Plan",
                   line=dict(color=YELLOW, width=2, dash="dot")),
    ])
    fig.update_layout(
        title=f"Динаміка вартості Значення / ЕЕ_кВт — {title}",
        height=300,
        margin=dict(t=45, b=50, l=10, r=10),
        barmode="group",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"cost_chart_{article_idx}_{active_tt}")

# ── Article block — absolute ─────────────────────────────────────────────────

def render_article_block(title, table_df, df, df_filtered,
                          col_tt, col_article, col_month, col_value, col_plf,
                          group_factors, tt_val, article_idx, active_tt=None, col_division=None,
                          metric_label="Значення"):
    metric_suffix = "" if metric_label == "Значення" else f" ({metric_label})"
    rows_cfg = [
        (f"План{metric_suffix}",                "Plan",      "#ffffff", "#333333"),
        (f"Факт{metric_suffix}",                "Fact",      "#e8d5f5", PURPLE),
        (f"Average{metric_suffix}",             "Average",   "#fde8e8", RED_LINE),
        ("Дельта Fact−Average", "Delta",       "#fff9e0", "#b8860b"),
        ("Дельта Fact−Plan",    "DeltaPlan",   "#eaf7ea", GREEN_HDR),
        ("Відхилення Fact/Plan %", "DeltaPlan_%", "#eefaf7", TEAL_HDR),
    ]
    th = _th(GREEN_HDR)

    if active_tt is None:
        active_tt = _render_slicer(article_idx, "abs", df_filtered, col_tt, col_article, title, col_division)

    if active_tt != "__ALL__":
        df_filt_tt = df_filtered[df_filtered[col_tt] == active_tt].copy()
        display_df = build_article_monthly(
            df, df_filt_tt, col_tt, col_article, col_month, col_value,
            col_plf, title, [active_tt], group_factors
        )
    else:
        display_df = table_df

    tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)
    active_tt_label = _tt_display_label(active_tt, tt_display_map) if active_tt != "__ALL__" else active_tt
    badge = (f'<span style="margin-left:10px;background:{PURPLE};color:white;font-size:0.78rem;'
             f'padding:2px 10px;border-radius:10px;">📍 {active_tt_label}</span>'
             if active_tt != "__ALL__" else "")
    st.markdown(f"""
    <div style="margin-top:20px;margin-bottom:4px;">
      <span style="background:{GREEN_HDR};color:white;font-weight:700;padding:4px 14px;
                   font-size:0.9rem;border-radius:2px;">{title}</span>{badge}
      <span style="margin-left:10px;background:#f4f0fa;color:#5b2d8e;font-size:0.72rem;
                   padding:2px 8px;border-radius:3px;">📌 Метрика: {metric_label}</span>
    </div>""", unsafe_allow_html=True)

    html = (f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;'
            f'margin-bottom:6px;"><thead><tr>'
            f'<th style="{th}">Показник</th>'
            + "".join(f'<th style="{th}">{m}</th>' for m in MONTHS_LIST)
            + f'<th style="{th}">Разом</th></tr></thead><tbody>')
    for label, col, bg, color in rows_cfg:
        vals = [display_df.loc[m, col] for m in range(1, 13)]

        # Для % відхилення Fact/Plan підсумок рахуємо не як суму місячних %,
        # а як загальний Fact / загальний Plan − 1.
        if col == "DeltaPlan_%":
            plan_total = sum(display_df.loc[m, "Plan"] for m in range(1, 13))
            fact_total = sum(display_df.loc[m, "Fact"] for m in range(1, 13))
            total = ((fact_total / plan_total - 1) * 100) if plan_total != 0 else np.nan
        else:
            total = sum(vals)

        html += f'<tr style="background:{bg};"><td style="{TL}color:{color};">{label}</td>'
        for v in vals:
            html += _fmt_pct_html(v) if col == "DeltaPlan_%" else _fmt_abs_html(v)

        if pd.isna(total):
            total_html = ""
        elif col == "DeltaPlan_%":
            total_html = f"{total:.2f}%"
        else:
            total_html = f"{total:,.0f}"

        total_color = "color:#c0392b;" if pd.notna(total) and total < 0 else ""
        html += f'<td style="{TD}font-weight:700;{total_color}background:{"white !important" if pd.isna(total) else "transparent"};">{total_html}</td></tr>'
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    facts       = [display_df.loc[m, "Fact"] for m in range(1, 13)]
    nz          = [f for f in facts if f != 0]
    avg_monthly = np.mean(nz) if nz else 0
    total_fact  = sum(facts)
    total_plan  = sum(display_df.loc[m, "Plan"] for m in range(1, 13))
    total_delta = total_fact - total_plan
    pct         = ((total_fact / total_plan - 1) * 100) if total_plan != 0 else None
    pct_str     = (f"{'+' if pct >= 0 else ''}{pct:.1f}%" if pct is not None else "—")
    pct_color   = RED_LINE if (pct or 0) > 0 else GREEN_HDR
    delta_color = RED_LINE if total_delta > 0 else GREEN_HDR

    best_pills = worst_pills = ""
    if active_tt == "__ALL__":
        sub = df_filtered[(df_filtered[col_article] == title) & (df_filtered[col_plf] == "F")].copy()
        sub[col_value] = pd.to_numeric(sub[col_value], errors="coerce")
        if not sub.empty and col_tt in sub.columns:
            tt_totals   = sub.groupby(col_tt, observed=True)[col_value].sum().dropna().sort_values()
            n           = min(3, len(tt_totals))
            best_series = _display_tt_series_index(tt_totals.head(n), df_filtered, col_tt, col_division)
            worst_series = _display_tt_series_index(tt_totals.tail(n).iloc[::-1], df_filtered, col_tt, col_division)
            best_pills  = _make_pills(best_series, "#1b5e20", "#e8f5e9")
            worst_pills = _make_pills(worst_series, "#7f0000", "#ffebee")

    best_block = f"""
      <div style="flex:1;min-width:220px;">
        <div style="color:#888;font-size:0.71rem;margin-bottom:4px;text-transform:uppercase;">✅ Найекономніші магазини (мін. Fact)</div>
        <div>{best_pills or '<span style="color:#aaa;font-size:0.75rem;">немає даних</span>'}</div>
      </div>
      <div style="flex:1;min-width:220px;">
        <div style="color:#888;font-size:0.71rem;margin-bottom:4px;text-transform:uppercase;">❌ Найбільш витратні магазини (макс. Fact)</div>
        <div>{worst_pills or '<span style="color:#aaa;font-size:0.75rem;">немає даних</span>'}</div>
      </div>""" if active_tt == "__ALL__" else ""

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:10px;background:#f9f6ff;
                border:1px solid #d0baf5;border-radius:6px;padding:10px 16px;margin:6px 0 10px 0;">
      <div style="min-width:130px;">
        <div style="color:#888;font-size:0.71rem;text-transform:uppercase;">Серед. Fact / міс. ({metric_label})</div>
        <div style="font-size:1.1rem;font-weight:700;color:{PURPLE};">{avg_monthly:,.0f}</div>
      </div>
      <div style="min-width:130px;">
        <div style="color:#888;font-size:0.71rem;text-transform:uppercase;">Δ Fact − Plan</div>
        <div style="font-size:1.1rem;font-weight:700;color:{delta_color};">{('+' if total_delta > 0 else '')}{total_delta:,.0f}</div>
      </div>
      <div style="min-width:100px;">
        <div style="color:#888;font-size:0.71rem;text-transform:uppercase;">% до плану</div>
        <div style="font-size:1.1rem;font-weight:700;color:{pct_color};">{pct_str}</div>
      </div>
      {best_block}
    </div>""", unsafe_allow_html=True)

    fig = go.Figure([
        go.Bar(x=MONTHS_LIST, y=display_df["Plan"],    name="План",    marker_color=GREY),
        go.Bar(x=MONTHS_LIST, y=display_df["Fact"],    name="Факт",    marker_color=PURPLE),
        go.Scatter(x=MONTHS_LIST, y=display_df["Average"], name="Average",
                   line=dict(color=RED_LINE, width=3)),
        go.Scatter(x=MONTHS_LIST, y=display_df["Delta"],   name="Дельта Fact−Average",
                   line=dict(color=YELLOW, dash="dot")),
        go.Scatter(x=MONTHS_LIST, y=display_df["DeltaPlan"], name="Дельта Fact−Plan",
                   line=dict(color=GREEN_HDR, width=2, dash="dash")),
    ])
    fig.update_layout(height=320, margin=dict(t=30, b=50, l=10, r=10),
                      barmode="group", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.18, x=0))
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{article_idx}_{active_tt}")



# ── Article block — ratio ────────────────────────────────────────────────────

def render_ratio_article_block(title, table_df, df, df_filtered,
                                col_tt, col_article, col_month, col_ratio, col_plf,
                                tt_val, article_idx, group_factors=None, active_tt=None, col_division=None):
    if group_factors is None:
        group_factors = []

    rows_cfg = [
        ("План %",                "Plan",      "#e8f8f8", TEAL_HDR),
        ("Факт %",                "Fact",      "#d0f0f0", TEAL),
        ("Average %",             "Average",   "#fde8e8", RED_LINE),
        ("Дельта % Fact−Average", "Delta",     "#fff9e0", ORANGE),
        ("Дельта % Fact−Plan",    "DeltaPlan", "#eaf7ea", GREEN_HDR),
    ]
    th = _th(TEAL_HDR)

    if active_tt is None:
        active_tt = _render_slicer(article_idx, "rat", df_filtered, col_tt, col_article, title, col_division)

    if active_tt != "__ALL__":
        df_filt_tt = df_filtered[df_filtered[col_tt] == active_tt].copy()
        display_df = build_ratio_monthly(
            df_filt_tt, col_tt, col_article, col_month, col_ratio, col_plf,
            title, [active_tt], df_all=df, group_factors=group_factors
        )
    else:
        display_df = table_df

    tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)
    active_tt_label = _tt_display_label(active_tt, tt_display_map) if active_tt != "__ALL__" else active_tt
    badge = (f'<span style="margin-left:10px;background:{TEAL};color:white;font-size:0.78rem;'
             f'padding:2px 10px;border-radius:10px;">📍 {active_tt_label}</span>'
             if active_tt != "__ALL__" else "")
    st.markdown(f"""
    <div style="margin-top:12px;margin-bottom:4px;">
      <span style="background:{TEAL_HDR};color:white;font-weight:700;padding:4px 14px;
                   font-size:0.85rem;border-radius:2px;">📊 % в ТО — {title}</span>{badge}
      <span style="margin-left:10px;background:#fff3cd;color:#856404;font-size:0.7rem;
                   padding:2px 8px;border-radius:3px;">💡 Average% = (Average₍абс₎ / ТО) × 100</span>
    </div>""", unsafe_allow_html=True)

    html = (f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;'
            f'margin-bottom:6px;"><thead><tr>'
            f'<th style="{th}">Показник</th>'
            + "".join(f'<th style="{th}">{m}</th>' for m in MONTHS_LIST)
            + f'<th style="{th}">Серед.</th></tr></thead><tbody>')
    for label, col, bg, color in rows_cfg:
        vals    = [display_df.loc[m, col] for m in range(1, 13)]
        nz      = [v for v in vals if v != 0]
        summary = np.mean(nz) if nz else 0.0
        html   += f'<tr style="background:{bg};"><td style="{TL}color:{color};">{label}</td>'
        for v in vals:
            html += _fmt_pct_html(v)
        summary_html = "" if pd.isna(summary) else f"{summary:.2f}%"
        html += f'<td style="{TD}font-weight:700;background:{"white !important" if pd.isna(summary) else "transparent"};">{summary_html}</td></tr>'
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    facts       = [display_df.loc[m, "Fact"] for m in range(1, 13)]
    nz          = [f for f in facts if f != 0]
    avg_monthly = np.mean(nz) if nz else 0.0
    avg_global  = np.mean([display_df.loc[m, "Average"] for m in range(1, 13)
                            if display_df.loc[m, "Average"] != 0] or [0])
    total_delta = np.mean([display_df.loc[m, "Delta"] for m in range(1, 13)
                            if display_df.loc[m, "Fact"] != 0] or [0])
    delta_color = RED_LINE if total_delta > 0 else GREEN_HDR

    best_pills = worst_pills = ""
    if active_tt == "__ALL__":
        sub = df_filtered[df_filtered[col_article] == title].copy()
        sub[col_ratio] = pd.to_numeric(sub[col_ratio], errors="coerce")
        if col_plf and col_plf in sub.columns:
            sub = sub[sub[col_plf] == "F"]
        if not sub.empty and col_tt in sub.columns:
            tt_avgs     = sub.groupby(col_tt, observed=True)[col_ratio].mean().dropna().sort_values()
            n           = min(3, len(tt_avgs))
            best_series = _display_tt_series_index(tt_avgs.head(n), df_filtered, col_tt, col_division)
            worst_series = _display_tt_series_index(tt_avgs.tail(n).iloc[::-1], df_filtered, col_tt, col_division)
            best_pills  = _make_pct_pills(best_series, "#1b5e20", "#e8f5e9")
            worst_pills = _make_pct_pills(worst_series, "#7f0000", "#ffebee")

    best_block = f"""
      <div style="flex:1;min-width:220px;">
        <div style="color:#555;font-size:0.71rem;text-transform:uppercase;">✅ Найекономніші магазини (мін. %)</div>
        <div>{best_pills or '<span style="color:#aaa;font-size:0.75rem;">немає даних</span>'}</div>
      </div>
      <div style="flex:1;min-width:220px;">
        <div style="color:#555;font-size:0.71rem;text-transform:uppercase;">❌ Найбільш витратні магазини (макс. %)</div>
        <div>{worst_pills or '<span style="color:#aaa;font-size:0.75rem;">немає даних</span>'}</div>
      </div>""" if active_tt == "__ALL__" else ""

    st.markdown(f"""
    <div style="display:flex;flex-wrap:wrap;gap:10px;background:#e8f8f8;
                border:1px solid #8ecece;border-radius:6px;padding:10px 16px;margin:6px 0 10px 0;">
      <div style="min-width:130px;">
        <div style="color:#555;font-size:0.71rem;text-transform:uppercase;">Серед. Fact % / міс.</div>
        <div style="font-size:1.1rem;font-weight:700;color:{TEAL};">{avg_monthly:.2f}%</div>
      </div>
      <div style="min-width:130px;">
        <div style="color:#555;font-size:0.71rem;text-transform:uppercase;">Average % (норма)</div>
        <div style="font-size:1.1rem;font-weight:700;color:{RED_LINE};">{avg_global:.2f}%</div>
      </div>
      <div style="min-width:130px;">
        <div style="color:#555;font-size:0.71rem;text-transform:uppercase;">Δ Fact − Average</div>
        <div style="font-size:1.1rem;font-weight:700;color:{delta_color};">{('+' if total_delta > 0 else '')}{total_delta:.2f}%</div>
      </div>
      {best_block}
    </div>""", unsafe_allow_html=True)

    fig = go.Figure([
        go.Bar(x=MONTHS_LIST, y=display_df["Plan"],    name="План %",  marker_color="#8ecece", opacity=0.8),
        go.Bar(x=MONTHS_LIST, y=display_df["Fact"],    name="Факт %",  marker_color=TEAL, opacity=0.95),
        go.Scatter(x=MONTHS_LIST, y=display_df["Average"], name="Average %",
                   line=dict(color=RED_LINE, width=3)),
        go.Scatter(x=MONTHS_LIST, y=display_df["Delta"],   name="Δ % Fact−Average",
                   line=dict(color=ORANGE, dash="dot")),
        go.Scatter(x=MONTHS_LIST, y=display_df["DeltaPlan"], name="Δ % Fact−Plan",
                   line=dict(color=GREEN_HDR, width=2, dash="dash")),
    ])

    # ✅ Одна спільна вісь Y для всіх показників %:
    # План %, Факт %, Average % і Δ % більше не розносяться на різні масштаби.
    y_max = pd.to_numeric(
        display_df[["Plan", "Fact", "Average", "Delta", "DeltaPlan"]].stack(),
        errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).dropna()

    if not y_max.empty:
        y_min_val = float(y_max.min())
        y_max_val = float(y_max.max())
        pad = max(abs(y_min_val), abs(y_max_val), 1) * 0.12
        y_range = [min(0, y_min_val - pad), max(0, y_max_val + pad)]
    else:
        y_range = [0, 1]

    fig.update_layout(
        height=320, margin=dict(t=30, b=50, l=10, r=10),
        barmode="group", hovermode="x unified",
        yaxis=dict(
            title="% в ТО",
            tickformat=".2f",
            ticksuffix="%",
            showgrid=True,
            zeroline=True,
            range=y_range,
        ),
        legend=dict(orientation="h", y=-0.18, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ratio_chart_{article_idx}_{active_tt}")



# ── Ratio heatmap section ─────────────────────────────────────────────────────

def render_ratio_heatmap_section(df, df_filtered, col_tt, col_article,
                                  col_month, col_ratio, col_plf,
                                  articles_to_show, ratio_mode, group_factors=None,
                                  col_division=None, col_year=None,
                                  show_download_button=True):
    if group_factors is None:
        group_factors = []

    heat, tt_table, val_col = build_ratio_heat_data(
        df, df_filtered, col_tt, col_article, col_month,
        col_ratio, col_plf, articles_to_show, ratio_mode,
        group_factors=group_factors,
    )

    st.markdown(f"""
    <div style="margin-bottom:6px;">
      <span style="background:{TEAL_HDR};color:white;font-weight:700;
                   padding:4px 14px;font-size:0.9rem;border-radius:2px;">
        🌡️ Карта аномалій % в ТО
      </span>
      <span style="margin-left:12px;color:#666;font-size:0.8rem;">
        Режим: <b>{ratio_mode}</b> · значення у відсотках
      </span>
    </div>""", unsafe_allow_html=True)

    heat_display = _replace_tt_index_with_division(heat, df_filtered, col_tt, col_division)
    ratio_heatmap_title = _build_article_period_title(
        articles=articles_to_show, df_context=df_filtered, col_article=col_article,
        col_month=col_month, col_year=col_year, suffix=f"Heatmap % в ТО — {ratio_mode}"
    )

    st.dataframe(
        heat_display.style
            .background_gradient(cmap="RdYlGn_r", axis=None)
            .apply(_style_white_na, axis=None)
            .format(lambda v: f"{v:.2f}%" if pd.notna(v) else "", na_rep=""),
        use_container_width=True,
    )

    st.markdown(f"""
    <div style="margin:14px 0 6px 0;">
      <span style="background:{TEAL_HDR};color:white;font-weight:700;
                   padding:4px 14px;font-size:0.9rem;border-radius:2px;">
        🏆 TOP / ANTITOP — % в ТО
      </span>
    </div>""", unsafe_allow_html=True)

    rating_options = _get_heatmap_rating_factor_options(
        df_filtered,
        col_tt=col_tt,
        col_division=col_division,
        group_factors=group_factors,
    )
    if not rating_options:
        rating_options = [col_tt]

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        ratio_rating_col = st.selectbox(
            "Рейтинг % в ТО дивимось по:",
            options=rating_options,
            index=0,
            key=f"ratio_heatmap_rating_factor_{ratio_mode}",
        )
    with rc2:
        n_tt = st.slider(
            "Кількість позицій (% в ТО)",
            1, 100, 10,
            key=f"ratio_heatmap_rating_topn_{ratio_mode}",
        )

    sum_val = _prepare_heatmap_rating_table(
        tt_table, df_filtered, col_tt, val_col, ratio_rating_col, aggfunc="mean"
    )
    top = sum_val.sort_values(val_col, ascending=True).head(n_tt)
    antitop = sum_val.sort_values(val_col, ascending=False).head(n_tt)
    top_display = top.copy()
    antitop_display = antitop.copy()
    fmt_fn = lambda v: f"{v:.2f}%" if pd.notna(v) else "-"

    ratio_top_title = _build_article_period_title(
        articles=articles_to_show, df_context=df_filtered, col_article=col_article,
        col_month=col_month, col_year=col_year, suffix=f"TOP / ANTITOP % в ТО — {ratio_rating_col}"
    )

    ca, cb = st.columns(2)
    with ca:
        st.write(f"✅ Top (найменший %) по: {ratio_rating_col}")
        st.dataframe(
            top_display[["Підрозділ", val_col]].set_index("Підрозділ")
                .style.background_gradient(cmap="RdYlGn", subset=[val_col])
                .apply(_style_white_na, axis=None)
                .format({val_col: fmt_fn}),
            use_container_width=True
        )
    with cb:
        st.write(f"❌ Antitop (найбільший %) по: {ratio_rating_col}")
        st.dataframe(
            antitop_display[["Підрозділ", val_col]].set_index("Підрозділ")
                .style.background_gradient(cmap="RdYlGn_r", subset=[val_col])
                .apply(_style_white_na, axis=None)
                .format({val_col: fmt_fn}),
            use_container_width=True
        )

    ratio_export_block = {
        "title": ratio_heatmap_title,
        "heat_df": heat_display,
        "top_df": top_display,
        "antitop_df": antitop_display,
        "val_col": val_col,
        "header_color": TEAL_HDR,
        "percent": True,
        "sheet_prefix": "Heatmap_%ТО",
    }

    if show_download_button:
        st.download_button(
            "⬇️ Скачати Heatmap % в ТО в Excel",
            data=_heatmap_block_to_excel_bytes(
                title=ratio_heatmap_title,
                heat_df=heat_display,
                top_df=top_display,
                antitop_df=antitop_display,
                val_col=val_col,
                header_color=TEAL_HDR,
                percent=True,
            ),
            file_name=_safe_xlsx_filename(f"heatmap_ratio_{ratio_mode}"),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_ratio_heatmap_excel_{ratio_mode}_{val_col}",
            use_container_width=True,
        )

    return ratio_export_block


# ── Dynamic HTML export: 3 analytical tabs ───────────────────────────────────
def export_excel(df, df_filtered, col_tt, col_article, col_month, col_value,
                 col_plf, articles_to_show, tt_val, group_factors, metric_col,
                 mode, pivot_df=None, df_tt_agg=None, col_ratio=None,
                 col_division=None, ratio_mode="Delta"):
    """
    Експорт усіх блоків дашборду в Excel.
    Розрахунки залишаються по col_tt, а для відображення використовується Підрозділ.
    """
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage

    NUM_FMT = '# ##0;-# ##0;-'
    PCT_FMT = '+0.0%;-0.0%;-'
    PCT_VALUE_FMT = '0.00"%"'

    def hdr_fill(h):
        h = str(h).lstrip("#")
        return PatternFill("solid", start_color=h, end_color=h)

    def thin_border():
        s = Side(style="thin", color="AAAAAA")
        return Border(left=s, right=s, top=s, bottom=s)

    def scw(ws, ci, w):
        ws.column_dimensions[get_column_letter(ci)].width = w

    def safe_sheet_name(name):
        invalid = ['\\', '/', '*', '?', ':', '[', ']']
        s = str(name) if name is not None else "Аркуш"
        for ch in invalid:
            s = s.replace(ch, "_")
        s = s.strip()[:31] or "Аркуш"
        base = s[:28]
        i = 1
        while s in wb.sheetnames:
            suffix = f"_{i}"
            s = (base[:31-len(suffix)] + suffix)[:31]
            i += 1
        return s

    def write_df_sheet(title, data, header_color="5b2d8e", index=True,
                       number_format=NUM_FMT, pct_cols=None, first_col_width=32):
        ws = wb.create_sheet(safe_sheet_name(title))
        ws.freeze_panes = "B2"
        pct_cols = set(pct_cols or [])

        if data is None or data.empty:
            ws.cell(row=1, column=1, value="Немає даних")
            return ws

        out = data.copy()
        if index:
            out = out.reset_index()
        out = out.replace([np.inf, -np.inf], np.nan)

        for ci, h in enumerate(out.columns, 1):
            c = ws.cell(row=1, column=ci, value=str(h))
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = hdr_fill(header_color)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = thin_border()
        ws.row_dimensions[1].height = 28

        for ri, (_, row) in enumerate(out.iterrows(), 2):
            for ci, col_name in enumerate(out.columns, 1):
                v = row[col_name]
                c = ws.cell(row=ri, column=ci)
                c.border = thin_border()
                c.font = Font(name="Arial", size=9)
                if pd.isna(v):
                    c.value = None
                    c.fill = hdr_fill("FFFFFF")
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    c.value = float(v)
                    c.number_format = PCT_VALUE_FMT if str(col_name) in pct_cols or "%" in str(col_name) else number_format
                    c.alignment = Alignment(horizontal="right")
                    if float(v) < 0:
                        c.font = Font(name="Arial", size=9, color="C0392B")
                else:
                    c.value = str(v)
                    c.alignment = Alignment(horizontal="left")
        ws.column_dimensions["A"].width = first_col_width
        for ci in range(2, len(out.columns) + 1):
            scw(ws, ci, 12)
        return ws

    def display_label(tt_value, display_map):
        return _tt_display_label(tt_value, display_map) if display_map else tt_value

    wb = Workbook()
    wb.remove(wb.active)
    tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)

    # 1) Зведена таблиця по статтях
    ws_p = wb.create_sheet("Зведена_таблиця")
    ws_p.freeze_panes = "B2"
    header = ["Стаття"] + MONTHS_LIST + ["РАЗОМ"]
    for ci, h in enumerate(header, 1):
        c = ws_p.cell(row=1, column=ci, value=h)
        c.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        c.fill = hdr_fill("2e7d32")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = thin_border()
    ws_p.row_dimensions[1].height = 28
    ws_p.column_dimensions["A"].width = 36
    for ci in range(2, len(header) + 1):
        scw(ws_p, ci, 11)

    for ri, article in enumerate(articles_to_show, 2):
        tdf = build_article_monthly(
            df, df_filtered, col_tt, col_article, col_month,
            col_value, col_plf, article, tt_val, group_factors
        )
        vals = [article] + [tdf.loc[m, metric_col] for m in range(1, 13)]
        vals.append(sum(tdf.loc[m, metric_col] for m in range(1, 13)))
        for ci, v in enumerate(vals, 1):
            c = ws_p.cell(row=ri, column=ci, value=v)
            c.border = thin_border()
            c.font = Font(name="Arial", size=9)
            if ci == 1:
                c.alignment = Alignment(horizontal="left")
            else:
                c.number_format = NUM_FMT
                c.alignment = Alignment(horizontal="right")
                if isinstance(v, (int, float, np.integer, np.floating)) and v < 0:
                    c.font = Font(name="Arial", size=9, color="C0392B")

    # 2) Всі основні блоки одним листом
    main_rows = []
    for article in articles_to_show:
        tdf = build_article_monthly(
            df, df_filtered, col_tt, col_article, col_month,
            col_value, col_plf, article, tt_val, group_factors
        )
        for label, key in [("План", "Plan"), ("Факт", "Fact"), ("Average", "Average"), ("Дельта Fact−Average", "Delta"), ("Дельта Fact−Plan", "DeltaPlan")]:
            row = {"Стаття": article, "Показник": label}
            vals = [tdf.loc[m, key] for m in range(1, 13)]
            for m in range(1, 13):
                row[MONTH_LABELS[m]] = tdf.loc[m, key]
            row["РАЗОМ"] = sum(vals)
            main_rows.append(row)
    write_df_sheet("Основні_таблиці", pd.DataFrame(main_rows), index=False, header_color="2e7d32")

    # 3) Окремі листи по кожній статті + графік
    row_labels = ["План", "Факт", "Average", "Дельта Fact−Average", "Дельта Fact−Plan"]
    row_keys = ["Plan", "Fact", "Average", "Delta", "DeltaPlan"]
    row_fills = ["FFFFFF", "e8d5f5", "fde8e8", "fff9e0", "eaf7ea"]
    row_colors = ["333333", "5b2d8e", "c0392b", "b8860b", "2e7d32"]

    for article in articles_to_show:
        tdf = build_article_monthly(
            df, df_filtered, col_tt, col_article,
            col_month, col_value, col_plf, article, tt_val, group_factors
        )
        ws = wb.create_sheet(safe_sheet_name(article))
        ws.freeze_panes = "B3"
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=14)
        tc = ws.cell(row=1, column=1, value=article)
        tc.font = Font(bold=True, color="FFFFFF", name="Arial", size=12)
        tc.fill = hdr_fill("5b2d8e")
        tc.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 22

        headers = ["Показник"] + MONTHS_LIST + ["РАЗОМ"]
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=2, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = hdr_fill("2e7d32")
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin_border()

        for ri, (label, key, fill_hex, color_hex) in enumerate(zip(row_labels, row_keys, row_fills, row_colors), 3):
            vals = [tdf.loc[m, key] for m in range(1, 13)]
            total = sum(vals)
            nc = ws.cell(row=ri, column=1, value=label)
            nc.font = Font(bold=True, color=color_hex, name="Arial", size=9)
            nc.fill = hdr_fill(fill_hex)
            nc.border = thin_border()
            nc.alignment = Alignment(horizontal="left")
            for ci, v in enumerate(vals, 2):
                c = ws.cell(row=ri, column=ci, value=float(v) if pd.notna(v) else None)
                c.number_format = NUM_FMT
                c.fill = hdr_fill(fill_hex)
                c.border = thin_border()
                c.alignment = Alignment(horizontal="right")
                c.font = Font(name="Arial", size=9, color="C0392B" if pd.notna(v) and v < 0 else color_hex)
            tc2 = ws.cell(row=ri, column=14, value=float(total) if pd.notna(total) else None)
            tc2.number_format = NUM_FMT
            tc2.fill = hdr_fill(fill_hex)
            tc2.border = thin_border()
            tc2.alignment = Alignment(horizontal="right")
            tc2.font = Font(bold=True, name="Arial", size=9, color="C0392B" if total < 0 else color_hex)

        ws.column_dimensions["A"].width = 12
        for ci in range(2, 15):
            scw(ws, ci, 11)

        fig = go.Figure([
            go.Bar(x=MONTHS_LIST, y=[tdf.loc[m, "Plan"] for m in range(1, 13)], name="План", marker_color="#c0c0c0"),
            go.Bar(x=MONTHS_LIST, y=[tdf.loc[m, "Fact"] for m in range(1, 13)], name="Факт", marker_color="#5b2d8e"),
            go.Scatter(x=MONTHS_LIST, y=[tdf.loc[m, "Average"] for m in range(1, 13)], mode="lines+markers", name="Average", line=dict(color="#c0392b", width=2.5)),
            go.Scatter(x=MONTHS_LIST, y=[tdf.loc[m, "Delta"] for m in range(1, 13)], mode="lines+markers", name="Дельта Fact−Average", line=dict(color="#f0c000", width=2), yaxis="y2"),
            go.Scatter(x=MONTHS_LIST, y=[tdf.loc[m, "DeltaPlan"] for m in range(1, 13)], mode="lines+markers", name="Дельта Fact−Plan", line=dict(color="#2e7d32", width=2, dash="dash"), yaxis="y2"),
        ])
        fig.update_layout(
            barmode="group", height=320, width=900, plot_bgcolor="white", paper_bgcolor="white",
            title=dict(text=f"Аналіз — {article}", x=0.5, font=dict(size=12)),
            yaxis2=dict(overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="v", x=1.06, y=1, font=dict(size=9)),
            margin=dict(t=40, b=30, l=55, r=130), font=dict(family="Arial"),
        )
        try:
            import plotly.io as pio
            img_obj = XLImage(io.BytesIO(pio.to_image(fig, format="png", scale=1.5)))
            img_obj.anchor = "A7"
            ws.add_image(img_obj)
        except Exception:
            ws.cell(row=7, column=1, value="⚠️ Графік недоступний (pip install kaleido)")

    # 4) % в ТО: основні блоки + зведена
    if col_ratio:
        ratio_rows = []
        ratio_summary_rows = []
        for article in articles_to_show:
            rdf = build_ratio_monthly(
                df_filtered, col_tt, col_article, col_month,
                col_ratio, col_plf, article, tt_val,
                df_all=df, group_factors=group_factors
            )
            for label, key in [("План %", "Plan"), ("Факт %", "Fact"), ("Average %", "Average"), ("Δ %", "Delta")]:
                row = {"Стаття": article, "Показник": label}
                vals = [rdf.loc[m, key] for m in range(1, 13)]
                for m in range(1, 13):
                    row[MONTH_LABELS[m]] = rdf.loc[m, key]
                nz = [v for v in vals if v != 0]
                row["Серед."] = np.mean(nz) if nz else 0.0
                ratio_rows.append(row)

            row = {"Стаття": article}
            vals = [rdf.loc[m, "Fact"] for m in range(1, 13)]
            for m in range(1, 13):
                row[MONTH_LABELS[m]] = vals[m-1]
            nz = [v for v in vals if v != 0]
            row["Серед."] = np.mean(nz) if nz else 0.0
            ratio_summary_rows.append(row)

        write_df_sheet("Основні_%_в_ТО", pd.DataFrame(ratio_rows), index=False, header_color="085f63")
        write_df_sheet("% в ТО_зведена", pd.DataFrame(ratio_summary_rows).set_index("Стаття"), header_color="085f63")

    # 5) Зведена по Підрозділ
    if df_tt_agg is not None and not df_tt_agg.empty:
        df_tt_export = df_tt_agg.copy()
        if "ТТ" in df_tt_export.columns:
            df_tt_export["Підрозділ"] = df_tt_export["ТТ"].map(lambda v: display_label(v, tt_display_map))
            rest_cols = [c for c in df_tt_export.columns if c not in ("Підрозділ", "ТТ")]
            df_tt_export = ["Підрозділ"] and df_tt_export[["Підрозділ"] + rest_cols]
        write_df_sheet("Підрозділ_Зведена", df_tt_export, index=False, header_color="5b2d8e")

    # 6) Heatmap + TOP/ANTITOP абсолютні
    if group_factors:
        heat, tt_table, val_col = build_heat_data(
            df, df_filtered, col_tt, col_article, col_month, col_value,
            col_plf, group_factors, articles_to_show, mode
        )
        heat_display = _replace_tt_index_with_division(heat, df_filtered, col_tt, col_division)
        write_df_sheet("Heatmap", heat_display, header_color="5b2d8e")

        rating_options = _get_heatmap_rating_factor_options(
            df_filtered, col_tt=col_tt, col_division=col_division, group_factors=group_factors
        ) or [col_tt]
        heatmap_rating_col = rating_options[0]
        sum_val = _prepare_heatmap_rating_table(
            tt_table, df_filtered, col_tt, val_col, heatmap_rating_col, aggfunc="sum"
        )
        top_df = sum_val.sort_values(val_col, ascending=True).head(50)[["Підрозділ", val_col]]
        anti_df = sum_val.sort_values(val_col, ascending=False).head(50)[["Підрозділ", val_col]]
        top_anti = pd.concat({"TOP_економія": top_df.reset_index(drop=True), "ANTITOP_переліміт": anti_df.reset_index(drop=True)}, axis=1)
        write_df_sheet("TOP_ANTITOP", top_anti, index=False, header_color="5b2d8e")

        # 7) Heatmap + TOP/ANTITOP % в ТО
        if col_ratio:
            heat_r, tt_table_r, val_col_r = build_ratio_heat_data(
                df, df_filtered, col_tt, col_article, col_month,
                col_ratio, col_plf, articles_to_show, ratio_mode,
                group_factors=group_factors
            )
            heat_r_display = _replace_tt_index_with_division(heat_r, df_filtered, col_tt, col_division)
            write_df_sheet("Heatmap_%_ТО", heat_r_display, header_color="085f63", number_format=PCT_VALUE_FMT)

            ratio_rating_options = _get_heatmap_rating_factor_options(
                df_filtered, col_tt=col_tt, col_division=col_division, group_factors=group_factors
            ) or [col_tt]
            ratio_rating_col = ratio_rating_options[0]
            sum_val_r = _prepare_heatmap_rating_table(
                tt_table_r, df_filtered, col_tt, val_col_r, ratio_rating_col, aggfunc="mean"
            )
            top_r = sum_val_r.sort_values(val_col_r, ascending=True).head(50)[["Підрозділ", val_col_r]]
            anti_r = sum_val_r.sort_values(val_col_r, ascending=False).head(50)[["Підрозділ", val_col_r]]
            top_anti_r = pd.concat({"TOP_%": top_r.reset_index(drop=True), "ANTITOP_%": anti_r.reset_index(drop=True)}, axis=1)
            write_df_sheet("TOP_ANTITOP_%_ТО", top_anti_r, index=False, header_color="085f63", number_format=PCT_VALUE_FMT)

        # 8) Аналіз комбінації факторів
        combo_rows = []
        combo_ratio_rows = []
        selected_factors = list(group_factors)
        combo_col = "Комбінація факторів"
        for article in articles_to_show:
            art_all = _prep(df[df[col_article] == article].copy(), col_month)
            all_fact = _fact_rows(art_all, col_plf).copy()
            if selected_factors and not all_fact.empty:
                all_fact[col_value] = pd.to_numeric(all_fact[col_value], errors="coerce")
                all_fact[combo_col], valid_selected_factors = _make_combo_col(all_fact, selected_factors)
                if not valid_selected_factors:
                    continue
                combo_impact = all_fact.groupby([combo_col, "_m"], as_index=False, observed=True)[col_value].agg(
                    Середнє="mean", Кількість="count", Сума="sum", Відхилення="std"
                )
                combo_impact["Стаття"] = article
                combo_impact["Місяць"] = combo_impact["_m"].map(MONTH_LABELS)
                combo_rows.append(combo_impact[["Стаття", combo_col, "Місяць", "Середнє", "Кількість", "Сума", "Відхилення"]])

            if col_ratio and selected_factors:
                art_all_r = _prep(df[df[col_article] == article].copy(), col_month)
                all_fact_r = _fact_rows(art_all_r, col_plf).copy()
                if not all_fact_r.empty and col_ratio in all_fact_r.columns:
                    all_fact_r[col_ratio] = pd.to_numeric(all_fact_r[col_ratio], errors="coerce")
                    all_fact_r[combo_col], valid_selected_factors = _make_combo_col(all_fact_r, selected_factors)
                    if not valid_selected_factors:
                        continue
                    combo_r = all_fact_r.groupby([combo_col, "_m"], as_index=False, observed=True)[col_ratio].agg(
                        Середнє="mean", Кількість="count", Сума="sum", Відхилення="std"
                    )
                    combo_r["Стаття"] = article
                    combo_r["Місяць"] = combo_r["_m"].map(MONTH_LABELS)
                    combo_ratio_rows.append(combo_r[["Стаття", combo_col, "Місяць", "Середнє", "Кількість", "Сума", "Відхилення"]])

        if combo_rows:
            write_df_sheet("Комбінації_факторів", pd.concat(combo_rows, ignore_index=True), index=False, header_color="e67e22")
        if combo_ratio_rows:
            write_df_sheet("Комбінації_%_ТО", pd.concat(combo_ratio_rows, ignore_index=True), index=False, header_color="e67e22", number_format=PCT_VALUE_FMT)

    # 9) Статистичний аналіз факторів
    if group_factors:
        stat_rows = []
        combo_stat_rows = []
        for article in articles_to_show:
            stat_df, _ = analyze_statistical_factor_models(df, col_article, col_value, col_plf, article, group_factors)
            if stat_df is not None and not stat_df.empty:
                stat_df = stat_df.copy()
                stat_df.insert(0, "Стаття", article)
                stat_rows.append(stat_df)

            combo_factors = group_factors[:min(2, len(group_factors))]
            combo_stat, _ = analyze_combination_statistical_impact(df, col_article, col_value, col_plf, article, combo_factors)
            if combo_stat is not None and not combo_stat.empty:
                combo_stat = combo_stat.copy()
                combo_stat.insert(0, "Стаття", article)
                combo_stat_rows.append(combo_stat)

        if stat_rows:
            write_df_sheet("Статистика_факторів", pd.concat(stat_rows, ignore_index=True), index=False, header_color="2c3e50")
        if combo_stat_rows:
            write_df_sheet("Статистика_комбінацій", pd.concat(combo_stat_rows, ignore_index=True), index=False, header_color="2c3e50")

    # 10) Дані після фільтрів
    raw = df_filtered.copy()
    if col_tt in raw.columns:
        raw.insert(0, "Підрозділ_відображення", raw[col_tt].map(lambda v: display_label(v, tt_display_map)))
    if len(raw) <= 50000:
        write_df_sheet("Дані_після_фільтрів", raw, index=False, header_color="607d8b", first_col_width=28)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


# ── Main ──────────────────────────────────────────────────────────────────────


@st.cache_data(show_spinner=False, ttl=3600)
def get_excel_sheet_names(file_bytes, file_name):
    """Кешований список аркушів, щоб pd.ExcelFile не запускався на кожен rerun."""
    import os
    ext = os.path.splitext(file_name)[1].lower()
    engine = "pyxlsb" if ext == ".xlsb" else None
    if engine:
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)
    else:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names


def main():
    st.set_page_config(page_title="СІМІ Dashboard", layout="wide")
    st.markdown("""
    <style>
    .simi-header { background:linear-gradient(90deg,#5b2d8e 0%,#7b52ae 100%);
                   color:white;padding:10px 18px;border-radius:4px;margin-bottom:4px; }
    .simi-logo   { font-size:2rem;font-weight:900;color:#f0c000;letter-spacing:1px;
                   margin-right:24px;vertical-align:middle; }
    .simi-store  { font-size:1.1rem;font-weight:700;color:white;vertical-align:middle; }
    .simi-meta-grid { display:grid;grid-template-columns:repeat(5,1fr);gap:4px 12px;margin-top:6px; }
    .simi-meta-item { font-size:0.78rem;color:#e0d0f8; }
    .simi-meta-val  { font-size:0.85rem;font-weight:700;color:white; }
    .article-selector { background:#f4f0fa;border:2px solid #5b2d8e;
                        border-radius:8px;padding:12px 16px;margin-bottom:14px; }
    .block-sep      { border-top:2px solid #5b2d8e;margin:16px 0 10px 0; }
    .block-sep-teal { border-top:2px solid #0d7377;margin:16px 0 10px 0; }
    .ratio-section-banner { background:linear-gradient(90deg,#085f63 0%,#0d7377 100%);
                            color:white;padding:8px 18px;border-radius:4px;
                            margin:8px 0 6px 0;font-size:0.95rem;font-weight:700; }
    div[data-testid="stButton"]>button[kind="primary"]   { background-color:#5b2d8e!important;color:white!important;border:2px solid #5b2d8e!important;font-weight:700!important;font-size:0.72rem!important; }
    div[data-testid="stButton"]>button[kind="secondary"] { background-color:#f4f0fa!important;color:#5b2d8e!important;border:1px solid #c9b6e8!important;font-size:0.72rem!important; }
    div[data-testid="stButton"]>button[kind="secondary"]:hover { background-color:#e8d5f5!important;border-color:#5b2d8e!important; }
    </style>""", unsafe_allow_html=True)

    file = st.file_uploader("📂 Завантажте Excel", type=["xlsx", "xlsb"])
    if file is None:
        st.info("Завантажте Excel-файл для початку роботи.")
        st.stop()

    file_bytes = file.getvalue()
    sheet_names = get_excel_sheet_names(file_bytes, file.name)
    sheet_name = st.selectbox("Аркуш", sheet_names)
    df = load_excel(file_bytes, file.name, sheet_name)
    cols       = df.columns.tolist()
    _init_column_state(df, file.name, sheet_name)

    with st.expander("⚙️ Налаштування колонок", expanded=False):
        st.caption("Колонки утотожнюються автоматично за назвою. За потреби їх можна змінити вручну.")

        if st.button("🔄 Автоутотожнити колонки", key="remap_columns_btn"):
            auto_cols = auto_map_columns(df)
            for key, val in auto_cols.items():
                st.session_state[key] = val if val in cols else None
            st.rerun()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            col_tt   = _select_col("TT (Магазин)", cols, "col_tt")
            col_year = _select_col("Year", cols, "col_year")
        with c2:
            col_month = _select_col("Month", cols, "col_month")
            col_value = _select_col("Значення", cols, "col_value")
            col_kwh = _select_col(
                "ЕЕ_кВт",
                cols,
                "col_kwh",
                allow_empty=True,
                help="Колонка для розрахунку вартості: Значення / ЕЕ_кВт"
            )
        with c3:
            col_plf     = _select_col("PL / F", cols, "col_plf")
            col_article = _select_col("Стаття бюджету", cols, "col_article")
        with c4:
            col_level0 = _select_col("Level_0", cols, "col_level0")
            col_ratio = _select_col(
                "% в ТО без акцизу та без ПДВ",
                cols,
                "col_ratio",
                allow_empty=True,
                help="Колонка з відсотком % в ТО. Оберіть '—', щоб приховати блок."
            )

    with st.expander("🏪 Колонки шапки магазину", expanded=False):
        st.caption("Колонки шапки також утотожнюються автоматично, але їх можна змінити вручну.")
        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            col_city = _select_col("Місто", cols, "col_city", allow_empty=True)
            col_area = _select_col("Площа", cols, "col_area", allow_empty=True)
        with sh2:
            col_format = _select_col("Формат ТО", cols, "col_format", allow_empty=True)
            col_format2 = _select_col("Формат2", cols, "col_format2", allow_empty=True)
        with sh3:
            col_division = _select_col("Підрозділ (назва в шапці)", cols, "col_division", allow_empty=True)
            col_rik = _select_col("Рік", cols, "col_rik", allow_empty=True)
            col_mega = _select_col("Мегасегмент (резерв)", cols, "col_mega", allow_empty=True)
        col_mis = None  # Місяць у шапці не показуємо

    required_cols = {
        "TT (Магазин)": col_tt,
        "Year": col_year,
        "Month": col_month,
        "Значення": col_value,
        "PL / F": col_plf,
        "Стаття бюджету": col_article,
        "Level_0": col_level0,
    }
    missing_required = [name for name, value in required_cols.items() if not value]
    if missing_required:
        st.error("Не утотожнено обов’язкові колонки: " + ", ".join(missing_required))
        st.stop()

    # Sidebar
    st.sidebar.markdown("## 🔍 Фільтри")
    year_val   = st.sidebar.multiselect("Year",    sorted(df[col_year].dropna().unique(),   key=str))
    month_val  = st.sidebar.multiselect("Month",   sorted(df[col_month].dropna().unique(),  key=str))
    level0_val = st.sidebar.multiselect("Level_0", sorted(df[col_level0].dropna().unique(), key=str))

    st.sidebar.markdown("### ➕ Додаткові фільтри")
    fixed_cols    = {col_tt, col_year, col_month, col_level0}
    extra_filters = {}

    remaining = [c for c in cols if c not in fixed_cols]
    for i in range(1, 7):
        key_col = f"extra_filter_col{i}"
        key_val = f"extra_filter_val{i}"
        prev_extra = list(extra_filters.keys())
        options    = ["— не обрано —"] + [c for c in remaining if c not in prev_extra]
        chosen_col = st.sidebar.selectbox(f"Стовпець {i}", options, key=key_col)
        if chosen_col != "— не обрано —":
            chosen_val = st.sidebar.multiselect(
                f"Значення «{chosen_col}»",
                sorted(df[chosen_col].dropna().unique(), key=str), key=key_val,
            )
            if chosen_val:
                extra_filters[chosen_col] = chosen_val
        else:
            break

    # Pre-filter for TT list (без впливу на df — повний датасет зберігається)
    df_pre = df
    if year_val:   df_pre = df_pre[df_pre[col_year].isin(year_val)]
    if month_val:  df_pre = df_pre[df_pre[col_month].isin(month_val)]
    if level0_val: df_pre = df_pre[df_pre[col_level0].isin(level0_val)]
    for col_e, vals_e in extra_filters.items():
        df_pre = df_pre[df_pre[col_e].isin(vals_e)]

    visible_tts = sorted(df_pre[col_tt].dropna().unique(), key=str)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏪 ТТ за поточними фільтрами")
    if visible_tts:
        st.sidebar.caption(f"Знайдено: {len(visible_tts)} магазинів")
        tt_search    = st.sidebar.text_input("🔎 Пошук ТТ", value="",
                                              placeholder="Введіть назву...", key="tt_search")
        filtered_tts = [tt for tt in visible_tts if tt_search.lower() in str(tt).lower()] \
                       if tt_search else visible_tts
        b1, b2 = st.sidebar.columns(2)
        with b1:
            if st.button("✅ Всі", key="tt_select_all", use_container_width=True):
                st.session_state["tt_multiselect"] = filtered_tts
        with b2:
            if st.button("✖ Жодного", key="tt_clear_all", use_container_width=True):
                st.session_state["tt_multiselect"] = []
        tt_val = st.sidebar.multiselect(
            "Оберіть ТТ:", options=filtered_tts,
            default=st.session_state.get("tt_multiselect", []), key="tt_multiselect",
        )
    else:
        st.sidebar.warning("Немає ТТ за обраними фільтрами.")
        tt_val = []

    st.sidebar.markdown("---")
    mode       = st.sidebar.selectbox("Mode (Heatmap)",       ["Delta", "Delta %", "Z-score", "Fact", "Average"])
    ratio_mode = st.sidebar.selectbox("Mode (% в ТО Heatmap)", ["Delta", "Delta %", "Fact", "Average"], key="ratio_mode")

    options = [c for c in df.columns if c not in [col_value, col_plf, col_article]]
    group_factors = st.sidebar.multiselect(
        "Фактори групування (Average/Std)",
        options=options,
        default=[col_tt] if col_tt in options else [],
        placeholder="Оберіть стовпці"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👁️ Відображення")
    show_ratio_section = st.sidebar.checkbox("Показати блок «% в ТО»", value=True,
                                              key="show_ratio_section") if col_ratio else False
    show_cost_section = st.sidebar.checkbox(
        "Показати блок «Вартість Значення/ЕЕ_кВт»",
        value=True,
        key="show_cost_section",
        help="Показується тільки для статей Електроенергія та Водопостачання."
    ) if col_kwh else False
    show_ratio_heatmap = st.sidebar.checkbox("Показати Heatmap % в ТО", value=True,
                                              key="show_ratio_heatmap") if col_ratio else False

    # Перемикач метрики для перших основних таблиць/графіків.
    if col_kwh and col_kwh in df.columns:
        main_metric_choice = st.sidebar.radio(
            "Метрика для основних таблиць",
            options=["Значення", "ЕЕ_кВт"],
            index=0,
            horizontal=True,
            key="main_metric_choice",
            help="Перемикає перші основні таблиці та всі повʼязані абсолютні розрахунки."
        )
        col_main_value = col_kwh if main_metric_choice == "ЕЕ_кВт" else col_value
    else:
        main_metric_choice = "Значення"
        col_main_value = col_value

    # Одноразова підготовка важких колонок після вибору мапінгу.
    # Далі всі функції бачать готову _m і не перераховують місяці десятки разів.
    df = _prep(df, col_month)
    _ensure_numeric_inplace(df, [col_value, col_kwh, col_ratio])

    # ── apply_filters: df залишається ПОВНИМ (для норм), df_filtered — для відображення ──
    def apply_filters(d):
        if tt_val:     d = d[d[col_tt].isin(tt_val)]
        if year_val:   d = d[d[col_year].isin(year_val)]
        if month_val:  d = d[d[col_month].isin(month_val)]
        if level0_val: d = d[d[col_level0].isin(level0_val)]
        for col_e, vals_e in extra_filters.items():
            d = d[d[col_e].isin(vals_e)]
        return d

    df_filtered = apply_filters(df)

    def _get_active_single_tt_from_filter_or_slicer():
        """
        Повертає один активний ТТ тільки якщо він вибраний:
        1) у боковому фільтрі ТТ; або
        2) у компактному слайсері на вкладці аналізу.
        Якщо вибрано 0 або більше 1 ТТ — повертає None.
        """
        if tt_val and len(tt_val) == 1:
            return tt_val[0]

        active_slicer_tts = []
        for key, value in st.session_state.items():
            if str(key).startswith("shared_slicer_tt_") and value not in (None, "__ALL__"):
                active_slicer_tts.append(value)

        active_slicer_tts = list(dict.fromkeys(active_slicer_tts))
        return active_slicer_tts[0] if len(active_slicer_tts) == 1 else None

    active_header_tt = _get_active_single_tt_from_filter_or_slicer()
    show_store_meta = active_header_tt is not None
    df_meta_source = (
        df_filtered[df_filtered[col_tt] == active_header_tt].copy()
        if show_store_meta
        else pd.DataFrame(columns=df_filtered.columns)
    )

    def get_meta(col):
        # Метадані магазину показуємо тільки для одного активного ТТ.
        if not show_store_meta:
            return ""
        if col is None or col == "—" or col not in df_meta_source.columns:
            return "—"

        vals = df_meta_source[col].dropna().unique()
        if len(vals) == 0:
            return "—"

        return str(vals[0])

    def get_meta_year(col):
        # Рік у шапці показуємо як ціле число без .0 / коми.
        raw = get_meta(col)
        if raw in (None, "", "—"):
            return "—"
        try:
            return str(int(float(str(raw).replace(",", "."))))
        except Exception:
            return str(raw).replace(".0", "")

    def get_meta_format2():
        # За вимогою: поле "Мегасегмент" у шапці показує значення з колонки Формат2.
        val = get_meta(col_format2) if "col_format2" in locals() else "—"
        return val if val not in (None, "", "—") else get_meta(col_mega)

    articles_all = sorted(df[col_article].dropna().unique(), key=str)

    st.markdown('<div class="article-selector">', unsafe_allow_html=True)
    sel_col1, sel_col2, sel_col3 = st.columns([3, 1, 1])
    with sel_col1:
        st.markdown("**🎯 Стаття бюджету для аналізу**")
        selected_article = st.selectbox("article_selector", articles_all,
                                         key="global_article", label_visibility="collapsed")
    with sel_col2:
        st.markdown("&nbsp;")
        show_all = st.checkbox("Показати всі статті", value=False, key="show_all")
    with sel_col3:
        st.markdown("&nbsp;")
        multi_sel = st.multiselect("Або обери кілька:", articles_all,
                                    default=[], key="multi_article")
    st.markdown('</div>', unsafe_allow_html=True)

    articles_to_show = (articles_all if show_all
                        else multi_sel if multi_sel
                        else [selected_article])

    if len(articles_to_show) == 1:
        st.info(f"📌 Показується стаття: **{articles_to_show[0]}**")
    else:
        st.info(f"📌 Показується {len(articles_to_show)} статей: {', '.join(articles_to_show)}")

    # Якщо вибраний один магазин — у шапці показуємо значення зі стовпця "Підрозділ".
    # Якщо колонка не знайдена або порожня — безпечно повертаємось до ТТ.
    if show_store_meta:
        division_name = get_meta(col_division) if "col_division" in locals() else "—"
        store_name = division_name if division_name not in (None, "", "—") else str(active_header_tt)
    else:
        store_name = (", ".join(str(v) for v in tt_val) if tt_val else "Всі магазини")

    meta_html = ""
    if show_store_meta:
        meta_html = f"""
      <div class="simi-meta-grid">
        <div><span class="simi-meta-item">Місто </span><span class="simi-meta-val">{get_meta(col_city)}</span></div>
        <div><span class="simi-meta-item">Площа </span><span class="simi-meta-val">{get_meta(col_area)}</span></div>
        <div><span class="simi-meta-item">Формат ТО </span><span class="simi-meta-val">{get_meta(col_format)}</span></div>
        <div><span class="simi-meta-item">Мегасегмент </span><span class="simi-meta-val">{get_meta_format2()}</span></div>
        <div><span class="simi-meta-item">Рік </span><span class="simi-meta-val">{get_meta_year(col_rik)}</span></div>
      </div>"""

    st.markdown(f"""
    <div class="simi-header">
      <span class="simi-logo">СіМі</span>
      <span class="simi-store">{store_name}</span>
      {meta_html}
    </div>""", unsafe_allow_html=True)

    # ── Lazy tabs layout ──────────────────────────────────────────────────────
    # st.tabs() рахує всі вкладки одразу. segmented_control рендерить тільки активну,
    # тому перемикання та оновлення даних значно швидші без кешування.
    active_tab = st.segmented_control(
        "Вкладка",
        [
            "📊 Аналіз по статтях",
            "🧩 Комбінації факторів",
            "📋 Зведені таблиці",
            "🌡️ Heatmap / TOP",
            "📊 Статистика факторів",
            "📥 Експорт",
        ],
        default="📊 Аналіз по статтях",
        key="active_main_tab",
    )

    metric_col = "Fact"
    pivot_df = pd.DataFrame()
    df_tt_agg = pd.DataFrame()

    if active_tab == "📊 Аналіз по статтях":
            # ── Article blocks ────────────────────────────────────────────────────────
            for art_idx, article in enumerate(articles_to_show):
                st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

                shared_active_tt = _render_shared_tt_slicer(
                    art_idx, df_filtered, col_tt, col_article, article, col_division
                )

                tdf = build_article_monthly(
                    df, df_filtered, col_tt, col_article, col_month, col_main_value,
                    col_plf, article, tt_val, group_factors
                )
                render_article_block(
                    title=article, table_df=tdf,
                    df=df, df_filtered=df_filtered,
                    col_tt=col_tt, col_article=col_article,
                    col_month=col_month, col_value=col_main_value, col_plf=col_plf,
                    group_factors=group_factors, tt_val=tt_val, article_idx=art_idx,
                    active_tt=shared_active_tt,
                    col_division=col_division,
                    metric_label=main_metric_choice,
                )

                if col_ratio and show_ratio_section:
                    rdf = build_ratio_monthly(
                        df_filtered, col_tt, col_article, col_month, col_ratio, col_plf,
                        article, tt_val, df_all=df, group_factors=group_factors
                    )
                    st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
                    render_ratio_article_block(
                        title=article, table_df=rdf,
                        df=df, df_filtered=df_filtered,
                        col_tt=col_tt, col_article=col_article,
                        col_month=col_month, col_ratio=col_ratio, col_plf=col_plf,
                        tt_val=tt_val, article_idx=art_idx, group_factors=group_factors,
                        active_tt=shared_active_tt,
                        col_division=col_division,
                    )

                if show_cost_section and col_kwh and _is_utility_cost_article(article):
                    cost_df = build_cost_monthly(
                        df_filtered, col_tt, col_article, col_month,
                        col_value, col_kwh, col_plf, article, tt_val
                    )
                    st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
                    render_cost_article_block(
                        title=article, table_df=cost_df,
                        df_filtered=df_filtered,
                        col_tt=col_tt, col_article=col_article,
                        col_month=col_month, col_value=col_value,
                        col_kwh=col_kwh, col_plf=col_plf,
                        tt_val=tt_val, article_idx=art_idx,
                        active_tt=shared_active_tt,
                        col_division=col_division,
                    )


    if active_tab == "🧩 Комбінації факторів":
            # ── Combined factor impact analysis ───────────────────────────────
            render_combined_factor_impact_tab(
                df=df,
                df_filtered=df_filtered,
                col_tt=col_tt,
                col_article=col_article,
                col_month=col_month,
                col_value=col_main_value,
                col_ratio=col_ratio,
                col_plf=col_plf,
                articles_to_show=articles_to_show,
                group_factors=group_factors,
            )

    if active_tab == "📋 Зведені таблиці":
            # ── Pivot table ───────────────────────────────────────────────────────────
            st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
            st.subheader("📋 Зведена таблиця")
            pivot_metric = st.radio("Метрика", ["Fact", "Plan", "Delta (Fact-Plan)"], horizontal=True)
            col_map_d    = {"Fact": "Fact", "Plan": "Plan", "Delta (Fact-Plan)": "Delta"}
            metric_col   = col_map_d[pivot_metric]

            rows_pivot = []
            for article in articles_to_show:
                tdf = build_article_monthly(df, df_filtered, col_tt, col_article,
                                            col_month, col_main_value, col_plf, article, tt_val, group_factors)
                row = {"Стаття": article}
                for m in range(1, 13):
                    row[MONTH_LABELS[m]] = tdf.loc[m, metric_col]
                row["РАЗОМ"] = sum(tdf.loc[m, metric_col] for m in range(1, 13))
                rows_pivot.append(row)

            pivot_df = pd.DataFrame(rows_pivot).set_index("Стаття")
            cmap_p   = "RdYlGn_r" if pivot_metric == "Delta (Fact-Plan)" else "Blues"
            st.dataframe(
                pivot_df.style
                    .background_gradient(cmap=cmap_p, axis=None)
                    .apply(_style_white_na, axis=None)
                    .format(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-", na_rep="-"),
                use_container_width=True,
            )

            # ── % в ТО зведена ───────────────────────────────────────────────────────
            if col_ratio and show_ratio_section:
                st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
                st.markdown('<div class="ratio-section-banner">📊 Зведена таблиця — % в ТО без акцизу та без ПДВ</div>',
                            unsafe_allow_html=True)
                ratio_pivot_metric = st.radio("Метрика (% в ТО)", ["Fact", "Plan", "Average", "Delta"],
                                               horizontal=True, key="ratio_pivot_metric")
                rows_ratio_pivot = []
                for article in articles_to_show:
                    rdf = build_ratio_monthly(
                        df_filtered, col_tt, col_article, col_month, col_ratio, col_plf,
                        article, tt_val, df_all=df, group_factors=group_factors
                    )
                    row  = {"Стаття": article}
                    vals = [rdf.loc[m, ratio_pivot_metric] for m in range(1, 13)]
                    for m in range(1, 13):
                        row[MONTH_LABELS[m]] = rdf.loc[m, ratio_pivot_metric]
                    nz = [v for v in vals if v != 0]
                    row["Серед."] = np.mean(nz) if nz else 0.0
                    rows_ratio_pivot.append(row)

                ratio_pivot_df = pd.DataFrame(rows_ratio_pivot).set_index("Стаття")
                st.dataframe(
                    ratio_pivot_df.style
                        .background_gradient(cmap="RdYlGn_r", axis=None)
                        .apply(_style_white_na, axis=None)
                        .format(lambda v: f"{v:.2f}%" if pd.notna(v) else "-", na_rep="-"),
                    use_container_width=True,
                )

            # ── TT Pivot ──────────────────────────────────────────────────────────────
            st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
            st.subheader("📋 Зведена таблиця в розрізі Підрозділ")
            tt_pivot_metric = st.radio("Метрика (Підрозділ)", ["Fact", "Plan", "Delta (Fact-Plan)"],
                                        horizontal=True, key="tt_pivot_metric")
            tt_metric_col   = col_map_d[tt_pivot_metric]
            show_pct        = st.checkbox("Показати % відхилення (Fact vs Plan)", value=True, key="show_pct")
            show_months     = st.checkbox("Розгорнути по місяцях", value=False, key="tt_show_months")

            df_tt_agg = build_tt_pivot(
                df_filtered, col_tt, col_article, col_month, col_main_value, col_plf, articles_to_show
            )

            if df_tt_agg.empty:
                st.info("Немає даних для побудови таблиці по ТТ.")
            else:
                if show_months:
                    display_cols = ["ТТ"]
                    col_labels   = {"ТТ": "Підрозділ"}
                    for m in range(1, 13):
                        ml = MONTH_LABELS[m]
                        if tt_metric_col in ("Fact", "Delta"):
                            display_cols.append(f"fact_{ml}")
                            col_labels[f"fact_{ml}"] = f"{ml} Fact"
                        if tt_metric_col == "Plan":
                            display_cols.append(f"plan_{ml}")
                            col_labels[f"plan_{ml}"] = f"{ml} Plan"
                        if show_pct and tt_metric_col != "Plan":
                            display_cols.append(f"pct_{ml}")
                            col_labels[f"pct_{ml}"] = f"{ml} %"
                else:
                    display_cols = ["ТТ"]
                    col_labels   = {"ТТ": "Підрозділ"}

                if tt_metric_col == "Fact":
                    display_cols += ["Fact_РАЗОМ"]
                    col_labels["Fact_РАЗОМ"] = "Fact РАЗОМ"
                elif tt_metric_col == "Plan":
                    display_cols += ["Plan_РАЗОМ"]
                    col_labels["Plan_РАЗОМ"] = "Plan РАЗОМ"
                else:
                    display_cols += ["Fact_РАЗОМ", "Plan_РАЗОМ", "Delta_РАЗОМ"]
                    col_labels.update({"Fact_РАЗОМ": "Fact РАЗОМ",
                                       "Plan_РАЗОМ": "Plan РАЗОМ", "Delta_РАЗОМ": "Δ РАЗОМ"})

                if show_pct:
                    display_cols.append("Pct_РАЗОМ")
                    col_labels["Pct_РАЗОМ"] = "% відхил."

                df_display = df_tt_agg[display_cols].rename(columns=col_labels).set_index("Підрозділ")

                tt_display_map = _build_tt_display_map(df_filtered, col_tt, col_division)
                if tt_display_map:
                    df_display.index = [_tt_display_label(v, tt_display_map) for v in df_display.index]
                    df_display.index.name = "Підрозділ"

                sort_col_label = ("% відхил." if show_pct else
                                  "Δ РАЗОМ"    if tt_metric_col == "Delta (Fact-Plan)" else
                                  "Fact РАЗОМ" if tt_metric_col == "Fact" else "Plan РАЗОМ")
                if sort_col_label in df_display.columns:
                    df_display = df_display.sort_values(sort_col_label, ascending=True)

                total_row = df_display.sum(numeric_only=True)
                if "% відхил." in df_display.columns:
                    plan_sum = df_tt_agg["Plan_РАЗОМ"].sum()
                    fact_sum = df_tt_agg["Fact_РАЗОМ"].sum()
                    total_row["% відхил."] = (fact_sum / plan_sum - 1) * 100 if plan_sum != 0 else None
                total_row.name = "🟰 РАЗОМ"
                df_display     = pd.concat([df_display, total_row.to_frame().T])

                pct_cols = [c for c in df_display.columns if "%" in c]
                num_cols = [c for c in df_display.columns if "%" not in c]
                fmt_dict = {c: (lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-") for c in num_cols}
                fmt_dict.update({c: (lambda v: f"{'+' if v > 0 else ''}{v:.1f}%" if pd.notna(v) else "-")
                                 for c in pct_cols})

                styled = df_display.style.apply(_style_white_na, axis=None).format(fmt_dict, na_rep="")
                if pct_cols:
                    styled = styled.background_gradient(cmap="RdYlGn_r",
                                subset=pd.IndexSlice[df_display.index[:-1], pct_cols], axis=None)
                delta_cols = [c for c in num_cols if "Δ" in c]
                other_cols = [c for c in num_cols if "Δ" not in c]
                if delta_cols:
                    styled = styled.background_gradient(cmap="RdYlGn_r",
                                subset=pd.IndexSlice[df_display.index[:-1], delta_cols], axis=None)
                if other_cols:
                    styled = styled.background_gradient(cmap="Blues",
                                subset=pd.IndexSlice[df_display.index[:-1], other_cols], axis=None)
                styled = styled.apply(
                    lambda row: ["font-weight:bold;border-top:2px solid #5b2d8e;" for _ in row]
                    if row.name == "🟰 РАЗОМ" else ["" for _ in row], axis=1
                )
                st.dataframe(styled, use_container_width=True, height=500)
                st.download_button(
                    "⬇️ Завантажити CSV (ТТ-зведена)",
                    data=df_display.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                    file_name="tt_pivot.csv", mime="text/csv", key="tt_pivot_csv",
                )

    if active_tab == "🌡️ Heatmap / TOP":
            # ── Heatmap ───────────────────────────────────────────────────────────────
            st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
            st.subheader("🌡️ Карта аномалій по магазинах")
            heatmap_export_blocks = []

            if group_factors:
                heat, tt_table, val_col = build_heat_data(
                    df, df_filtered, col_tt, col_article, col_month, col_main_value,
                    col_plf, group_factors, articles_to_show, mode
                )
                heat_display = _replace_tt_index_with_division(heat, df_filtered, col_tt, col_division)
                heatmap_title = _build_article_period_title(
                    articles=articles_to_show, df_context=df_filtered, col_article=col_article,
                    col_month=col_month, col_year=col_year, suffix=f"Heatmap — {mode}"
                )

                st.dataframe(
                    heat_display.style
                        .background_gradient(cmap="RdYlGn_r", axis=None)
                        .apply(_style_white_na, axis=None)
                        .format(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "", na_rep=""),
                    use_container_width=True,
                )

                st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
                st.subheader("🏆 TOP / ANTITOP Heatmap")

                rating_options = _get_heatmap_rating_factor_options(
                    df_filtered,
                    col_tt=col_tt,
                    col_division=col_division,
                    group_factors=group_factors,
                )
                if not rating_options:
                    rating_options = [col_tt]

                rc1, rc2 = st.columns([2, 1])
                with rc1:
                    heatmap_rating_col = st.selectbox(
                        "Рейтинг дивимось по:",
                        options=rating_options,
                        index=0,
                        key=f"heatmap_rating_factor_{mode}",
                    )
                with rc2:
                    n_tt = st.slider(
                        "Кількість позицій",
                        1, 100, 10,
                        key=f"heatmap_rating_topn_{mode}",
                    )

                sum_val = _prepare_heatmap_rating_table(
                    tt_table, df_filtered, col_tt, val_col, heatmap_rating_col, aggfunc="sum"
                )
                top = sum_val.sort_values(val_col, ascending=True).head(n_tt)
                antitop = sum_val.sort_values(val_col, ascending=False).head(n_tt)
                top_display = top.copy()
                antitop_display = antitop.copy()
                fmt_abs = lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-"
                heatmap_top_title = _build_article_period_title(
                    articles=articles_to_show, df_context=df_filtered, col_article=col_article,
                    col_month=col_month, col_year=col_year,
                    suffix=f"TOP / ANTITOP Heatmap — {heatmap_rating_col}"
                )
                ca, cb = st.columns(2)
                with ca:
                    st.write(f"✅ Top (економія) по: {heatmap_rating_col}")
                    st.dataframe(
                        top_display[["Підрозділ", val_col]].set_index("Підрозділ")
                            .style.background_gradient(cmap="RdYlGn", subset=[val_col])
                            .apply(_style_white_na, axis=None)
                            .format({val_col: fmt_abs}),
                        use_container_width=True
                    )
                with cb:
                    st.write(f"❌ Antitop (переліміт) по: {heatmap_rating_col}")
                    st.dataframe(
                        antitop_display[["Підрозділ", val_col]].set_index("Підрозділ")
                            .style.background_gradient(cmap="RdYlGn_r", subset=[val_col])
                            .apply(_style_white_na, axis=None)
                            .format({val_col: fmt_abs}),
                        use_container_width=True
                    )

                heatmap_export_blocks = [{
                    "title": heatmap_title,
                    "heat_df": heat_display,
                    "top_df": top_display,
                    "antitop_df": antitop_display,
                    "val_col": val_col,
                    "header_color": PURPLE,
                    "percent": (mode == "Delta %"),
                    "sheet_prefix": "Heatmap",
                }]
            else:
                st.info("Оберіть фактори групування в боковому меню для побудови Heatmap.")

            # ── % в ТО Heatmap ────────────────────────────────────────────────────────
            if col_ratio and show_ratio_heatmap:
                st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
                ratio_export_block = render_ratio_heatmap_section(
                    df, df_filtered, col_tt, col_article, col_month,
                    col_ratio, col_plf, articles_to_show, ratio_mode,
                    group_factors=group_factors,
                    col_division=col_division,
                    col_year=col_year,
                    show_download_button=False,
                )
                if ratio_export_block:
                    heatmap_export_blocks.append(ratio_export_block)


            if heatmap_export_blocks:
                st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇️ Скачати всі Heatmap-блоки в одному Excel",
                    data=_heatmap_blocks_to_excel_bytes(
                        heatmap_export_blocks,
                        workbook_title="Всі Heatmap-блоки"
                    ),
                    file_name=_safe_xlsx_filename("all_heatmap_blocks"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_all_heatmap_blocks_{mode}_{ratio_mode}",
                    use_container_width=True,
                )
    if active_tab == "📊 Статистика факторів":
            # ── Statistical factor analysis ──────────────────────────────────────────
            render_statistical_analysis_tab(
                df=df,
                col_article=col_article,
                col_value=col_main_value,
                col_plf=col_plf,
                articles_to_show=articles_to_show,
                group_factors=group_factors,
            )

    if active_tab == "📥 Експорт":
            # ── Export ────────────────────────────────────────────────────────────────
            st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
            st.subheader("📥 Експорт в Excel")

            # Дані для експорту формуємо тільки при відкритті вкладки експорту.
            metric_col = "Fact"
            rows_pivot = []
            for article in articles_to_show:
                tdf = build_article_monthly(
                    df, df_filtered, col_tt, col_article, col_month, col_main_value,
                    col_plf, article, tt_val, group_factors
                )
                row = {"Стаття": article}
                for m in range(1, 13):
                    row[MONTH_LABELS[m]] = tdf.loc[m, metric_col]
                row["РАЗОМ"] = sum(tdf.loc[m, metric_col] for m in range(1, 13))
                rows_pivot.append(row)
            pivot_df = pd.DataFrame(rows_pivot).set_index("Стаття") if rows_pivot else pd.DataFrame()
            df_tt_agg = build_tt_pivot(
                df_filtered, col_tt, col_article, col_month, col_main_value, col_plf, articles_to_show
            )

            export_sections = [
                "✅ Зведена таблиця (статті)",
                "✅ Основні таблиці всіх блоків",
                "✅ Листи по кожній статті (з графіком)",
            ]
            if col_ratio:
                export_sections += [
                    "✅ Основні блоки % в ТО",
                    "✅ % в ТО — зведена таблиця",
                ]
            if df_tt_agg is not None and not df_tt_agg.empty:
                export_sections.append("✅ Підрозділ-Зведена таблиця")
            if group_factors:
                export_sections += [
                    "✅ Heatmap аномалій",
                    "✅ TOP / ANTITOP",
                    "✅ Аналіз комбінації факторів",
                    "✅ Статистичний аналіз факторів",
                ]
            if group_factors and col_ratio:
                export_sections += [
                    "✅ Heatmap % в ТО",
                    "✅ TOP / ANTITOP % в ТО",
                    "✅ Аналіз комбінації факторів % в ТО",
                ]
            export_sections.append("✅ Дані після фільтрів")

            st.markdown("**Файл міститиме аркуші:**")
            for s in export_sections:
                st.markdown(f"- {s}")

            # ── Експорт Excel: кнопка показується одразу, файл формується тільки по кліку ──
            export_signature = repr({
                "rows": int(len(df_filtered)),
                "cols": tuple(map(str, df_filtered.columns)),
                "articles": tuple(map(str, articles_to_show or [])),
                "tt": tuple(map(str, tt_val or [])),
                "group_factors": tuple(map(str, group_factors or [])),
                "metric_col": str(metric_col),
                "mode": str(mode),
                "ratio_mode": str(ratio_mode),
                "has_ratio": bool(col_ratio),
                "has_tt_agg": bool(df_tt_agg is not None and not df_tt_agg.empty),
            })

            if st.session_state.get("simi_export_signature") != export_signature:
                st.session_state["simi_export_signature"] = export_signature
                st.session_state["simi_export_excel_bytes"] = None

            export_col1, export_col2 = st.columns([1, 1])

            with export_col1:
                prepare_export = st.button(
                    "📦 Підготувати Excel-файл",
                    key="prepare_simi_dashboard_excel",
                    use_container_width=True,
                )

            if prepare_export:
                with st.spinner("⏳ Формую Excel-файл..."):
                    st.session_state["simi_export_excel_bytes"] = export_excel(
                        df, df_filtered, col_tt, col_article, col_month, col_main_value,
                        col_plf, articles_to_show, tt_val, group_factors, metric_col,
                        mode, pivot_df, df_tt_agg=df_tt_agg, col_ratio=col_ratio,
                        col_division=col_division, ratio_mode=ratio_mode,
                    )
                st.success("✅ Excel-файл готовий. Натисни кнопку завантаження.")

            with export_col2:
                if st.session_state.get("simi_export_excel_bytes") is not None:
                    st.download_button(
                        label="⬇️ Скачати дашборд як Excel",
                        data=st.session_state["simi_export_excel_bytes"],
                        file_name="simi_dashboard.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_simi_dashboard_excel",
                        use_container_width=True,
                    )
                else:
                    st.button(
                        "⬇️ Скачати дашборд як Excel",
                        key="download_simi_dashboard_excel_disabled",
                        disabled=True,
                        use_container_width=True,
                        help="Спочатку натисни «Підготувати Excel-файл»."
                    )


if __name__ == "__main__":
    main()
