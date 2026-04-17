import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import io

pd.set_option("styler.render.max_elements", 2**31 - 1)

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
_MONTHS = list(range(1, 13))
_MONTH_LABEL_LIST = [MONTH_LABELS[m] for m in _MONTHS]

PURPLE    = "#5b2d8e"
GREY      = "#c0c0c0"
RED_LINE  = "#c0392b"
YELLOW    = "#f0c000"
GREEN_HDR = "#2e7d32"
TEAL      = "#0d7377"
TEAL_HDR  = "#085f63"
ORANGE    = "#e67e22"

_EMPTY_12 = pd.DataFrame(
    0.0, index=_MONTHS,
    columns=["Plan", "Fact", "Average", "Delta"]
)
_EMPTY_12.index.name = "month"


# ═══════════════════════════════════════════════════════════════
# DATA LOADING & HELPERS
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="⏳ Завантаження файлу…")
def load_excel(file_bytes: bytes, file_name: str, sheet_name: str) -> pd.DataFrame:
    import os
    buf = io.BytesIO(file_bytes)
    ext = os.path.splitext(file_name)[1].lower()
    engine = "pyxlsb" if ext == ".xlsb" else None
    df = pd.read_excel(buf, sheet_name=sheet_name, engine=engine)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def preprocess_df(df: pd.DataFrame, col_value: str, col_month: str,
                  col_ratio: str | None) -> pd.DataFrame:
    """Одноразово додає _m, числові колонки — кешується."""
    df = df.copy()
    df[col_value] = pd.to_numeric(df[col_value], errors="coerce")
    if col_ratio:
        df[col_ratio] = pd.to_numeric(df[col_ratio], errors="coerce")
    # _m — числовий місяць
    s = df[col_month]
    if pd.api.types.is_numeric_dtype(s):
        df["_m"] = s.astype(int)
    else:
        df["_m"] = s.astype(str).str.strip().map(MONTH_MAP).fillna(0).astype(int)
    return df


def get_month_num(series: pd.Series) -> pd.Series:
    """Повертає числовий місяць. Якщо вже int — просто кастуємо."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.astype(str).str.strip().map(MONTH_MAP).fillna(0).astype(int)


# ═══════════════════════════════════════════════════════════════
# CORE CALCULATION — основний показник
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_article_monthly(df: pd.DataFrame, df_filtered: pd.DataFrame,
                          col_tt: str, col_article: str, col_month: str,
                          col_value: str, col_plf: str,
                          selected_art: str,
                          selected_tts: tuple,          # tuple для hashability
                          group_factors: tuple) -> pd.DataFrame:
    group_factors = list(group_factors)

    # Фільтр без .copy() — нам не треба мутувати
    art_all  = df[df[col_article] == selected_art]
    art_filt = df_filtered[df_filtered[col_article] == selected_art]

    # _m вже є після preprocess_df; якщо ні — рахуємо
    if "_m" not in art_all.columns:
        art_all  = art_all.assign(_m=get_month_num(art_all[col_month]))
        art_filt = art_filt.assign(_m=get_month_num(art_filt[col_month]))

    if art_filt.empty:
        return _EMPTY_12.copy()

    sel_tts = list(selected_tts) if selected_tts else art_filt[col_tt].dropna().unique().tolist()

    fact_rows = art_filt[art_filt[col_plf] == "F"]
    plan_rows = art_filt[art_filt[col_plf] == "PL"]

    plan = (plan_rows.groupby("_m")[col_value].sum()
            .reindex(_MONTHS, fill_value=0).rename("Plan"))
    fact = (fact_rows.groupby("_m")[col_value].sum()
            .reindex(_MONTHS, fill_value=0).rename("Fact"))

    all_fact = art_all[art_all[col_plf] == "F"]
    if group_factors:
        global_avg_std = (all_fact
                          .groupby(group_factors + [col_article], as_index=False)[col_value]
                          .agg(Average_Calc="mean", Std="std"))
    else:
        global_avg_std = (all_fact
                          .groupby([col_article, "_m"], as_index=False)[col_value]
                          .agg(Average_Calc="mean"))
        global_avg_std["Std"] = np.nan

    tt_grp = list(dict.fromkeys([col_tt] + group_factors + ["_m", col_article]))
    tt_table = (fact_rows.groupby(tt_grp, as_index=False)[col_value]
                .sum().rename(columns={col_value: "Fact"}))

    merge_cols = list(dict.fromkeys(group_factors + [col_article]))
    if merge_cols:
        tt_table = tt_table.merge(global_avg_std, on=merge_cols, how="left")
    elif "_m" in global_avg_std.columns:
        tt_table = tt_table.merge(global_avg_std, on=[col_article, "_m"], how="left")
    else:
        tt_table["Average_Calc"] = np.nan

    tt_table["Fact"]         = tt_table["Fact"].fillna(0)
    tt_table["Average_Calc"] = tt_table["Average_Calc"].fillna(0)
    tt_table.loc[tt_table["Fact"] == 0, "Average_Calc"] = 0

    dynamic_average = (tt_table[tt_table[col_tt].isin(sel_tts)]
                       .groupby("_m")["Average_Calc"].sum()
                       .reindex(_MONTHS, fill_value=0).rename("Average"))

    merged = (pd.DataFrame(index=_MONTHS)
              .join(plan).join(fact).join(dynamic_average)
              .fillna(0))
    merged.index.name = "month"
    merged.loc[merged["Fact"] == 0, "Average"] = 0
    merged["Delta"] = merged["Fact"] - merged["Average"]
    return merged


# ═══════════════════════════════════════════════════════════════
# CORE CALCULATION — % в ТО
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_ratio_monthly(df_filtered: pd.DataFrame, col_tt: str, col_article: str,
                        col_month: str, col_ratio: str, col_plf: str,
                        selected_art: str, selected_tts: tuple,
                        df_all: pd.DataFrame | None = None,
                        group_factors: tuple = ()) -> pd.DataFrame:
    group_factors = list(group_factors)

    src_all  = df_all if df_all is not None else df_filtered
    art_all  = src_all[src_all[col_article] == selected_art]
    art_filt = df_filtered[df_filtered[col_article] == selected_art]

    if "_m" not in art_all.columns:
        art_all  = art_all.assign(_m=get_month_num(art_all[col_month]))
        art_filt = art_filt.assign(_m=get_month_num(art_filt[col_month]))

    if art_filt.empty:
        return _EMPTY_12.copy()

    sel_tts  = list(selected_tts) if selected_tts else art_filt[col_tt].dropna().unique().tolist()
    has_plf  = bool(col_plf and col_plf in art_filt.columns)

    fact_rows = art_filt[art_filt[col_plf] == "F"] if has_plf else art_filt
    plan_rows = art_filt[art_filt[col_plf] == "PL"] if has_plf else pd.DataFrame(columns=art_filt.columns)

    fact = (fact_rows.groupby("_m")[col_ratio].mean()
            .reindex(_MONTHS, fill_value=np.nan).rename("Fact"))
    plan = (plan_rows.groupby("_m")[col_ratio].mean()
            .reindex(_MONTHS, fill_value=np.nan).rename("Plan"))

    f_src = art_all[art_all[col_plf] == "F"] if has_plf else art_all

    if group_factors:
        global_avg = (f_src.groupby(group_factors + [col_article], as_index=False)[col_ratio]
                      .mean().rename(columns={col_ratio: "Average_Calc"}))
    else:
        global_avg = (f_src.groupby([col_tt, "_m"], as_index=False)[col_ratio]
                      .mean().rename(columns={col_ratio: "Average_Calc"}))

    tt_grp = list(dict.fromkeys([col_tt] + group_factors + ["_m", col_article]))
    tt_table = (fact_rows.groupby(tt_grp, as_index=False)[col_ratio]
                .mean().rename(columns={col_ratio: "Fact_tt"}))

    if group_factors:
        merge_cols = list(dict.fromkeys(group_factors + [col_article]))
        tt_table = tt_table.merge(global_avg, on=merge_cols, how="left")
    else:
        tt_table = tt_table.merge(global_avg, on=[col_tt, "_m"], how="left")

    tt_table["Fact_tt"]      = tt_table["Fact_tt"].fillna(0)
    tt_table["Average_Calc"] = tt_table["Average_Calc"].fillna(0)
    tt_table.loc[tt_table["Fact_tt"] == 0, "Average_Calc"] = 0

    dynamic_average = (tt_table[tt_table[col_tt].isin(sel_tts)]
                       .groupby("_m")["Average_Calc"].mean()
                       .reindex(_MONTHS, fill_value=0).rename("Average"))

    merged = (pd.DataFrame(index=_MONTHS)
              .join(plan).join(fact).join(dynamic_average)
              .fillna(0.0))
    merged.index.name = "month"
    merged.loc[merged["Fact"] == 0, "Average"] = 0
    merged["Delta"] = merged["Fact"] - merged["Average"]
    return merged


@st.cache_data(show_spinner=False)
def build_ratio_heat_data(df: pd.DataFrame, df_filtered: pd.DataFrame,
                           col_tt: str, col_article: str, col_month: str,
                           col_ratio: str, col_plf: str,
                           articles_to_show: tuple, mode: str,
                           group_factors: tuple = ()):
    group_factors = list(group_factors)
    arts = list(articles_to_show)

    # df вже preprocessed (_m і числові є)
    has_plf = bool(col_plf and col_plf in df_filtered.columns)

    filt_fact = df_filtered[df_filtered[col_article].isin(arts)]
    if has_plf:
        filt_fact = filt_fact[filt_fact[col_plf] == "F"]

    all_fact = df[df[col_article].isin(arts)]
    if has_plf:
        all_fact = all_fact[all_fact[col_plf] == "F"]

    if group_factors:
        grp_cols  = list(dict.fromkeys(group_factors + [col_article, "_m"]))
        global_avg = (all_fact.groupby(grp_cols)[col_ratio].mean()
                      .reset_index().rename(columns={col_ratio: "Average_Calc"}))
        merge_on = grp_cols
    else:
        global_avg = (all_fact.groupby([col_article, "_m"])[col_ratio].mean()
                      .reset_index().rename(columns={col_ratio: "Average_Calc"}))
        merge_on = [col_article, "_m"]

    tt_grp   = list(dict.fromkeys([col_tt] + group_factors + [col_article, "_m"]))
    tt_table = (filt_fact.groupby(tt_grp, as_index=False)[col_ratio]
                .mean().rename(columns={col_ratio: "Fact"}))
    tt_table = tt_table.merge(global_avg, on=merge_on, how="left")
    tt_table["Delta"]   = tt_table["Fact"] - tt_table["Average_Calc"]
    tt_table["Delta_%"] = tt_table["Delta"] / tt_table["Average_Calc"].replace(0, np.nan)
    tt_table["Std"]     = np.nan
    tt_table["Z"]       = np.nan

    val_col = {"Delta": "Delta", "Delta %": "Delta_%",
               "Fact": "Fact", "Average": "Average_Calc"}.get(mode, "Delta")

    heat = tt_table.pivot_table(index=col_tt, columns="_m", values=val_col, aggfunc="mean")
    for m in _MONTHS:
        if m not in heat.columns:
            heat[m] = None
    heat = heat[sorted(heat.columns)]
    heat.columns = [MONTH_LABELS.get(int(c), str(c)) for c in heat.columns]
    heat["РАЗОМ"] = heat.mean(axis=1, numeric_only=True)
    return heat, tt_table, val_col


# ═══════════════════════════════════════════════════════════════
# TT PIVOT — векторизована версія (без подвійного Python-циклу)
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def build_tt_pivot(df_filtered: pd.DataFrame, col_tt: str, col_article: str,
                   col_month: str, col_value: str, col_plf: str,
                   articles_to_show: tuple) -> pd.DataFrame:
    arts = list(articles_to_show)
    sub  = df_filtered[df_filtered[col_article].isin(arts)].copy()
    if sub.empty:
        return pd.DataFrame()

    if "_m" not in sub.columns:
        sub["_m"] = get_month_num(sub[col_month])

    # Vectorized pivot: (TT, _m) → plan / fact
    plan_sub = sub[sub[col_plf] == "PL"]
    fact_sub = sub[sub[col_plf] == "F"]

    def _monthly_pivot(df_sub):
        if df_sub.empty:
            return pd.DataFrame(columns=[col_tt] + [f"m{m}" for m in _MONTHS])
        piv = (df_sub.groupby([col_tt, "_m"])[col_value].sum()
               .unstack(fill_value=0)
               .reindex(columns=_MONTHS, fill_value=0))
        piv.columns = [f"m{m}" for m in _MONTHS]
        return piv.reset_index()

    plan_piv = _monthly_pivot(plan_sub).rename(columns={f"m{m}": f"plan_{MONTH_LABELS[m]}" for m in _MONTHS})
    fact_piv = _monthly_pivot(fact_sub).rename(columns={f"m{m}": f"fact_{MONTH_LABELS[m]}" for m in _MONTHS})

    all_tts  = pd.DataFrame({col_tt: sorted(sub[col_tt].dropna().unique(), key=str)})
    df_agg   = all_tts.merge(plan_piv, on=col_tt, how="left").merge(fact_piv, on=col_tt, how="left")

    plan_cols  = [f"plan_{MONTH_LABELS[m]}" for m in _MONTHS]
    fact_cols  = [f"fact_{MONTH_LABELS[m]}" for m in _MONTHS]
    for col in plan_cols + fact_cols:
        if col not in df_agg.columns:
            df_agg[col] = 0.0
    df_agg[plan_cols + fact_cols] = df_agg[plan_cols + fact_cols].fillna(0)
    df_agg.rename(columns={col_tt: "ТТ"}, inplace=True)

    df_agg["Plan_РАЗОМ"]  = df_agg[plan_cols].sum(axis=1)
    df_agg["Fact_РАЗОМ"]  = df_agg[fact_cols].sum(axis=1)
    df_agg["Delta_РАЗОМ"] = df_agg["Fact_РАЗОМ"] - df_agg["Plan_РАЗОМ"]

    # Векторизований % — np.where замість apply
    plan_s = df_agg["Plan_РАЗОМ"].to_numpy()
    fact_s = df_agg["Fact_РАЗОМ"].to_numpy()
    df_agg["Pct_РАЗОМ"] = np.where(
        plan_s != 0, (fact_s / plan_s - 1) * 100, np.nan
    )

    for m in _MONTHS:
        ml = MONTH_LABELS[m]
        df_agg[f"delta_{ml}"] = df_agg[f"fact_{ml}"] - df_agg[f"plan_{ml}"]
        p = df_agg[f"plan_{ml}"].to_numpy()
        f = df_agg[f"fact_{ml}"].to_numpy()
        df_agg[f"pct_{ml}"] = np.where(p != 0, (f / p - 1) * 100, np.nan)

    return df_agg


@st.cache_data(show_spinner=False)
def build_heat_data(df: pd.DataFrame, df_filtered: pd.DataFrame,
                    col_tt: str, col_article: str, col_month: str,
                    col_value: str, col_plf: str, group_factors: tuple,
                    articles_to_show: tuple, mode: str):
    gf   = list(group_factors)
    arts = list(articles_to_show)

    all_fact = df[df[col_plf] == "F"]
    global_avg_std = (all_fact
                      .groupby(gf + [col_article], as_index=False)[col_value]
                      .agg(Average_Calc="mean", Std="std"))

    data_heat = df_filtered[(df_filtered[col_plf] == "F") &
                             (df_filtered[col_article].isin(arts))]
    if "_m" not in data_heat.columns:
        data_heat = data_heat.assign(_m=get_month_num(data_heat[col_month]))

    tt_grp   = list(dict.fromkeys([col_tt] + gf + ["_m", col_article]))
    tt_table = (data_heat.groupby(tt_grp, as_index=False)[col_value]
                .sum().rename(columns={col_value: "Fact"}))
    merge_cols = list(dict.fromkeys(gf + [col_article]))
    tt_table   = tt_table.merge(global_avg_std, on=merge_cols, how="left")
    tt_table["Delta"]   = tt_table["Fact"] - tt_table["Average_Calc"]
    tt_table["Delta_%"] = tt_table["Delta"] / tt_table["Average_Calc"].replace(0, np.nan)
    tt_table["Z"]       = tt_table["Delta"] / tt_table["Std"].replace(0, np.nan)

    val_col = {"Delta": "Delta", "Delta %": "Delta_%",
               "Z-score": "Z", "Fact": "Fact", "Average": "Average_Calc"}[mode]

    heat = tt_table.pivot_table(index=col_tt, columns="_m", values=val_col, aggfunc="sum")
    for m in _MONTHS:
        if m not in heat.columns:
            heat[m] = None
    heat = heat[sorted(heat.columns)]
    heat.columns = [MONTH_LABELS.get(int(c), str(c)) for c in heat.columns]
    heat["РАЗОМ"] = heat.sum(axis=1, numeric_only=True)
    return heat, tt_table, val_col


# ═══════════════════════════════════════════════════════════════
# HELPERS — перетворення аргументів для кешованих функцій
# ═══════════════════════════════════════════════════════════════

def _t(lst) -> tuple:
    """list → tuple для @st.cache_data hashability."""
    return tuple(lst) if lst else ()


# ═══════════════════════════════════════════════════════════════
# PILL HELPERS
# ═══════════════════════════════════════════════════════════════

def _make_pills(series, color, bg, fmt_fn) -> str:
    parts = []
    for tt, val in series.items():
        sign = "+" if val > 0 else ""
        parts.append(
            f'<span style="display:inline-block;background:{bg};color:{color};'
            f'border-radius:4px;padding:2px 9px;margin:2px 3px;font-size:0.75rem;'
            f'font-weight:600;white-space:nowrap;">'
            f'{tt}\u00a0<span style="opacity:.7;font-weight:400;">'
            f'({sign}{fmt_fn(val)})</span></span>'
        )
    return "".join(parts)


# ═══════════════════════════════════════════════════════════════
# SLICER WIDGET (спільний для обох блоків)
# ═══════════════════════════════════════════════════════════════

def _render_tt_slicer(article_title, df_filtered, col_tt, col_article,
                      skey, active_tt, prefix):
    available_tts = sorted(
        df_filtered[df_filtered[col_article] == article_title][col_tt].dropna().unique(),
        key=str,
    )
    if not available_tts:
        return

    VISIBLE_ITEMS = 24  # 4 rows × 6 cols
    COLS_PER_ROW  = 6

    with st.expander(f"🏪 Слайсер по ТТ{' (% в ТО)' if prefix == 'ratio' else ''} — клікни для деталізації",
                     expanded=False):
        search_key = f"{prefix}_slicer_search_{skey}"
        search_val = st.text_input("🔎 Пошук магазину", value="",
                                   placeholder="Введіть назву…", key=search_key)
        filtered_tts = [t for t in available_tts
                        if search_val.lower() in str(t).lower()] if search_val else available_tts

        all_options  = ["__ALL__"] + list(filtered_tts)
        show_all_key = f"{prefix}_slicer_showall_{skey}"
        if show_all_key not in st.session_state:
            st.session_state[show_all_key] = False
        show_all_tts  = st.session_state[show_all_key]
        items_to_show = all_options if show_all_tts else all_options[:VISIBLE_ITEMS]

        for row_start in range(0, len(items_to_show), COLS_PER_ROW):
            chunk = items_to_show[row_start:row_start + COLS_PER_ROW]
            for ci, tt_opt in enumerate(st.columns(len(chunk))):
                opt   = chunk[ci]
                label = "🔁 Всі" if opt == "__ALL__" else str(opt)
                with tt_opt:
                    if st.button(label,
                                 key=f"{prefix}_slicer_{skey}_{row_start}_{ci}_{hash(str(opt))}",
                                 type="primary" if active_tt == opt else "secondary",
                                 use_container_width=True):
                        st.session_state[skey] = opt
                        st.rerun()

        total = len(all_options)
        if total > VISIBLE_ITEMS:
            if st.button("▲ Згорнути" if show_all_tts else f"▼ Показати ще {total - VISIBLE_ITEMS} магазинів",
                         key=f"{prefix}_slicer_toggle_{skey}"):
                st.session_state[show_all_key] = not show_all_tts
                st.rerun()

        st.caption(f"📍 Показано тільки: **{active_tt}**"
                   if active_tt != "__ALL__"
                   else f"Показано всі ТТ · знайдено: {len(filtered_tts)}")


# ═══════════════════════════════════════════════════════════════
# ARTICLE BLOCK RENDERER — основний
# ═══════════════════════════════════════════════════════════════

def render_article_block(title, table_df, chart_title,
                         df=None, df_filtered=None,
                         col_tt=None, col_article=None,
                         col_month=None, col_value=None, col_plf=None,
                         group_factors=None, tt_val=None,
                         article_idx=0):
    skey = f"slicer_tt_{article_idx}"
    if skey not in st.session_state:
        st.session_state[skey] = "__ALL__"
    active_tt = st.session_state[skey]

    if active_tt != "__ALL__" and df is not None and df_filtered is not None:
        df_filt_tt = df_filtered[df_filtered[col_tt] == active_tt]
        display_df = build_article_monthly(
            df, df_filt_tt, col_tt, col_article, col_month, col_value,
            col_plf, title, (active_tt,), _t(group_factors or [])
        )
    else:
        display_df = table_df

    th = ("background:#2e7d32;color:white;font-weight:bold;border:1px solid #aaa;"
          "padding:4px 8px;text-align:center;font-size:0.78rem;")
    td = "border:1px solid #ccc;padding:3px 7px;text-align:right;font-size:0.78rem;"
    tl = "border:1px solid #ccc;padding:3px 7px;font-size:0.78rem;font-weight:600;white-space:nowrap;"

    badge = (f'<span style="margin-left:10px;background:{PURPLE};color:white;font-size:0.78rem;'
             f'padding:2px 10px;border-radius:10px;">📍 {active_tt}</span>'
             if active_tt != "__ALL__" else "")
    st.markdown(
        f'<div style="margin-top:20px;margin-bottom:4px;">'
        f'<span style="background:{GREEN_HDR};color:white;font-weight:700;padding:4px 14px;'
        f'font-size:0.9rem;border-radius:2px;">{title}</span>{badge}</div>',
        unsafe_allow_html=True)

    rows_def = [
        ("План",    "Plan",    "#ffffff", "#333333"),
        ("Факт",    "Fact",    "#e8d5f5", PURPLE),
        ("Average", "Average", "#fde8e8", RED_LINE),
        ("Дельта",  "Delta",   "#fff9e0", "#b8860b"),
    ]
    month_cols = _MONTH_LABEL_LIST
    html_parts = [
        f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;margin-bottom:6px;">',
        f'<thead><tr><th style="{th}">Показник</th>',
        "".join(f'<th style="{th}">{m}</th>' for m in month_cols),
        f'<th style="{th}">Разом</th></tr></thead><tbody>',
    ]
    for label, col, bg, color in rows_def:
        vals  = display_df[col].values  # numpy array — швидше ніж .loc loop
        total = vals.sum()
        html_parts.append(f'<tr style="background:{bg};"><td style="{tl}color:{color};">{label}</td>')
        for v in vals:
            neg = "color:#c0392b;" if v < 0 else ""
            html_parts.append(f'<td style="{td}{neg}">{v:,.0f}</td>')
        html_parts.append(f'<td style="{td}font-weight:700;">{total:,.0f}</td></tr>')
    html_parts.append("</tbody></table></div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────
    facts = display_df["Fact"].values
    nz    = facts[facts != 0]
    avg_monthly = nz.mean() if len(nz) else 0.0
    total_fact  = facts.sum()
    total_plan  = display_df["Plan"].values.sum()
    total_delta = total_fact - total_plan
    pct_vs_plan = (total_fact / total_plan - 1) * 100 if total_plan != 0 else None
    pct_str     = (f"{'+' if pct_vs_plan >= 0 else ''}{pct_vs_plan:.1f}%"
                   if pct_vs_plan is not None else "—")
    pct_color   = RED_LINE if (pct_vs_plan or 0) > 0 else GREEN_HDR
    delta_color = RED_LINE if total_delta > 0 else GREEN_HDR

    best_pills = worst_pills = ""
    if active_tt == "__ALL__" and df_filtered is not None and col_tt:
        sub = df_filtered[(df_filtered[col_article] == title) & (df_filtered[col_plf] == "F")]
        if not sub.empty:
            tt_totals = sub.groupby(col_tt)[col_value].sum().dropna().sort_values()
            n = min(3, len(tt_totals))
            best_pills  = _make_pills(tt_totals.head(n),          "#1b5e20", "#e8f5e9",
                                      lambda v: f"{v:,.0f}".replace(",", " "))
            worst_pills = _make_pills(tt_totals.tail(n).iloc[::-1], "#7f0000", "#ffebee",
                                      lambda v: f"{v:,.0f}".replace(",", " "))

    best_block = (
        f'<div style="flex:1;min-width:220px;"><div style="color:#888;font-size:0.71rem;'
        f'margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;">✅ Кращі магазини (мін. Fact)</div>'
        f'<div>{best_pills or "<span style=\'color:#aaa;font-size:0.75rem;\'>немає даних</span>"}</div></div>'
        f'<div style="flex:1;min-width:220px;"><div style="color:#888;font-size:0.71rem;'
        f'margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;">❌ Гірші магазини (макс. Fact)</div>'
        f'<div>{worst_pills or "<span style=\'color:#aaa;font-size:0.75rem;\'>немає даних</span>"}</div></div>'
    ) if active_tt == "__ALL__" else ""

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;'
        f'background:#f9f6ff;border:1px solid #d0baf5;border-radius:6px;padding:10px 16px;margin:6px 0 10px 0;">'
        f'<div style="min-width:130px;"><div style="color:#888;font-size:0.71rem;margin-bottom:2px;'
        f'text-transform:uppercase;letter-spacing:.04em;">Серед. Fact / міс.</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{PURPLE};">{avg_monthly:,.0f}</div></div>'
        f'<div style="min-width:130px;"><div style="color:#888;font-size:0.71rem;margin-bottom:2px;'
        f'text-transform:uppercase;letter-spacing:.04em;">Δ Fact − Plan</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{delta_color};">'
        f'{"+" if total_delta > 0 else ""}{total_delta:,.0f}</div></div>'
        f'<div style="min-width:100px;"><div style="color:#888;font-size:0.71rem;margin-bottom:2px;'
        f'text-transform:uppercase;letter-spacing:.04em;">% до плану</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{pct_color};">{pct_str}</div></div>'
        f'{best_block}</div>',
        unsafe_allow_html=True)

    x_axis = _MONTH_LABEL_LIST
    fig = go.Figure([
        go.Bar(x=x_axis, y=display_df["Plan"].values,    name="План",    marker_color=GREY),
        go.Bar(x=x_axis, y=display_df["Fact"].values,    name="Факт",    marker_color=PURPLE),
        go.Scatter(x=x_axis, y=display_df["Average"].values, name="Average",
                   line=dict(color=RED_LINE, width=3)),
        go.Scatter(x=x_axis, y=display_df["Delta"].values,   name="Дельта",
                   line=dict(color=YELLOW, dash="dot")),
    ])
    fig.update_layout(height=350, margin=dict(t=30, b=20, l=10, r=10),
                      barmode="group", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{article_idx}_{active_tt}")

    _render_tt_slicer(title, df_filtered, col_tt, col_article, skey, active_tt, "main")


# ═══════════════════════════════════════════════════════════════
# RATIO ARTICLE BLOCK RENDERER — % в ТО
# ═══════════════════════════════════════════════════════════════

def render_ratio_article_block(title, table_df,
                                df=None, df_filtered=None,
                                col_tt=None, col_article=None,
                                col_month=None, col_ratio=None, col_plf=None,
                                tt_val=None, article_idx=0, group_factors=None):
    if group_factors is None:
        group_factors = []

    skey = f"ratio_slicer_tt_{article_idx}"
    if skey not in st.session_state:
        st.session_state[skey] = "__ALL__"
    active_tt = st.session_state[skey]

    if active_tt != "__ALL__" and df is not None and df_filtered is not None:
        df_filt_tt = df_filtered[df_filtered[col_tt] == active_tt]
        display_df = build_ratio_monthly(
            df_filt_tt, col_tt, col_article, col_month,
            col_ratio, col_plf, title, (active_tt,),
            df_all=df, group_factors=_t(group_factors),
        )
    else:
        display_df = table_df

    th = (f"background:{TEAL_HDR};color:white;font-weight:bold;border:1px solid #aaa;"
          f"padding:4px 8px;text-align:center;font-size:0.78rem;")
    td = "border:1px solid #ccc;padding:3px 7px;text-align:right;font-size:0.78rem;"
    tl = "border:1px solid #ccc;padding:3px 7px;font-size:0.78rem;font-weight:600;white-space:nowrap;"

    badge = (f'<span style="margin-left:10px;background:{TEAL};color:white;font-size:0.78rem;'
             f'padding:2px 10px;border-radius:10px;">📍 {active_tt}</span>'
             if active_tt != "__ALL__" else "")
    st.markdown(
        f'<div style="margin-top:12px;margin-bottom:4px;">'
        f'<span style="background:{TEAL_HDR};color:white;font-weight:700;padding:4px 14px;'
        f'font-size:0.85rem;border-radius:2px;">📊 % в ТО — {title}</span>{badge}</div>',
        unsafe_allow_html=True)

    rows_def = [
        ("План %",    "Plan",    "#e8f8f8", TEAL_HDR),
        ("Факт %",    "Fact",    "#d0f0f0", TEAL),
        ("Average %", "Average", "#fde8e8", RED_LINE),
        ("Дельта %",  "Delta",   "#fff9e0", ORANGE),
    ]
    html_parts = [
        f'<div style="overflow-x:auto;"><table style="border-collapse:collapse;width:100%;margin-bottom:6px;">',
        f'<thead><tr><th style="{th}">Показник</th>',
        "".join(f'<th style="{th}">{m}</th>' for m in _MONTH_LABEL_LIST),
        f'<th style="{th}">Серед.</th></tr></thead><tbody>',
    ]
    for label, col, bg, color in rows_def:
        vals    = display_df[col].values
        nz      = vals[vals != 0]
        summary = nz.mean() if len(nz) else 0.0
        html_parts.append(f'<tr style="background:{bg};"><td style="{tl}color:{color};">{label}</td>')
        for v in vals:
            neg = "color:#c0392b;" if v < 0 else ""
            html_parts.append(f'<td style="{td}{neg}">{v:.2f}%</td>')
        html_parts.append(f'<td style="{td}font-weight:700;">{summary:.2f}%</td></tr>')
    html_parts.append("</tbody></table></div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)

    # ── Metrics ────────────────────────────────────────────────
    facts      = display_df["Fact"].values
    nz_facts   = facts[facts != 0]
    avg_monthly = nz_facts.mean() if len(nz_facts) else 0.0
    avg_arr     = display_df["Average"].values
    nz_avg      = avg_arr[avg_arr != 0]
    avg_global  = nz_avg.mean() if len(nz_avg) else 0.0
    delta_arr   = display_df["Delta"].values
    nz_delta    = delta_arr[facts != 0]
    total_delta = nz_delta.mean() if len(nz_delta) else 0.0
    delta_color = RED_LINE if total_delta > 0 else GREEN_HDR

    best_pills = worst_pills = ""
    if active_tt == "__ALL__" and df_filtered is not None and col_tt:
        sub = df_filtered[df_filtered[col_article] == title]
        if col_plf and col_plf in sub.columns:
            sub = sub[sub[col_plf] == "F"]
        if not sub.empty:
            tt_avgs = sub.groupby(col_tt)[col_ratio].mean().dropna().sort_values()
            n = min(3, len(tt_avgs))
            best_pills  = _make_pills(tt_avgs.head(n),          "#1b5e20", "#e8f5e9",
                                      lambda v: f"{v:.2f}%")
            worst_pills = _make_pills(tt_avgs.tail(n).iloc[::-1], "#7f0000", "#ffebee",
                                      lambda v: f"{v:.2f}%")

    best_block = (
        f'<div style="flex:1;min-width:220px;"><div style="color:#888;font-size:0.71rem;'
        f'margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;">✅ Кращі магазини (мін. %)</div>'
        f'<div>{best_pills or "<span style=\'color:#aaa;font-size:0.75rem;\'>немає даних</span>"}</div></div>'
        f'<div style="flex:1;min-width:220px;"><div style="color:#888;font-size:0.71rem;'
        f'margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em;">❌ Гірші магазини (макс. %)</div>'
        f'<div>{worst_pills or "<span style=\'color:#aaa;font-size:0.75rem;\'>немає даних</span>"}</div></div>'
    ) if active_tt == "__ALL__" else ""

    st.markdown(
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;'
        f'background:#e8f8f8;border:1px solid #8ecece;border-radius:6px;padding:10px 16px;margin:6px 0 10px 0;">'
        f'<div style="min-width:130px;"><div style="color:#555;font-size:0.71rem;margin-bottom:2px;'
        f'text-transform:uppercase;letter-spacing:.04em;">Серед. Fact % / міс.</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{TEAL};">{avg_monthly:.2f}%</div></div>'
        f'<div style="min-width:130px;"><div style="color:#555;font-size:0.71rem;margin-bottom:2px;'
        f'text-transform:uppercase;letter-spacing:.04em;">Average % (норма)</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{RED_LINE};">{avg_global:.2f}%</div></div>'
        f'<div style="min-width:130px;"><div style="color:#555;font-size:0.71rem;margin-bottom:2px;'
        f'text-transform:uppercase;letter-spacing:.04em;">Δ Fact − Average</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{delta_color};">'
        f'{"+" if total_delta > 0 else ""}{total_delta:.2f}%</div></div>'
        f'{best_block}</div>',
        unsafe_allow_html=True)

    x_axis = _MONTH_LABEL_LIST
    fig = go.Figure([
        go.Bar(x=x_axis, y=display_df["Plan"].values,    name="План %",    marker_color="#8ecece", opacity=0.8),
        go.Bar(x=x_axis, y=display_df["Fact"].values,    name="Факт %",    marker_color=TEAL,       opacity=0.95),
        go.Scatter(x=x_axis, y=display_df["Average"].values, name="Average %",
                   line=dict(color=RED_LINE, width=3)),
        go.Scatter(x=x_axis, y=display_df["Delta"].values,   name="Δ %",
                   line=dict(color=ORANGE, dash="dot"), yaxis="y2"),
    ])
    fig.update_layout(
        height=320, margin=dict(t=30, b=20, l=10, r=10),
        barmode="group", hovermode="x unified",
        yaxis=dict(tickformat=".2f", ticksuffix="%"),
        yaxis2=dict(overlaying="y", side="right", tickformat=".2f", ticksuffix="%", showgrid=False),
        legend=dict(orientation="h", y=-0.15, x=0),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"ratio_chart_{article_idx}_{active_tt}")

    _render_tt_slicer(title, df_filtered, col_tt, col_article, skey, active_tt, "ratio")


# ═══════════════════════════════════════════════════════════════
# RATIO HEATMAP + TOP/ANTITOP SECTION
# ═══════════════════════════════════════════════════════════════

def render_ratio_heatmap_section(df, df_filtered, col_tt, col_article,
                                  col_month, col_ratio, col_plf,
                                  articles_to_show, ratio_mode, group_factors=None):
    if group_factors is None:
        group_factors = []

    heat, tt_table, val_col = build_ratio_heat_data(
        df, df_filtered, col_tt, col_article, col_month,
        col_ratio, col_plf, _t(articles_to_show), ratio_mode, _t(group_factors),
    )

    st.markdown(
        f'<div style="margin-bottom:6px;">'
        f'<span style="background:{TEAL_HDR};color:white;font-weight:700;'
        f'padding:4px 14px;font-size:0.9rem;border-radius:2px;">🌡️ Карта аномалій % в ТО</span>'
        f'<span style="margin-left:12px;color:#666;font-size:0.8rem;">Режим: <b>{ratio_mode}</b> · значення у відсотках</span>'
        f'</div>', unsafe_allow_html=True)

    st.dataframe(
        heat.style.background_gradient(cmap="RdYlGn_r", axis=None).highlight_null(color="white")
            .format(lambda v: f"{v:.2f}%" if pd.notna(v) else "", na_rep=""),
        use_container_width=True)

    st.markdown(
        f'<div style="margin:14px 0 6px 0;">'
        f'<span style="background:{TEAL_HDR};color:white;font-weight:700;'
        f'padding:4px 14px;font-size:0.9rem;border-radius:2px;">🏆 TOP / ANTITOP магазинів — % в ТО</span>'
        f'</div>', unsafe_allow_html=True)

    sum_val = tt_table.groupby(col_tt)[val_col].mean().reset_index()
    n_tt    = st.slider("Кількість магазинів (% в ТО)", 1, 100, 10, key="ratio_n_tt_slider")
    top     = sum_val.nsmallest(n_tt, val_col)
    antitop = sum_val.nlargest(n_tt, val_col)

    ca, cb = st.columns(2)
    fmt_pct_fn = {val_col: lambda v: f"{v:.2f}%" if pd.notna(v) else "-"}
    with ca:
        st.write("✅ Top (найменший %)")
        st.dataframe(top.style.background_gradient(cmap="RdYlGn", subset=[val_col]).format(fmt_pct_fn),
                     use_container_width=True)
    with cb:
        st.write("❌ Antitop (найбільший %)")
        st.dataframe(antitop.style.background_gradient(cmap="RdYlGn_r", subset=[val_col]).format(fmt_pct_fn),
                     use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# EXCEL EXPORT
# ═══════════════════════════════════════════════════════════════

def export_excel(df, df_filtered, col_tt, col_article, col_month, col_value,
                 col_plf, articles_to_show, tt_val, group_factors, metric_col,
                 mode, pivot_df, df_tt_agg=None, col_ratio=None):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage

    NUM_FMT = '# ##0;-# ##0;-'
    PCT_FMT = '+0.0%;-0.0%;-'

    # Pre-build shared style objects (реuse замість new per-cell)
    _border_side = Side(style="thin", color="AAAAAA")
    _thin_border = Border(left=_border_side, right=_border_side,
                          top=_border_side, bottom=_border_side)

    def hdr_fill(h):
        h = h.lstrip("#")
        return PatternFill("solid", start_color=h, end_color=h)

    def scw(ws, ci, w):
        ws.column_dimensions[get_column_letter(ci)].width = w

    month_labels_list = _MONTH_LABEL_LIST
    wb = Workbook()
    wb.remove(wb.active)

    arts   = list(articles_to_show)
    gf     = list(group_factors)
    tt_tpl = _t(tt_val)

    # Pre-compute all article DataFrames once (cache hits if already called)
    art_dfs = {art: build_article_monthly(df, df_filtered, col_tt, col_article,
                                           col_month, col_value, col_plf,
                                           art, tt_tpl, _t(gf))
               for art in arts}

    # ── 1. Зведена таблиця ────────────────────────────────────────
    ws_p = wb.create_sheet("Зведена_таблиця")
    ws_p.freeze_panes = "B2"
    header = ["Стаття"] + month_labels_list + ["РАЗОМ"]
    hdr_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    hdr_fill_g = hdr_fill("2e7d32")
    hdr_align  = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for ci, h in enumerate(header, 1):
        c = ws_p.cell(row=1, column=ci, value=h)
        c.font = hdr_font; c.fill = hdr_fill_g
        c.alignment = hdr_align; c.border = _thin_border
    ws_p.row_dimensions[1].height = 28
    ws_p.column_dimensions["A"].width = 36
    for ci in range(2, len(header) + 1):
        scw(ws_p, ci, 11)

    for ri, article in enumerate(arts, 2):
        tdf  = art_dfs[article]
        vals = [article] + list(tdf[metric_col].values) + [tdf[metric_col].sum()]
        for ci, v in enumerate(vals, 1):
            c = ws_p.cell(row=ri, column=ci, value=v)
            c.border = _thin_border
            if ci == 1:
                c.font = Font(name="Arial", size=9)
                c.alignment = Alignment(horizontal="left")
            else:
                c.number_format = NUM_FMT
                c.alignment = Alignment(horizontal="right")
                c.font = Font(name="Arial", size=9,
                              color="C0392B" if isinstance(v, (int, float)) and v < 0 else "000000")

    # ── 2. Листи по статтях ───────────────────────────────────────
    row_labels = ["План", "Факт", "Average", "Дельта"]
    row_keys   = ["Plan", "Fact", "Average", "Delta"]
    row_fills  = ["FFFFFF", "e8d5f5", "fde8e8", "fff9e0"]
    row_colors = ["333333", "5b2d8e", "c0392b", "b8860b"]

    for article in arts:
        tdf  = art_dfs[article]
        safe = article[:28].replace("/", "_").replace("\\", "_")
        ws   = wb.create_sheet(safe)
        ws.freeze_panes = "B3"

        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=15)
        tc = ws.cell(row=1, column=1, value=article)
        tc.font = Font(bold=True, color="FFFFFF", name="Arial", size=12)
        tc.fill = hdr_fill("5b2d8e")
        tc.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 22

        for ci, h in enumerate(["Показник"] + month_labels_list + ["РАЗОМ"], 1):
            c = ws.cell(row=2, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = hdr_fill_g; c.alignment = hdr_align; c.border = _thin_border
        ws.row_dimensions[2].height = 18

        for ri, (label, key, fill_hex, color_hex) in enumerate(
                zip(row_labels, row_keys, row_fills, row_colors), 3):
            vals  = list(tdf[key].values)
            total = sum(vals)
            nc = ws.cell(row=ri, column=1, value=label)
            nc.font = Font(bold=True, color=color_hex, name="Arial", size=9)
            nc.fill = hdr_fill(fill_hex); nc.border = _thin_border
            nc.alignment = Alignment(horizontal="left")
            row_fill = hdr_fill(fill_hex)
            for ci, v in enumerate(vals, 2):
                c = ws.cell(row=ri, column=ci, value=v)
                c.number_format = NUM_FMT; c.fill = row_fill
                c.border = _thin_border
                c.alignment = Alignment(horizontal="right")
                c.font = Font(name="Arial", size=9, color="C0392B" if v < 0 else color_hex)
            tc2 = ws.cell(row=ri, column=14, value=total)
            tc2.number_format = NUM_FMT; tc2.fill = row_fill
            tc2.border = _thin_border
            tc2.alignment = Alignment(horizontal="right")
            tc2.font = Font(bold=True, name="Arial", size=9,
                            color="C0392B" if total < 0 else color_hex)

        ws.column_dimensions["A"].width = 12
        for ci in range(2, 15):
            scw(ws, ci, 11)

        fig = go.Figure([
            go.Bar(x=month_labels_list, y=list(tdf["Plan"].values),
                   name="План", marker_color="#c0c0c0", opacity=0.9),
            go.Bar(x=month_labels_list, y=list(tdf["Fact"].values),
                   name="Факт", marker_color="#5b2d8e", opacity=0.95),
            go.Scatter(x=month_labels_list, y=list(tdf["Average"].values),
                       mode="lines+markers", name="Average",
                       line=dict(color="#c0392b", width=2.5), marker=dict(size=8)),
            go.Scatter(x=month_labels_list, y=list(tdf["Delta"].values),
                       mode="lines+markers", name="Дельта",
                       line=dict(color="#f0c000", width=2), marker=dict(size=7), yaxis="y2"),
        ])
        fig.update_layout(
            barmode="group", height=320, width=900,
            plot_bgcolor="white", paper_bgcolor="white",
            title=dict(text=f"Аналіз — {article}", x=0.5, font=dict(size=12)),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="#ececec"),
            yaxis2=dict(overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="v", x=1.06, y=1, font=dict(size=9)),
            margin=dict(t=40, b=30, l=55, r=130),
            font=dict(family="Arial"),
        )
        try:
            import plotly.io as pio
            img_obj = XLImage(io.BytesIO(pio.to_image(fig, format="png", scale=1.5)))
            img_obj.anchor = "A7"
            ws.add_image(img_obj)
        except Exception:
            ws.cell(row=7, column=1, value="⚠️ Графік недоступний (pip install kaleido)")

    # ── 3. % в ТО ─────────────────────────────────────────────────
    if col_ratio:
        ratio_dfs = {art: build_ratio_monthly(df_filtered, col_tt, col_article, col_month,
                                               col_ratio, col_plf, art, tt_tpl,
                                               df_all=df, group_factors=_t(gf))
                     for art in arts}

        ws_ratio = wb.create_sheet("% в ТО_зведена")
        ws_ratio.freeze_panes = "B2"
        r_header = ["Стаття"] + month_labels_list + ["Серед."]
        teal_fill = hdr_fill("085f63")
        teal_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        for ci, h in enumerate(r_header, 1):
            c = ws_ratio.cell(row=1, column=ci, value=h)
            c.font = teal_font; c.fill = teal_fill
            c.alignment = hdr_align; c.border = _thin_border
        ws_ratio.row_dimensions[1].height = 28
        ws_ratio.column_dimensions["A"].width = 36
        for ci in range(2, len(r_header) + 1):
            scw(ws_ratio, ci, 11)

        r_row_labels = ["План %", "Факт %", "Average %", "Δ %"]
        r_row_keys   = ["Plan", "Fact", "Average", "Delta"]
        r_row_fills  = ["e8f8f8", "d0f0f0", "fde8e8", "fff9e0"]
        r_row_colors = ["085f63", "0d7377", "c0392b", "e67e22"]

        for art_ri, article in enumerate(arts):
            rdf      = ratio_dfs[article]
            base_row = 2 + art_ri * 5
            ws_ratio.merge_cells(start_row=base_row, start_column=1,
                                  end_row=base_row, end_column=len(r_header))
            tc = ws_ratio.cell(row=base_row, column=1, value=article)
            tc.font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
            tc.fill = hdr_fill("0d7377")
            tc.alignment = Alignment(horizontal="left", vertical="center")
            ws_ratio.row_dimensions[base_row].height = 18

            for sub_ri, (label, key, fill_hex, color_hex) in enumerate(
                    zip(r_row_labels, r_row_keys, r_row_fills, r_row_colors), base_row + 1):
                vals    = list(rdf[key].values)
                nz      = [v for v in vals if v != 0]
                summary = float(np.mean(nz)) if nz else 0.0
                nc = ws_ratio.cell(row=sub_ri, column=1, value=label)
                nc.font = Font(bold=True, color=color_hex, name="Arial", size=9)
                nc.fill = hdr_fill(fill_hex); nc.border = _thin_border
                row_fill = hdr_fill(fill_hex)
                for ci, v in enumerate(vals, 2):
                    c = ws_ratio.cell(row=sub_ri, column=ci, value=round(v, 4))
                    c.number_format = '0.00"%"'; c.fill = row_fill
                    c.border = _thin_border
                    c.alignment = Alignment(horizontal="right")
                    c.font = Font(name="Arial", size=9,
                                  color="C0392B" if v < 0 else color_hex)
                sc = ws_ratio.cell(row=sub_ri, column=14, value=round(summary, 4))
                sc.number_format = '0.00"%"'; sc.fill = row_fill
                sc.border = _thin_border
                sc.alignment = Alignment(horizontal="right")
                sc.font = Font(bold=True, name="Arial", size=9,
                               color="C0392B" if summary < 0 else color_hex)

    # ── 4. ТТ-Зведена ─────────────────────────────────────────────
    if df_tt_agg is not None and not df_tt_agg.empty:
        ws_tt = wb.create_sheet("ТТ_Зведена")
        ws_tt.freeze_panes = "B2"
        tt_export_cols = (
            ["ТТ"]
            + [f"fact_{MONTH_LABELS[m]}" for m in _MONTHS]
            + [f"plan_{MONTH_LABELS[m]}" for m in _MONTHS]
            + [f"delta_{MONTH_LABELS[m]}" for m in _MONTHS]
            + ["Fact_РАЗОМ", "Plan_РАЗОМ", "Delta_РАЗОМ", "Pct_РАЗОМ"]
        )
        tt_export_cols = [c for c in tt_export_cols if c in df_tt_agg.columns]
        tt_header_labels = {"ТТ": "ТТ", "Fact_РАЗОМ": "Fact РАЗОМ",
                            "Plan_РАЗОМ": "Plan РАЗОМ", "Delta_РАЗОМ": "Δ РАЗОМ",
                            "Pct_РАЗОМ": "% відхил."}
        for m in _MONTHS:
            ml = MONTH_LABELS[m]
            tt_header_labels.update({f"fact_{ml}": f"{ml} Fact",
                                     f"plan_{ml}": f"{ml} Plan",
                                     f"delta_{ml}": f"{ml} Δ"})

        pur_fill = hdr_fill("5b2d8e")
        for ci, h in enumerate([tt_header_labels.get(c, c) for c in tt_export_cols], 1):
            c = ws_tt.cell(row=1, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = pur_fill
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = _thin_border
        ws_tt.row_dimensions[1].height = 28
        ws_tt.column_dimensions["A"].width = 22
        for ci in range(2, len(tt_export_cols) + 1):
            scw(ws_tt, ci, 10)

        df_tt_sorted = (df_tt_agg[tt_export_cols]
                        .sort_values("Delta_РАЗОМ" if "Delta_РАЗОМ" in df_tt_agg.columns
                                     else tt_export_cols[1], ascending=False)
                        .reset_index(drop=True))

        for ri, row in df_tt_sorted.iterrows():
            for ci, col_name in enumerate(tt_export_cols, 1):
                v = row[col_name]
                c = ws_tt.cell(row=ri + 2, column=ci)
                c.border = _thin_border
                if col_name == "ТТ":
                    c.value = str(v) if pd.notna(v) else ""
                    c.font = Font(name="Arial", size=9)
                    c.alignment = Alignment(horizontal="left")
                elif col_name == "Pct_РАЗОМ":
                    c.value = float(v) / 100 if pd.notna(v) else None
                    c.number_format = PCT_FMT
                    c.alignment = Alignment(horizontal="right")
                    clr = "C0392B" if pd.notna(v) and v > 0 else ("2E7D32" if pd.notna(v) and v < 0 else "000000")
                    c.font = Font(name="Arial", size=9, color=clr)
                else:
                    c.value = float(v) if pd.notna(v) else None
                    c.number_format = NUM_FMT
                    c.alignment = Alignment(horizontal="right")
                    c.font = Font(name="Arial", size=9,
                                  color="C0392B" if pd.notna(v) and isinstance(v, (int, float)) and v < 0 else "000000")

        total_ri = len(df_tt_sorted) + 2
        lav_fill = hdr_fill("e8d5f5")
        for ci, col_name in enumerate(tt_export_cols, 1):
            c = ws_tt.cell(row=total_ri, column=ci)
            c.border = _thin_border; c.fill = lav_fill
            c.font = Font(bold=True, name="Arial", size=9)
            if col_name == "ТТ":
                c.value = "🟰 РАЗОМ"; c.alignment = Alignment(horizontal="left")
            elif col_name == "Pct_РАЗОМ":
                ps = df_tt_sorted["Plan_РАЗОМ"].sum() if "Plan_РАЗОМ" in df_tt_sorted else 0
                fs = df_tt_sorted["Fact_РАЗОМ"].sum() if "Fact_РАЗОМ" in df_tt_sorted else 0
                c.value = (fs / ps - 1) if ps != 0 else None
                c.number_format = PCT_FMT; c.alignment = Alignment(horizontal="right")
            else:
                s = df_tt_sorted[col_name].sum() if col_name in df_tt_sorted else 0
                c.value = float(s) if pd.notna(s) else None
                c.number_format = NUM_FMT; c.alignment = Alignment(horizontal="right")

    # ── 5. Heatmap основний ───────────────────────────────────────
    if gf:
        heat, tt_table, val_col = build_heat_data(
            df, df_filtered, col_tt, col_article, col_month, col_value,
            col_plf, _t(gf), _t(arts), mode)

        ws_h = wb.create_sheet("Heatmap")
        ws_h.freeze_panes = "B2"
        heat_header = [col_tt] + month_labels_list + ["РАЗОМ"]
        for ci, h in enumerate(heat_header, 1):
            c = ws_h.cell(row=1, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = pur_fill; c.alignment = hdr_align; c.border = _thin_border

        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            cmap_obj = plt.get_cmap("RdYlGn_r")
            flat = heat.values.flatten().astype(float)
            flat = flat[~np.isnan(flat)]
            norm = mcolors.Normalize(vmin=float(flat.min()), vmax=float(flat.max()))
            use_cmap = True
        except Exception:
            use_cmap = False

        for ri, (idx, row) in enumerate(heat.iterrows(), 2):
            ws_h.cell(row=ri, column=1, value=idx).border = _thin_border
            for ci, col_name in enumerate(heat.columns, 2):
                v = row[col_name]
                c = ws_h.cell(row=ri, column=ci)
                c.border = _thin_border; c.alignment = Alignment(horizontal="right")
                if pd.isna(v):
                    c.fill = hdr_fill("FFFFFF")
                else:
                    c.value = float(v); c.number_format = NUM_FMT
                    if use_cmap:
                        rgba = cmap_obj(norm(float(v)))
                        c.fill = hdr_fill("{:02X}{:02X}{:02X}".format(
                            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)))
                    c.font = Font(name="Arial", size=9)

        ws_h.column_dimensions["A"].width = 20
        for ci in range(2, len(heat_header) + 1):
            scw(ws_h, ci, 11)

        ws_top     = wb.create_sheet("TOP_ANTITOP")
        sum_val    = tt_table.groupby(col_tt)[val_col].sum().reset_index()
        top_df     = sum_val.nsmallest(50, val_col)
        antitop_df = sum_val.nlargest(50, val_col)

        def write_block(start_col, title_text, df_block, cmap_name):
            tc2 = ws_top.cell(row=1, column=start_col, value=title_text)
            tc2.font = Font(bold=True, color="FFFFFF", name="Arial", size=11)
            tc2.fill = pur_fill
            ws_top.merge_cells(start_row=1, start_column=start_col,
                               end_row=1, end_column=start_col + 1)
            for ci2, h2 in enumerate([col_tt, val_col], start_col):
                c = ws_top.cell(row=2, column=ci2, value=h2)
                c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
                c.fill = hdr_fill_g; c.border = _thin_border
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                vals_arr = df_block[val_col].values.astype(float)
                nm = mcolors.Normalize(vmin=float(vals_arr.min()), vmax=float(vals_arr.max()))
                cm2 = plt.get_cmap(cmap_name); use_c = True
            except Exception:
                use_c = False
            for ri2, row2 in enumerate(df_block.itertuples(index=False), 3):
                tt_v  = getattr(row2, col_tt, "")
                val_v = getattr(row2, val_col, 0)
                ws_top.cell(row=ri2, column=start_col, value=tt_v).border = _thin_border
                ws_top.cell(row=ri2, column=start_col).font = Font(name="Arial", size=9)
                c2 = ws_top.cell(row=ri2, column=start_col + 1,
                                  value=float(val_v) if pd.notna(val_v) else None)
                c2.number_format = NUM_FMT; c2.border = _thin_border
                c2.alignment = Alignment(horizontal="right")
                if use_c and pd.notna(val_v):
                    rgba2 = cm2(nm(float(val_v)))
                    hx2 = "{:02X}{:02X}{:02X}".format(
                        int(rgba2[0]*255), int(rgba2[1]*255), int(rgba2[2]*255))
                    c2.fill = hdr_fill(hx2)
                    lum = (0.299*rgba2[0] + 0.587*rgba2[1] + 0.114*rgba2[2])
                    c2.font = Font(name="Arial", size=9,
                                   color="000000" if lum > 0.5 else "FFFFFF")
                else:
                    c2.font = Font(name="Arial", size=9)
            ws_top.column_dimensions[get_column_letter(start_col)].width = 22
            ws_top.column_dimensions[get_column_letter(start_col + 1)].width = 14

        write_block(1, "✅ TOP (економія)",      top_df,     "RdYlGn")
        write_block(4, "❌ ANTITOP (переліміт)", antitop_df, "RdYlGn_r")

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="СІМІ Dashboard", layout="wide")
    st.markdown("""
    <style>
    .simi-header{background:linear-gradient(90deg,#5b2d8e 0%,#7b52ae 100%);
      color:white;padding:10px 18px;border-radius:4px;margin-bottom:4px;}
    .simi-logo{font-size:2rem;font-weight:900;color:#f0c000;
      letter-spacing:1px;margin-right:24px;vertical-align:middle;}
    .simi-store{font-size:1.1rem;font-weight:700;color:white;vertical-align:middle;}
    .simi-meta-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:4px 12px;margin-top:6px;}
    .simi-meta-item{font-size:0.78rem;color:#e0d0f8;}
    .simi-meta-val{font-size:0.85rem;font-weight:700;color:white;}
    .article-selector{background:#f4f0fa;border:2px solid #5b2d8e;
      border-radius:8px;padding:12px 16px;margin-bottom:14px;}
    .block-sep{border-top:2px solid #5b2d8e;margin:16px 0 10px 0;}
    .block-sep-teal{border-top:2px solid #0d7377;margin:16px 0 10px 0;}
    .ratio-section-banner{background:linear-gradient(90deg,#085f63 0%,#0d7377 100%);
      color:white;padding:8px 18px;border-radius:4px;margin:8px 0 6px 0;
      font-size:0.95rem;font-weight:700;}
    div[data-testid="stButton"]>button[kind="primary"]{
      background-color:#5b2d8e!important;color:white!important;
      border:2px solid #5b2d8e!important;font-weight:700!important;font-size:0.72rem!important;}
    div[data-testid="stButton"]>button[kind="secondary"]{
      background-color:#f4f0fa!important;color:#5b2d8e!important;
      border:1px solid #c9b6e8!important;font-size:0.72rem!important;}
    div[data-testid="stButton"]>button[kind="secondary"]:hover{
      background-color:#e8d5f5!important;border-color:#5b2d8e!important;}
    </style>""", unsafe_allow_html=True)

    file = st.file_uploader("📂 Завантажте Excel", type=["xlsx", "xlsb"])
    if file is None:
        st.info("Завантажте Excel-файл для початку роботи.")
        st.stop()

    file_bytes = file.read()
    xl         = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_name = st.selectbox("Аркуш", xl.sheet_names)
    df_raw     = load_excel(file_bytes, file.name, sheet_name)
    cols       = df_raw.columns.tolist()

    with st.expander("⚙️ Налаштування колонок", expanded=True):
        c1, c2, c3, c4, _ = st.columns(5)
        with c1:
            col_tt   = st.selectbox("TT (Магазин)", cols)
            col_year = st.selectbox("Year",          cols)
        with c2:
            col_month   = st.selectbox("Month",    cols)
            col_value   = st.selectbox("Значення", cols)
        with c3:
            col_plf     = st.selectbox("PL / F",         cols)
            col_article = st.selectbox("Стаття бюджету", cols)
        with c4:
            col_level0 = st.selectbox("Level_0", cols)
            ratio_candidates = ["— не обрано —"] + cols
            default_ratio_idx = 0
            for i, c in enumerate(cols):
                if "%" in c and "то" in c.lower():
                    default_ratio_idx = i + 1; break
            col_ratio = st.selectbox(
                "% в ТО без акцизу та без ПДВ", ratio_candidates,
                index=default_ratio_idx,
                help="Колонка з відсотком % в ТО. Оберіть 'не обрано' щоб приховати блок.")
            if col_ratio == "— не обрано —":
                col_ratio = None

    # ── Препроцесинг (кешується) ──────────────────────────────────
    df = preprocess_df(df_raw, col_value, col_month, col_ratio)

    with st.expander("🏪 Колонки шапки магазину", expanded=False):
        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            col_city  = st.selectbox("Місто",  ["—"] + cols)
            col_area  = st.selectbox("Площа",  ["—"] + cols)
        with sh2:
            col_format = st.selectbox("Формат ТО",   ["—"] + cols)
            col_mega   = st.selectbox("Мегасегмент", ["—"] + cols)
        with sh3:
            col_rik = st.selectbox("Рік",            ["—"] + cols)
            col_mis = st.selectbox("Місяць (шапка)", ["—"] + cols)

    # ── Sidebar filters ───────────────────────────────────────────
    st.sidebar.markdown("## 🔍 Фільтри")
    year_val   = st.sidebar.multiselect("Year",    sorted(df[col_year].dropna().unique(),   key=str))
    month_val  = st.sidebar.multiselect("Month",   sorted(df[col_month].dropna().unique(),  key=str))
    level0_val = st.sidebar.multiselect("Level_0", sorted(df[col_level0].dropna().unique(), key=str))

    st.sidebar.markdown("### ➕ Додаткові фільтри")
    fixed_cols = {col_tt, col_year, col_month, col_level0}
    extra_col  = st.sidebar.selectbox(
        "Стовпець для фільтру",
        ["— не обрано —"] + [c for c in cols if c not in fixed_cols],
        key="extra_filter_col")
    extra_filters = {}
    if extra_col != "— не обрано —":
        extra_val = st.sidebar.multiselect(f"Значення «{extra_col}»",
                                           sorted(df[extra_col].dropna().unique(), key=str),
                                           key="extra_filter_val")
        if extra_val:
            extra_filters[extra_col] = extra_val
    remaining_cols = [c for c in cols if c not in fixed_cols and c != extra_col]
    if extra_col != "— не обрано —":
        extra_col2 = st.sidebar.selectbox("Ще стовпець", ["— не обрано —"] + remaining_cols,
                                          key="extra_filter_col2")
        if extra_col2 != "— не обрано —":
            extra_val2 = st.sidebar.multiselect(f"Значення «{extra_col2}»",
                                                sorted(df[extra_col2].dropna().unique(), key=str),
                                                key="extra_filter_val2")
            if extra_val2:
                extra_filters[extra_col2] = extra_val2
            remaining_cols2 = [c for c in remaining_cols if c != extra_col2]
            extra_col3 = st.sidebar.selectbox("Ще стовпець", ["— не обрано —"] + remaining_cols2,
                                              key="extra_filter_col3")
            if extra_col3 != "— не обрано —":
                extra_val3 = st.sidebar.multiselect(f"Значення «{extra_col3}»",
                                                    sorted(df[extra_col3].dropna().unique(), key=str),
                                                    key="extra_filter_val3")
                if extra_val3:
                    extra_filters[extra_col3] = extra_val3

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏪 ТТ за поточними фільтрами")

    # Без .copy() — просто boolean mask chain
    df_pre = df
    if year_val:   df_pre = df_pre[df_pre[col_year].isin(year_val)]
    if month_val:  df_pre = df_pre[df_pre[col_month].isin(month_val)]
    if level0_val: df_pre = df_pre[df_pre[col_level0].isin(level0_val)]
    for col_e, vals_e in extra_filters.items():
        df_pre = df_pre[df_pre[col_e].isin(vals_e)]

    visible_tts = sorted(df_pre[col_tt].dropna().unique(), key=str)

    if visible_tts:
        st.sidebar.caption(f"Знайдено: {len(visible_tts)} магазинів")
        tt_search = st.sidebar.text_input("🔎 Пошук ТТ", value="",
                                          placeholder="Введіть назву…", key="tt_search")
        filtered_tts = ([tt for tt in visible_tts if tt_search.lower() in str(tt).lower()]
                        if tt_search else visible_tts)
        bc1, bc2 = st.sidebar.columns(2)
        with bc1:
            if st.button("✅ Всі", key="tt_select_all", use_container_width=True):
                st.session_state["tt_multiselect"] = filtered_tts
        with bc2:
            if st.button("✖ Жодного", key="tt_clear_all", use_container_width=True):
                st.session_state["tt_multiselect"] = []

        tt_val = st.sidebar.multiselect("Оберіть ТТ:", options=filtered_tts,
                                        default=st.session_state.get("tt_multiselect", []),
                                        key="tt_multiselect")
    else:
        st.sidebar.warning("Немає ТТ за обраними фільтрами.")
        tt_val = []

    st.sidebar.markdown("---")
    mode = st.sidebar.selectbox("Mode (Heatmap)", ["Delta", "Delta %", "Z-score", "Fact", "Average"])
    ratio_mode = st.sidebar.selectbox("Mode (% в ТО Heatmap)", ["Delta", "Delta %", "Fact", "Average"],
                                      key="ratio_mode")
    group_factors = st.sidebar.multiselect(
        "Фактори групування (Average/Std)",
        options=[c for c in df.columns if c not in [col_value, col_plf, col_article, "_m"]],
        default=[col_tt] if col_tt in df.columns else [])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👁️ Відображення")
    show_ratio_section = st.sidebar.checkbox("Показати блок «% в ТО»", value=True,
                                             key="show_ratio_section") if col_ratio else False
    show_ratio_heatmap = st.sidebar.checkbox("Показати Heatmap % в ТО", value=True,
                                             key="show_ratio_heatmap") if col_ratio else False

    # ── Apply filters (no copy) ───────────────────────────────────
    def apply_filters(d):
        if tt_val:     d = d[d[col_tt].isin(tt_val)]
        if year_val:   d = d[d[col_year].isin(year_val)]
        if month_val:  d = d[d[col_month].isin(month_val)]
        if level0_val: d = d[d[col_level0].isin(level0_val)]
        for col, vals in extra_filters.items():
            d = d[d[col].isin(vals)]
        return d

    df_filtered = apply_filters(df)

    def get_meta(col):
        if col == "—": return "—"
        vals = df_filtered[col].dropna().unique()
        return str(vals[0]) if len(vals) else "—"

    # ── Article selector ─────────────────────────────────────────
    articles_all = sorted(df[col_article].dropna().unique(), key=str)
    st.markdown('<div class="article-selector">', unsafe_allow_html=True)
    sel_col1, sel_col2, sel_col3 = st.columns([3, 1, 1])
    with sel_col1:
        st.markdown("**🎯 Стаття бюджету для аналізу** — обери одну або увімкни «Всі»")
        selected_article = st.selectbox("article_selector", articles_all,
                                        key="global_article", label_visibility="collapsed")
    with sel_col2:
        st.markdown("&nbsp;")
        show_all = st.checkbox("Показати всі статті", value=False, key="show_all")
    with sel_col3:
        st.markdown("&nbsp;")
        multi_sel = st.multiselect("Або обери кілька:", articles_all, default=[], key="multi_article")
    st.markdown('</div>', unsafe_allow_html=True)

    if show_all:
        articles_to_show = articles_all
    elif multi_sel:
        articles_to_show = multi_sel
    else:
        articles_to_show = [selected_article]

    st.info(f"📌 Показується {len(articles_to_show)} {'стаття' if len(articles_to_show)==1 else 'статей'}: "
            f"**{articles_to_show[0]}**" if len(articles_to_show) == 1
            else f"📌 Показується {len(articles_to_show)} статей: {', '.join(articles_to_show)}")

    store_name = ", ".join(str(v) for v in tt_val) if tt_val else "Всі магазини"
    st.markdown(
        f'<div class="simi-header">'
        f'<span class="simi-logo">СіМі</span>'
        f'<span class="simi-store">{store_name}</span>'
        f'<div class="simi-meta-grid">'
        f'<div><span class="simi-meta-item">Місто </span><span class="simi-meta-val">{get_meta(col_city)}</span></div>'
        f'<div><span class="simi-meta-item">Площа </span><span class="simi-meta-val">{get_meta(col_area)}</span></div>'
        f'<div><span class="simi-meta-item">Формат ТО </span><span class="simi-meta-val">{get_meta(col_format)}</span></div>'
        f'<div><span class="simi-meta-item">Мегасегмент </span><span class="simi-meta-val">{get_meta(col_mega)}</span></div>'
        f'<div><span class="simi-meta-item">Рік </span><span class="simi-meta-val">{get_meta(col_rik)}</span></div>'
        f'<div><span class="simi-meta-item">Місяць </span><span class="simi-meta-val">{get_meta(col_mis)}</span></div>'
        f'</div></div>', unsafe_allow_html=True)

    tt_tpl = _t(tt_val)
    gf_tpl = _t(group_factors)

    # ═══ ARTICLE BLOCKS ═══════════════════════════════════════════
    for art_idx, article in enumerate(articles_to_show):
        st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)

        tdf = build_article_monthly(df, df_filtered, col_tt, col_article,
                                    col_month, col_value, col_plf,
                                    article, tt_tpl, gf_tpl)

        if col_ratio and show_ratio_section:
            rdf = build_ratio_monthly(df_filtered, col_tt, col_article, col_month,
                                      col_ratio, col_plf, article, tt_tpl,
                                      df_all=df, group_factors=gf_tpl)
            col_main, col_rat = st.columns([1, 1], gap="medium")
            with col_main:
                render_article_block(title=article, table_df=tdf,
                                     chart_title=f"Аналіз — {article}",
                                     df=df, df_filtered=df_filtered,
                                     col_tt=col_tt, col_article=col_article,
                                     col_month=col_month, col_value=col_value, col_plf=col_plf,
                                     group_factors=group_factors, tt_val=tt_val,
                                     article_idx=art_idx)
            with col_rat:
                render_ratio_article_block(title=article, table_df=rdf,
                                           df=df, df_filtered=df_filtered,
                                           col_tt=col_tt, col_article=col_article,
                                           col_month=col_month, col_ratio=col_ratio, col_plf=col_plf,
                                           tt_val=tt_val, article_idx=art_idx,
                                           group_factors=group_factors)
        else:
            render_article_block(title=article, table_df=tdf,
                                 chart_title=f"Аналіз — {article}",
                                 df=df, df_filtered=df_filtered,
                                 col_tt=col_tt, col_article=col_article,
                                 col_month=col_month, col_value=col_value, col_plf=col_plf,
                                 group_factors=group_factors, tt_val=tt_val,
                                 article_idx=art_idx)

    # ═══ PIVOT TABLE ══════════════════════════════════════════════
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("📋 Зведена таблиця")
    pivot_metric = st.radio("Метрика", ["Fact", "Plan", "Delta (Fact-Plan)"], horizontal=True)
    col_map_d  = {"Fact": "Fact", "Plan": "Plan", "Delta (Fact-Plan)": "Delta"}
    metric_col = col_map_d[pivot_metric]

    rows_pivot = []
    for article in articles_to_show:
        tdf = build_article_monthly(df, df_filtered, col_tt, col_article,
                                    col_month, col_value, col_plf,
                                    article, tt_tpl, gf_tpl)
        row = {"Стаття": article}
        for m in _MONTHS:
            row[MONTH_LABELS[m]] = tdf.loc[m, metric_col]
        row["РАЗОМ"] = tdf[metric_col].sum()
        rows_pivot.append(row)

    pivot_df = pd.DataFrame(rows_pivot).set_index("Стаття")
    cmap_p   = "RdYlGn_r" if pivot_metric == "Delta (Fact-Plan)" else "Blues"
    st.dataframe(
        pivot_df.style.background_gradient(cmap=cmap_p, axis=None)
            .format(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-", na_rep="-"),
        use_container_width=True)

    # ═══ % в ТО — ЗВЕДЕНА ════════════════════════════════════════
    if col_ratio and show_ratio_section:
        st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
        st.markdown('<div class="ratio-section-banner">📊 Зведена таблиця — % в ТО без акцизу та без ПДВ</div>',
                    unsafe_allow_html=True)
        ratio_pivot_metric = st.radio("Метрика (% в ТО)", ["Fact", "Plan", "Average", "Delta"],
                                      horizontal=True, key="ratio_pivot_metric")
        rows_ratio_pivot = []
        for article in articles_to_show:
            rdf = build_ratio_monthly(df_filtered, col_tt, col_article, col_month,
                                      col_ratio, col_plf, article, tt_tpl,
                                      df_all=df, group_factors=gf_tpl)
            row = {"Стаття": article}
            vals_m = rdf[ratio_pivot_metric].values
            for m in _MONTHS:
                row[MONTH_LABELS[m]] = rdf.loc[m, ratio_pivot_metric]
            nz = vals_m[vals_m != 0]
            row["Серед."] = float(nz.mean()) if len(nz) else 0.0
            rows_ratio_pivot.append(row)

        ratio_pivot_df = pd.DataFrame(rows_ratio_pivot).set_index("Стаття")
        st.dataframe(
            ratio_pivot_df.style.background_gradient(cmap="RdYlGn_r", axis=None)
                .format(lambda v: f"{v:.2f}%" if pd.notna(v) else "-", na_rep="-"),
            use_container_width=True)

    # ═══ TT PIVOT ═════════════════════════════════════════════════
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("📋 Зведена таблиця в розрізі ТТ")

    tt_pivot_metric = st.radio("Метрика (ТТ)", ["Fact", "Plan", "Delta (Fact-Plan)"],
                               horizontal=True, key="tt_pivot_metric")
    tt_metric_col = col_map_d[tt_pivot_metric]
    show_pct      = st.checkbox("Показати % відхилення (Fact vs Plan)", value=True, key="show_pct")
    show_months   = st.checkbox("Розгорнути по місяцях", value=False, key="tt_show_months")

    df_tt_agg = build_tt_pivot(df_filtered, col_tt, col_article, col_month,
                               col_value, col_plf, _t(articles_to_show))

    if df_tt_agg.empty:
        st.info("Немає даних для побудови таблиці по ТТ.")
    else:
        display_cols = ["ТТ"]
        col_labels   = {"ТТ": "ТТ"}

        if show_months:
            for m in _MONTHS:
                ml = MONTH_LABELS[m]
                if tt_metric_col in ("Fact", "Delta"):
                    display_cols.append(f"fact_{ml}"); col_labels[f"fact_{ml}"] = f"{ml} Fact"
                if tt_metric_col == "Plan":
                    display_cols.append(f"plan_{ml}"); col_labels[f"plan_{ml}"] = f"{ml} Plan"
                if show_pct and tt_metric_col != "Plan":
                    display_cols.append(f"pct_{ml}"); col_labels[f"pct_{ml}"] = f"{ml} %"

        if tt_metric_col == "Fact":
            display_cols += ["Fact_РАЗОМ"]; col_labels["Fact_РАЗОМ"] = "Fact РАЗОМ"
        elif tt_metric_col == "Plan":
            display_cols += ["Plan_РАЗОМ"]; col_labels["Plan_РАЗОМ"] = "Plan РАЗОМ"
        else:
            display_cols += ["Fact_РАЗОМ", "Plan_РАЗОМ", "Delta_РАЗОМ"]
            col_labels.update({"Fact_РАЗОМ": "Fact РАЗОМ", "Plan_РАЗОМ": "Plan РАЗОМ", "Delta_РАЗОМ": "Δ РАЗОМ"})
        if show_pct:
            display_cols.append("Pct_РАЗОМ"); col_labels["Pct_РАЗОМ"] = "% відхил."

        df_display = df_tt_agg[display_cols].rename(columns=col_labels).set_index("ТТ")
        sort_col_label = ("% відхил." if show_pct else
                          ("Δ РАЗОМ" if tt_metric_col == "Delta (Fact-Plan)" else
                           "Fact РАЗОМ" if tt_metric_col == "Fact" else "Plan РАЗОМ"))
        if sort_col_label in df_display.columns:
            df_display = df_display.sort_values(sort_col_label, ascending=True)

        total_row = df_display.sum(numeric_only=True)
        if "% відхил." in df_display.columns:
            ps = df_tt_agg["Plan_РАЗОМ"].sum()
            fs = df_tt_agg["Fact_РАЗОМ"].sum()
            total_row["% відхил."] = (fs / ps - 1) * 100 if ps != 0 else None
        total_row.name = "🟰 РАЗОМ"
        df_display = pd.concat([df_display, total_row.to_frame().T])

        pct_cols       = [c for c in df_display.columns if "%" in c]
        num_cols       = [c for c in df_display.columns if "%" not in c]
        delta_num_cols = [c for c in num_cols if "Δ" in c]
        other_num_cols = [c for c in num_cols if "Δ" not in c]

        fmt_dict = {c: (lambda v: "-" if pd.isna(v) else f"{v:,.0f}".replace(",", " "))
                    for c in num_cols}
        fmt_dict.update({c: (lambda v: "-" if pd.isna(v) else f"{'+'if v>0 else ''}{v:.1f}%")
                         for c in pct_cols})

        styled = df_display.style.format(fmt_dict, na_rep="-")
        if pct_cols:
            styled = styled.background_gradient(cmap="RdYlGn_r",
                subset=pd.IndexSlice[df_display.index[:-1], pct_cols], axis=None)
        if delta_num_cols:
            styled = styled.background_gradient(cmap="RdYlGn_r",
                subset=pd.IndexSlice[df_display.index[:-1], delta_num_cols], axis=None)
        if other_num_cols:
            styled = styled.background_gradient(cmap="Blues",
                subset=pd.IndexSlice[df_display.index[:-1], other_num_cols], axis=None)
        styled = styled.apply(
            lambda row: ["font-weight:bold;border-top:2px solid #5b2d8e;" for _ in row]
            if row.name == "🟰 РАЗОМ" else ["" for _ in row], axis=1)

        st.dataframe(styled, use_container_width=True, height=500)
        csv_bytes = df_display.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("⬇️ Завантажити CSV (ТТ-зведена)", data=csv_bytes,
                           file_name="tt_pivot.csv", mime="text/csv", key="tt_pivot_csv")

    # ═══ HEATMAP ══════════════════════════════════════════════════
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("🌡️ Карта аномалій по магазинах")

    if group_factors:
        heat, tt_table, val_col = build_heat_data(
            df, df_filtered, col_tt, col_article, col_month, col_value,
            col_plf, gf_tpl, _t(articles_to_show), mode)

        st.dataframe(
            heat.style.background_gradient(cmap="RdYlGn_r", axis=None).highlight_null(color="white")
                .format(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "", na_rep=""),
            use_container_width=True)

        st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
        st.subheader("🏆 TOP / ANTITOP магазинів")
        sum_val = tt_table.groupby(col_tt)[val_col].sum().reset_index()
        n_tt    = st.slider("Кількість магазинів", 1, 100, 10)
        fmt_num_fn = {val_col: lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-"}
        ca, cb = st.columns(2)
        with ca:
            st.write("✅ Top (економія)")
            st.dataframe(sum_val.nsmallest(n_tt, val_col).style
                         .background_gradient(cmap="RdYlGn", subset=[val_col])
                         .format(fmt_num_fn))
        with cb:
            st.write("❌ Antitop (переліміт)")
            st.dataframe(sum_val.nlargest(n_tt, val_col).style
                         .background_gradient(cmap="RdYlGn_r", subset=[val_col])
                         .format(fmt_num_fn))
    else:
        st.info("Оберіть фактори групування в боковому меню для побудови Heatmap.")

    # ═══ % в ТО — HEATMAP ════════════════════════════════════════
    if col_ratio and show_ratio_heatmap:
        st.markdown('<div class="block-sep-teal"></div>', unsafe_allow_html=True)
        render_ratio_heatmap_section(
            df, df_filtered, col_tt, col_article, col_month,
            col_ratio, col_plf, articles_to_show, ratio_mode,
            group_factors=group_factors)

    # ═══ EXPORT ═══════════════════════════════════════════════════
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("📥 Експорт в Excel")

    export_sections = ["✅ Зведена таблиця (статті)", "✅ Листи по кожній статті (з графіком)"]
    if df_tt_agg is not None and not df_tt_agg.empty:
        export_sections.append("✅ ТТ-Зведена таблиця")
    if col_ratio:
        export_sections.append("✅ % в ТО — зведена таблиця")
    if group_factors:
        export_sections += ["✅ Heatmap аномалій", "✅ TOP / ANTITOP магазинів"]

    st.markdown("**Файл міститиме аркуші:**")
    for s in export_sections:
        st.markdown(f"- {s}")

    st.download_button(
        label="⬇️ Скачати дашборд як Excel",
        data=export_excel(df, df_filtered, col_tt, col_article, col_month, col_value,
                          col_plf, articles_to_show, tt_val, group_factors, metric_col,
                          mode, pivot_df, df_tt_agg=df_tt_agg, col_ratio=col_ratio),
        file_name="simi_dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
