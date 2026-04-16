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

PURPLE    = "#5b2d8e"
GREY      = "#c0c0c0"
RED_LINE  = "#c0392b"
YELLOW    = "#f0c000"
GREEN_HDR = "#2e7d32"


@st.cache_data
def load_excel(file, sheet_name):
    import os
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".xlsb":
        df = pd.read_excel(file, sheet_name=sheet_name, engine="pyxlsb")
    else:
        df = pd.read_excel(file, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df


def get_month_num(series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.astype(str).str.strip().map(MONTH_MAP).fillna(0).astype(int)


def build_article_monthly(df, df_filtered, col_tt, col_article, col_month,
                          col_value, col_plf, selected_art, selected_tts, group_factors):
    art_all  = df[df[col_article] == selected_art].copy()
    art_filt = df_filtered[df_filtered[col_article] == selected_art].copy()
    art_all["_m"]  = get_month_num(art_all[col_month])
    art_filt["_m"] = get_month_num(art_filt[col_month])

    if art_filt.empty:
        merged = pd.DataFrame(index=range(1, 13),
                              columns=["Plan", "Fact", "Average", "Delta"]).fillna(0)
        merged.index.name = "month"
        return merged

    if not selected_tts:
        selected_tts = art_filt[col_tt].dropna().unique().tolist()

    plan = (art_filt[art_filt[col_plf] == "PL"]
            .groupby("_m")[col_value].sum()
            .reindex(range(1, 13), fill_value=0).rename("Plan"))
    fact = (art_filt[art_filt[col_plf] == "F"]
            .groupby("_m")[col_value].sum()
            .reindex(range(1, 13), fill_value=0).rename("Fact"))

    if group_factors:
        global_avg_std = (
            art_all[art_all[col_plf] == "F"]
            .groupby(group_factors + [col_article], as_index=False)[col_value]
            .agg(Average_Calc="mean", Std="std")
        )
    else:
        global_avg_std = (
            art_all[art_all[col_plf] == "F"]
            .groupby([col_article, "_m"], as_index=False)[col_value]
            .agg(Average_Calc="mean")
        )
        if "Std" not in global_avg_std.columns:
            global_avg_std["Std"] = np.nan

    tt_grp = list(dict.fromkeys([col_tt] + group_factors + ["_m", col_article]))
    tt_table = (
        art_filt[art_filt[col_plf] == "F"]
        .groupby(tt_grp, as_index=False)[col_value]
        .sum()
        .rename(columns={col_value: "Fact"})
    )

    merge_cols = list(dict.fromkeys(group_factors + [col_article]))
    if merge_cols:
        tt_table = pd.merge(tt_table, global_avg_std, on=merge_cols, how="left")
    else:
        if "_m" in global_avg_std.columns:
            tt_table = pd.merge(tt_table, global_avg_std, on=[col_article, "_m"], how="left")
        else:
            tt_table["Average_Calc"] = np.nan
            tt_table["Std"] = np.nan

    tt_table["Fact"]         = tt_table["Fact"].fillna(0)
    tt_table["Average_Calc"] = tt_table["Average_Calc"].fillna(0)
    tt_table.loc[tt_table["Fact"] == 0, "Average_Calc"] = 0

    tt_table_sel    = tt_table[tt_table[col_tt].isin(selected_tts)].copy()
    dynamic_average = (tt_table_sel.groupby("_m")["Average_Calc"]
                       .sum()
                       .reindex(range(1, 13), fill_value=0)
                       .rename("Average"))

    merged = pd.DataFrame(index=range(1, 13)).join(plan).join(fact).join(dynamic_average)
    merged = merged.fillna(0)
    merged.index.name = "month"
    merged.loc[merged["Fact"] == 0, "Average"] = 0
    merged["Delta"] = merged["Fact"] - merged["Average"]

    return merged


def render_article_block(title, table_df, chart_title,
                         df=None, df_filtered=None,
                         col_tt=None, col_article=None,
                         col_month=None, col_value=None, col_plf=None,
                         group_factors=None, tt_val=None,
                         article_idx=0):
    """
    Renders the article block.
    Below the chart — a TT slicer: clicking a TT button re-renders
    the table & chart filtered to that specific store only.
    article_idx – unique integer key per article to avoid widget key collisions.
    """
    rows = {
        "План":    ("Plan",    "#ffffff", "#333333"),
        "Факт":    ("Fact",    "#e8d5f5", PURPLE),
        "Average": ("Average", "#fde8e8", RED_LINE),
        "Дельта":  ("Delta",   "#fff9e0", "#b8860b"),
    }
    month_cols = [MONTH_LABELS[m] for m in range(1, 13)]

    th = ("background:#2e7d32;color:white;font-weight:bold;border:1px solid #aaa;"
          "padding:4px 8px;text-align:center;font-size:0.78rem;")
    td = "border:1px solid #ccc;padding:3px 7px;text-align:right;font-size:0.78rem;"
    tl = "border:1px solid #ccc;padding:3px 7px;font-size:0.78rem;font-weight:600;white-space:nowrap;"

    # ── Session-state key for current TT selection per article ──
    skey = f"slicer_tt_{article_idx}"
    if skey not in st.session_state:
        st.session_state[skey] = "__ALL__"

    active_tt = st.session_state[skey]

    # ── If a specific TT is selected, recompute table_df for that TT ──
    if active_tt != "__ALL__" and df is not None and df_filtered is not None:
        # filter df_filtered to single TT
        df_filt_tt = df_filtered[df_filtered[col_tt] == active_tt].copy()
        display_df = build_article_monthly(
            df, df_filt_tt, col_tt, col_article, col_month, col_value,
            col_plf, title, [active_tt], group_factors or []
        )
    else:
        display_df = table_df

    # ── Header ──────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:20px; margin-bottom:4px;">
      <span style="background:{GREEN_HDR};color:white;font-weight:700;padding:4px 14px;
                   font-size:0.9rem;border-radius:2px;">{title}</span>
      {"" if active_tt == "__ALL__" else
       f'<span style="margin-left:10px;background:{PURPLE};color:white;font-size:0.78rem;'
       f'padding:2px 10px;border-radius:10px;">📍 {active_tt}</span>'}
    </div>
    """, unsafe_allow_html=True)

    # ── Data table ───────────────────────────────────────────────
    html = f"""
    <div style="overflow-x:auto;">
    <table style="border-collapse:collapse;width:100%;margin-bottom:6px;">
      <thead><tr>
        <th style="{th}">Показник</th>
        {"".join(f'<th style="{th}">{m}</th>' for m in month_cols)}
        <th style="{th}">Разом</th>
      </tr></thead><tbody>
    """
    for label, (col, bg, color) in rows.items():
        vals  = [display_df.loc[m, col] for m in range(1, 13)]
        total = sum(vals)
        html += f'<tr style="background:{bg};">'
        html += f'<td style="{tl}color:{color};">{label}</td>'
        for v in vals:
            neg = "color:#c0392b;" if v < 0 else ""
            html += f'<td style="{td}{neg}">{v:,.0f}</td>'
        html += f'<td style="{td}font-weight:700;">{total:,.0f}</td></tr>'
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

    # ── Metrics panel ────────────────────────────────────────────
    if (df_filtered is not None and col_tt and col_article
            and col_value and col_plf and col_month):

        df_for_metrics = (
            df_filtered[df_filtered[col_tt] == active_tt].copy()
            if active_tt != "__ALL__" else df_filtered.copy()
        )

        facts          = [display_df.loc[m, "Fact"] for m in range(1, 13)]
        non_zero_facts = [f for f in facts if f != 0]
        avg_monthly    = np.mean(non_zero_facts) if non_zero_facts else 0
        total_fact     = sum(facts)
        total_plan     = sum(display_df.loc[m, "Plan"] for m in range(1, 13))
        total_delta    = total_fact - total_plan
        pct_vs_plan    = ((total_fact / total_plan - 1) * 100) if total_plan != 0 else None

        pct_str   = (f"{'+' if pct_vs_plan >= 0 else ''}{pct_vs_plan:.1f}%"
                     if pct_vs_plan is not None else "—")
        pct_color = RED_LINE if (pct_vs_plan or 0) > 0 else GREEN_HDR

        sub = df_for_metrics[
            (df_for_metrics[col_article] == title) &
            (df_for_metrics[col_plf] == "F")
        ].copy()
        sub[col_value] = pd.to_numeric(sub[col_value], errors="coerce")

        best_pills  = ""
        worst_pills = ""

        if not sub.empty and col_tt in sub.columns and active_tt == "__ALL__":
            tt_totals = (
                sub.groupby(col_tt)[col_value]
                .sum().dropna().sort_values()
            )
            n = min(3, len(tt_totals))

            def make_pills(series, color, bg):
                pills = ""
                for tt, val in series.items():
                    sign    = "+" if val > 0 else ""
                    val_fmt = f"{val:,.0f}".replace(",", " ")
                    pills  += (
                        f'<span style="display:inline-block;background:{bg};color:{color};'
                        f'border-radius:4px;padding:2px 9px;margin:2px 3px;font-size:0.75rem;'
                        f'font-weight:600;white-space:nowrap;">'
                        f'{tt}&nbsp;<span style="opacity:.7;font-weight:400;">'
                        f'({sign}{val_fmt})</span></span>'
                    )
                return pills

            best_pills  = make_pills(tt_totals.head(n),          "#1b5e20", "#e8f5e9")
            worst_pills = make_pills(tt_totals.tail(n).iloc[::-1], "#7f0000", "#ffebee")

        delta_color = RED_LINE if total_delta > 0 else GREEN_HDR

        best_block = f"""
          <div style="flex:1;min-width:220px;">
            <div style="color:#888;font-size:0.71rem;margin-bottom:4px;text-transform:uppercase;
                        letter-spacing:.04em;">✅ Кращі магазини (мін. Fact)</div>
            <div>{best_pills if best_pills else
                  '<span style="color:#aaa;font-size:0.75rem;">немає даних</span>'}</div>
          </div>
          <div style="flex:1;min-width:220px;">
            <div style="color:#888;font-size:0.71rem;margin-bottom:4px;text-transform:uppercase;
                        letter-spacing:.04em;">❌ Гірші магазини (макс. Fact)</div>
            <div>{worst_pills if worst_pills else
                  '<span style="color:#aaa;font-size:0.75rem;">немає даних</span>'}</div>
          </div>
        """ if active_tt == "__ALL__" else ""

        metrics_html = f"""
        <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start;
                    background:#f9f6ff;border:1px solid #d0baf5;border-radius:6px;
                    padding:10px 16px;margin:6px 0 10px 0;">
          <div style="min-width:130px;">
            <div style="color:#888;font-size:0.71rem;margin-bottom:2px;text-transform:uppercase;
                        letter-spacing:.04em;">Серед. Fact / міс.</div>
            <div style="font-size:1.1rem;font-weight:700;color:{PURPLE};">{avg_monthly:,.0f}</div>
          </div>
          <div style="min-width:130px;">
            <div style="color:#888;font-size:0.71rem;margin-bottom:2px;text-transform:uppercase;
                        letter-spacing:.04em;">Δ Fact − Plan</div>
            <div style="font-size:1.1rem;font-weight:700;color:{delta_color};">
              {('+' if total_delta > 0 else '')}{total_delta:,.0f}
            </div>
          </div>
          <div style="min-width:100px;">
            <div style="color:#888;font-size:0.71rem;margin-bottom:2px;text-transform:uppercase;
                        letter-spacing:.04em;">% до плану</div>
            <div style="font-size:1.1rem;font-weight:700;color:{pct_color};">{pct_str}</div>
          </div>
          {best_block}
        </div>
        """
        st.markdown(metrics_html, unsafe_allow_html=True)

    # ── Plotly Chart ─────────────────────────────────────────────
    x_axis = [MONTH_LABELS[m] for m in range(1, 13)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x_axis, y=display_df["Plan"],    name="План",    marker_color=GREY))
    fig.add_trace(go.Bar(x=x_axis, y=display_df["Fact"],    name="Факт",    marker_color=PURPLE))
    fig.add_trace(go.Scatter(x=x_axis, y=display_df["Average"], name="Average",
                             line=dict(color=RED_LINE, width=3)))
    fig.add_trace(go.Scatter(x=x_axis, y=display_df["Delta"],   name="Дельта",
                             line=dict(color=YELLOW, dash="dot")))
    fig.update_layout(
        height=350, margin=dict(t=30, b=20, l=10, r=10),
        barmode="group", hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{article_idx}_{active_tt}")

    # ═══════════════════════════════════════════════════════════════
    # ═══════════════════════════════════════════════════════════════
    # TT SLICER — collapsible
    # ═══════════════════════════════════════════════════════════════
    if df_filtered is not None and col_tt is not None:
        available_tts = sorted(
        df_filtered[df_filtered[col_article] == title][col_tt].dropna().unique(),
        key=str
    )

    if available_tts:
        with st.expander("🏪 Слайсер по ТТ — клікни для деталізації", expanded=False):

            # Build options
            all_options = ["__ALL__"] + list(available_tts)

            CHUNK = 12
            chunks = [all_options[i:i + CHUNK] for i in range(0, len(all_options), CHUNK)]

            for chunk_idx, chunk in enumerate(chunks):
                cols = st.columns(len(chunk))
                for ci, tt_opt in enumerate(chunk):
                    label = "🔁 Всі" if tt_opt == "__ALL__" else str(tt_opt)
                    is_active = (active_tt == tt_opt)
                    btn_type = "primary" if is_active else "secondary"

                    with cols[ci]:
                        if st.button(
                            label,
                            key=f"slicer_{article_idx}_{chunk_idx}_{ci}_{hash(str(tt_opt))}",
                            type=btn_type,
                            use_container_width=True,
                        ):
                            st.session_state[skey] = tt_opt
                            st.rerun()

            # Hint
            if active_tt != "__ALL__":
                st.caption(f"📍 Показано тільки: **{active_tt}**")
            else:
                st.caption("Показано всі ТТ")
        


def export_excel(df, df_filtered, col_tt, col_article, col_month, col_value,
                 col_plf, articles_to_show, tt_val, group_factors, metric_col,
                 mode, pivot_df):
    import openpyxl
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.drawing.image import Image as XLImage

    NUM_FMT = '# ##0;-# ##0;-'

    def hdr_fill(h):
        h = h.lstrip("#")
        return PatternFill("solid", start_color=h, end_color=h)

    def thin_border():
        s = Side(style="thin", color="AAAAAA")
        return Border(left=s, right=s, top=s, bottom=s)

    def scw(ws, ci, w):
        ws.column_dimensions[get_column_letter(ci)].width = w

    month_labels_list = [MONTH_LABELS[m] for m in range(1, 13)]
    wb = Workbook()
    wb.remove(wb.active)

    # ── 1. Зведена таблиця ──────────────────────────────────────
    ws_p = wb.create_sheet("Зведена_таблиця")
    ws_p.freeze_panes = "B2"
    header = ["Стаття"] + month_labels_list + ["РАЗОМ"]
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
        tdf = build_article_monthly(df, df_filtered, col_tt, col_article,
                                    col_month, col_value, col_plf, article, tt_val, group_factors)
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
                if isinstance(v, (int, float)) and v < 0:
                    c.font = Font(name="Arial", size=9, color="C0392B")

    # ── 2. Листи по статтях з графіками ─────────────────────────
    row_labels = ["План", "Факт", "Average", "Дельта"]
    row_keys   = ["Plan", "Fact", "Average", "Delta"]
    row_fills  = ["FFFFFF", "e8d5f5", "fde8e8", "fff9e0"]
    row_colors = ["333333", "5b2d8e", "c0392b", "b8860b"]

    for article in articles_to_show:
        tdf  = build_article_monthly(df, df_filtered, col_tt, col_article,
                                     col_month, col_value, col_plf, article, tt_val, group_factors)
        safe = article[:28].replace("/", "_").replace("\\", "_")
        ws   = wb.create_sheet(safe)
        ws.freeze_panes = "B3"

        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=15)
        tc = ws.cell(row=1, column=1, value=article)
        tc.font = Font(bold=True, color="FFFFFF", name="Arial", size=12)
        tc.fill = hdr_fill("5b2d8e")
        tc.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 22

        headers = ["Показник"] + month_labels_list + ["РАЗОМ"]
        for ci, h in enumerate(headers, 1):
            c = ws.cell(row=2, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = hdr_fill("2e7d32")
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin_border()
        ws.row_dimensions[2].height = 18

        for ri, (label, key, fill_hex, color_hex) in enumerate(
                zip(row_labels, row_keys, row_fills, row_colors), 3):
            vals  = [tdf.loc[m, key] for m in range(1, 13)]
            total = sum(vals)
            nc = ws.cell(row=ri, column=1, value=label)
            nc.font = Font(bold=True, color=color_hex, name="Arial", size=9)
            nc.fill = hdr_fill(fill_hex)
            nc.border = thin_border()
            nc.alignment = Alignment(horizontal="left")
            for ci, v in enumerate(vals, 2):
                c = ws.cell(row=ri, column=ci, value=v)
                c.number_format = NUM_FMT
                c.fill = hdr_fill(fill_hex)
                c.border = thin_border()
                c.alignment = Alignment(horizontal="right")
                c.font = Font(name="Arial", size=9, color="C0392B" if v < 0 else color_hex)
            tc2 = ws.cell(row=ri, column=14, value=total)
            tc2.number_format = NUM_FMT
            tc2.fill = hdr_fill(fill_hex)
            tc2.border = thin_border()
            tc2.alignment = Alignment(horizontal="right")
            tc2.font = Font(bold=True, name="Arial", size=9,
                            color="C0392B" if total < 0 else color_hex)

        ws.column_dimensions["A"].width = 12
        for ci in range(2, 15):
            scw(ws, ci, 11)

        x = month_labels_list
        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=[tdf.loc[m, "Plan"] for m in range(1, 13)],
                             name="План", marker_color="#c0c0c0", opacity=0.9))
        fig.add_trace(go.Bar(x=x, y=[tdf.loc[m, "Fact"] for m in range(1, 13)],
                             name="Факт", marker_color="#5b2d8e", opacity=0.95))
        fig.add_trace(go.Scatter(x=x, y=[tdf.loc[m, "Average"] for m in range(1, 13)],
                                 mode="lines+markers", name="Average",
                                 line=dict(color="#c0392b", width=2.5), marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=x, y=[tdf.loc[m, "Delta"] for m in range(1, 13)],
                                 mode="lines+markers", name="Дельта",
                                 line=dict(color="#f0c000", width=2),
                                 marker=dict(size=7), yaxis="y2"))
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
            img_bytes  = pio.to_image(fig, format="png", scale=1.5)
            img_stream = io.BytesIO(img_bytes)
            img_obj    = XLImage(img_stream)
            img_obj.anchor = "A7"
            ws.add_image(img_obj)
        except Exception:
            ws.cell(row=7, column=1,
                    value="⚠️ Графік недоступний (встановіть kaleido: pip install kaleido)")

    # ── 3. Heatmap ───────────────────────────────────────────────
    if group_factors:
        ws_h = wb.create_sheet("Heatmap")
        ws_h.freeze_panes = "B2"
        heat_header = [col_tt] + month_labels_list + ["РАЗОМ"]
        for ci, h in enumerate(heat_header, 1):
            c = ws_h.cell(row=1, column=ci, value=h)
            c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
            c.fill = hdr_fill("5b2d8e")
            c.alignment = Alignment(horizontal="center", vertical="center")
            c.border = thin_border()

        df_num2 = df.copy()
        df_num2[col_value] = pd.to_numeric(df_num2[col_value], errors="coerce")
        global_avg_std2 = (
            df_num2[df_num2[col_plf] == "F"]
            .groupby(group_factors + [col_article], as_index=False)[col_value]
            .agg(Average_Calc="mean", Std="std")
        )
        data_heat2 = df_filtered[
            (df_filtered[col_plf] == "F") &
            (df_filtered[col_article].isin(articles_to_show))
        ].copy()
        data_heat2["_m"] = get_month_num(data_heat2[col_month])
        tt_grp2 = list(dict.fromkeys([col_tt] + group_factors + ["_m", col_article]))
        tt_table2 = (
            data_heat2.groupby(tt_grp2, as_index=False)[col_value]
            .sum().rename(columns={col_value: "Fact"})
        )
        merge_cols2 = list(dict.fromkeys(group_factors + [col_article]))
        tt_table2 = pd.merge(tt_table2, global_avg_std2, on=merge_cols2, how="left")
        tt_table2["Delta"]   = tt_table2["Fact"] - tt_table2["Average_Calc"]
        tt_table2["Delta_%"] = tt_table2["Delta"] / tt_table2["Average_Calc"].replace(0, np.nan)
        tt_table2["Z"]       = tt_table2["Delta"] / tt_table2["Std"].replace(0, np.nan)

        vc = {"Delta": "Delta", "Delta %": "Delta_%", "Z-score": "Z",
              "Fact": "Fact", "Average": "Average_Calc"}.get(mode, "Delta")

        heat2 = tt_table2.pivot_table(index=col_tt, columns="_m", values=vc, aggfunc="sum")
        for m in range(1, 13):
            if m not in heat2.columns:
                heat2[m] = None
        heat2 = heat2[sorted(heat2.columns)]
        heat2.columns = [MONTH_LABELS.get(int(c), str(c)) for c in heat2.columns]
        heat2["РАЗОМ"] = heat2.sum(axis=1, numeric_only=True)

        try:
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            cmap_obj = plt.get_cmap("RdYlGn_r")
            flat = heat2.values.flatten().astype(float)
            flat = flat[~np.isnan(flat)]
            norm = mcolors.Normalize(vmin=float(np.nanmin(flat)), vmax=float(np.nanmax(flat)))
            use_cmap = True
        except Exception:
            use_cmap = False

        for ri, (idx, row) in enumerate(heat2.iterrows(), 2):
            c0 = ws_h.cell(row=ri, column=1, value=idx)
            c0.border = thin_border()
            c0.font = Font(name="Arial", size=9)
            for ci, col_name in enumerate(heat2.columns, 2):
                v = row[col_name]
                c = ws_h.cell(row=ri, column=ci)
                c.border = thin_border()
                c.alignment = Alignment(horizontal="right")
                if pd.isna(v):
                    c.value = None
                    c.fill = hdr_fill("FFFFFF")
                else:
                    c.value = float(v)
                    c.number_format = NUM_FMT
                    if use_cmap:
                        rgba = cmap_obj(norm(float(v)))
                        hx = "{:02X}{:02X}{:02X}".format(
                            int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
                        c.fill = hdr_fill(hx)
                    c.font = Font(name="Arial", size=9, color="000000")

        ws_h.column_dimensions["A"].width = 20
        for ci in range(2, len(heat_header) + 1):
            scw(ws_h, ci, 11)

        # ── 4. TOP / ANTITOP ─────────────────────────────────────
        ws_top     = wb.create_sheet("TOP_ANTITOP")
        sum_val2   = tt_table2.groupby(col_tt)[vc].sum().reset_index()
        top_df     = sum_val2.sort_values(vc, ascending=True).head(50)
        antitop_df = sum_val2.sort_values(vc, ascending=False).head(50)

        def write_block(start_col, title_text, df_block, cmap_name):
            tc = ws_top.cell(row=1, column=start_col, value=title_text)
            tc.font = Font(bold=True, color="FFFFFF", name="Arial", size=11)
            tc.fill = hdr_fill("5b2d8e")
            ws_top.merge_cells(start_row=1, start_column=start_col,
                               end_row=1, end_column=start_col + 1)
            for ci2, h2 in enumerate([col_tt, vc], start_col):
                c = ws_top.cell(row=2, column=ci2, value=h2)
                c.font = Font(bold=True, color="FFFFFF", name="Arial", size=9)
                c.fill = hdr_fill("2e7d32")
                c.border = thin_border()
            try:
                import matplotlib.pyplot as plt
                import matplotlib.colors as mcolors
                vals_arr = df_block[vc].values.astype(float)
                nm = mcolors.Normalize(vmin=float(np.nanmin(vals_arr)),
                                       vmax=float(np.nanmax(vals_arr)))
                cm2     = plt.get_cmap(cmap_name)
                use_c   = True
            except Exception:
                use_c = False
            for ri2, row2 in enumerate(df_block.itertuples(index=False), 3):
                tt_v  = getattr(row2, col_tt, "")
                val_v = getattr(row2, vc, 0)
                ws_top.cell(row=ri2, column=start_col, value=tt_v).border = thin_border()
                ws_top.cell(row=ri2, column=start_col).font = Font(name="Arial", size=9)
                c2 = ws_top.cell(row=ri2, column=start_col + 1,
                                  value=float(val_v) if pd.notna(val_v) else None)
                c2.number_format = NUM_FMT
                c2.border = thin_border()
                c2.alignment = Alignment(horizontal="right")
                if use_c and pd.notna(val_v):
                    rgba2 = cm2(nm(float(val_v)))
                    hx2 = "{:02X}{:02X}{:02X}".format(
                        int(rgba2[0]*255), int(rgba2[1]*255), int(rgba2[2]*255))
                    c2.fill = hdr_fill(hx2)
                    r2, g2, b2   = int(rgba2[0]*255), int(rgba2[1]*255), int(rgba2[2]*255)
                    luminance    = (0.299*r2 + 0.587*g2 + 0.114*b2) / 255
                    font_color   = "000000" if luminance > 0.5 else "FFFFFF"
                    c2.font = Font(name="Arial", size=9, color=font_color)
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


def main():
    st.set_page_config(page_title="СІМІ Dashboard", layout="wide")
    st.markdown("""
    <style>
    .simi-header {
        background: linear-gradient(90deg, #5b2d8e 0%, #7b52ae 100%);
        color: white; padding: 10px 18px; border-radius: 4px; margin-bottom: 4px;
    }
    .simi-logo  { font-size:2rem; font-weight:900; color:#f0c000;
                  letter-spacing:1px; margin-right:24px; vertical-align:middle; }
    .simi-store { font-size:1.1rem; font-weight:700; color:white; vertical-align:middle; }
    .simi-meta-grid { display:grid; grid-template-columns:repeat(6,1fr);
                      gap:4px 12px; margin-top:6px; }
    .simi-meta-item { font-size:0.78rem; color:#e0d0f8; }
    .simi-meta-val  { font-size:0.85rem; font-weight:700; color:white; }
    .article-selector {
        background:#f4f0fa; border:2px solid #5b2d8e;
        border-radius:8px; padding:12px 16px; margin-bottom:14px;
    }
    .block-sep { border-top:2px solid #5b2d8e; margin:16px 0 10px 0; }

    /* Slicer button tweaks */
    div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #5b2d8e !important;
    color: white !important;
    border: 2px solid #5b2d8e !important;
    font-weight: 700 !important;
    font-size: 0.60rem !important;
    padding: 0.25rem 0.4rem !important;
    line-height: 1.1 !important;
}

div[data-testid="stButton"] > button[kind="secondary"] {
    background-color: #f4f0fa !important;
    color: #5b2d8e !important;
    border: 1px solid #c9b6e8 !important;
    font-size: 0.60rem !important;
    padding: 0.25rem 0.4rem !important;
    line-height: 1.1 !important;
}

div[data-testid="stButton"] > button:hover {
    transform: scale(0.98);
}
    }
    div[data-testid="stButton"] > button[kind="secondary"]:hover {
        background-color: #e8d5f5 !important;
        border-color: #5b2d8e !important;
    }
    </style>
    """, unsafe_allow_html=True)

    file = st.file_uploader("📂 Завантажте Excel", type=["xlsx", "xlsb"])
    if file is None:
        st.info("Завантажте Excel-файл для початку роботи.")
        st.stop()

    xl         = pd.ExcelFile(file)
    sheet_name = st.selectbox("Аркуш", xl.sheet_names)
    df         = load_excel(file, sheet_name)
    cols       = df.columns.tolist()

    with st.expander("⚙️ Налаштування колонок", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            col_tt      = st.selectbox("TT (Магазин)",   cols)
            col_year    = st.selectbox("Year",            cols)
        with c2:
            col_month   = st.selectbox("Month",          cols)
            col_value   = st.selectbox("Значення",       cols)
        with c3:
            col_plf     = st.selectbox("PL / F",         cols)
            col_article = st.selectbox("Стаття бюджету", cols)
        with c4:
            col_level0  = st.selectbox("Level_0",        cols)

    with st.expander("🏪 Колонки шапки магазину", expanded=False):
        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            col_city   = st.selectbox("Місто",          ["—"] + cols)
            col_area   = st.selectbox("Площа",          ["—"] + cols)
        with sh2:
            col_format = st.selectbox("Формат ТО",     ["—"] + cols)
            col_mega   = st.selectbox("Мегасегмент",   ["—"] + cols)
        with sh3:
            col_rik    = st.selectbox("Рік",            ["—"] + cols)
            col_mis    = st.selectbox("Місяць (шапка)", ["—"] + cols)

    # ── Sidebar filters ──────────────────────────────────────────
    st.sidebar.markdown("## 🔍 Фільтри")
    year_val   = st.sidebar.multiselect("Year",    sorted(df[col_year].dropna().unique(),   key=str))
    month_val  = st.sidebar.multiselect("Month",   sorted(df[col_month].dropna().unique(),  key=str))
    level0_val = st.sidebar.multiselect("Level_0", sorted(df[col_level0].dropna().unique(), key=str))

    st.sidebar.markdown("### ➕ Додаткові фільтри")
    fixed_cols = {col_tt, col_year, col_month, col_level0}
    extra_col  = st.sidebar.selectbox(
        "Стовпець для фільтру",
        ["— не обрано —"] + [c for c in cols if c not in fixed_cols],
        key="extra_filter_col",
    )
    extra_filters = {}
    if extra_col != "— не обрано —":
        extra_val = st.sidebar.multiselect(
            f"Значення «{extra_col}»",
            sorted(df[extra_col].dropna().unique(), key=str),
            key="extra_filter_val",
        )
        if extra_val:
            extra_filters[extra_col] = extra_val
    remaining_cols = [c for c in cols if c not in fixed_cols and c != extra_col]
    if extra_col != "— не обрано —":
        extra_col2 = st.sidebar.selectbox(
            "Ще стовпець", ["— не обрано —"] + remaining_cols, key="extra_filter_col2",
        )
        if extra_col2 != "— не обрано —":
            extra_val2 = st.sidebar.multiselect(
                f"Значення «{extra_col2}»",
                sorted(df[extra_col2].dropna().unique(), key=str),
                key="extra_filter_val2",
            )
            if extra_val2:
                extra_filters[extra_col2] = extra_val2
            remaining_cols2 = [c for c in remaining_cols if c != extra_col2]
            extra_col3 = st.sidebar.selectbox(
                "Ще стовпець", ["— не обрано —"] + remaining_cols2, key="extra_filter_col3",
            )
            if extra_col3 != "— не обрано —":
                extra_val3 = st.sidebar.multiselect(
                    f"Значення «{extra_col3}»",
                    sorted(df[extra_col3].dropna().unique(), key=str),
                    key="extra_filter_val3",
                )
                if extra_val3:
                    extra_filters[extra_col3] = extra_val3

    # ── Dynamic TT list ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏪 ТТ за поточними фільтрами")

    df_pre = df.copy()
    if year_val:
        df_pre = df_pre[df_pre[col_year].isin(year_val)]
    if month_val:
        df_pre = df_pre[df_pre[col_month].isin(month_val)]
    if level0_val:
        df_pre = df_pre[df_pre[col_level0].isin(level0_val)]
    for col_e, vals_e in extra_filters.items():
        df_pre = df_pre[df_pre[col_e].isin(vals_e)]

    visible_tts = sorted(df_pre[col_tt].dropna().unique(), key=str)

    if visible_tts:
        st.sidebar.caption(f"Знайдено: {len(visible_tts)} магазинів")
        tt_search = st.sidebar.text_input(
            "🔎 Пошук ТТ", value="", placeholder="Введіть назву...", key="tt_search"
        )
        filtered_tts = (
            [tt for tt in visible_tts if tt_search.lower() in str(tt).lower()]
            if tt_search else visible_tts
        )
        btn_col1, btn_col2 = st.sidebar.columns(2)
        with btn_col1:
            if st.button("✅ Всі", key="tt_select_all", use_container_width=True):
                st.session_state["tt_multiselect"] = filtered_tts
        with btn_col2:
            if st.button("✖ Жодного", key="tt_clear_all", use_container_width=True):
                st.session_state["tt_multiselect"] = []

        tt_val = st.sidebar.multiselect(
            "Оберіть ТТ:",
            options=filtered_tts,
            default=st.session_state.get("tt_multiselect", []),
            key="tt_multiselect",
        )
    else:
        st.sidebar.warning("Немає ТТ за обраними фільтрами.")
        tt_val = []

    st.sidebar.markdown("---")

    mode = st.sidebar.selectbox(
        "Mode (Heatmap)", ["Delta", "Delta %", "Z-score", "Fact", "Average"]
    )
    group_factors = st.sidebar.multiselect(
        "Фактори групування (Average/Std)",
        options=[c for c in df.columns if c not in [col_value, col_plf, col_article]],
        default=[col_tt] if col_tt in df.columns else [],
    )

    df[col_value] = pd.to_numeric(df[col_value], errors="coerce")

    def apply_filters(d):
        if tt_val:     d = d[d[col_tt].isin(tt_val)]
        if year_val:   d = d[d[col_year].isin(year_val)]
        if month_val:  d = d[d[col_month].isin(month_val)]
        if level0_val: d = d[d[col_level0].isin(level0_val)]
        for col, vals in extra_filters.items():
            d = d[d[col].isin(vals)]
        return d

    df_filtered = apply_filters(df).copy()
    df_store    = df_filtered

    def get_meta(col):
        if col == "—":
            return "—"
        vals = df_store[col].dropna().unique()
        return str(vals[0]) if len(vals) else "—"

    articles_all = sorted(df[col_article].dropna().unique(), key=str)
    st.markdown('<div class="article-selector">', unsafe_allow_html=True)
    sel_col1, sel_col2, sel_col3 = st.columns([3, 1, 1])
    with sel_col1:
        st.markdown("**🎯 Стаття бюджету для аналізу** — обери одну статтю або увімкни «Всі»")
        selected_article = st.selectbox(
            "article_selector", articles_all, key="global_article",
            label_visibility="collapsed",
        )
    with sel_col2:
        st.markdown("&nbsp;")
        show_all = st.checkbox("Показати всі статті", value=False, key="show_all")
    with sel_col3:
        st.markdown("&nbsp;")
        multi_sel = st.multiselect(
            "Або обери кілька:", articles_all, default=[], key="multi_article",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    if show_all:
        articles_to_show = articles_all
    elif multi_sel:
        articles_to_show = multi_sel
    else:
        articles_to_show = [selected_article]

    if len(articles_to_show) == 1:
        st.info(f"📌 Показується стаття: **{articles_to_show[0]}**")
    else:
        st.info(f"📌 Показується {len(articles_to_show)} статей: {', '.join(articles_to_show)}")

    store_name = ", ".join(str(v) for v in tt_val) if tt_val else "Всі магазини"
    st.markdown(f"""
    <div class="simi-header">
      <span class="simi-logo">СіМі</span>
      <span class="simi-store">{store_name}</span>
      <div class="simi-meta-grid">
        <div><span class="simi-meta-item">Місто </span>
             <span class="simi-meta-val">{get_meta(col_city)}</span></div>
        <div><span class="simi-meta-item">Площа </span>
             <span class="simi-meta-val">{get_meta(col_area)}</span></div>
        <div><span class="simi-meta-item">Формат ТО </span>
             <span class="simi-meta-val">{get_meta(col_format)}</span></div>
        <div><span class="simi-meta-item">Мегасегмент </span>
             <span class="simi-meta-val">{get_meta(col_mega)}</span></div>
        <div><span class="simi-meta-item">Рік </span>
             <span class="simi-meta-val">{get_meta(col_rik)}</span></div>
        <div><span class="simi-meta-item">Місяць </span>
             <span class="simi-meta-val">{get_meta(col_mis)}</span></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Article blocks ───────────────────────────────────────────
    for art_idx, article in enumerate(articles_to_show):
        st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
        tdf = build_article_monthly(
            df, df_filtered, col_tt, col_article,
            col_month, col_value, col_plf, article, tt_val, group_factors
        )
        render_article_block(
            title=article,
            table_df=tdf,
            chart_title=f"Аналіз середньомісячного показника — {article}",
            df=df,
            df_filtered=df_filtered,
            col_tt=col_tt,
            col_article=col_article,
            col_month=col_month,
            col_value=col_value,
            col_plf=col_plf,
            group_factors=group_factors,
            tt_val=tt_val,
            article_idx=art_idx,
        )

    # ── Pivot table ──────────────────────────────────────────────
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("📋 Зведена таблиця")
    pivot_metric = st.radio("Метрика", ["Fact", "Plan", "Delta (Fact-Plan)"], horizontal=True)
    col_map_d    = {"Fact": "Fact", "Plan": "Plan", "Delta (Fact-Plan)": "Delta"}
    metric_col   = col_map_d[pivot_metric]

    rows_pivot = []
    for article in articles_to_show:
        tdf = build_article_monthly(df, df_filtered, col_tt, col_article,
                                    col_month, col_value, col_plf, article, tt_val, group_factors)
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
            .format(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-", na_rep="-"),
        use_container_width=True,
    )

    # ── TT pivot ─────────────────────────────────────────────────
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("📋 Зведена таблиця в розрізі ТТ")

    tt_pivot_metric = st.radio(
        "Метрика (ТТ)", ["Fact", "Plan", "Delta (Fact-Plan)"], horizontal=True,
        key="tt_pivot_metric"
    )
    tt_metric_col = col_map_d[tt_pivot_metric]

    show_pct    = st.checkbox("Показати % відхилення (Fact vs Plan)", value=True, key="show_pct")
    show_months = st.checkbox("Розгорнути по місяцях", value=False, key="tt_show_months")

    df_num_tt = df_filtered.copy()
    df_num_tt[col_value] = pd.to_numeric(df_num_tt[col_value], errors="coerce")
    df_num_tt["_m"] = get_month_num(df_num_tt[col_month])

    all_tts = sorted(df_filtered[col_tt].dropna().unique(), key=str)

    rows_tt = []
    for tt in all_tts:
        sub = df_num_tt[df_num_tt[col_tt] == tt]
        for article in articles_to_show:
            sub_a  = sub[sub[col_article] == article]
            plan_m = sub_a[sub_a[col_plf] == "PL"].groupby("_m")[col_value].sum()
            fact_m = sub_a[sub_a[col_plf] == "F"].groupby("_m")[col_value].sum()
            row    = {"ТТ": tt, "Стаття": article}
            for m in range(1, 13):
                row[f"plan_{MONTH_LABELS[m]}"] = plan_m.get(m, 0)
                row[f"fact_{MONTH_LABELS[m]}"] = fact_m.get(m, 0)
            row["Plan_РАЗОМ"]  = sum(plan_m.get(m, 0) for m in range(1, 13))
            row["Fact_РАЗОМ"]  = sum(fact_m.get(m, 0) for m in range(1, 13))
            row["Delta_РАЗОМ"] = row["Fact_РАЗОМ"] - row["Plan_РАЗОМ"]
            row["Pct_РАЗОМ"]   = (
                (row["Fact_РАЗОМ"] / row["Plan_РАЗОМ"] - 1) * 100
                if row["Plan_РАЗОМ"] != 0 else None
            )
            rows_tt.append(row)

    df_tt_pivot = pd.DataFrame(rows_tt)

    if df_tt_pivot.empty:
        st.info("Немає даних для побудови таблиці по ТТ.")
    else:
        agg_cols_plan = [f"plan_{MONTH_LABELS[m]}" for m in range(1, 13)]
        agg_cols_fact = [f"fact_{MONTH_LABELS[m]}" for m in range(1, 13)]
        numeric_cols  = agg_cols_plan + agg_cols_fact + ["Plan_РАЗОМ", "Fact_РАЗОМ", "Delta_РАЗОМ"]

        df_tt_agg = df_tt_pivot.groupby("ТТ")[numeric_cols].sum().reset_index()
        df_tt_agg["Pct_РАЗОМ"] = df_tt_agg.apply(
            lambda r: (r["Fact_РАЗОМ"] / r["Plan_РАЗОМ"] - 1) * 100
            if r["Plan_РАЗОМ"] != 0 else None, axis=1
        )
        for m in range(1, 13):
            ml = MONTH_LABELS[m]
            df_tt_agg[f"delta_{ml}"] = df_tt_agg[f"fact_{ml}"] - df_tt_agg[f"plan_{ml}"]
            df_tt_agg[f"pct_{ml}"] = df_tt_agg.apply(
                lambda r, ml=ml: (r[f"fact_{ml}"] / r[f"plan_{ml}"] - 1) * 100
                if r[f"plan_{ml}"] != 0 else None, axis=1
            )

        if show_months:
            display_cols = ["ТТ"]
            col_labels   = {"ТТ": "ТТ"}
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
            col_labels   = {"ТТ": "ТТ"}

        if tt_metric_col == "Fact":
            display_cols += ["Fact_РАЗОМ"]
            col_labels["Fact_РАЗОМ"] = "Fact РАЗОМ"
        elif tt_metric_col == "Plan":
            display_cols += ["Plan_РАЗОМ"]
            col_labels["Plan_РАЗОМ"] = "Plan РАЗОМ"
        else:
            display_cols += ["Fact_РАЗОМ", "Plan_РАЗОМ", "Delta_РАЗОМ"]
            col_labels.update({
                "Fact_РАЗОМ":  "Fact РАЗОМ",
                "Plan_РАЗОМ":  "Plan РАЗОМ",
                "Delta_РАЗОМ": "Δ РАЗОМ",
            })

        if show_pct:
            display_cols.append("Pct_РАЗОМ")
            col_labels["Pct_РАЗОМ"] = "% відхил."

        df_display = df_tt_agg[display_cols].rename(columns=col_labels).set_index("ТТ")

        sort_col_label = "% відхил." if show_pct else (
            "Δ РАЗОМ"    if tt_metric_col == "Delta (Fact-Plan)" else
            "Fact РАЗОМ" if tt_metric_col == "Fact" else "Plan РАЗОМ"
        )
        if sort_col_label in df_display.columns:
            df_display = df_display.sort_values(sort_col_label, ascending=True)

        total_row = df_display.sum(numeric_only=True)
        if "% відхил." in df_display.columns:
            plan_sum = df_tt_agg["Plan_РАЗОМ"].sum()
            fact_sum = df_tt_agg["Fact_РАЗОМ"].sum()
            total_row["% відхил."] = (
                (fact_sum / plan_sum - 1) * 100 if plan_sum != 0 else None
            )
        total_row.name = "🟰 РАЗОМ"
        df_display = pd.concat([df_display, total_row.to_frame().T])

        pct_cols = [c for c in df_display.columns if "%" in c]
        num_cols = [c for c in df_display.columns if "%" not in c]

        def fmt_num(v):
            if pd.isna(v): return "-"
            return f"{v:,.0f}".replace(",", " ")

        def fmt_pct(v):
            if pd.isna(v): return "-"
            sign = "+" if v > 0 else ""
            return f"{sign}{v:.1f}%"

        fmt_dict = {c: fmt_num for c in num_cols}
        fmt_dict.update({c: fmt_pct for c in pct_cols})

        styled = df_display.style.format(fmt_dict, na_rep="-")

        if pct_cols:
            styled = styled.background_gradient(
                cmap="RdYlGn_r",
                subset=pd.IndexSlice[df_display.index[:-1], pct_cols], axis=None
            )
        delta_num_cols = [c for c in num_cols if "Δ" in c]
        other_num_cols = [c for c in num_cols if "Δ" not in c]
        if delta_num_cols:
            styled = styled.background_gradient(
                cmap="RdYlGn_r",
                subset=pd.IndexSlice[df_display.index[:-1], delta_num_cols], axis=None
            )
        if other_num_cols:
            styled = styled.background_gradient(
                cmap="Blues",
                subset=pd.IndexSlice[df_display.index[:-1], other_num_cols], axis=None
            )

        styled = styled.apply(
            lambda row: [
                "font-weight:bold; border-top:2px solid #5b2d8e;" for _ in row
            ] if row.name == "🟰 РАЗОМ" else ["" for _ in row],
            axis=1,
        )

        st.dataframe(styled, use_container_width=True, height=500)

        csv_bytes = df_display.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "⬇️ Завантажити CSV (ТТ-зведена)",
            data=csv_bytes,
            file_name="tt_pivot.csv",
            mime="text/csv",
            key="tt_pivot_csv",
        )

    # ── Anomaly heatmap ──────────────────────────────────────────
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("🌡️ Карта аномалій по магазинах")

    if group_factors:
        df_num = df.copy()
        df_num[col_value] = pd.to_numeric(df_num[col_value], errors="coerce")
        global_avg_std = (
            df_num[df_num[col_plf] == "F"]
            .groupby(group_factors + [col_article], as_index=False)[col_value]
            .agg(Average_Calc="mean", Std="std")
        )
        data_heat = df_filtered[
            (df_filtered[col_plf] == "F") &
            (df_filtered[col_article].isin(articles_to_show))
        ].copy()
        data_heat["_m"] = get_month_num(data_heat[col_month])
        tt_grp = list(dict.fromkeys([col_tt] + group_factors + ["_m", col_article]))
        tt_table = (
            data_heat.groupby(tt_grp, as_index=False)[col_value]
            .sum().rename(columns={col_value: "Fact"})
        )
        merge_cols = list(dict.fromkeys(group_factors + [col_article]))
        tt_table = pd.merge(tt_table, global_avg_std, on=merge_cols, how="left")
        tt_table["Delta"]   = tt_table["Fact"] - tt_table["Average_Calc"]
        tt_table["Delta_%"] = tt_table["Delta"] / tt_table["Average_Calc"].replace(0, np.nan)
        tt_table["Z"]       = tt_table["Delta"] / tt_table["Std"].replace(0, np.nan)

        val_col = {
            "Delta": "Delta", "Delta %": "Delta_%",
            "Z-score": "Z", "Fact": "Fact", "Average": "Average_Calc",
        }[mode]

        heat = tt_table.pivot_table(index=col_tt, columns="_m", values=val_col, aggfunc="sum")
        for m in range(1, 13):
            if m not in heat.columns:
                heat[m] = None
        heat = heat[sorted(heat.columns)]
        heat.columns = [MONTH_LABELS.get(int(c), str(c)) for c in heat.columns]
        heat["РАЗОМ"] = heat.sum(axis=1, numeric_only=True)

        st.dataframe(
            heat.style
                .background_gradient(cmap="RdYlGn_r", axis=None)
                .highlight_null(color="white")
                .format(lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "",
                        na_rep=""),
            use_container_width=True,
        )

        st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
        st.subheader("🏆 TOP / ANTITOP магазинів")
        sum_val = tt_table.groupby(col_tt)[val_col].sum().reset_index()
        n_tt    = st.slider("Кількість магазинів", 1, 100, 10)
        top     = sum_val.sort_values(val_col, ascending=True).head(n_tt)
        antitop = sum_val.sort_values(val_col, ascending=False).head(n_tt)
        ca, cb  = st.columns(2)
        with ca:
            st.write("✅ Top (економія)")
            st.dataframe(
                top.style
                   .background_gradient(cmap="RdYlGn", subset=[val_col])
                   .format({val_col: lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-"})
            )
        with cb:
            st.write("❌ Antitop (переліміт)")
            st.dataframe(
                antitop.style
                       .background_gradient(cmap="RdYlGn_r", subset=[val_col])
                       .format({val_col: lambda v: f"{v:,.0f}".replace(",", " ") if pd.notna(v) else "-"})
            )
    else:
        st.info("Оберіть фактори групування в боковому меню для побудови Heatmap.")

    # ── Export ───────────────────────────────────────────────────
    st.markdown('<div class="block-sep"></div>', unsafe_allow_html=True)
    st.subheader("📥 Експорт в Excel")
    st.download_button(
        label="⬇️ Скачати дашборд як Excel",
        data=export_excel(
            df, df_filtered, col_tt, col_article, col_month, col_value,
            col_plf, articles_to_show, tt_val, group_factors, metric_col,
            mode, pivot_df
        ),
        file_name="simi_dashboard.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()
