"""
Расширенная аналитика: устойчивое потребление.
Читает document.csv, выполняет 10 блоков анализа,
генерирует единый report.html.
"""
import io
import base64
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

FREQ_MAP = {"Никогда": 1, "Редко": 2, "Иногда": 3, "Часто": 4, "Постоянно": 5}
PALETTE = ["#4e79a7", "#e15759", "#59a14f", "#f28e2b", "#76b7b2",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def score_freq(series: pd.Series) -> pd.Series:
    def _map(x):
        if pd.isna(x):
            return np.nan
        for key, val in FREQ_MAP.items():
            if key in str(x):
                return val
        return np.nan
    return series.map(_map)


def cramers_v(ct: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))) if min(r, k) > 1 else 0.0


def rank_biserial(u: float, n1: int, n2: int) -> float:
    return 1.0 - (2.0 * u) / (n1 * n2)


def epsilon_squared(h: float, n: int) -> float:
    return h / (n - 1)


def effect_label(val: float, thresholds: tuple) -> str:
    if abs(val) < thresholds[0]:
        return "малый"
    elif abs(val) < thresholds[1]:
        return "средний"
    return "большой"


def ols_regression(y_arr, X_arr, feature_names):
    """OLS regression returning dict with all statistics."""
    y_arr = np.asarray(y_arr, dtype=float)
    X_arr = np.asarray(X_arr, dtype=float)
    n, p = X_arr.shape
    X_i = np.column_stack([np.ones(n), X_arr])
    XtX = X_i.T @ X_i
    Xty = X_i.T @ y_arr
    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(X_i, y_arr, rcond=None)[0]
    y_hat = X_i @ beta
    resid = y_arr - y_hat
    sse = float(resid @ resid)
    df_resid = max(n - p - 1, 1)
    mse = sse / df_resid
    try:
        cov = mse * np.linalg.inv(XtX)
    except Exception:
        cov = mse * np.linalg.pinv(XtX)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_vals = beta / np.where(se > 0, se, np.nan)
    p_vals = 2 * (1 - stats.t.cdf(np.abs(t_vals), df_resid))
    ci_lo = beta - 1.96 * se
    ci_hi = beta + 1.96 * se
    ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
    r_sq = 1 - sse / ss_tot if ss_tot > 0 else 0
    r_sq_adj = 1 - (1 - r_sq) * (n - 1) / df_resid
    f_stat = ((ss_tot - sse) / max(p, 1)) / mse if mse > 0 else 0
    f_p = 1 - stats.f.cdf(f_stat, p, df_resid)
    return {
        "names": ["(Intercept)"] + list(feature_names),
        "betas": beta, "se": se, "t": t_vals, "p": p_vals,
        "ci_lo": ci_lo, "ci_hi": ci_hi,
        "r_sq": r_sq, "r_sq_adj": r_sq_adj,
        "f_stat": f_stat, "f_p": f_p, "n": n, "df_resid": df_resid,
    }


# ---------------------------------------------------------------------------
# Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    for enc in ("cp1251", "utf-8"):
        try:
            df = pd.read_csv("document.csv", sep=";", encoding=enc)
            if df.shape[1] >= 10:
                break
        except Exception:
            continue

    col_names = [
        "Timestamp", "age", "gender", "education", "living", "city", "income",
        "q2_1_waste", "q2_2_eco", "q2_3_energy",
        "q3_1_lamp", "q3_2_contribution", "q3_3_store",
        "q3_4_consumption", "q3_5_logic",
        "q4_1_friends", "q4_2_neighbors", "q4_3_opinion",
        "q5_1_barriers",
    ]
    if df.shape[1] >= len(col_names):
        df.columns = col_names + list(df.columns[len(col_names):])
    else:
        df.columns = col_names[: df.shape[1]]

    df["q2_1_num"] = score_freq(df["q2_1_waste"])
    df["q2_2_num"] = score_freq(df["q2_2_eco"])
    df["q2_3_num"] = score_freq(df["q2_3_energy"])
    df["sustainable_index"] = df["q2_1_num"] + df["q2_2_num"] + df["q2_3_num"]

    df["generation"] = df["age"].apply(
        lambda x: "zoom" if any(t in str(x) for t in ("18-22", "23-25"))
        else ("millennial" if any(t in str(x) for t in ("26-28", "29-35", "35+")) else "other")
    )

    def _store(x):
        s = str(x)
        if "Нет" in s and "удобство" in s:
            return 1
        if "Может быть" in s:
            return 2
        if "Да" in s and "если" in s:
            return 3
        if "Да" in s and "готов" in s:
            return 4
        return np.nan
    df["q3_3_num"] = df["q3_3_store"].apply(_store)

    def _friends(x):
        s = str(x)
        if "Почти никто" in s: return 1
        if "Немного" in s: return 2
        if "Половина" in s: return 3
        if "Большинство" in s: return 4
        if "Почти все" in s: return 5
        return np.nan
    df["q4_1_num"] = df["q4_1_friends"].apply(_friends)

    def _neigh(x):
        s = str(x)
        if "Очень маловероятно" in s: return 1
        if "Скорее нет" in s: return 2
        if "Нейтрально" in s or "затрудняюсь" in s: return 3
        if "Скорее да" in s: return 4
        if "Очень вероятно" in s or "все делают" in s: return 5
        return np.nan
    df["q4_2_num"] = df["q4_2_neighbors"].apply(_neigh)

    def _opinion(x):
        s = str(x)
        if "Не важно" in s or "только для себя" in s: return 1
        if "Слабо" in s: return 2
        if "Нейтрально" in s: return 3
        if "Довольно важно" in s: return 4
        if "Очень важно" in s or "тренде" in s: return 5
        return np.nan
    df["q4_3_num"] = df["q4_3_opinion"].apply(_opinion)

    df["barrier_infra"] = df["q5_1_barriers"].str.contains(
        "инфраструктур", case=False, na=False).astype(int)
    df["barrier_lazy"] = df["q5_1_barriers"].str.contains(
        "Лень|нет времени|забываю", case=False, na=False).astype(int)
    df["barrier_unbelief"] = df["q5_1_barriers"].str.contains(
        "Не верю", na=False).astype(int)
    df["barrier_inconvenient"] = df["q5_1_barriers"].str.contains(
        "неудобно|специальные магазины", case=False, na=False).astype(int)
    df["barrier_expense"] = df["q5_1_barriers"].str.contains(
        "дорого|нет денег", case=False, na=False).astype(int)
    df["barrier_nothink"] = df["q5_1_barriers"].str.contains(
        "не думаю", case=False, na=False).astype(int)

    df["q3_1_led"] = df["q3_1_lamp"].str.contains("LED|Опция Б", na=False).astype(int)
    df["q3_2_agree"] = df["q3_2_contribution"].str.contains("Согласен", na=False).astype(int)

    df["city_msk"] = df["city"].str.contains(
        "Москва|Санкт-Петербург", case=False, na=False).astype(int)

    return df


# ============================================================================
# СЕКЦИИ ОТЧЁТА
# ============================================================================

def section_effect_sizes(df: pd.DataFrame) -> str:
    practices = [("q2_1_waste", "q2_1_num", "Сортировка"),
                 ("q2_2_eco", "q2_2_num", "Эко-товары"),
                 ("q2_3_energy", "q2_3_num", "Экономия")]
    factors = [("age", "Возраст"), ("income", "Доход"), ("education", "Образование")]

    rows_chi = []
    for fcol, flabel in factors:
        for pcol, _, plabel in practices:
            ct = pd.crosstab(df[fcol], df[pcol])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            v = cramers_v(ct.values)
            eff = effect_label(v, (0.10, 0.30))
            rows_chi.append((f"{flabel} x {plabel}", f"{chi2:.1f}", dof, f"{p:.4f}", f"{v:.3f}", eff))

    df_gen = df[df["generation"].isin(["zoom", "millennial"])]
    rows_mw = []
    for _, ncol, plabel in practices:
        x = df_gen.loc[df_gen["generation"] == "zoom", ncol].dropna()
        y = df_gen.loc[df_gen["generation"] == "millennial", ncol].dropna()
        u, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        r = rank_biserial(u, len(x), len(y))
        eff = effect_label(r, (0.10, 0.30))
        rows_mw.append((f"Зумеры vs Миллениалы: {plabel}", f"{u:.0f}", f"{p:.4f}", f"{r:.3f}", eff))

    rows_kw = []
    for fcol, flabel in [("income", "Доход"), ("education", "Образование")]:
        for _, ncol, plabel in practices:
            groups = [g[ncol].dropna().values for _, g in df.groupby(fcol)]
            groups = [g for g in groups if len(g) > 0]
            h, p = stats.kruskal(*groups)
            n_total = sum(len(g) for g in groups)
            eps2 = epsilon_squared(h, n_total)
            eff = effect_label(eps2, (0.01, 0.06))
            rows_kw.append((f"{flabel} x {plabel}", f"{h:.2f}", f"{p:.4f}", f"{eps2:.4f}", eff))

    all_effects = []
    for r in rows_chi:
        all_effects.append((r[0], "Cramer V", float(r[4])))
    for r in rows_mw:
        all_effects.append((r[0], "Rank-biserial r", abs(float(r[3]))))
    for r in rows_kw:
        all_effects.append((r[0], "ε²", float(r[3])))

    fig, ax = plt.subplots(figsize=(14, 9))
    labels = [e[0] for e in all_effects]
    vals = [e[2] for e in all_effects]
    colors = ["#e15759" if v >= 0.30 else "#f28e2b" if v >= 0.10 else "#76b7b2" for v in vals]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Размер эффекта")
    ax.set_title("Размеры эффекта для всех тестов")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    legend_elements = [Patch(facecolor="#76b7b2", label="Малый"),
                       Patch(facecolor="#f28e2b", label="Средний"),
                       Patch(facecolor="#e15759", label="Большой")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()
    img = fig_to_base64(fig)

    chi_rows = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
        f"<td><b>{r[4]}</b></td><td class='eff-{r[5]}'>{r[5]}</td></tr>"
        for r in rows_chi
    )
    mw_rows = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td>"
        f"<td><b>{r[3]}</b></td><td class='eff-{r[4]}'>{r[4]}</td></tr>"
        for r in rows_mw
    )
    kw_rows = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td>"
        f"<td><b>{r[3]}</b></td><td class='eff-{r[4]}'>{r[4]}</td></tr>"
        for r in rows_kw
    )

    return f"""
    <div class="card" id="sect1">
      <h2>1. Размеры эффекта (Effect Sizes)</h2>
      <p class="method">Cramer's V для χ²-тестов, Rank-biserial r для Манна–Уитни,
      ε² для Краскела–Уоллиса. Пороги: малый &lt;0.10, средний 0.10–0.30, большой &gt;0.30.</p>
      <div class="two-col">
        <div class="tables-col">
          <h3>Хи-квадрат → Cramer's V</h3>
          <table><tr><th>Тест</th><th>χ²</th><th>df</th><th>p</th><th>V</th><th>Эффект</th></tr>{chi_rows}</table>
          <h3>Манна–Уитни → Rank-biserial r</h3>
          <table><tr><th>Тест</th><th>U</th><th>p</th><th>r</th><th>Эффект</th></tr>{mw_rows}</table>
          <h3>Краскела–Уоллиса → ε²</h3>
          <table><tr><th>Тест</th><th>H</th><th>p</th><th>ε²</th><th>Эффект</th></tr>{kw_rows}</table>
        </div>
        <div class="chart-col">
          <img src="data:image/png;base64,{img}" alt="Effect sizes">
        </div>
      </div>
      <div class="insight">
        <b>Вывод:</b> Наибольшие эффекты — связь возраста с эко-товарами и сортировкой. Доход и образование
        практически не дифференцируют экологичное поведение.
      </div>
    </div>"""


def section_correlation(df: pd.DataFrame) -> str:
    vars_cols = ["sustainable_index", "q3_3_num", "q3_1_led", "q3_2_agree",
                 "q4_1_num", "q4_2_num", "q4_3_num",
                 "barrier_infra", "barrier_lazy", "barrier_unbelief",
                 "barrier_inconvenient", "barrier_expense", "barrier_nothink",
                 "city_msk"]
    var_labels = ["Индекс", "Готов. к магаз.", "Выбор LED", "Согласие с вкладом",
                  "Эко-друзья", "След. за сосед.", "Важн. мнения",
                  "Барьер: инфрастр.", "Барьер: лень", "Барьер: не верю",
                  "Барьер: неудобно", "Барьер: дорого", "Барьер: не думаю",
                  "Москва/СПб"]
    sub = df[vars_cols].dropna()
    n_vars = len(vars_cols)

    rho_mat = np.zeros((n_vars, n_vars))
    p_mat = np.ones((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                rho_mat[i, j] = 1.0
            else:
                r, p = stats.spearmanr(sub.iloc[:, i], sub.iloc[:, j])
                rho_mat[i, j] = r
                p_mat[i, j] = p

    all_p = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            all_p.append(p_mat[i, j])

    def holm_correction(pvals):
        n = len(pvals)
        order = np.argsort(pvals)
        corrected = np.empty(n)
        cummax = 0.0
        for rank, idx in enumerate(order):
            adj = pvals[idx] * (n - rank)
            cummax = max(cummax, adj)
            corrected[idx] = min(cummax, 1.0)
        return corrected

    corrected_p = holm_correction(np.array(all_p))

    idx = 0
    p_corrected_mat = np.ones((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            p_corrected_mat[i, j] = corrected_p[idx]
            p_corrected_mat[j, i] = corrected_p[idx]
            idx += 1

    fig, ax = plt.subplots(figsize=(14, 12))
    annot = np.empty_like(rho_mat, dtype=object)
    for i in range(n_vars):
        for j in range(n_vars):
            star = ""
            if i != j and p_corrected_mat[i, j] < 0.001:
                star = "***"
            elif i != j and p_corrected_mat[i, j] < 0.01:
                star = "**"
            elif i != j and p_corrected_mat[i, j] < 0.05:
                star = "*"
            annot[i, j] = f"{rho_mat[i, j]:.2f}{star}"

    sns.heatmap(rho_mat, annot=annot, fmt="", xticklabels=var_labels,
                yticklabels=var_labels, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax, square=True,
                cbar_kws={"shrink": 0.75, "label": "Spearman ρ"})
    ax.set_title("Корреляционная матрица (Spearman, поправка Holm)")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    fig.tight_layout()
    img = fig_to_base64(fig)

    return f"""
    <div class="card" id="sect2">
      <h2>2. Корреляционная матрица (Spearman)</h2>
      <p class="method">Ранговые корреляции между поведенческими переменными, барьерами и контрольными факторами.
      p-values скорректированы методом Holm. * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001. N = {len(sub)}.</p>
      <div class="chart-center">
        <img src="data:image/png;base64,{img}" alt="Correlation heatmap">
      </div>
      <div class="insight">
        <b>Вывод:</b> Социальные переменные (эко-друзья, следование за соседями) наиболее сильно
        коррелируют с индексом. Барьеры показывают отрицательные корреляции с устойчивым поведением.
      </div>
    </div>"""


def section_generation_comparison(df: pd.DataFrame) -> str:
    df_gen = df[df["generation"].isin(["zoom", "millennial"])].copy()
    gen_label_map = {"zoom": "Зумеры", "millennial": "Миллениалы"}
    df_gen["gen_label"] = df_gen["generation"].map(gen_label_map)
    zoom = df_gen[df_gen["generation"] == "zoom"]
    mill = df_gen[df_gen["generation"] == "millennial"]
    n_z, n_m = len(zoom), len(mill)

    # --- Figure 1: Violin plots for practices + index ---
    plot_vars = [("q2_1_num", "Сортировка"), ("q2_2_num", "Эко-товары"),
                 ("q2_3_num", "Экономия"), ("sustainable_index", "Индекс")]
    fig1, axes1 = plt.subplots(1, 4, figsize=(18, 6))
    for i, (col, label) in enumerate(plot_vars):
        ax = axes1[i]
        data_z = zoom[col].dropna().values
        data_m = mill[col].dropna().values
        if len(data_z) > 1 and len(data_m) > 1:
            parts = ax.violinplot([data_z, data_m], positions=[0, 1],
                                  showmeans=True, showmedians=True, widths=0.7)
            for j, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(PALETTE[j])
                pc.set_alpha(0.7)
            parts["cmeans"].set_color("black")
            parts["cmedians"].set_color("gray")
            parts["cmedians"].set_linestyle("--")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Зумеры", "Миллениалы"], fontsize=11)
        ax.set_title(label, fontsize=13)
        m_z, m_m = np.nanmean(data_z), np.nanmean(data_m)
        y_top = max(np.nanmax(data_z), np.nanmax(data_m)) if len(data_z) and len(data_m) else 5
        ax.text(0, y_top * 1.02, f"M={m_z:.2f}", ha="center", fontsize=10, color=PALETTE[0], weight="bold")
        ax.text(1, y_top * 1.02, f"M={m_m:.2f}", ha="center", fontsize=10, color=PALETTE[1], weight="bold")
    fig1.suptitle("Распределения по поколениям", fontsize=15, y=1.02)
    fig1.tight_layout()
    img_violin = fig_to_base64(fig1)

    # --- Figure 2: Grouped bar chart of means ---
    compare_vars = [("q2_1_num", "Сортировка"), ("q2_2_num", "Эко-товары"),
                    ("q2_3_num", "Экономия"), ("q3_3_num", "Готовн.\nк магаз."),
                    ("q4_1_num", "Эко-друзья"), ("q4_2_num", "След. за\nсоседями"),
                    ("q4_3_num", "Важн.\nмнения")]
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(compare_vars))
    w = 0.35
    z_means = [zoom[c].mean() for c, _ in compare_vars]
    m_means = [mill[c].mean() for c, _ in compare_vars]
    ax2.bar(x - w / 2, z_means, w, label=f"Зумеры (n={n_z})", color=PALETTE[0], edgecolor="white")
    ax2.bar(x + w / 2, m_means, w, label=f"Миллениалы (n={n_m})", color=PALETTE[1], edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([l for _, l in compare_vars], fontsize=11)
    ax2.set_ylabel("Среднее значение")
    ax2.set_title("Средние значения поведенческих переменных по поколениям")
    ax2.legend(fontsize=11)
    for i, (zv, mv) in enumerate(zip(z_means, m_means)):
        ax2.text(i - w / 2, zv + 0.05, f"{zv:.2f}", ha="center", fontsize=9)
        ax2.text(i + w / 2, mv + 0.05, f"{mv:.2f}", ha="center", fontsize=9)
    fig2.tight_layout()
    img_means = fig_to_base64(fig2)

    # --- Figure 3: Barrier comparison ---
    barrier_cols = ["barrier_infra", "barrier_lazy", "barrier_unbelief",
                    "barrier_inconvenient", "barrier_expense", "barrier_nothink"]
    barrier_labels = ["Нет инфраструктуры", "Лень / нет времени", "Не верю в пользу",
                      "Неудобно", "Дорого", "Не думаю об этом"]
    z_bar_pct = [zoom[c].sum() / max(n_z, 1) * 100 for c in barrier_cols]
    m_bar_pct = [mill[c].sum() / max(n_m, 1) * 100 for c in barrier_cols]

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    y = np.arange(len(barrier_cols))
    h = 0.35
    ax3.barh(y - h / 2, z_bar_pct, h, label="Зумеры", color=PALETTE[0], edgecolor="white")
    ax3.barh(y + h / 2, m_bar_pct, h, label="Миллениалы", color=PALETTE[1], edgecolor="white")
    ax3.set_yticks(y)
    ax3.set_yticklabels(barrier_labels, fontsize=11)
    ax3.set_xlabel("% респондентов")
    ax3.set_title("Барьеры по поколениям")
    ax3.legend(fontsize=11)
    for i, (zv, mv) in enumerate(zip(z_bar_pct, m_bar_pct)):
        ax3.text(zv + 0.5, i - h / 2, f"{zv:.0f}%", va="center", fontsize=10)
        ax3.text(mv + 0.5, i + h / 2, f"{mv:.0f}%", va="center", fontsize=10)
    ax3.invert_yaxis()
    fig3.tight_layout()
    img_barriers = fig_to_base64(fig3)

    # --- Figure 4: Radar chart ---
    radar_vars = [c for c, _ in compare_vars]
    radar_labels = [l.replace("\n", " ") for _, l in compare_vars]
    angles = np.linspace(0, 2 * np.pi, len(radar_vars), endpoint=False).tolist()
    angles += angles[:1]

    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, polar=True)
    z_vals = [zoom[c].mean() for c in radar_vars] + [zoom[radar_vars[0]].mean()]
    m_vals = [mill[c].mean() for c in radar_vars] + [mill[radar_vars[0]].mean()]
    ax4.plot(angles, z_vals, "o-", linewidth=2, label="Зумеры", color=PALETTE[0], markersize=6)
    ax4.fill(angles, z_vals, alpha=0.15, color=PALETTE[0])
    ax4.plot(angles, m_vals, "s-", linewidth=2, label="Миллениалы", color=PALETTE[1], markersize=6)
    ax4.fill(angles, m_vals, alpha=0.15, color=PALETTE[1])
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_labels, fontsize=10)
    ax4.set_title("Профили поколений", fontsize=14, y=1.12)
    ax4.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig4.tight_layout()
    img_radar = fig_to_base64(fig4)

    # --- Figure 5: LED and agreement ---
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
    z_led = zoom["q3_1_led"].mean() * 100
    m_led = mill["q3_1_led"].mean() * 100
    bars_led = ax5a.bar(["Зумеры", "Миллениалы"], [z_led, m_led],
                        color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    ax5a.set_ylabel("% выбравших LED")
    ax5a.set_title("Выбор LED-лампочки")
    ax5a.set_ylim(0, 105)
    for bar, val in zip(bars_led, [z_led, m_led]):
        ax5a.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
                  ha="center", fontsize=12, weight="bold")

    z_agree = zoom["q3_2_agree"].mean() * 100
    m_agree = mill["q3_2_agree"].mean() * 100
    bars_agr = ax5b.bar(["Зумеры", "Миллениалы"], [z_agree, m_agree],
                        color=[PALETTE[0], PALETTE[1]], edgecolor="white", width=0.5)
    ax5b.set_ylabel("% согласных")
    ax5b.set_title("Согласие с личным вкладом")
    ax5b.set_ylim(0, 105)
    for bar, val in zip(bars_agr, [z_agree, m_agree]):
        ax5b.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{val:.1f}%",
                  ha="center", fontsize=12, weight="bold")
    fig5.tight_layout()
    img_led = fig_to_base64(fig5)

    # --- Figure 6: Heatmaps for consumption & logic patterns ---
    fig6, (ax6a, ax6b) = plt.subplots(2, 1, figsize=(16, 12))
    ct_cons = pd.crosstab(df_gen["gen_label"], df_gen["q3_4_consumption"], normalize="index") * 100
    short_cons = [str(c)[:40] + "…" if len(str(c)) > 40 else str(c) for c in ct_cons.columns]
    sns.heatmap(ct_cons.values, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=short_cons, yticklabels=ct_cons.index.tolist(), ax=ax6a,
                cbar_kws={"label": "%"}, linewidths=0.5)
    ax6a.set_title("Паттерн потребления по поколениям (%)", fontsize=13)
    ax6a.tick_params(axis="x", rotation=15, labelsize=9)

    ct_logic = pd.crosstab(df_gen["gen_label"], df_gen["q3_5_logic"], normalize="index") * 100
    short_logic = [str(c)[:40] + "…" if len(str(c)) > 40 else str(c) for c in ct_logic.columns]
    sns.heatmap(ct_logic.values, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=short_logic, yticklabels=ct_logic.index.tolist(), ax=ax6b,
                cbar_kws={"label": "%"}, linewidths=0.5)
    ax6b.set_title("Логика принятия решений по поколениям (%)", fontsize=13)
    ax6b.tick_params(axis="x", rotation=15, labelsize=9)
    fig6.tight_layout()
    img_heatmap = fig_to_base64(fig6)

    # --- Mann-Whitney table ---
    all_test_vars = [("q2_1_num", "Сортировка"), ("q2_2_num", "Эко-товары"),
                     ("q2_3_num", "Экономия"), ("sustainable_index", "Индекс"),
                     ("q3_3_num", "Готовн. к магаз."), ("q4_1_num", "Эко-друзья"),
                     ("q4_2_num", "След. за сосед."), ("q4_3_num", "Важн. мнения"),
                     ("q3_1_led", "Выбор LED"), ("q3_2_agree", "Согласие с вкладом")]
    mw_rows = []
    for col, label in all_test_vars:
        xz = zoom[col].dropna()
        xm = mill[col].dropna()
        if len(xz) < 2 or len(xm) < 2:
            mw_rows.append((label, "—", "—", "—", "—", "—", "—"))
            continue
        u, p = stats.mannwhitneyu(xz, xm, alternative="two-sided")
        r = rank_biserial(u, len(xz), len(xm))
        eff = effect_label(r, (0.10, 0.30))
        mw_rows.append((label, f"{xz.mean():.2f}", f"{xm.mean():.2f}",
                         f"{u:.0f}", f"{p:.4f}", f"{r:.3f}", eff))

    mw_html = ""
    for r in mw_rows:
        p_str = r[4]
        is_sig = p_str != "—" and float(p_str) < 0.05
        bo, bc = ("<b>", "</b>") if is_sig else ("", "")
        mw_html += (f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
                    f"<td>{bo}{r[4]}{bc}</td><td>{r[5]}</td>"
                    f"<td class='eff-{r[6]}'>{r[6]}</td></tr>")

    return f"""
    <div class="card" id="sect3">
      <h2>3. Сравнение поколений: Зумеры vs Миллениалы</h2>
      <p class="method">Ключевое сравнение двух поколений по всем поведенческим переменным.
      Зумеры (18–25 лет, n={n_z}), Миллениалы (26–35+, n={n_m}).</p>

      <h3>Распределения по практикам и индексу</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_violin}" alt="Violin plots">
      </div>

      <h3>Средние значения поведенческих переменных</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_means}" alt="Means comparison">
      </div>

      <h3>Профили поколений (радарная диаграмма)</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_radar}" alt="Radar chart">
      </div>

      <h3>Барьеры по поколениям</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_barriers}" alt="Barriers by generation">
      </div>

      <h3>Выбор LED-лампочки и согласие с личным вкладом</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_led}" alt="LED and agreement">
      </div>

      <h3>Паттерны потребления и логика решений</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_heatmap}" alt="Consumption heatmap">
      </div>

      <h3>Статистические тесты (Mann-Whitney U)</h3>
      <table>
        <tr><th>Переменная</th><th>M (Зум.)</th><th>M (Милл.)</th><th>U</th><th>p</th><th>r</th><th>Эффект</th></tr>
        {mw_html}
      </table>

      <div class="insight">
        <b>Вывод:</b> Зумеры демонстрируют более высокие средние по практикам сортировки и
        эко-товаров. Профили поколений различаются по структуре барьеров и социальному давлению.
      </div>
    </div>"""


def _make_forest_plot(res, title, figsize=(14, 8)):
    """Forest plot for OLS result, excluding intercept."""
    names = res["names"][1:]
    betas = res["betas"][1:]
    ci_lo = res["ci_lo"][1:]
    ci_hi = res["ci_hi"][1:]
    p_vals = res["p"][1:]

    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))
    colors = ["#e15759" if p < 0.05 else "#bab0ac" for p in p_vals]
    ax.barh(y_pos, betas, color=colors, edgecolor="white", height=0.55, alpha=0.85)
    for i in range(len(names)):
        if not np.isnan(ci_lo[i]):
            ax.plot([ci_lo[i], ci_hi[i]], [i, i], color="black", linewidth=1.2)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("β (коэффициент)")
    ax.set_title(title, fontsize=13)
    ax.invert_yaxis()
    legend_elements = [Patch(facecolor="#e15759", label="p < 0.05"),
                       Patch(facecolor="#bab0ac", label="p ≥ 0.05")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()
    return fig


def _make_regression_table(res, title):
    """HTML table from OLS result."""
    rows_html = ""
    for i in range(len(res["names"])):
        p_val = res["p"][i]
        star = ""
        if p_val < 0.001: star = "***"
        elif p_val < 0.01: star = "**"
        elif p_val < 0.05: star = "*"
        bold_o = "<b>" if p_val < 0.05 else ""
        bold_c = "</b>" if p_val < 0.05 else ""
        rows_html += (f"<tr><td>{res['names'][i]}</td>"
                      f"<td>{res['betas'][i]:.3f}</td>"
                      f"<td>{res['se'][i]:.3f}</td>"
                      f"<td>{res['t'][i]:.2f}</td>"
                      f"<td>{bold_o}{p_val:.4f}{star}{bold_c}</td>"
                      f"<td>[{res['ci_lo'][i]:.3f}; {res['ci_hi'][i]:.3f}]</td></tr>")
    return f"""<h3>{title}</h3>
    <div class="metrics">
      <div class="metric-box" style="border-left: 4px solid #4e79a7;">
        <div class="metric-value">{res['r_sq']:.3f}</div>
        <div class="metric-label">R²</div>
      </div>
      <div class="metric-box" style="border-left: 4px solid #59a14f;">
        <div class="metric-value">{res['r_sq_adj']:.3f}</div>
        <div class="metric-label">R² adj.</div>
      </div>
      <div class="metric-box" style="border-left: 4px solid #f28e2b;">
        <div class="metric-value">{res['f_stat']:.2f}</div>
        <div class="metric-label">F (p={res['f_p']:.2e})</div>
      </div>
      <div class="metric-box" style="border-left: 4px solid #76b7b2;">
        <div class="metric-value">{res['n']}</div>
        <div class="metric-label">N</div>
      </div>
    </div>
    <table>
      <tr><th>Предиктор</th><th>β</th><th>SE</th><th>t</th><th>p</th><th>95% CI</th></tr>
      {rows_html}
    </table>"""


def section_regression(df: pd.DataFrame) -> str:
    predictors_base = ["q4_1_num", "q4_2_num", "q4_3_num",
                       "barrier_infra", "barrier_lazy", "barrier_unbelief",
                       "barrier_inconvenient", "barrier_expense",
                       "q3_3_num", "q3_1_led", "q3_2_agree"]
    pred_labels_base = ["Эко-друзья", "След. за сосед.", "Важн. мнения",
                        "Барьер: инфрастр.", "Барьер: лень", "Барьер: не верю",
                        "Барьер: неудобно", "Барьер: дорого",
                        "Готовн. к магаз.", "Выбор LED", "Согласие с вкладом"]

    female = df["gender"].str.contains("Женский", na=False).astype(float)
    gen_zoom_col = (df["generation"] == "zoom").astype(float)

    # --- General model ---
    df_all = df[df["generation"].isin(["zoom", "millennial"])].copy()
    df_all["female"] = female.reindex(df_all.index)
    df_all["gen_zoom"] = gen_zoom_col.reindex(df_all.index)

    gen_preds = predictors_base + ["female", "gen_zoom"]
    gen_labels = pred_labels_base + ["Женщина", "Зумер (vs миллениал)"]
    sub_gen = df_all[gen_preds + ["sustainable_index"]].dropna()

    imgs = {}
    tables = {}

    try:
        gen_res = ols_regression(sub_gen["sustainable_index"].values,
                                 sub_gen[gen_preds].values, gen_labels)
        fig_gen = _make_forest_plot(gen_res, "Общая модель: все респонденты")
        imgs["gen"] = fig_to_base64(fig_gen)
        tables["gen"] = _make_regression_table(gen_res, "Общая модель (Зумеры + Миллениалы)")
    except Exception as e:
        imgs["gen"] = ""
        tables["gen"] = f"<p style='color:red'>Ошибка общей модели: {e}</p>"

    # --- Zoomers model ---
    sub_preds = predictors_base + ["female"]
    sub_labels = pred_labels_base + ["Женщина"]

    df_zoom = df_all[df_all["generation"] == "zoom"].copy()
    sub_z = df_zoom[sub_preds + ["sustainable_index"]].dropna()
    try:
        zoom_res = ols_regression(sub_z["sustainable_index"].values,
                                  sub_z[sub_preds].values, sub_labels)
        fig_z = _make_forest_plot(zoom_res, f"Зумеры (n={zoom_res['n']})")
        imgs["zoom"] = fig_to_base64(fig_z)
        tables["zoom"] = _make_regression_table(zoom_res, f"Зумеры (n={zoom_res['n']})")
    except Exception as e:
        zoom_res = None
        imgs["zoom"] = ""
        tables["zoom"] = f"<p style='color:red'>Ошибка модели зумеров: {e}</p>"

    # --- Millennials model ---
    df_mill = df_all[df_all["generation"] == "millennial"].copy()
    sub_m = df_mill[sub_preds + ["sustainable_index"]].dropna()
    try:
        mill_res = ols_regression(sub_m["sustainable_index"].values,
                                  sub_m[sub_preds].values, sub_labels)
        fig_m = _make_forest_plot(mill_res, f"Миллениалы (n={mill_res['n']})")
        imgs["mill"] = fig_to_base64(fig_m)
        tables["mill"] = _make_regression_table(mill_res, f"Миллениалы (n={mill_res['n']})")
    except Exception as e:
        mill_res = None
        imgs["mill"] = ""
        tables["mill"] = f"<p style='color:red'>Ошибка модели миллениалов: {e}</p>"

    # --- Comparative forest plot ---
    img_comp = ""
    if zoom_res and mill_res:
        common_names = sub_labels
        z_b = zoom_res["betas"][1:]
        z_lo = zoom_res["ci_lo"][1:]
        z_hi = zoom_res["ci_hi"][1:]
        z_p = zoom_res["p"][1:]
        m_b = mill_res["betas"][1:]
        m_lo = mill_res["ci_lo"][1:]
        m_hi = mill_res["ci_hi"][1:]
        m_p = mill_res["p"][1:]

        fig_c, ax_c = plt.subplots(figsize=(14, 10))
        y_pos = np.arange(len(common_names))
        hh = 0.3
        z_colors = ["#4e79a7" if p < 0.05 else "#a0c4e8" for p in z_p]
        m_colors = ["#e15759" if p < 0.05 else "#f0a0a0" for p in m_p]
        ax_c.barh(y_pos - hh / 2, z_b, hh, color=z_colors, edgecolor="white", alpha=0.85)
        ax_c.barh(y_pos + hh / 2, m_b, hh, color=m_colors, edgecolor="white", alpha=0.85)
        for i in range(len(common_names)):
            ax_c.plot([z_lo[i], z_hi[i]], [i - hh / 2, i - hh / 2], color="#2c3e50", linewidth=1)
            ax_c.plot([m_lo[i], m_hi[i]], [i + hh / 2, i + hh / 2], color="#2c3e50", linewidth=1)
        ax_c.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_c.set_yticks(y_pos)
        ax_c.set_yticklabels(common_names, fontsize=10)
        ax_c.set_xlabel("β (коэффициент)")
        ax_c.set_title("Сравнение коэффициентов: Зумеры vs Миллениалы", fontsize=13)
        ax_c.invert_yaxis()
        legend_el = [Patch(facecolor="#4e79a7", label="Зумеры (p<0.05)"),
                     Patch(facecolor="#a0c4e8", label="Зумеры (p≥0.05)"),
                     Patch(facecolor="#e15759", label="Миллениалы (p<0.05)"),
                     Patch(facecolor="#f0a0a0", label="Миллениалы (p≥0.05)")]
        ax_c.legend(handles=legend_el, loc="lower right", fontsize=9)
        fig_c.tight_layout()
        img_comp = fig_to_base64(fig_c)

    gen_img = f'<img src="data:image/png;base64,{imgs["gen"]}" alt="General model">' if imgs["gen"] else ""
    z_img = f'<img src="data:image/png;base64,{imgs["zoom"]}" alt="Zoom model">' if imgs["zoom"] else ""
    m_img = f'<img src="data:image/png;base64,{imgs["mill"]}" alt="Mill model">' if imgs["mill"] else ""
    c_img = f'<img src="data:image/png;base64,{img_comp}" alt="Comparison">' if img_comp else ""

    return f"""
    <div class="card" id="sect4">
      <h2>4. Множественная регрессия (OLS)</h2>
      <p class="method">Зависимая переменная — индекс устойчивого поведения (3–15).
      Сначала общая модель, затем раздельные модели по поколениям для выявления
      поведенческих особенностей. Красные полосы — значимые предикторы (p&lt;0.05).</p>

      {tables["gen"]}
      <div class="chart-center">{gen_img}</div>

      {tables["zoom"]}
      <div class="chart-center">{z_img}</div>

      {tables["mill"]}
      <div class="chart-center">{m_img}</div>

      <h3>Сравнение коэффициентов</h3>
      <div class="chart-center">{c_img}</div>

      <div class="insight">
        <b>Вывод:</b> Общая модель выявляет ключевые предикторы устойчивого поведения.
        Раздельные модели показывают, какие факторы работают по-разному для зумеров и миллениалов —
        это позволяет точнее таргетировать интервенции.
      </div>
    </div>"""


def section_gender(df: pd.DataFrame) -> str:
    practices = [("q2_1_num", "Сортировка"), ("q2_2_num", "Эко-товары"),
                 ("q2_3_num", "Экономия"), ("sustainable_index", "Индекс")]
    males = df[df["gender"].str.contains("Мужской", na=False)]
    females = df[df["gender"].str.contains("Женский", na=False)]
    nm, nf = len(males), len(females)

    rows = []
    for col, label in practices:
        xm, xf = males[col].dropna(), females[col].dropna()
        u, p = stats.mannwhitneyu(xm, xf, alternative="two-sided")
        r = rank_biserial(u, len(xm), len(xf))
        rows.append((label, f"{xm.mean():.2f}", f"{xf.mean():.2f}",
                      f"{u:.0f}", f"{p:.4f}", f"{r:.3f}"))

    # --- Figure 1: Grouped bar chart ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(practices))
    w = 0.35
    m_means = [males[col].mean() for col, _ in practices]
    f_means = [females[col].mean() for col, _ in practices]
    ax1.bar(x - w / 2, m_means, w, label="Мужчины", color=PALETTE[0], edgecolor="white")
    ax1.bar(x + w / 2, f_means, w, label="Женщины", color=PALETTE[1], edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p[1] for p in practices], fontsize=11)
    ax1.set_ylabel("Среднее значение")
    ax1.set_title("Средние баллы по полу")
    ax1.legend(fontsize=11)
    for i, (mv, fv) in enumerate(zip(m_means, f_means)):
        ax1.text(i - w / 2, mv + 0.08, f"{mv:.2f}", ha="center", fontsize=9)
        ax1.text(i + w / 2, fv + 0.08, f"{fv:.2f}", ha="center", fontsize=9)
    fig1.tight_layout()
    img_bar = fig_to_base64(fig1)

    # --- Figure 2: Box plots by gender ---
    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 6))
    for i, (col, label) in enumerate(practices):
        ax = axes2[i]
        data_m = males[col].dropna().values
        data_f = females[col].dropna().values
        bp = ax.boxplot([data_m, data_f], patch_artist=True, widths=0.6,
                        labels=["М", "Ж"])
        bp["boxes"][0].set_facecolor(PALETTE[0])
        bp["boxes"][1].set_facecolor(PALETTE[1])
        for box in bp["boxes"]:
            box.set_alpha(0.7)
        ax.set_title(label, fontsize=12)
        ax.set_ylabel("Балл")
    fig2.suptitle("Распределения по полу", fontsize=14, y=1.02)
    fig2.tight_layout()
    img_box = fig_to_base64(fig2)

    # --- Figure 3: Barriers by gender ---
    barrier_cols = ["barrier_infra", "barrier_lazy", "barrier_unbelief",
                    "barrier_inconvenient", "barrier_expense", "barrier_nothink"]
    barrier_labels = ["Нет инфраструктуры", "Лень / нет времени", "Не верю в пользу",
                      "Неудобно", "Дорого", "Не думаю об этом"]
    m_bar_pct = [males[c].sum() / max(nm, 1) * 100 for c in barrier_cols]
    f_bar_pct = [females[c].sum() / max(nf, 1) * 100 for c in barrier_cols]

    fig3, ax3 = plt.subplots(figsize=(14, 7))
    y = np.arange(len(barrier_cols))
    h = 0.35
    ax3.barh(y - h / 2, m_bar_pct, h, label="Мужчины", color=PALETTE[0], edgecolor="white")
    ax3.barh(y + h / 2, f_bar_pct, h, label="Женщины", color=PALETTE[1], edgecolor="white")
    ax3.set_yticks(y)
    ax3.set_yticklabels(barrier_labels, fontsize=11)
    ax3.set_xlabel("% респондентов")
    ax3.set_title("Барьеры по полу")
    ax3.legend(fontsize=11)
    for i, (mv, fv) in enumerate(zip(m_bar_pct, f_bar_pct)):
        ax3.text(mv + 0.5, i - h / 2, f"{mv:.0f}%", va="center", fontsize=10)
        ax3.text(fv + 0.5, i + h / 2, f"{fv:.0f}%", va="center", fontsize=10)
    ax3.invert_yaxis()
    fig3.tight_layout()
    img_barriers = fig_to_base64(fig3)

    # --- Figure 4: Social variables by gender ---
    social_vars = [("q4_1_num", "Эко-друзья"), ("q4_2_num", "След. за сосед."),
                   ("q4_3_num", "Важн. мнения")]
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(social_vars))
    m_soc = [males[c].mean() for c, _ in social_vars]
    f_soc = [females[c].mean() for c, _ in social_vars]
    ax4.bar(x - w / 2, m_soc, w, label="Мужчины", color=PALETTE[0], edgecolor="white")
    ax4.bar(x + w / 2, f_soc, w, label="Женщины", color=PALETTE[1], edgecolor="white")
    ax4.set_xticks(x)
    ax4.set_xticklabels([l for _, l in social_vars], fontsize=11)
    ax4.set_ylabel("Среднее значение")
    ax4.set_title("Социальные переменные по полу")
    ax4.legend(fontsize=11)
    for i, (mv, fv) in enumerate(zip(m_soc, f_soc)):
        ax4.text(i - w / 2, mv + 0.05, f"{mv:.2f}", ha="center", fontsize=9)
        ax4.text(i + w / 2, fv + 0.05, f"{fv:.2f}", ha="center", fontsize=9)
    fig4.tight_layout()
    img_social = fig_to_base64(fig4)

    tbl = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
        f"<td>{r[4]}</td><td>{r[5]}</td></tr>" for r in rows
    )

    # Chi-squared for LED and agreement
    chi_extra = ""
    for col, label in [("q3_1_led", "Выбор LED"), ("q3_2_agree", "Согласие с вкладом")]:
        ct = pd.crosstab(df["gender"], df[col])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            v = cramers_v(ct.values)
            chi_extra += f"<p>{label}: χ²={chi2:.2f}, p={p:.4f}, Cramer V={v:.3f}</p>"

    return f"""
    <div class="card" id="sect5">
      <h2>5. Гендерный анализ</h2>
      <p class="method">Манна–Уитни U-тест по каждой шкале. Мужчины n={nm}, Женщины n={nf}.</p>

      <h3>Средние баллы</h3>
      <div class="two-col">
        <div class="tables-col">
          <table>
            <tr><th>Шкала</th><th>M (М)</th><th>M (Ж)</th><th>U</th><th>p</th><th>r</th></tr>
            {tbl}
          </table>
          {chi_extra}
        </div>
        <div class="chart-col">
          <img src="data:image/png;base64,{img_bar}" alt="Gender bar">
        </div>
      </div>

      <h3>Распределения по полу</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_box}" alt="Gender boxplots">
      </div>

      <h3>Барьеры по полу</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_barriers}" alt="Barriers by gender">
      </div>

      <h3>Социальные переменные по полу</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_social}" alt="Social by gender">
      </div>

      <div class="insight">
        <b>Вывод:</b> {"Значимых гендерных различий не обнаружено (p > 0.05)."
        if all(float(r[4]) > 0.05 for r in rows) else
        "Обнаружены значимые гендерные различия по некоторым шкалам."}
      </div>
    </div>"""


def section_city(df: pd.DataFrame) -> str:
    practices = [("q2_1_num", "Сортировка"), ("q2_2_num", "Эко-товары"),
                 ("q2_3_num", "Экономия"), ("sustainable_index", "Индекс")]
    msk = df[df["city_msk"] == 1]
    oth = df[df["city_msk"] == 0]

    rows_city = []
    for col, label in practices:
        xm, xo = msk[col].dropna(), oth[col].dropna()
        if len(xm) < 2 or len(xo) < 2:
            rows_city.append((label, "—", "—", "—", "—", "—"))
            continue
        u, p = stats.mannwhitneyu(xm, xo, alternative="two-sided")
        r = rank_biserial(u, len(xm), len(xo))
        rows_city.append((label, f"{xm.mean():.2f}", f"{xo.mean():.2f}",
                          f"{u:.0f}", f"{p:.4f}", f"{r:.3f}"))

    # --- Figure 1: City comparison (boxplots) ---
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    x = np.arange(len(practices))
    bp_data_msk = [msk[col].dropna().values for col, _ in practices]
    bp_data_oth = [oth[col].dropna().values for col, _ in practices]
    w_box = 0.35
    positions_msk = x - w_box / 2
    positions_oth = x + w_box / 2
    bp1 = ax1.boxplot(bp_data_msk, positions=positions_msk, widths=w_box * 0.8,
                      patch_artist=True, manage_ticks=False)
    bp2 = ax1.boxplot(bp_data_oth, positions=positions_oth, widths=w_box * 0.8,
                      patch_artist=True, manage_ticks=False)
    for patch in bp1["boxes"]:
        patch.set_facecolor(PALETTE[0])
        patch.set_alpha(0.7)
    for patch in bp2["boxes"]:
        patch.set_facecolor(PALETTE[1])
        patch.set_alpha(0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label for _, label in practices], fontsize=11)
    ax1.set_ylabel("Балл")
    ax1.set_title("Москва/СПб vs Другие города")
    ax1.legend(handles=[Patch(facecolor=PALETTE[0], alpha=0.7, label="Мск/СПб"),
                        Patch(facecolor=PALETTE[1], alpha=0.7, label="Другие")],
               fontsize=11)
    fig1.tight_layout()
    img_city = fig_to_base64(fig1)

    # --- Figure 2: Living type ---
    living_groups = df.groupby("living")["sustainable_index"].apply(
        lambda x: x.dropna().values).to_dict()
    living_labels_raw = list(living_groups.keys())
    living_data = [living_groups[k] for k in living_labels_raw]
    living_labels_short = [l[:45] + "…" if len(l) > 45 else l for l in living_labels_raw]

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    bp3 = ax2.boxplot(living_data, patch_artist=True, vert=False, widths=0.6)
    for i, patch in enumerate(bp3["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    ax2.set_yticklabels(living_labels_short, fontsize=10)
    ax2.set_xlabel("Индекс устойчивого поведения")
    ax2.set_title("Индекс по типу проживания")

    if len(living_data) >= 2:
        h_val, p_kw = stats.kruskal(*[np.array(d) for d in living_data if len(d) > 0])
        kw_text = f"Kruskal-Wallis H = {h_val:.2f}, p = {p_kw:.4f}"
    else:
        kw_text = "Недостаточно групп"
    ax2.set_title(f"Индекс по типу проживания ({kw_text})")
    fig2.tight_layout()
    img_living = fig_to_base64(fig2)

    tbl_city = "".join(
        f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td>"
        f"<td>{r[4]}</td><td>{r[5]}</td></tr>" for r in rows_city
    )
    nm_c, no_c = len(msk), len(oth)

    return f"""
    <div class="card" id="sect6">
      <h2>6. Город и тип проживания</h2>
      <p class="method">Москва/СПб (n={nm_c}) vs другие города (n={no_c}).
      Тип проживания: {kw_text}.</p>

      <div class="two-col">
        <div class="tables-col">
          <h3>Москва/СПб vs другие (Mann-Whitney)</h3>
          <table>
            <tr><th>Шкала</th><th>M (Мск)</th><th>M (Др.)</th><th>U</th><th>p</th><th>r</th></tr>
            {tbl_city}
          </table>
        </div>
        <div class="chart-col">
          <img src="data:image/png;base64,{img_city}" alt="City comparison">
        </div>
      </div>

      <h3>Тип проживания</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_living}" alt="Living type">
      </div>

      <div class="insight">
        <b>Вывод:</b> Тип города и формат проживания дают дополнительный контекст
        различий в экологичном поведении.
      </div>
    </div>"""


def section_consumption(df: pd.DataFrame) -> str:
    def short_label(s, max_len=45):
        s = str(s)
        return s[:max_len] + "…" if len(s) > max_len else s

    q34_vc = df["q3_4_consumption"].dropna().value_counts()
    q34_labels = [short_label(x) for x in q34_vc.index]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    bars = ax1.barh(range(len(q34_vc)), q34_vc.values, color=PALETTE[:len(q34_vc)])
    ax1.set_yticks(range(len(q34_vc)))
    ax1.set_yticklabels(q34_labels, fontsize=9)
    ax1.set_xlabel("Количество")
    ax1.set_title("Паттерны потребления (q3.4)")
    ax1.invert_yaxis()
    for bar, val in zip(bars, q34_vc.values):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, str(val),
                 va="center", fontsize=10)

    q35_vc = df["q3_5_logic"].dropna().value_counts()
    q35_labels = [short_label(x) for x in q35_vc.index]
    ax2 = axes[1]
    bars2 = ax2.barh(range(len(q35_vc)), q35_vc.values, color=PALETTE[:len(q35_vc)])
    ax2.set_yticks(range(len(q35_vc)))
    ax2.set_yticklabels(q35_labels, fontsize=9)
    ax2.set_xlabel("Количество")
    ax2.set_title("Логика принятия решений (q3.5)")
    ax2.invert_yaxis()
    for bar, val in zip(bars2, q35_vc.values):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2, str(val),
                 va="center", fontsize=10)
    fig.tight_layout()
    img_dist = fig_to_base64(fig)

    groups_34 = df.groupby("q3_4_consumption")["sustainable_index"].apply(
        lambda x: x.dropna().values)
    groups_34 = {k: v for k, v in groups_34.items() if len(v) > 0}
    if len(groups_34) >= 2:
        h34, p34 = stats.kruskal(*groups_34.values())
        kw34 = f"H = {h34:.2f}, p = {p34:.4f}"
    else:
        kw34 = "Недостаточно групп"

    fig2, ax = plt.subplots(figsize=(14, 7))
    bp_data = list(groups_34.values())
    bp_labels = [short_label(k, 35) for k in groups_34.keys()]
    bp = ax.boxplot(bp_data, patch_artist=True, vert=False, widths=0.6)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.7)
    ax.set_yticklabels(bp_labels, fontsize=9)
    ax.set_xlabel("Индекс устойчивого поведения")
    ax.set_title(f"Индекс по паттерну потребления (KW: {kw34})")
    fig2.tight_layout()
    img_box = fig_to_base64(fig2)

    groups_35 = df.groupby("q3_5_logic")["sustainable_index"].apply(
        lambda x: x.dropna().values)
    groups_35 = {k: v for k, v in groups_35.items() if len(v) > 0}
    if len(groups_35) >= 2:
        h35, p35 = stats.kruskal(*groups_35.values())
        kw35 = f"H = {h35:.2f}, p = {p35:.4f}"
    else:
        kw35 = "Недостаточно групп"

    ct_gen = pd.crosstab(df["generation"], df["q3_4_consumption"])
    if ct_gen.shape[0] >= 2 and ct_gen.shape[1] >= 2:
        chi2_gen, p_gen, _, _ = stats.chi2_contingency(ct_gen)
        v_gen = cramers_v(ct_gen.values)
        gen_test = f"χ² = {chi2_gen:.1f}, p = {p_gen:.4f}, Cramer V = {v_gen:.3f}"
    else:
        gen_test = "—"

    return f"""
    <div class="card" id="sect7">
      <h2>7. Паттерны потребления и логика решений</h2>
      <p class="method">Анализ q3.4 (модель потребления) и q3.5 (логика выбора) —
      переменные, не задействованные в основной модели.</p>
      <div class="metrics">
        <div class="metric-box" style="border-left: 4px solid #f28e2b;">
          <div class="metric-value" style="font-size:14px">{kw34}</div>
          <div class="metric-label">q3.4 → Индекс (KW)</div>
        </div>
        <div class="metric-box" style="border-left: 4px solid #59a14f;">
          <div class="metric-value" style="font-size:14px">{kw35}</div>
          <div class="metric-label">q3.5 → Индекс (KW)</div>
        </div>
      </div>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_dist}" alt="Consumption patterns">
      </div>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_box}" alt="Index by pattern">
      </div>
      <p>Связь поколения с паттерном потребления: {gen_test}</p>
      <div class="insight">
        <b>Вывод:</b> Тип потребительского поведения и логика принятия решений дополняют
        картину экологичного поведения и могут объяснить часть «необъяснённой» дисперсии.
      </div>
    </div>"""


def section_clusters(df: pd.DataFrame) -> str:
    cluster_vars = ["q2_1_num", "q2_2_num", "q2_3_num",
                    "q3_3_num", "q3_1_led", "q3_2_agree",
                    "q4_1_num", "q4_2_num", "q4_3_num",
                    "barrier_infra", "barrier_lazy", "barrier_unbelief",
                    "barrier_inconvenient", "barrier_expense"]
    cluster_labels = ["Сортировка", "Эко-товары", "Экономия",
                      "Готов. магаз.", "Выбор LED", "Согласие",
                      "Эко-друзья", "След. сосед.", "Важн. мнения",
                      "Б: инфрастр.", "Б: лень", "Б: не верю",
                      "Б: неудобно", "Б: дорого"]
    sub = df[cluster_vars + ["generation", "sustainable_index"]].dropna()
    X = sub[cluster_vars].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    K_range = range(2, 7)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    k_opt = 3
    km = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    sub = sub.copy()
    sub["cluster"] = km.fit_predict(X_scaled)
    profiles = sub.groupby("cluster")[cluster_vars].mean()

    cl_index_means = sub.groupby("cluster")["sustainable_index"].mean()
    sorted_cl = cl_index_means.sort_values().index.tolist()
    name_map = {sorted_cl[0]: "Скептики",
                sorted_cl[1]: "Умеренные практики",
                sorted_cl[2]: "Экоактивисты"}
    sub["cluster_name"] = sub["cluster"].map(name_map)

    # --- Figure: Radar + PCA ---
    angles = np.linspace(0, 2 * np.pi, len(cluster_vars), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, polar=True)
    for cl_idx in range(k_opt):
        cl = sorted_cl[cl_idx]
        vals = profiles.loc[cl].values.tolist()
        vals += vals[:1]
        ax1.plot(angles, vals, "o-", linewidth=1.8, label=name_map[cl],
                 color=PALETTE[cl_idx], markersize=4)
        ax1.fill(angles, vals, alpha=0.1, color=PALETTE[cl_idx])
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(cluster_labels, fontsize=7)
    ax1.set_title("Профили кластеров", fontsize=12, y=1.12)
    ax1.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.4, 1.12))

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    ax2 = fig.add_subplot(122)
    for cl_idx in range(k_opt):
        cl = sorted_cl[cl_idx]
        mask = sub["cluster"].values == cl
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], c=PALETTE[cl_idx],
                    label=name_map[cl], alpha=0.6, s=30, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax2.set_title("PCA-проекция кластеров")
    ax2.legend(fontsize=10)
    fig.tight_layout()
    img = fig_to_base64(fig)

    # --- Figure 2: Elbow plot ---
    fig_elbow, ax_e = plt.subplots(figsize=(8, 5))
    ax_e.plot(list(K_range), inertias, "o-", color=PALETTE[0], linewidth=2, markersize=8)
    ax_e.axvline(k_opt, color=PALETTE[1], linestyle="--", linewidth=1.5, label=f"k = {k_opt}")
    ax_e.set_xlabel("Число кластеров (k)")
    ax_e.set_ylabel("Инерция")
    ax_e.set_title("Метод локтя (Elbow)")
    ax_e.legend(fontsize=10)
    fig_elbow.tight_layout()
    img_elbow = fig_to_base64(fig_elbow)

    cl_sizes = sub["cluster"].value_counts()
    cl_gen = pd.crosstab(sub["cluster"], sub["generation"], normalize="index") * 100

    profile_rows = ""
    for cl_idx in range(k_opt):
        cl = sorted_cl[cl_idx]
        vals = " ".join(f"<td>{profiles.loc[cl, v]:.2f}</td>" for v in cluster_vars)
        gen_info = ""
        for g in ["zoom", "millennial"]:
            if g in cl_gen.columns:
                lbl = "зум" if g == "zoom" else "милл"
                gen_info += f"{lbl}: {cl_gen.loc[cl, g]:.0f}% "
        profile_rows += (f"<tr><td><b>{name_map[cl]}</b></td><td>{cl_sizes[cl]}</td>"
                        f"<td>{cl_index_means[cl]:.1f}</td>{vals}<td>{gen_info}</td></tr>")

    hdr = "".join(f"<th>{l}</th>" for l in cluster_labels)

    return f"""
    <div class="card" id="sect8">
      <h2>8. Кластерный анализ (K-Means, k={k_opt})</h2>
      <p class="method">K-Means на {len(cluster_vars)} стандартизированных переменных.
      Кластеры названы по среднему индексу устойчивости. N = {len(sub)}.</p>

      <h3>Метод локтя</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img_elbow}" alt="Elbow">
      </div>

      <h3>Профили и PCA</h3>
      <div class="chart-center">
        <img src="data:image/png;base64,{img}" alt="Cluster analysis">
      </div>

      <table>
        <tr><th>Кластер</th><th>n</th><th>Индекс (M)</th>{hdr}<th>Поколения</th></tr>
        {profile_rows}
      </table>

      <div class="insight">
        <b>Вывод:</b> Три кластера отражают уровни экологичного поведения:
        «{name_map[sorted_cl[0]]}» (низкий индекс), «{name_map[sorted_cl[1]]}» (средний)
        и «{name_map[sorted_cl[2]]}» (высокий). Различия в составе поколений помогают
        понять, как возраст связан с типом поведенческого профиля.
      </div>
    </div>"""


def section_barriers(df: pd.DataFrame) -> str:
    barrier_cols = ["barrier_infra", "barrier_lazy", "barrier_unbelief",
                    "barrier_inconvenient", "barrier_expense", "barrier_nothink"]
    barrier_labels = ["Нет инфраструктуры", "Лень / нет времени", "Не верю в пользу",
                      "Неудобно", "Дорого", "Не думаю об этом"]

    freqs = df[barrier_cols].sum()
    pcts = freqs / len(df) * 100

    df = df.copy()
    df["n_barriers"] = df[barrier_cols].sum(axis=1)
    rho, p_rho = stats.spearmanr(df["n_barriers"], df["sustainable_index"],
                                  nan_policy="omit")

    combos = []
    for _, row in df.iterrows():
        active = [barrier_labels[i] for i, c in enumerate(barrier_cols) if row[c] == 1]
        if active:
            combos.append(tuple(sorted(active)))
    combo_counts = Counter(combos).most_common(7)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    y_pos = np.arange(len(barrier_cols))
    colors = [PALETTE[i] for i in range(len(barrier_cols))]
    ax1.barh(y_pos, pcts.values, color=colors, edgecolor="white", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(barrier_labels, fontsize=11)
    ax1.set_xlabel("% респондентов")
    ax1.set_title("Распространённость барьеров")
    ax1.invert_yaxis()
    for i, (v, pct) in enumerate(zip(freqs.values, pcts.values)):
        ax1.text(pct + 0.5, i, f"{int(v)} ({pct:.0f}%)", va="center", fontsize=10)

    ax2 = axes[1]
    valid = df[["n_barriers", "sustainable_index"]].dropna()
    ax2.scatter(valid["n_barriers"], valid["sustainable_index"],
                alpha=0.35, s=30, c=PALETTE[0], edgecolors="white", linewidths=0.3)
    z = np.polyfit(valid["n_barriers"], valid["sustainable_index"], 1)
    x_line = np.linspace(valid["n_barriers"].min(), valid["n_barriers"].max(), 50)
    ax2.plot(x_line, np.polyval(z, x_line), color=PALETTE[1], linewidth=2.5)
    ax2.set_xlabel("Количество барьеров")
    ax2.set_ylabel("Индекс устойчивого поведения")
    ax2.set_title(f"Барьеры vs Индекс (ρ = {rho:.3f}, p = {p_rho:.4f})")
    fig.tight_layout()
    img = fig_to_base64(fig)

    mean_barriers = df["n_barriers"].mean()
    combo_rows = "".join(
        f"<tr><td>{' + '.join(c[0])}</td><td>{c[1]}</td></tr>"
        for c in combo_counts
    )

    return f"""
    <div class="card" id="sect9">
      <h2>9. Детальный анализ барьеров</h2>
      <p class="method">Частота барьеров, среднее кол-во барьеров на респондента,
      корреляция с индексом, популярные комбинации.</p>
      <div class="metrics">
        <div class="metric-box" style="border-left: 4px solid #e15759;">
          <div class="metric-value">{mean_barriers:.2f}</div>
          <div class="metric-label">Среднее кол-во барьеров</div>
        </div>
        <div class="metric-box" style="border-left: 4px solid #4e79a7;">
          <div class="metric-value">{rho:.3f}</div>
          <div class="metric-label">ρ (барьеры → индекс)</div>
        </div>
        <div class="metric-box" style="border-left: 4px solid #59a14f;">
          <div class="metric-value">{p_rho:.4f}</div>
          <div class="metric-label">p-value</div>
        </div>
      </div>
      <div class="two-col">
        <div class="chart-col" style="flex:2">
          <img src="data:image/png;base64,{img}" alt="Barriers analysis">
        </div>
        <div class="tables-col">
          <h3>Популярные комбинации барьеров</h3>
          <table>
            <tr><th>Комбинация</th><th>n</th></tr>
            {combo_rows}
          </table>
        </div>
      </div>
      <div class="insight">
        <b>Вывод:</b> {"Количество барьеров значимо связано с индексом (p < 0.05) — чем больше барьеров, тем ниже экологичное поведение." if p_rho < 0.05 else "Связь числа барьеров с индексом статистически незначима."}
      </div>
    </div>"""


def section_normality(df: pd.DataFrame) -> str:
    idx_all = df["sustainable_index"].dropna()
    df_gen = df[df["generation"].isin(["zoom", "millennial"])]
    idx_zoom = df_gen.loc[df_gen["generation"] == "zoom", "sustainable_index"].dropna()
    idx_mill = df_gen.loc[df_gen["generation"] == "millennial", "sustainable_index"].dropna()

    tests = []
    for data, label in [(idx_all, "Все"), (idx_zoom, "Зумеры"), (idx_mill, "Миллениалы")]:
        w, p = stats.shapiro(data)
        tests.append((label, len(data), f"{data.mean():.2f}", f"{data.median():.1f}",
                       f"{w:.4f}", f"{p:.4f}"))

    def bootstrap_ci(data, func, n_boot=5000, alpha=0.05):
        rng = np.random.default_rng(42)
        boot_stats = [func(rng.choice(data, size=len(data), replace=True)) for _ in range(n_boot)]
        lo = np.percentile(boot_stats, 100 * alpha / 2)
        hi = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return lo, hi

    ci_data = []
    for data, label in [(idx_all, "Все"), (idx_zoom, "Зумеры"), (idx_mill, "Миллениалы")]:
        arr = data.values
        mean_ci = bootstrap_ci(arr, np.mean)
        med_ci = bootstrap_ci(arr, np.median)
        ci_data.append((label, data.mean(), mean_ci, data.median(), med_ci))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax1 = axes[0]
    ax1.hist(idx_all, bins=range(3, 17), color=PALETTE[0], alpha=0.7, edgecolor="white",
             density=True, label="Данные")
    x_norm = np.linspace(3, 15, 100)
    ax1.plot(x_norm, stats.norm.pdf(x_norm, idx_all.mean(), idx_all.std()),
             color=PALETTE[1], linewidth=2, label="Норм. распр.")
    ax1.set_xlabel("Индекс")
    ax1.set_ylabel("Плотность")
    ax1.set_title(f"Распределение индекса (Shapiro-Wilk p = {tests[0][5]})")
    ax1.legend(fontsize=10)

    ax2 = axes[1]
    labels_ci = [d[0] for d in ci_data]
    means = [d[1] for d in ci_data]
    mean_lo = [d[2][0] for d in ci_data]
    mean_hi = [d[2][1] for d in ci_data]
    medians = [d[3] for d in ci_data]
    med_lo = [d[4][0] for d in ci_data]
    med_hi = [d[4][1] for d in ci_data]
    y = np.arange(len(labels_ci))
    ax2.errorbar(means, y - 0.1, xerr=[[m - lo for m, lo in zip(means, mean_lo)],
                 [hi - m for m, hi in zip(means, mean_hi)]], fmt="o",
                 color=PALETTE[0], label="Среднее ± 95% CI", capsize=5, markersize=8)
    ax2.errorbar(medians, y + 0.1, xerr=[[m - lo for m, lo in zip(medians, med_lo)],
                 [hi - m for m, hi in zip(medians, med_hi)]], fmt="s",
                 color=PALETTE[1], label="Медиана ± 95% CI", capsize=5, markersize=8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels_ci, fontsize=11)
    ax2.set_xlabel("Индекс устойчивого поведения")
    ax2.set_title("Доверительные интервалы (bootstrap, 5000)")
    ax2.legend(fontsize=10)
    fig.tight_layout()
    img = fig_to_base64(fig)

    test_rows = "".join(
        f"<tr><td>{t[0]}</td><td>{t[1]}</td><td>{t[2]}</td><td>{t[3]}</td>"
        f"<td>{t[4]}</td><td>{t[5]}</td></tr>" for t in tests
    )
    ci_rows = "".join(
        f"<tr><td>{d[0]}</td><td>{d[1]:.2f} [{d[2][0]:.2f}; {d[2][1]:.2f}]</td>"
        f"<td>{d[3]:.1f} [{d[4][0]:.1f}; {d[4][1]:.1f}]</td></tr>"
        for d in ci_data
    )

    return f"""
    <div class="card" id="sect10">
      <h2>10. Нормальность и доверительные интервалы</h2>
      <p class="method">Тест Шапиро–Уилка на нормальность индекса.
      Bootstrap (5000 итераций) 95% CI для средних и медиан.</p>
      <div class="two-col">
        <div class="tables-col">
          <h3>Shapiro-Wilk</h3>
          <table>
            <tr><th>Группа</th><th>n</th><th>M</th><th>Mdn</th><th>W</th><th>p</th></tr>
            {test_rows}
          </table>
          <h3>95% доверительные интервалы</h3>
          <table>
            <tr><th>Группа</th><th>Среднее [95% CI]</th><th>Медиана [95% CI]</th></tr>
            {ci_rows}
          </table>
        </div>
        <div class="chart-col">
          <img src="data:image/png;base64,{img}" alt="Normality & CI">
        </div>
      </div>
      <div class="insight">
        <b>Вывод:</b> {"Распределение значимо отклоняется от нормального — это обосновывает использование непараметрических тестов." if float(tests[0][5]) < 0.05 else "Распределение не отличается значимо от нормального."}
      </div>
    </div>"""


# ============================================================================
# HTML-ШАБЛОН
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Расширенная аналитика: Устойчивое потребление</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f4f5f7; color: #333; line-height: 1.55; }}
  .sidebar {{ position: fixed; top: 0; left: 0; width: 240px; height: 100vh;
              background: #2c3e50; color: #ecf0f1; padding: 20px 12px; overflow-y: auto; z-index: 100; }}
  .sidebar h1 {{ font-size: 15px; margin-bottom: 18px; line-height: 1.3; color: #fff; }}
  .sidebar a {{ display: block; color: #bdc3c7; text-decoration: none; padding: 6px 8px;
                font-size: 12.5px; border-radius: 4px; margin-bottom: 2px; }}
  .sidebar a:hover {{ background: #34495e; color: #fff; }}
  .main {{ margin-left: 240px; padding: 30px 35px; max-width: 1200px; }}
  .header {{ text-align: center; margin-bottom: 35px; }}
  .header h1 {{ font-size: 26px; color: #2c3e50; }}
  .header p {{ color: #666; margin-top: 6px; }}
  .card {{ background: #fff; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.07);
           padding: 28px 30px; margin-bottom: 30px; }}
  .card h2 {{ color: #2c3e50; margin-bottom: 10px; font-size: 20px; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
  .card h3 {{ color: #34495e; margin: 14px 0 6px; font-size: 15px; }}
  .method {{ color: #777; font-size: 13px; margin-bottom: 14px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 12.5px; }}
  th, td {{ border: 1px solid #e0e0e0; padding: 5px 8px; text-align: left; }}
  th {{ background: #f7f8fa; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafbfc; }}
  .metrics {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 14px 0; }}
  .metric-box {{ background: #f9fafb; border-radius: 8px; padding: 14px 20px;
                 min-width: 140px; flex: 1; }}
  .metric-value {{ font-size: 28px; font-weight: 700; color: #2c3e50; }}
  .metric-label {{ font-size: 12px; color: #888; margin-top: 2px; }}
  .two-col {{ display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }}
  .tables-col {{ flex: 1; min-width: 280px; }}
  .chart-col {{ flex: 1; min-width: 300px; }}
  .chart-col img, .chart-center img {{ max-width: 100%; height: auto; border-radius: 6px; }}
  .chart-center {{ text-align: center; margin: 12px 0; }}
  .insight {{ background: #eef6ff; border-left: 4px solid #4e79a7; padding: 10px 16px;
              margin-top: 14px; border-radius: 0 6px 6px 0; font-size: 13px; }}
  .eff-малый {{ color: #76b7b2; font-weight: 600; }}
  .eff-средний {{ color: #f28e2b; font-weight: 600; }}
  .eff-большой {{ color: #e15759; font-weight: 600; }}
  @media (max-width: 900px) {{
    .sidebar {{ display: none; }}
    .main {{ margin-left: 0; padding: 16px; }}
    .two-col {{ flex-direction: column; }}
  }}
</style>
</head>
<body>

<nav class="sidebar">
  <h1>Устойчивое потребление</h1>
  <a href="#sect1">1. Размеры эффекта</a>
  <a href="#sect2">2. Корреляции</a>
  <a href="#sect3">3. Зумеры vs Миллениалы</a>
  <a href="#sect4">4. Регрессия</a>
  <a href="#sect5">5. Гендер</a>
  <a href="#sect6">6. Город</a>
  <a href="#sect7">7. Потребление</a>
  <a href="#sect8">8. Кластеры</a>
  <a href="#sect9">9. Барьеры</a>
  <a href="#sect10">10. Нормальность</a>
</nav>

<div class="main">
  <div class="header">
    <h1>Расширенная аналитика: устойчивое потребление</h1>
    <p>N = {n_total} | Зумеры: {n_zoom} | Миллениалы: {n_mill}</p>
  </div>
  {sections}
</div>

</body>
</html>"""


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Загрузка данных...")
    df = load_data()
    n_total = len(df)
    n_zoom = (df["generation"] == "zoom").sum()
    n_mill = (df["generation"] == "millennial").sum()

    print(f"N = {n_total}, zoom = {n_zoom}, millennial = {n_mill}")

    sections = []
    section_funcs = [
        ("1. Effect Sizes", section_effect_sizes),
        ("2. Correlation", section_correlation),
        ("3. Generation Comparison", section_generation_comparison),
        ("4. Regression", section_regression),
        ("5. Gender", section_gender),
        ("6. City", section_city),
        ("7. Consumption", section_consumption),
        ("8. Clusters", section_clusters),
        ("9. Barriers", section_barriers),
        ("10. Normality", section_normality),
    ]

    for name, func in section_funcs:
        print(f"  Генерация: {name}...")
        try:
            sections.append(func(df))
        except Exception as e:
            sections.append(
                f'<div class="card"><h2>{name}</h2>'
                f'<p style="color:red">Ошибка: {e}</p></div>'
            )

    html = HTML_TEMPLATE.format(
        n_total=n_total, n_zoom=n_zoom, n_mill=n_mill,
        sections="\n".join(sections)
    )

    with open("report.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("Готово! Открой report.html в браузере.")


if __name__ == "__main__":
    main()
