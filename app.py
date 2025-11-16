from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html, no_update

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "final_data.csv"

COLUMN_MAP: Dict[str, str] = {
    "Unnamed: 0": "연도",
    "상호": "상호",
    "브랜드": "브랜드",
    "가입비\n(가맹비)": "가입비",
    "교육비": "교육비",
    "보증금": "보증금",
    "기타비용\n(인테리어\n비용포함)": "기타비용",
    "합계\n(창업비용\n지수)": "창업비용합계",
    "인테리어면적당(3.3㎡)\n비용": "면적당인테리어비용",
    "기준면적\n(㎡)인테리어": "기준면적",
    "총 비용": "총비용",
    "가맹점수": "가맹점수",
    "신규개점": "신규개점",
    "계약종료": "계약종료",
    "계약해지": "계약해지",
    "명의변경": "명의변경",
    "가맹점 평균 매출액 ": "평균매출액",
    "가맹점 면적(3.3㎡)당 평균매출액": "면적당매출액",
}

NUMERIC_COLUMNS: Tuple[str, ...] = (
    "가입비",
    "교육비",
    "보증금",
    "기타비용",
    "창업비용합계",
    "면적당인테리어비용",
    "기준면적",
    "총비용",
    "가맹점수",
    "신규개점",
    "계약종료",
    "계약해지",
    "명의변경",
    "평균매출액",
    "면적당매출액",
)

RADAR_FIELDS = [
    "가맹점수",
    "신규개점",
    "폐점수",
    "평균매출액",
    "면적당매출액",
    "창업비용합계",
]

COLOR_SCALE = [
    "#F94144",
    "#F3722C",
    "#F9C74F",
    "#90BE6D",
    "#43AA8B",
    "#577590",
    "#277DA1",
    "#B5179E",
    "#7209B7",
    "#4CC9F0",
]
DEFAULT_BRAND_COUNT = 5
MAX_BRAND_SELECTION = 10

STORE_BUCKETS = {
    "store_1500_plus": {"label": "가맹점수 1500개 이상", "min": 1500, "max": None},
    "store_1499_1000": {"label": "1000~1499개", "min": 1000, "max": 1499},
    "store_999_500": {"label": "500~999개", "min": 500, "max": 999},
    "store_499_100": {"label": "100~499개", "min": 100, "max": 499},
    "store_under_99": {"label": "99개 이하", "min": 0, "max": 99},
    "store_custom": {"label": "직접 입력", "custom": True},
}
SALES_BUCKETS = {
    "sales_400_plus": {"label": "평균 매출액 400백만원 이상", "min": 400, "max": None},
    "sales_300_400": {"label": "300~400백만원", "min": 300, "max": 400},
    "sales_200_300": {"label": "200~300백만원", "min": 200, "max": 300},
    "sales_under_200": {"label": "200백만원 이하", "min": 0, "max": 200},
    "sales_custom": {"label": "직접 입력", "custom": True},
}
COST_BUCKETS = {
    "cost_30000_plus": {"label": "창업비용 3만만원 이상", "min": 30000, "max": None},
    "cost_20000_30000": {"label": "2만~3만만원", "min": 20000, "max": 30000},
    "cost_10000_20000": {"label": "1만~2만만원", "min": 10000, "max": 20000},
    "cost_under_10000": {"label": "1만만원 이하", "min": 0, "max": 10000},
    "cost_custom": {"label": "직접 입력", "custom": True},
}

DEFAULT_STORE_FILTER = [
    "store_1500_plus",
    "store_1499_1000",
    "store_999_500",
    "store_499_100",
]

BRAND_TOKEN_STOPWORDS = {"coffe", "coffee", "커피", "카페", "주", "㈜", "(주)", "주식회사"}
OWNER_TOKEN_STOPWORDS = {"coffe", "coffee", "커피", "카페", "주", "㈜", "(주)", "주식회사", "limited", "ltd", "company", "comp", "inc", "corp"}
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


def _clean_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _reorder_parenthetical(text: str) -> str:
    txt = _clean_whitespace(text)
    if "(" in txt and txt.endswith(")"):
        head, tail = txt.split("(", 1)
        tail = tail[:-1]
        parts = [_clean_whitespace(head), _clean_whitespace(tail)]
        if all(parts):
            ordered = sorted(parts, key=lambda x: x.lower())
            return f"{ordered[0]}({ordered[1]})"
    return txt


def _canonical_tokens(text: str, stopwords: set[str]) -> str:
    tokens = [
        token for token in TOKEN_PATTERN.findall(text.lower())
        if token not in stopwords
    ]
    if not tokens:
        return text.lower()
    return "|".join(sorted(tokens))


BRAND_ALIAS_RAW = {
    "더리터(THE LITER)": "더리터(THE LITER)",
    "THE LITER(더리터)": "더리터(THE LITER)",
    "더리터": "더리터(THE LITER)",
    "더리터 (THE LITER)": "더리터(THE LITER)",
    "할리스(HOLLYS)": "할리스(HOLLYS)",
    "할리스": "할리스(HOLLYS)",
    "할리스커피": "할리스(HOLLYS)",
}

BRAND_PRESET_NAMES = [
    "더리터(THE LITER)",
    "할리스(HOLLYS)",
]

def _build_preset_targets():
    targets = {}
    for name in BRAND_PRESET_NAMES:
        key = _canonical_tokens(name, BRAND_TOKEN_STOPWORDS)
        targets[key] = name
    return targets


BRAND_CANONICAL_TARGETS = _build_preset_targets()
BRAND_NORMALIZATION_CACHE: Dict[str, str] = {}
BRAND_NORMALIZATION_WARNINGS: set[str] = set()


def canonicalize_owner(owner: str | None) -> str:
    if not owner or pd.isna(owner):
        return ""
    return _canonical_tokens(str(owner), OWNER_TOKEN_STOPWORDS)


def normalize_brand_name(name: str, owner: str | None = None) -> str:
    if pd.isna(name):
        return name
    text = _reorder_parenthetical(str(name))
    if text in BRAND_ALIAS_RAW:
        return BRAND_ALIAS_RAW[text]
    tokens_key = _canonical_tokens(text, BRAND_TOKEN_STOPWORDS)
    owner_key = canonicalize_owner(owner)
    compound_key = f"{tokens_key}|{owner_key}" if owner_key else tokens_key

    for key in (tokens_key, compound_key):
        if key in BRAND_CANONICAL_TARGETS:
            normalized = BRAND_CANONICAL_TARGETS[key]
            BRAND_NORMALIZATION_CACHE[tokens_key] = normalized
            if owner_key:
                BRAND_NORMALIZATION_CACHE[compound_key] = normalized
            return normalized
        if key in BRAND_NORMALIZATION_CACHE:
            return BRAND_NORMALIZATION_CACHE[key]

    normalized = text
    BRAND_NORMALIZATION_CACHE[tokens_key] = normalized
    if owner_key:
        BRAND_NORMALIZATION_CACHE[compound_key] = normalized
    if tokens_key not in BRAND_CANONICAL_TARGETS:
        BRAND_NORMALIZATION_WARNINGS.add(text)
    return normalized


def clean_numeric(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (float, int, np.number)):
        return float(value)
    text = str(value).strip()
    if not text or text == "-":
        return np.nan
    text = text.replace(",", "")
    if "\n" in text:
        text = text.split("\n")[0]
    text = re.sub(r"[^0-9.\-]", "", text)
    return float(text) if text else np.nan


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["상호"] != "평균"].copy()
    df.rename(columns=COLUMN_MAP, inplace=True)
    df["브랜드"] = df["브랜드"].astype(str).str.strip()
    df["상호"] = df["상호"].astype(str).str.strip()
    df["브랜드"] = [
        normalize_brand_name(brand, owner)
        for brand, owner in zip(df["브랜드"], df["상호"])
    ]
    df["연도"] = pd.to_numeric(df["연도"], errors="coerce").astype("Int64")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    df["계약종료"] = df["계약종료"].fillna(0)
    df["계약해지"] = df["계약해지"].fillna(0)
    df["폐점수"] = df["계약종료"] + df["계약해지"]
    df["신규개점"] = df["신규개점"].fillna(0)
    df["가맹점수"] = df["가맹점수"].fillna(0)
    df["순증"] = df["신규개점"] - df["폐점수"]
    df["순증가율"] = np.where(
        df["가맹점수"] > 0, df["순증"] / df["가맹점수"] * 100, np.nan
    )
    df["창업비용_만원"] = df["창업비용합계"] / 10  # 원문 데이터는 천원 단위
    df["평균매출액_백만원"] = df["평균매출액"] / 1000
    df["연순이익"] = df["평균매출액"] * 0.3
    df["회수기간"] = np.where(
        df["연순이익"] > 0, df["창업비용합계"] / df["연순이익"], np.nan
    )
    if BRAND_NORMALIZATION_WARNINGS:
        sample = ", ".join(sorted(list(BRAND_NORMALIZATION_WARNINGS))[:5])
        print(
            f"[brand-normalize] 신규 표기 {len(BRAND_NORMALIZATION_WARNINGS)}건 발견 — 예: {sample}"
        )
    return df


df_all = load_dataset()
available_years = sorted(df_all["연도"].dropna().unique())
default_year = 2024 if 2024 in available_years else available_years[-1]
default_year_range = [
    available_years[0],
    available_years[-1],
]
brand_options = sorted(df_all["브랜드"].dropna().unique())
default_selection = []


def get_top_brands(year: int, top_n: int = 10, allowed_brands: List[str] | None = None) -> List[str]:
    year_df = df_all[df_all["연도"] == year]
    if allowed_brands is not None:
        year_df = year_df[year_df["브랜드"].isin(allowed_brands)]
    ranking = (
        year_df.groupby("브랜드")["가맹점수"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    return ranking.index.tolist()


default_top_brands = get_top_brands(default_year, DEFAULT_BRAND_COUNT)
default_selection = default_top_brands


def make_empty_fig(message: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=14),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=height,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def build_metric_card(title: str, value: str, subtitle: str = "", color: str = "primary"):
    return dbc.Card(
        [
            dbc.CardHeader(title, className="fw-bold small text-muted"),
            dbc.CardBody(
                [
                    html.Div(value, className=f"display-6 text-{color} fw-bold"),
                    html.Div(subtitle, className="text-muted small mt-1"),
                ]
            ),
        ],
        className="h-100 shadow-sm",
    )


def format_currency(value: float, unit: str = "만원") -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}{unit}"


def bucket_dropdown_options(bucket_dict: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
    return [{"label": cfg["label"], "value": key} for key, cfg in bucket_dict.items()]


def build_range_mask(
    series: pd.Series,
    selections: List[str] | None,
    custom_min: float | None,
    custom_max: float | None,
    bucket_dict: Dict[str, Dict[str, float]],
) -> pd.Series:
    if not selections:
        return pd.Series(True, index=series.index)
    mask = pd.Series(False, index=series.index)
    any_regular = False
    for key in selections:
        bucket = bucket_dict.get(key)
        if not bucket or bucket.get("custom"):
            continue
        cond = pd.Series(True, index=series.index)
        if bucket.get("min") is not None:
            cond &= series >= bucket["min"]
        if bucket.get("max") is not None:
            cond &= series <= bucket["max"]
        mask |= cond
        any_regular = True
    any_custom = any(bucket_dict.get(key, {}).get("custom") for key in selections)
    if any_custom:
        cond = pd.Series(True, index=series.index)
        if custom_min is not None:
            cond &= series >= custom_min
        if custom_max is not None:
            cond &= series <= custom_max
        mask |= cond
    if not any_regular and not any_custom:
        return pd.Series(True, index=series.index)
    return mask


def filter_brands_for_year(
    year: int,
    store_selection: List[str] | None,
    store_min: float | None,
    store_max: float | None,
    sales_selection: List[str] | None,
    sales_min: float | None,
    sales_max: float | None,
    cost_selection: List[str] | None,
    cost_min: float | None,
    cost_max: float | None,
) -> List[str]:
    df_year = df_all[df_all["연도"] == year]
    if df_year.empty:
        return []
    mask = build_range_mask(
        df_year["가맹점수"], store_selection, store_min, store_max, STORE_BUCKETS
    )
    mask &= build_range_mask(
        df_year["평균매출액_백만원"],
        sales_selection,
        sales_min,
        sales_max,
        SALES_BUCKETS,
    )
    mask &= build_range_mask(
        df_year["창업비용_만원"], cost_selection, cost_min, cost_max, COST_BUCKETS
    )
    filtered = df_year.loc[mask, "브랜드"].dropna().unique().tolist()
    return sorted(filtered)


def clamp_brand_list(brands: List[str]) -> List[str]:
    if not brands:
        return []
    return brands[:MAX_BRAND_SELECTION]


def build_color_map(brands: List[str]) -> Dict[str, str]:
    ordered: List[str] = []
    seen = set()
    for brand in brands:
        if brand and brand not in seen:
            ordered.append(brand)
            seen.add(brand)
    mapping = {}
    for idx, brand in enumerate(ordered):
        mapping[brand] = COLOR_SCALE[idx % len(COLOR_SCALE)]
    return mapping


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="프랜차이즈 커피 대시보드",
)
server = app.server

overview_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("필터", className="fw-bold mb-3"),
                        dbc.Label("기준 연도"),
                        dcc.Dropdown(
                            id="overview-year",
                            options=[{"label": f"{int(y)}년", "value": int(y)} for y in available_years],
                            value=int(default_year),
                            clearable=False,
                        ),
                        dbc.Label("브랜드 선택 (복수)", className="mt-3"),
                        dcc.Dropdown(
                            id="overview-brands",
                            options=[{"label": b, "value": b} for b in brand_options],
                            value=default_selection,
                            multi=True,
                            placeholder="브랜드를 선택하세요",
                        ),
                        dbc.ButtonGroup(
                            [
                                dbc.Button("상위 10 자동선택", id="btn-top10", color="primary"),
                                dbc.Button("선택 초기화", id="btn-reset", color="secondary"),
                            ],
                            className="mt-2 w-100",
                        ),
                        dbc.Label("시계열 범위", className="mt-3"),
                        dcc.RangeSlider(
                            id="overview-year-range",
                            min=int(available_years[0]),
                            max=int(available_years[-1]),
                            value=[int(default_year_range[0]), int(default_year_range[1])],
                            marks={int(y): str(int(y)) for y in available_years},
                            allowCross=False,
                            step=1,
                        ),
                        html.Div(
                            "※ 버튼을 눌러 기준 연도의 상위 10개 브랜드를 빠르게 불러올 수 있습니다.",
                            className="text-muted small mt-3",
                        ),
                    ],
                    md=3,
                    className="mb-4",
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.Span(
                                                    f"{int(default_year)}년 브랜드별 개점 vs 폐점",
                                                    id="overview-scatter-title",
                                                )
                                            ),
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    id="overview-scatter",
                                                    config={"displayModeBar": False},
                                                    style={"height": "360px"},
                                                )
                                            ),
                                        ],
                                        className="h-100 shadow-sm",
                                    ),
                                    md=6,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardHeader(
                                                html.Span(
                                                    f"{int(default_year)}년 브랜드별 평균 창업비용",
                                                    id="overview-cost-title",
                                                )
                                            ),
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    id="overview-cost",
                                                    config={"displayModeBar": False},
                                                    style={"height": "360px"},
                                                )
                                            ),
                                        ],
                                        className="h-100 shadow-sm",
                                    ),
                                    md=6,
                                ),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            dbc.Col(
                                dbc.Card(
                                    [
                                        dbc.CardHeader("브랜드별 평균 매출액 추이"),
                                        dbc.CardBody(
                                            dcc.Graph(
                                                id="overview-sales",
                                                config={"displayModeBar": False},
                                                style={"height": "420px"},
                                            )
                                        ),
                                    ],
                                    className="shadow-sm",
                                ),
                            )
                        ),
                    ],
                    md=9,
                ),
            ],
            className="g-3",
        )
    ],
    fluid=True,
    className="py-4",
)

compare_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("비교 연도"),
                        dcc.Dropdown(
                            id="radar-year",
                            options=[{"label": f"{int(y)}년", "value": int(y)} for y in available_years],
                            value=int(default_year),
                            clearable=False,
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        dbc.Label("브랜드 A"),
                        dcc.Dropdown(id="radar-brand-a", options=[], value=None, clearable=False),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Label("브랜드 B"),
                        dcc.Dropdown(id="radar-brand-b", options=[], value=None, clearable=False),
                    ],
                    md=5,
                ),
            ],
            className="g-3 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("브랜드 종합 비교 레이더 (정규화 점수)"),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="radar-graph",
                                    config={"displayModeBar": False},
                                    style={"height": "520px"},
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    lg=8,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("브랜드 A 지표"),
                                dbc.CardBody(id="brand-a-card"),
                            ],
                            className="mb-3 shadow-sm",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("브랜드 B 지표"),
                                dbc.CardBody(id="brand-b-card"),
                            ],
                            className="shadow-sm",
                        ),
                    ],
                    lg=4,
                ),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    className="py-4",
)

profit_marks = {round(p, 2): f"{int(p*100)}%" for p in np.linspace(0.1, 0.5, 5)}

detail_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("브랜드 선택"),
                        dcc.Dropdown(
                            id="detail-brand",
                            options=[{"label": b, "value": b} for b in brand_options],
                            value="메가엠지씨커피(MEGA MGC COFFEE)"
                            if "메가엠지씨커피(MEGA MGC COFFEE)" in brand_options
                            else brand_options[0],
                            clearable=False,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("연도 범위"),
                        dcc.RangeSlider(
                            id="detail-year-range",
                            min=int(available_years[0]),
                            max=int(available_years[-1]),
                            value=[int(available_years[0]), int(available_years[-1])],
                            marks={int(y): str(int(y)) for y in available_years},
                            allowCross=False,
                            step=1,
                        ),
                    ],
                    md=5,
                ),
                dbc.Col(
                    [
                        dbc.Label("순이익률 가정"),
                        dcc.Slider(
                            id="detail-profit",
                            min=0.1,
                            max=0.5,
                            step=0.05,
                            value=0.3,
                            marks=profit_marks,
                            tooltip={"placement": "top"},
                        ),
                    ],
                    md=3,
                ),
            ],
            className="g-3 mb-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("연도별 개점·폐점 추이"),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="detail-open-close",
                                    config={"displayModeBar": False},
                                    style={"height": "340px"},
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("투자 대비 수익 추이"),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="detail-invest",
                                    config={"displayModeBar": False},
                                    style={"height": "340px"},
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    lg=6,
                ),
            ],
            className="g-4",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("가맹점 순증가율"),
                            dbc.CardBody(
                                dcc.Graph(
                                    id="detail-net-growth",
                                    config={"displayModeBar": False},
                                    style={"height": "320px"},
                                )
                            ),
                        ],
                        className="shadow-sm",
                    ),
                    lg=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader("핵심 KPI"),
                            dbc.CardBody(id="detail-kpis"),
                        ],
                        className="shadow-sm",
                    ),
                    lg=6,
                ),
            ],
            className="g-4",
        ),
    ],
    fluid=True,
    className="py-4",
)

app.layout = dbc.Container(
    [
        html.H2("프랜차이즈 커피 브랜드 대시보드", className="mt-4 fw-bold"),
        html.P(
            "공정거래위원회 가맹사업 정보공개서를 기반으로 2017-2024년 데이터를 시각화했습니다.",
            className="text-muted",
        ),
        dbc.Card(
            [
                dbc.CardHeader("브랜드 필터", className="fw-bold"),
                dbc.CardBody(
                    [
                        html.P(
                            "가맹점 규모·매출·창업비용 구간을 선택하면 드롭다운에 노출되는 브랜드 수가 줄어듭니다. "
                            "‘직접 입력’을 선택하면 하단 입력값이 적용됩니다.",
                            className="text-muted small",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("가맹점수 구간"),
                                        dcc.Dropdown(
                                            id="store-range-filter",
                                            options=bucket_dropdown_options(STORE_BUCKETS),
                                            value=DEFAULT_STORE_FILTER,
                                            multi=True,
                                            placeholder="가맹점 규모를 선택하세요",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("최소(개)"),
                                                            dbc.Input(
                                                                id="store-custom-min",
                                                                type="number",
                                                                min=0,
                                                                step=10,
                                                                placeholder="예) 100",
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("최대(개)"),
                                                            dbc.Input(
                                                                id="store-custom-max",
                                                                type="number",
                                                                min=0,
                                                                step=10,
                                                                placeholder="예) 1500",
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-2 mt-2",
                                        ),
                                    ],
                                    lg=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("평균 매출액 구간 (백만원)"),
                                        dcc.Dropdown(
                                            id="sales-range-filter",
                                            options=bucket_dropdown_options(SALES_BUCKETS),
                                            value=[],
                                            multi=True,
                                            placeholder="매출 구간을 선택하세요",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("최소"),
                                                            dbc.Input(
                                                                id="sales-custom-min",
                                                                type="number",
                                                                min=0,
                                                                step=10,
                                                                placeholder="예) 200",
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("최대"),
                                                            dbc.Input(
                                                                id="sales-custom-max",
                                                                type="number",
                                                                min=0,
                                                                step=10,
                                                                placeholder="예) 450",
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-2 mt-2",
                                        ),
                                    ],
                                    lg=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("창업비용 구간 (만원)"),
                                        dcc.Dropdown(
                                            id="cost-range-filter",
                                            options=bucket_dropdown_options(COST_BUCKETS),
                                            value=[],
                                            multi=True,
                                            placeholder="창업비용 구간을 선택하세요",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("최소"),
                                                            dbc.Input(
                                                                id="cost-custom-min",
                                                                type="number",
                                                                min=0,
                                                                step=100,
                                                                placeholder="예) 5000",
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.InputGroupText("최대"),
                                                            dbc.Input(
                                                                id="cost-custom-max",
                                                                type="number",
                                                                min=0,
                                                                step=100,
                                                                placeholder="예) 20000",
                                                            ),
                                                        ]
                                                    ),
                                                    md=6,
                                                ),
                                            ],
                                            className="g-2 mt-2",
                                        ),
                                    ],
                                    lg=4,
                                ),
                            ],
                            className="g-4",
                        ),
                    ]
                ),
            ],
            className="mb-4 shadow-sm",
        ),
        dbc.Tabs(
            [
                dbc.Tab(overview_tab, label="브랜드 개요", tab_id="tab-overview"),
                dbc.Tab(compare_tab, label="브랜드 비교", tab_id="tab-compare"),
                dbc.Tab(detail_tab, label="브랜드 상세 분석", tab_id="tab-detail"),
            ],
            active_tab="tab-overview",
            className="mt-3",
        ),
        html.Footer(
            "데이터 출처: 공정거래위원회 가맹사업 정보공개서",
            className="text-muted small my-4",
        ),
    ],
    fluid=True,
)


@app.callback(
    Output("overview-brands", "options"),
    Output("overview-brands", "value"),
    Input("overview-year", "value"),
    Input("store-range-filter", "value"),
    Input("sales-range-filter", "value"),
    Input("cost-range-filter", "value"),
    Input("store-custom-min", "value"),
    Input("store-custom-max", "value"),
    Input("sales-custom-min", "value"),
    Input("sales-custom-max", "value"),
    Input("cost-custom-min", "value"),
    Input("cost-custom-max", "value"),
    Input("btn-top10", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("overview-brands", "value"),
)
def update_overview_brand_selection(
    year,
    store_filter,
    sales_filter,
    cost_filter,
    store_min,
    store_max,
    sales_min,
    sales_max,
    cost_min,
    cost_max,
    top_clicks,
    reset_clicks,
    current_value,
):
    year = int(year)
    filtered_brands = filter_brands_for_year(
        year,
        store_filter,
        store_min,
        store_max,
        sales_filter,
        sales_min,
        sales_max,
        cost_filter,
        cost_min,
        cost_max,
    )
    options = [{"label": brand, "value": brand} for brand in filtered_brands]
    ctx = dash.callback_context
    trigger = ctx.triggered_id if ctx.triggered else None
    if trigger == "btn-top10":
        new_value = get_top_brands(
            year,
            min(MAX_BRAND_SELECTION, 10),
            allowed_brands=filtered_brands,
        )
    elif trigger == "btn-reset":
        new_value = []
    else:
        current_value = current_value or []
        new_value = [b for b in current_value if b in filtered_brands]
        if not new_value:
            new_value = get_top_brands(
                year,
                DEFAULT_BRAND_COUNT,
                allowed_brands=filtered_brands,
            )
    new_value = clamp_brand_list(new_value or [])
    return options, new_value


@app.callback(
    Output("overview-scatter-title", "children"),
    Output("overview-cost-title", "children"),
    Output("overview-scatter", "figure"),
    Output("overview-cost", "figure"),
    Output("overview-sales", "figure"),
    Input("overview-year", "value"),
    Input("overview-brands", "value"),
    Input("overview-year-range", "value"),
)
def update_overview(year, brands, year_range):
    if not brands:
        msg_fig = make_empty_fig("브랜드를 선택해주세요.")
        year = int(year)
        scatter_title = f"{year}년 브랜드별 개점 vs 폐점"
        cost_title = f"{year}년 브랜드별 평균 창업비용"
        return scatter_title, cost_title, msg_fig, msg_fig, msg_fig
    selected = df_all[df_all["브랜드"].isin(brands)].copy()
    year = int(year)
    scatter_title = f"{year}년 브랜드별 개점 vs 폐점"
    cost_title = f"{year}년 브랜드별 평균 창업비용"
    scatter_df = selected[selected["연도"] == year]
    order_for_colors = []
    for b in brands or []:
        if b in scatter_df["브랜드"].values and b not in order_for_colors:
            order_for_colors.append(b)
    for b in scatter_df["브랜드"].unique():
        if b not in order_for_colors:
            order_for_colors.append(b)
    color_map = build_color_map(order_for_colors)
    if scatter_df.empty:
        scatter_fig = make_empty_fig("선택한 연도에 데이터가 없습니다.")
    else:
        scatter_fig = px.scatter(
            scatter_df,
            x="신규개점",
            y="폐점수",
            size="가맹점수",
            color="브랜드",
            hover_data={"가맹점수": True, "평균매출액_백만원": ":.1f"},
            size_max=60,
            color_discrete_map=color_map,
        )
        max_axis = max(
            scatter_df["신규개점"].max(),
            scatter_df["폐점수"].max(),
        )
        scatter_fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=max_axis,
            y1=max_axis,
            line=dict(color="#e74c3c", dash="dash"),
        )
        scatter_fig.update_layout(
            xaxis_title="신규개점 수",
            yaxis_title="폐점 수",
            legend_title="브랜드",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )

    cost_df = scatter_df.groupby("브랜드")["창업비용_만원"].mean().sort_values(ascending=False)
    if cost_df.empty:
        cost_fig = make_empty_fig("창업비용 데이터가 없습니다.")
    else:
        cost_fig = go.Figure(
            go.Bar(
                x=cost_df.values,
                y=cost_df.index,
                orientation="h",
                marker_color=[color_map.get(b, "#666666") for b in cost_df.index],
            )
        )
        cost_fig.update_layout(
            xaxis_title="평균 창업비용 (만원)",
            margin=dict(l=160, r=20, t=40, b=40),
            template="plotly_white",
        )

    start_year, end_year = sorted([int(year_range[0]), int(year_range[1])])
    time_df = (
        selected[(selected["연도"] >= start_year) & (selected["연도"] <= end_year)]
        .groupby(["연도", "브랜드"], as_index=False)["평균매출액_백만원"]
        .mean()
    )
    if time_df.empty:
        sales_fig = make_empty_fig("해당 범위에 매출 데이터가 없습니다.", height=380)
    else:
        sales_fig = px.line(
            time_df,
            x="연도",
            y="평균매출액_백만원",
            color="브랜드",
            markers=True,
            color_discrete_map=color_map,
        )
        sales_fig.update_layout(
            xaxis_title="연도",
            yaxis_title="평균 매출액 (백만원)",
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
            legend_title="브랜드",
        )
    return scatter_title, cost_title, scatter_fig, cost_fig, sales_fig


@app.callback(
    Output("radar-brand-a", "options"),
    Output("radar-brand-b", "options"),
    Output("radar-brand-a", "value"),
    Output("radar-brand-b", "value"),
    Input("radar-year", "value"),
    Input("store-range-filter", "value"),
    Input("sales-range-filter", "value"),
    Input("cost-range-filter", "value"),
    Input("store-custom-min", "value"),
    Input("store-custom-max", "value"),
    Input("sales-custom-min", "value"),
    Input("sales-custom-max", "value"),
    Input("cost-custom-min", "value"),
    Input("cost-custom-max", "value"),
    State("radar-brand-a", "value"),
    State("radar-brand-b", "value"),
)
def update_radar_options(
    year,
    store_filter,
    sales_filter,
    cost_filter,
    store_min,
    store_max,
    sales_min,
    sales_max,
    cost_min,
    cost_max,
    brand_a,
    brand_b,
):
    year = int(year)
    available = filter_brands_for_year(
        year,
        store_filter,
        store_min,
        store_max,
        sales_filter,
        sales_min,
        sales_max,
        cost_filter,
        cost_min,
        cost_max,
    )
    if not available:
        return [], [], None, None
    options = [{"label": b, "value": b} for b in available]
    defaults = get_top_brands(year, 2, allowed_brands=available)

    def pick(current, fallback_list):
        if current in available:
            return current
        for fb in fallback_list:
            if fb in available:
                return fb
        return available[0] if available else None

    value_a = pick(brand_a, defaults)
    value_b_candidates = [b for b in defaults if b != value_a]
    value_b = pick(brand_b, value_b_candidates or defaults)
    if value_a == value_b and len(available) > 1:
        value_b = next((b for b in available if b != value_a), None)
    return options, options, value_a, value_b


def build_brand_summary(row: pd.Series) -> List[html.Div]:
    if row is None or row.empty:
        return [html.P("데이터가 없습니다.", className="text-muted")]
    rows = [
        ("가맹점 수", f"{row['가맹점수']:,.0f}개"),
        ("신규개점", f"{row['신규개점']:,.0f}개"),
        ("폐점수", f"{row['폐점수']:,.0f}개"),
        ("평균 매출액", f"{row['평균매출액_백만원']:,.1f}백만원"),
        ("면적당 매출액", f"{row['면적당매출액']:,.0f}원/3.3㎡"),
        ("평균 창업비용", f"{row['창업비용_만원']:,.0f}만원"),
    ]
    return [
        html.H5(row["브랜드"], className="fw-bold"),
        html.Hr(),
        html.Ul([html.Li(f"{label}: {value}") for label, value in rows], className="mb-0"),
    ]


@app.callback(
    Output("radar-graph", "figure"),
    Output("brand-a-card", "children"),
    Output("brand-b-card", "children"),
    Input("radar-year", "value"),
    Input("radar-brand-a", "value"),
    Input("radar-brand-b", "value"),
)
def update_radar(year, brand_a, brand_b):
    if not brand_a or not brand_b or brand_a == brand_b:
        empty = make_empty_fig("서로 다른 두 브랜드를 선택하세요.", height=520)
        return empty, [html.P("브랜드 A 미선택", className="text-muted")], [
            html.P("브랜드 B 미선택", className="text-muted")
        ]
    year_df = df_all[df_all["연도"] == int(year)]
    subset = year_df[year_df["브랜드"].isin([brand_a, brand_b])]
    if subset.empty:
        empty = make_empty_fig("해당 연도 데이터가 없습니다.", height=520)
        return empty, [html.P("데이터 없음")], [html.P("데이터 없음")]

    top_brands_year = get_top_brands(int(year), 10)
    top_base = year_df[year_df["브랜드"].isin(top_brands_year)]
    if top_base.empty:
        top_base = subset
    max_values = top_base[RADAR_FIELDS].max().replace(0, np.nan)

    color_map = build_color_map([brand_a, brand_b])
    fig = go.Figure()
    for idx, brand in enumerate([brand_a, brand_b]):
        row = subset[subset["브랜드"] == brand].iloc[0]
        normalized = []
        for field in RADAR_FIELDS:
            max_val = max_values.get(field)
            raw = row[field]
            normalized.append((raw / max_val * 100) if max_val else 0)
        normalized.append(normalized[0])
        theta = RADAR_FIELDS + [RADAR_FIELDS[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=normalized,
                theta=theta,
                fill="toself",
                name=brand,
                marker=dict(color=color_map.get(brand, COLOR_SCALE[idx % len(COLOR_SCALE)])),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 110]),
        ),
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    )

    card_a = build_brand_summary(subset[subset["브랜드"] == brand_a].iloc[0])
    card_b = build_brand_summary(subset[subset["브랜드"] == brand_b].iloc[0])
    return fig, card_a, card_b


@app.callback(
    Output("detail-brand", "options"),
    Output("detail-brand", "value"),
    Input("detail-year-range", "value"),
    Input("store-range-filter", "value"),
    Input("sales-range-filter", "value"),
    Input("cost-range-filter", "value"),
    Input("store-custom-min", "value"),
    Input("store-custom-max", "value"),
    Input("sales-custom-min", "value"),
    Input("sales-custom-max", "value"),
    Input("cost-custom-min", "value"),
    Input("cost-custom-max", "value"),
    State("detail-brand", "value"),
)
def update_detail_brand_options(
    year_range,
    store_filter,
    sales_filter,
    cost_filter,
    store_min,
    store_max,
    sales_min,
    sales_max,
    cost_min,
    cost_max,
    current_brand,
):
    start_year, end_year = sorted([int(year_range[0]), int(year_range[1])])
    available = filter_brands_for_year(
        end_year,
        store_filter,
        store_min,
        store_max,
        sales_filter,
        sales_min,
        sales_max,
        cost_filter,
        cost_min,
        cost_max,
    )
    # Ensure brand exists within selected year range
    range_df = df_all[
        (df_all["연도"] >= start_year)
        & (df_all["연도"] <= end_year)
        & (df_all["브랜드"].isin(available))
    ]
    available_range = sorted(range_df["브랜드"].unique().tolist())
    options = [{"label": b, "value": b} for b in available_range]
    if current_brand in available_range:
        value = current_brand
    elif available_range:
        value = available_range[0]
    else:
        value = None
    return options, value


@app.callback(
    Output("store-custom-min", "disabled"),
    Output("store-custom-max", "disabled"),
    Output("sales-custom-min", "disabled"),
    Output("sales-custom-max", "disabled"),
    Output("cost-custom-min", "disabled"),
    Output("cost-custom-max", "disabled"),
    Input("store-range-filter", "value"),
    Input("sales-range-filter", "value"),
    Input("cost-range-filter", "value"),
)
def toggle_custom_inputs(store_filter, sales_filter, cost_filter):
    def needs_disabled(selection, bucket_dict):
        selection = selection or []
        return not any(bucket_dict.get(key, {}).get("custom") for key in selection)

    store_disabled = needs_disabled(store_filter, STORE_BUCKETS)
    sales_disabled = needs_disabled(sales_filter, SALES_BUCKETS)
    cost_disabled = needs_disabled(cost_filter, COST_BUCKETS)
    return (
        store_disabled,
        store_disabled,
        sales_disabled,
        sales_disabled,
        cost_disabled,
        cost_disabled,
    )


@app.callback(
    Output("detail-open-close", "figure"),
    Output("detail-invest", "figure"),
    Output("detail-net-growth", "figure"),
    Output("detail-kpis", "children"),
    Input("detail-brand", "value"),
    Input("detail-year-range", "value"),
    Input("detail-profit", "value"),
)
def update_detail(brand, year_range, profit_rate):
    start_year, end_year = sorted([int(year_range[0]), int(year_range[1])])
    subset = df_all[
        (df_all["브랜드"] == brand)
        & (df_all["연도"] >= start_year)
        & (df_all["연도"] <= end_year)
    ].copy()
    subset = subset.sort_values("연도")
    if subset.empty:
        empty = make_empty_fig("데이터가 없습니다.", height=320)
        return empty, empty, empty, html.P("선택한 조건에 해당하는 데이터가 없습니다.")

    subset["연순이익"] = subset["평균매출액"] * profit_rate
    subset["회수기간"] = np.where(
        subset["연순이익"] > 0, subset["창업비용합계"] / subset["연순이익"], np.nan
    )

    open_close_fig = go.Figure()
    open_close_fig.add_trace(
        go.Bar(
            x=subset["연도"],
            y=subset["신규개점"],
            name="신규개점",
            marker_color="#2ecc71",
        )
    )
    open_close_fig.add_trace(
        go.Bar(
            x=subset["연도"],
            y=-subset["폐점수"],
            name="폐점",
            marker_color="#e74c3c",
        )
    )
    open_close_fig.update_layout(
        barmode="relative",
        template="plotly_white",
        xaxis_title="연도",
        yaxis_title="매장 수 (위=개점, 아래=폐점)",
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    invest_fig = go.Figure()
    invest_fig.add_trace(
        go.Scatter(
            x=subset["연도"],
            y=subset["창업비용_만원"],
            name="창업비용 (만원)",
            mode="lines+markers",
            line=dict(color="#2980b9", width=4),
        )
    )
    invest_fig.add_trace(
        go.Scatter(
            x=subset["연도"],
            y=subset["평균매출액"] / 10,
            name="연매출 (만원)",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#c0392b", width=4),
        )
    )
    invest_fig.update_layout(
        template="plotly_white",
        xaxis_title="연도",
        yaxis=dict(title="창업비용 (만원)"),
        yaxis2=dict(
            title="연매출 (만원)",
            overlaying="y",
            side="right",
        ),
        margin=dict(l=40, r=40, t=40, b=20),
        legend=dict(orientation="h"),
    )

    net_growth_fig = go.Figure(
        go.Bar(
            x=subset["연도"],
            y=subset["순증가율"],
            marker_color=np.where(subset["순증가율"] >= 0, "#1abc9c", "#e67e22"),
        )
    )
    net_growth_fig.update_layout(
        template="plotly_white",
        xaxis_title="연도",
        yaxis_title="순증가율 (%)",
        margin=dict(l=20, r=20, t=40, b=20),
    )

    latest = subset.iloc[-1]
    growth = (
        (subset.iloc[-1]["가맹점수"] / subset.iloc[0]["가맹점수"] - 1) * 100
        if subset.iloc[0]["가맹점수"] > 0
        else np.nan
    )
    cards = dbc.Row(
        [
            dbc.Col(
                build_metric_card(
                    f"회수기간 ({int(latest['연도'])}년)",
                    f"{latest['회수기간']:.1f}년" if not pd.isna(latest["회수기간"]) else "N/A",
                    "순이익률 가정 반영",
                    "info",
                ),
                md=6,
            ),
            dbc.Col(
                build_metric_card(
                    "평균 순증가율",
                    f"{subset['순증가율'].mean():.1f}%" if subset["순증가율"].notna().any() else "N/A",
                    f"{start_year}~{end_year} 평균",
                    "success",
                ),
                md=6,
            ),
            dbc.Col(
                build_metric_card(
                    "누적 가맹점 성장",
                    f"{growth:.1f}%" if not pd.isna(growth) else "N/A",
                    "기간의 첫 연도 대비",
                    "warning",
                ),
                md=6,
                className="mt-3",
            ),
            dbc.Col(
                build_metric_card(
                    f"연 순이익 ({int(latest['연도'])}년)",
                    format_currency(latest["연순이익"] / 10, "만원")
                    if latest["연순이익"] > 0
                    else "N/A",
                    f"순이익률 {int(profit_rate*100)}% 가정",
                    "danger",
                ),
                md=6,
                className="mt-3",
            ),
        ],
        className="g-2",
    )
    return open_close_fig, invest_fig, net_growth_fig, cards


if __name__ == "__main__":
    app.run(debug=True)

