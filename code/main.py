import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="Steam K-Means 분석",
    layout="wide"
)

st.title("Steam 게임 K-Means 군집 분석")

st.write(
    "데이터셋의 열을 직접 선택하여 "
    "K-Means 군집화 결과를 비교할 수 있습니다."
)


# =========================================================
# 1. 데이터 불러오기
# =========================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "dotty708/common-traits-of-successful-games/"
    "refs/heads/main/data/"
    "games_march2025_popularity_reduced.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    with st.spinner("데이터를 불러오는 중..."):
        df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =========================================================
# 2. estimated_owners → 중앙값 변환
# =========================================================

# 파생 변수를 추가하기 전 원본 데이터 개요를 보관합니다.
original_shape = df.shape
column_summary = pd.DataFrame({
    "열 이름": df.columns,
    "자료형": df.dtypes.astype(str).values,
    "결측치 수": df.isna().sum().values,
    "결측 비율(%)": (df.isna().mean() * 100).round(2).values,
})

def convert_owners(value):

    try:
        text = str(value).replace(",", "").strip()

        parts = text.split("-")

        if len(parts) != 2:
            return np.nan

        minimum = float(parts[0].strip())
        maximum = float(parts[1].strip())

        if minimum < 0 or maximum < 0:
            return np.nan

        return (minimum + maximum) / 2

    except:
        return np.nan


if "estimated_owners" in df.columns:

    df["owners_midpoint"] = (
        df["estimated_owners"]
        .apply(convert_owners)
    )


# =========================================================
# 3. 기본 정보
# =========================================================

st.header("1. 데이터 정보")

col1, col2, col3 = st.columns(3)

col1.metric(
    "게임 수",
    f"{len(df):,}"
)

col2.metric(
    "원본 열 수",
    original_shape[1]
)

col3.metric(
    "결측값",
    f"{int(column_summary['결측치 수'].sum()):,}"
)


st.subheader("원본 열별 자료형과 결측치")
st.dataframe(column_summary, use_container_width=True, hide_index=True)

with st.expander("데이터셋 확인"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# 4. K-Means에 사용 가능한 수치형 열 찾기
# =========================================================

# 숫자형 열 자동 탐색
numeric_columns = (
    df.select_dtypes(
        include=[np.number]
    )
    .columns
    .tolist()
)


# appid는 게임 식별 번호이므로 분석 변수에서 제외
if "appid" in numeric_columns:
    numeric_columns.remove("appid")


# 추천 기본값
recommended_features = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "positive",
    "negative"
]

default_features = [
    column
    for column in recommended_features
    if column in numeric_columns
]


# =========================================================
# 5. 분석 설정
# =========================================================

st.header("2. 군집화 조건 설정")

st.write(
    "아래에서 K-Means에 사용할 열을 직접 선택하세요."
)


selected_features = st.multiselect(
    "군집화에 사용할 열",
    options=numeric_columns,
    default=default_features
)


# 최소 2개 필요
if len(selected_features) < 2:

    st.warning(
        "군집화를 위해 최소 2개의 열을 선택하세요."
    )

    st.stop()


# =========================================================
# 6. 로그 변환 설정
# =========================================================

recommended_log = [
    "peak_ccu",
    "positive",
    "negative"
]

default_log = [
    column
    for column in recommended_log
    if column in selected_features
]


log_features = st.multiselect(
    "log1p 변환을 적용할 열",
    options=selected_features,
    default=default_log
)


st.caption(
    "값의 차이가 매우 큰 변수에 log1p를 적용하면 "
    "일부 초대형 게임이 군집 결과를 지나치게 지배하는 것을 "
    "줄일 수 있습니다."
)


# =========================================================
# 7. K 선택
# =========================================================

k = st.slider(
    "군집 개수 K",
    min_value=2,
    max_value=8,
    value=3,
    step=1
)


# =========================================================
# 8. 선택한 조건 표시
# =========================================================

st.subheader("현재 분석 설정")

setting_col1, setting_col2, setting_col3 = (
    st.columns(3)
)

setting_col1.write("**선택한 변수**")

setting_col1.write(
    ", ".join(selected_features)
)

setting_col2.write("**log1p 적용**")

if log_features:

    setting_col2.write(
        ", ".join(log_features)
    )

else:

    setting_col2.write("없음")


setting_col3.write("**군집 개수**")

setting_col3.write(
    f"K = {k}"
)


# =========================================================
# 9. 선택된 데이터 만들기
# =========================================================

analysis_df = df.copy()

X = analysis_df[
    selected_features
].copy()


# 선택한 변수만 기준으로 결측치를 처리합니다.
# 원본 결측, 숫자 변환 실패, 무한대, 음수를 구분해 집계합니다.
missing_before = X.isna().sum()
converted = X.apply(pd.to_numeric, errors="coerce")
conversion_failures = (X.notna() & converted.isna()).sum()
infinite_mask = pd.DataFrame(
    np.isinf(converted.to_numpy(dtype=float)),
    index=converted.index,
    columns=converted.columns,
)
infinite_counts = infinite_mask.sum()
X = converted.mask(infinite_mask)

# 기존 앱의 규칙을 유지합니다: 선택 변수의 음수는 분석에서 제외합니다.
negative_mask = X.lt(0)
negative_counts = negative_mask.sum()
X = X.mask(negative_mask)

preprocessing_summary = pd.DataFrame({
    "분석 변수": selected_features,
    "기존 결측치": missing_before.reindex(selected_features).values,
    "숫자 변환 실패": conversion_failures.reindex(selected_features).values,
    "무한대": infinite_counts.reindex(selected_features).values,
    "음수 → 결측치": negative_counts.reindex(selected_features).values,
    "처리 후 결측치": X.isna().sum().reindex(selected_features).values,
})

# 하나의 행에 여러 결측치가 있어도 제외 행은 한 번만 셉니다.
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()
analysis_df = analysis_df.loc[valid_rows].copy()

original_count = len(df)
used_count = len(X)
removed_count = original_count - used_count

st.subheader("결측치 처리 방법 및 결과")
st.write(
    "선택한 분석 변수 중 하나라도 결측치가 있는 행은 제외했습니다. "
    "숫자로 변환할 수 없는 값과 무한대도 결측치로 처리했습니다. "
    "기존 앱의 규칙에 따라 음수도 결측치로 바꿨으며, "
    "평균이나 0으로 값을 채우지는 않았습니다. "
    "선택하지 않은 열에만 결측치가 있는 행은 유지합니다."
)
st.caption(
    "owners_midpoint의 결측치에는 estimated_owners가 비어 있거나 "
    "범위를 숫자로 해석하지 못한 경우가 포함됩니다. "
    "음수 제외 규칙은 해당 변수에서 음수가 유효한 값인지 확인한 뒤 적용해야 합니다."
)
st.dataframe(preprocessing_summary, use_container_width=True, hide_index=True)

col1, col2, col3 = st.columns(3)
col1.metric("처리 전 행 수", f"{original_count:,}")
col2.metric("처리 후 행 수", f"{used_count:,}")
col3.metric("제외한 행 수", f"{removed_count:,}")
st.caption(
    f"원본 데이터: {original_shape[0]:,}행 × {original_shape[1]}열 · "
    f"분석 입력: {used_count:,}행 × {len(selected_features)}개 변수. "
    "표의 결측치 수는 셀 개수이므로 합계가 제외 행 수와 다를 수 있습니다. "
    "분석 변수 선택에 따라 제외 행 수가 달라집니다."
)

if used_count < k:
    st.error("결측치 처리 후 분석 가능한 데이터 수보다 K가 큽니다. 변수나 K를 조정하세요.")
    st.stop()


# =========================================================
# 12. log1p 변환
# =========================================================

for column in log_features:

    X[column] = np.log1p(
        X[column]
    )


# =========================================================
# 13. StandardScaler
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# =========================================================
# 14. K-Means
# =========================================================

st.header("3. K-Means 실행")


with st.spinner("K-Means 계산 중..."):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(
        X_scaled
    )


analysis_df["cluster"] = labels

analysis_df["cluster_name"] = (
    analysis_df["cluster"]
    .apply(
        lambda x: f"군집 {x}"
    )
)


st.success(
    f"{used_count:,}개 게임을 이용하여 "
    f"K={k} 군집화를 완료했습니다."
)


# =========================================================
# 15. PCA
# =========================================================

st.header("4. K-Means 군집 분포")


pca = PCA(
    n_components=2
)

pca_result = pca.fit_transform(
    X_scaled
)


analysis_df["PCA1"] = (
    pca_result[:, 0]
)

analysis_df["PCA2"] = (
    pca_result[:, 1]
)


explained_variance = (
    pca.explained_variance_ratio_.sum()
    * 100
)


st.write(
    f"선택한 {len(selected_features)}개 특성을 "
    f"PCA를 이용하여 2차원으로 표현했습니다. "
    f"두 축의 설명력은 약 "
    f"**{explained_variance:.1f}%** 입니다."
)


# =========================================================
# 16. 그래프용 샘플
# =========================================================

if len(analysis_df) > 15000:

    plot_df = analysis_df.sample(
        15000,
        random_state=42
    )

    st.caption(
        "화면 렌더링 속도를 위해 그래프에는 "
        "15,000개 게임만 무작위로 표시합니다. "
        "K-Means 계산은 전체 데이터를 사용했습니다."
    )

else:

    plot_df = analysis_df


# =========================================================
# 17. PCA Plotly 그래프
# =========================================================

hover_columns = {}


for column in selected_features:

    hover_columns[column] = True


hover_columns["PCA1"] = False
hover_columns["PCA2"] = False


fig_pca = px.scatter(
    plot_df,
    x="PCA1",
    y="PCA2",
    color="cluster_name",
    hover_name="name",
    hover_data=hover_columns,
    opacity=0.55,
    title="선택한 특성에 따른 K-Means 군집"
)


fig_pca.update_layout(
    xaxis_title="PCA 1",
    yaxis_title="PCA 2",
    legend_title="군집"
)


st.plotly_chart(
    fig_pca,
    use_container_width=True
)


# =========================================================
# 18. 군집별 게임 수
# =========================================================

st.header("5. 군집별 게임 수")


cluster_counts = (
    analysis_df[
        "cluster_name"
    ]
    .value_counts()
    .sort_index()
    .reset_index()
)


cluster_counts.columns = [
    "군집",
    "게임 수"
]


fig_count = px.bar(
    cluster_counts,
    x="군집",
    y="게임 수",
    text_auto=",",
    title="군집별 게임 수"
)


st.plotly_chart(
    fig_count,
    use_container_width=True
)


# =========================================================
# 19. 군집별 평균
# =========================================================

st.header("6. 군집별 특성 평균")


cluster_profile = (
    analysis_df
    .groupby(
        "cluster_name"
    )[selected_features]
    .mean()
    .round(2)
    .reset_index()
)


st.dataframe(
    cluster_profile,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 20. 특정 특성 그래프
# =========================================================

st.subheader(
    "군집별 특성 비교"
)


compare_feature = st.selectbox(
    "그래프로 비교할 특성",
    options=selected_features
)


compare_df = (
    analysis_df
    .groupby(
        "cluster_name"
    )[compare_feature]
    .mean()
    .reset_index()
)


compare_df.columns = [
    "군집",
    "평균값"
]


fig_compare = px.bar(
    compare_df,
    x="군집",
    y="평균값",
    text_auto=".3s",
    title=(
        f"군집별 {compare_feature} 평균"
    )
)


fig_compare.update_layout(
    yaxis_title=compare_feature
)


st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# =========================================================
# 21. 두 변수 관계 직접 선택
# =========================================================

st.header("7. 두 특성의 관계")


x_feature = st.selectbox(
    "X축",
    options=selected_features,
    index=0
)


y_options = [
    column
    for column in selected_features
    if column != x_feature
]


y_feature = st.selectbox(
    "Y축",
    options=y_options
)


relationship_plot = plot_df[
    [
        "name",
        "cluster_name",
        x_feature,
        y_feature
    ]
].copy()


fig_relation = px.scatter(
    relationship_plot,
    x=x_feature,
    y=y_feature,
    color="cluster_name",
    hover_name="name",
    opacity=0.55,
    title=(
        f"{x_feature}와 "
        f"{y_feature}의 관계"
    )
)


fig_relation.update_layout(
    xaxis_title=x_feature,
    yaxis_title=y_feature,
    legend_title="군집"
)


st.plotly_chart(
    fig_relation,
    use_container_width=True
)


# =========================================================
# 22. 군집별 대표 게임
# =========================================================

st.header("8. 군집별 게임 확인")


# 정렬 기준도 직접 선택
sort_feature = st.selectbox(
    "대표 게임 정렬 기준",
    options=selected_features,
    index=(
        selected_features.index("peak_ccu")
        if "peak_ccu" in selected_features
        else 0
    )
)


for cluster_number in range(k):

    st.subheader(
        f"군집 {cluster_number}"
    )


    display_columns = [
        "name"
    ]


    # estimated_owners가 있으면 같이 표시
    if "estimated_owners" in analysis_df.columns:

        display_columns.append(
            "estimated_owners"
        )


    for column in selected_features:

        if column not in display_columns:

            display_columns.append(
                column
            )


    games = (
        analysis_df[
            analysis_df["cluster"]
            == cluster_number
        ][display_columns]
        .sort_values(
            sort_feature,
            ascending=False
        )
        .head(15)
    )


    st.dataframe(
        games,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 23. 분석 설명
# =========================================================

st.header("9. 분석 방법")


st.write(
    """
### 분석 과정

1. 데이터셋에서 사용자가 군집화에 사용할 특성을 선택합니다.
2. -1 등 음수로 기록된 비정상값을 결측값으로 처리합니다.
3. 선택한 핵심 특성이 누락된 게임은 군집화에서 제외합니다.
4. 사용자가 지정한 변수에 `log1p` 변환을 적용합니다.
5. StandardScaler를 이용해 변수들의 단위를 표준화합니다.
6. K-Means를 이용하여 게임들을 K개의 군집으로 분류합니다.
7. PCA를 이용해 군집 결과를 2차원으로 시각화합니다.
8. 각 군집의 평균값과 대표 게임을 비교하여 특징을 해석합니다.
"""
)
