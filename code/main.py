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
    page_title="Steam 게임 K-Means 분석",
    layout="wide"
)

st.title("Steam 게임 인기 특성 K-Means 군집 분석")

st.write(
    "연구 질문: **인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?**"
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
    with st.spinner("Steam 게임 데이터를 불러오는 중..."):
        df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


st.success("데이터를 정상적으로 불러왔습니다.")


# =========================================================
# 2. 기본 정보
# =========================================================

st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric(
    "전체 게임 수",
    f"{len(df):,}"
)

col2.metric(
    "전체 열 수",
    f"{len(df.columns):,}"
)

col3.metric(
    "전체 결측값",
    f"{df.isna().sum().sum():,}"
)

with st.expander("원본 데이터 일부 보기"):
    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# 3. estimated_owners 변환
# =========================================================

st.header("2. 데이터 전처리")


def convert_owners(value):
    """
    예:
    '20000000 - 50000000'
        ↓
    35000000
    """

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


df["owners_midpoint"] = df["estimated_owners"].apply(
    convert_owners
)


st.write(
    "`estimated_owners`는 정확한 소유자 수가 아니라 "
    "범위로 저장되어 있으므로 최솟값과 최댓값의 "
    "**중앙값(midpoint)** 으로 변환했습니다."
)

owner_example_columns = [
    "name",
    "estimated_owners",
    "owners_midpoint"
]

st.dataframe(
    df[owner_example_columns].head(10),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 4. 군집화 변수 선택
# =========================================================

st.header("3. K-Means 군집화 변수")

cluster_features = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "positive",
    "negative"
]

st.write(
    """
군집화에는 다음 5개 특성만 사용합니다.

- **price** : 게임 가격
- **owners_midpoint** : 추정 소유자 수 범위의 중앙값
- **peak_ccu** : 최대 동시 접속자 수
- **positive** : 긍정 리뷰 수
- **negative** : 부정 리뷰 수
"""
)


# =========================================================
# 5. 숫자 변환 및 이상값 처리
# =========================================================

for column in cluster_features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# 논리적으로 불가능한 음수값을 결측치로 처리
for column in cluster_features:

    df.loc[
        df[column] < 0,
        column
    ] = np.nan


before_count = len(df)

cluster_df = df.dropna(
    subset=cluster_features
).copy()

after_count = len(cluster_df)

removed_count = before_count - after_count


col1, col2, col3 = st.columns(3)

col1.metric(
    "원본 게임",
    f"{before_count:,}"
)

col2.metric(
    "군집화 사용 게임",
    f"{after_count:,}"
)

col3.metric(
    "제외된 게임",
    f"{removed_count:,}"
)


st.caption(
    "-1 등 논리적으로 불가능한 음수값과 "
    "군집화 핵심 변수의 결측값이 있는 게임은 "
    "K-Means 분석에서 제외했습니다."
)


# =========================================================
# 6. log1p 변환
# =========================================================

st.header("4. 로그 변환 및 표준화")

model_df = cluster_df[
    cluster_features
].copy()


# 사용자가 정한 세 변수만 log1p 적용
log_columns = [
    "peak_ccu",
    "positive",
    "negative"
]


for column in log_columns:

    model_df[column] = np.log1p(
        model_df[column]
    )


st.write(
    """
값의 분포가 매우 큰 다음 변수에는 **log1p 변환**을 적용했습니다.

- peak_ccu
- positive
- negative

`price`와 `owners_midpoint`는 로그 변환하지 않고,
이후 모든 변수를 StandardScaler로 표준화합니다.
"""
)


# =========================================================
# 7. 표준화
# =========================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    model_df
)


# =========================================================
# 8. K-Means 실행
# =========================================================

st.header("5. K-Means 군집화")

k = st.slider(
    "군집 개수 K",
    min_value=2,
    max_value=6,
    value=3,
    step=1
)


with st.spinner("K-Means 군집을 계산하는 중..."):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        X_scaled
    )


cluster_df["cluster"] = cluster_labels

cluster_df["cluster_name"] = (
    cluster_df["cluster"]
    .apply(lambda x: f"군집 {x}")
)


st.success(
    f"K = {k} 기준 군집화가 완료되었습니다."
)


# =========================================================
# 9. PCA 2차원 변환
# =========================================================

st.header("6. 군집 분포 시각화")

pca = PCA(
    n_components=2,
    random_state=42
)

pca_result = pca.fit_transform(
    X_scaled
)

cluster_df["PCA1"] = pca_result[:, 0]
cluster_df["PCA2"] = pca_result[:, 1]


explained = (
    pca.explained_variance_ratio_.sum() * 100
)


st.write(
    f"5차원 군집 데이터를 화면에서 확인하기 위해 "
    f"PCA로 2차원으로 축소했습니다. "
    f"두 축이 설명하는 분산은 약 **{explained:.1f}%** 입니다."
)


# 데이터가 너무 많으면 브라우저가 느려질 수 있으므로
# 그래프 표시만 샘플링
if len(cluster_df) > 15000:

    plot_df = cluster_df.sample(
        15000,
        random_state=42
    )

    st.caption(
        "그래프 렌더링 속도를 위해 15,000개 게임을 "
        "무작위 표본으로 표시합니다. "
        "K-Means 자체는 전체 데이터를 사용했습니다."
    )

else:
    plot_df = cluster_df


fig_pca = px.scatter(
    plot_df,
    x="PCA1",
    y="PCA2",
    color="cluster_name",
    hover_name="name",
    hover_data={
        "price": ":.2f",
        "owners_midpoint": ":,.0f",
        "peak_ccu": ":,.0f",
        "positive": ":,.0f",
        "negative": ":,.0f",
        "PCA1": False,
        "PCA2": False
    },
    title="K-Means 군집 분포 (PCA 2차원 시각화)",
    opacity=0.55
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
# 10. 군집별 게임 수
# =========================================================

st.header("7. 군집별 게임 수")

cluster_counts = (
    cluster_df["cluster_name"]
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

fig_count.update_layout(
    xaxis_title="군집",
    yaxis_title="게임 수"
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)


# =========================================================
# 11. 군집별 실제 평균값
# =========================================================

st.header("8. 군집별 특성 비교")


profile = (
    cluster_df
    .groupby("cluster_name")[
        cluster_features
    ]
    .mean()
    .reset_index()
)


profile_display = profile.copy()

profile_display.columns = [
    "군집",
    "평균 가격",
    "평균 추정 소유자 수",
    "평균 peak_ccu",
    "평균 긍정 리뷰",
    "평균 부정 리뷰"
]


st.subheader("군집별 실제 평균값")

st.dataframe(
    profile_display,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# 12. 원하는 지표를 선택해서 군집 비교
# =========================================================

st.subheader("군집별 지표 비교")

metric_options = {
    "평균 가격": "price",
    "평균 추정 소유자 수": "owners_midpoint",
    "평균 peak_ccu": "peak_ccu",
    "평균 긍정 리뷰": "positive",
    "평균 부정 리뷰": "negative"
}


selected_metric_name = st.selectbox(
    "비교할 지표",
    list(metric_options.keys())
)

selected_metric = metric_options[
    selected_metric_name
]


metric_df = (
    cluster_df
    .groupby("cluster_name")[selected_metric]
    .mean()
    .reset_index()
)

metric_df.columns = [
    "군집",
    selected_metric_name
]


fig_metric = px.bar(
    metric_df,
    x="군집",
    y=selected_metric_name,
    text_auto=".3s",
    title=f"군집별 {selected_metric_name}"
)

fig_metric.update_layout(
    xaxis_title="군집",
    yaxis_title=selected_metric_name
)

st.plotly_chart(
    fig_metric,
    use_container_width=True
)


# =========================================================
# 13. 실제 인기도 지표 관계
# =========================================================

st.header("9. peak_ccu와 긍정 리뷰 관계")


relationship_df = cluster_df[
    (cluster_df["peak_ccu"] > 0) &
    (cluster_df["positive"] > 0)
].copy()


# 그래프가 너무 무거워지는 것 방지
if len(relationship_df) > 15000:

    relationship_plot_df = relationship_df.sample(
        15000,
        random_state=42
    )

else:
    relationship_plot_df = relationship_df


fig_relationship = px.scatter(
    relationship_plot_df,
    x="positive",
    y="peak_ccu",
    color="cluster_name",
    hover_name="name",
    hover_data={
        "estimated_owners": True,
        "price": ":.2f",
        "negative": ":,.0f"
    },
    log_x=True,
    log_y=True,
    opacity=0.55,
    title="긍정 리뷰 수와 최대 동시 접속자 수"
)

fig_relationship.update_layout(
    xaxis_title="긍정 리뷰 수 (로그 축)",
    yaxis_title="peak_ccu (로그 축)",
    legend_title="군집"
)

st.plotly_chart(
    fig_relationship,
    use_container_width=True
)


# =========================================================
# 14. 군집별 대표 게임
# =========================================================

st.header("10. 각 군집의 대표 게임")


for cluster_number in range(k):

    st.subheader(
        f"군집 {cluster_number}"
    )

    games = (
        cluster_df[
            cluster_df["cluster"] == cluster_number
        ][
            [
                "name",
                "price",
                "estimated_owners",
                "peak_ccu",
                "positive",
                "negative"
            ]
        ]
        .sort_values(
            "peak_ccu",
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
# 15. 결과 해석 안내
# =========================================================

st.header("11. 결과 해석")

st.write(
    """
군집별 평균값과 대표 게임을 비교하여 각 군집의 특징을 해석합니다.

특히 다음 내용을 확인합니다.

1. 높은 동시 접속자 수를 가진 게임들이 같은 군집에 모이는가?
2. 추정 소유자 수가 많은 게임들이 특정 군집에 집중되는가?
3. 긍정 리뷰와 부정 리뷰의 규모에 따라 군집이 어떻게 나뉘는가?
4. 가격이 군집 구분에 영향을 주는가?
5. 유명한 게임들이 공통적으로 속하는 군집이 존재하는가?

이후 각 군집의 특징을 분석하여
'인기 게임이 공통적인 특성을 가지고 있는가'를 탐색합니다.
"""
)
