
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ==================================================
# 1. 페이지 및 데이터 설정
# ==================================================

st.title("Steam 게임 K-Means 군집 분석")

st.write(
    "Steam 게임의 긍정 리뷰 수와 부정 리뷰 수를 "
    "기준으로 게임을 3개의 군집으로 구분합니다."
)

st.info(
    "연구 질문: 인기 있는 게임은 공통적인 특성을 "
    "가지고 있는가?"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "dotty708/common-traits-of-successful-games/"
    "refs/heads/main/data/"
    "games_march2025_popularity_filtered.csv"
)

FEATURES = ["positive", "negative"]
K = 3


@st.cache_data(ttl=300)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


try:
    with st.spinner("Steam 데이터를 불러오는 중..."):
        raw_df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ==================================================
# 2. 데이터 정보
# ==================================================

st.header("1. 데이터 정보")

if st.button("GitHub 데이터 새로 불러오기"):
    load_data.clear()
    st.rerun()

required_columns = ["positive", "negative"]

missing_columns = [
    c for c in required_columns
    if c not in raw_df.columns
]

if missing_columns:
    st.error(
        f"필수 열이 없습니다: {missing_columns}"
    )
    st.stop()

source_count = len(raw_df)

positive = pd.to_numeric(
    raw_df["positive"], errors="coerce"
)

negative = pd.to_numeric(
    raw_df["negative"], errors="coerce"
)

# 결측치, -1, 음수, 무한대 제거
invalid_mask = (
    ~np.isfinite(positive)
    | ~np.isfinite(negative)
    | (positive < 0)
    | (negative < 0)
)

# 긍정·부정 리뷰가 모두 0인 게임 제외
no_review_mask = (
    (positive == 0)
    & (negative == 0)
    & ~invalid_mask
)

valid_mask = ~(invalid_mask | no_review_mask)

df = raw_df.loc[valid_mask].copy()

df["positive"] = positive.loc[valid_mask]
df["negative"] = negative.loc[valid_mask]

df["total_reviews"] = (
    df["positive"] + df["negative"]
)

df["positive_rate"] = (
    df["positive"] / df["total_reviews"] * 100
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("원본 게임 수", f"{source_count:,}")
col2.metric("분석 게임 수", f"{len(df):,}")
col3.metric(
    "리뷰 없는 게임 제외",
    f"{int(no_review_mask.sum()):,}"
)
col4.metric(
    "비정상값 제외",
    f"{int(invalid_mask.sum()):,}"
)

st.caption(
    "긍정·부정 리뷰가 모두 0인 게임과 "
    "비정상값이 있는 게임을 제외했습니다. "
    "한쪽 리뷰 수만 0인 게임은 유지합니다."
)

with st.expander("원본 데이터 확인"):
    st.dataframe(
        raw_df.head(20),
        use_container_width=True
    )

with st.expander("전처리 후 데이터 확인"):
    st.dataframe(
        df.head(20),
        use_container_width=True
    )

if len(df) < K:
    st.error("군집화할 게임 수가 부족합니다.")
    st.stop()


# ==================================================
# 3. 군집화 설정
# ==================================================

st.header("2. 군집화 조건")

c1, c2, c3 = st.columns(3)

c1.metric("사용 변수", "2개")
c2.metric("군집 개수", "K = 3")
c3.metric("알고리즘", "K-Means")

st.write("**사용 변수**")
st.write("- positive: 긍정 리뷰 수")
st.write("- negative: 부정 리뷰 수")

st.write("**전처리 순서**")
st.write(
    "결측·비정상값 제거 → log1p 변환 "
    "→ StandardScaler 표준화 → K-Means"
)

# IQR 극단값 확인
with st.expander("극단값(IQR) 점검"):
    outlier_records = []

    for column in FEATURES:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)

        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = int(
            (
                (df[column] < lower)
                | (df[column] > upper)
            ).sum()
        )

        outlier_records.append({
            "변수": column,
            "IQR 하한": lower,
            "IQR 상한": upper,
            "극단값 후보 수": count
        })

    st.dataframe(
        pd.DataFrame(outlier_records),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "극단값은 삭제하지 않습니다. "
        "실제로 리뷰가 많은 인기 게임일 수 있기 "
        "때문입니다."
    )


# ==================================================
# 4. K-Means 학습
# ==================================================

st.header("3. K-Means 군집화")

# 로그 변환
X_log = np.log1p(
    df[FEATURES].to_numpy(dtype=float)
)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 서로 다른 데이터 포인트 확인
unique_count = len(
    np.unique(X_scaled, axis=0)
)

if unique_count < K:
    st.error(
        "서로 다른 데이터 포인트가 "
        "3개 미만이어서 군집화할 수 없습니다."
    )
    st.stop()

with st.spinner("K-Means 학습 중..."):
    model = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=10
    )

    original_labels = model.fit_predict(
        X_scaled
    )

df["_original_cluster"] = original_labels

# 군집 번호를 리뷰 총량 중앙값 순으로 정렬
# 0 = 리뷰 총량 중앙값이 가장 작은 군집
# 2 = 리뷰 총량 중앙값이 가장 큰 군집

cluster_order = (
    df.groupby("_original_cluster")[
        "total_reviews"
    ]
    .median()
    .sort_values()
    .index
    .tolist()
)

cluster_mapping = {
    old_label: new_label
    for new_label, old_label
    in enumerate(cluster_order)
}

df["cluster"] = (
    df["_original_cluster"]
    .map(cluster_mapping)
    .astype(int)
)

df.drop(
    columns=["_original_cluster"],
    inplace=True
)

df["cluster_name"] = (
    "군집 " + df["cluster"].astype(str)
)
# 군집화 결과를 다른 페이지에 자동 전달
st.session_state["steam_clustered_df"] = df.copy()

st.success(
    f"{len(df):,}개 게임을 "
    "3개의 군집으로 분류했습니다."
)

st.caption(
    "군집 번호는 리뷰 총량 중앙값의 "
    "오름차순으로 정렬했습니다. "
    "번호가 흥행 성공 순위를 의미하지는 않습니다."
)


# ==================================================
# 5. 군집 분포 시각화
# ==================================================

st.header("4. 군집 분포")

# 그래프 렌더링을 위한 샘플링
if len(df) > 15000:
    plot_df = df.sample(
        n=15000,
        random_state=42
    ).copy()

    st.caption(
        "그래프에는 무작위 15,000개 게임을 "
        "표시합니다. 군집화와 통계 계산에는 "
        "전체 데이터를 사용했습니다."
    )
else:
    plot_df = df.copy()

plot_df["log_positive"] = np.log1p(
    plot_df["positive"]
)

plot_df["log_negative"] = np.log1p(
    plot_df["negative"]
)

hover_columns = [
    "positive",
    "negative",
    "total_reviews",
    "positive_rate"
]

fig = px.scatter(
    plot_df,
    x="log_positive",
    y="log_negative",
    color="cluster_name",
    hover_name=(
        "name"
        if "name" in plot_df.columns
        else None
    ),
    hover_data=hover_columns,
    opacity=0.5,
    render_mode="webgl",
    title="긍정·부정 리뷰 기반 군집 분포",
    labels={
        "log_positive": "log1p(긍정 리뷰 수)",
        "log_negative": "log1p(부정 리뷰 수)",
        "cluster_name": "군집"
    }
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.caption(
    "그래프의 X축과 Y축은 log1p 변환값입니다. "
    "마우스를 올리면 실제 리뷰 수를 확인할 수 있습니다."
)


# ==================================================
# 6. PCA 시각화
# ==================================================

with st.expander("PCA 2차원 시각화"):
    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(
        X_scaled
    )

    pca_df = df[
        ["cluster_name"]
    ].copy()

    pca_df["PCA1"] = pca_result[:, 0]
    pca_df["PCA2"] = pca_result[:, 1]

    if "name" in df.columns:
        pca_df["name"] = df["name"]

    if len(pca_df) > 15000:
        pca_df = pca_df.sample(
            15000,
            random_state=42
        )

    fig_pca = px.scatter(
        pca_df,
        x="PCA1",
        y="PCA2",
        color="cluster_name",
        hover_name=(
            "name"
            if "name" in pca_df.columns
            else None
        ),
        opacity=0.5,
        render_mode="webgl",
        title="PCA 기반 군집 시각화"
    )

    st.plotly_chart(
        fig_pca,
        use_container_width=True
    )

    st.caption(
        "입력 변수가 정확히 2개이므로 "
        "PCA의 두 축은 원래 표준화된 데이터의 "
        "정보를 모두 표현합니다."
    )


# ==================================================
# 7. 군집별 게임 수
# ==================================================

st.header("5. 군집별 게임 수")

cluster_counts = (
    df["cluster_name"]
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

st.dataframe(
    cluster_counts,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 8. 군집별 통계
# ==================================================

st.header("6. 군집별 특성 분석")

cluster_summary = (
    df.groupby("cluster")
    .agg(
        game_count=("cluster", "size"),
        positive_mean=("positive", "mean"),
        negative_mean=("negative", "mean"),
        positive_median=("positive", "median"),
        negative_median=("negative", "median"),
        total_median=("total_reviews", "median"),
        positive_rate_mean=("positive_rate", "mean"),
        positive_sum=("positive", "sum"),
        negative_sum=("negative", "sum")
    )
    .reset_index()
)

cluster_summary["total_positive_rate"] = (
    cluster_summary["positive_sum"]
    / (
        cluster_summary["positive_sum"]
        + cluster_summary["negative_sum"]
    )
    * 100
)

display_summary = cluster_summary[
    [
        "cluster",
        "game_count",
        "positive_mean",
        "negative_mean",
        "total_median",
        "positive_rate_mean",
        "total_positive_rate"
    ]
].copy()

display_summary.columns = [
    "군집",
    "게임 수",
    "긍정 리뷰 평균",
    "부정 리뷰 평균",
    "전체 리뷰 수 중앙값",
    "게임별 긍정 비율 평균(%)",
    "군집 전체 긍정 비율(%)"
]

st.dataframe(
    display_summary.round(2),
    use_container_width=True,
    hide_index=True
)

# 평균 리뷰 수 비교
mean_df = cluster_summary[
    ["cluster", "positive_mean", "negative_mean"]
].melt(
    id_vars="cluster",
    var_name="리뷰 유형",
    value_name="평균 리뷰 수"
)

mean_df["리뷰 유형"] = mean_df[
    "리뷰 유형"
].replace({
    "positive_mean": "긍정 리뷰",
    "negative_mean": "부정 리뷰"
})

mean_df["군집"] = (
    "군집 " + mean_df["cluster"].astype(str)
)

fig_mean = px.bar(
    mean_df,
    x="군집",
    y="평균 리뷰 수",
    color="리뷰 유형",
    barmode="group",
    title="군집별 긍정·부정 리뷰 평균"
)

st.plotly_chart(
    fig_mean,
    use_container_width=True
)

# 긍정 비율 비교
rate_df = cluster_summary.copy()

rate_df["군집"] = (
    "군집 " + rate_df["cluster"].astype(str)
)

fig_rate = px.bar(
    rate_df,
    x="군집",
    y="total_positive_rate",
    text_auto=".1f",
    title="군집별 전체 긍정 리뷰 비율",
    labels={
        "total_positive_rate": "긍정 리뷰 비율(%)"
    },
    range_y=[0, 100]
)

st.plotly_chart(
    fig_rate,
    use_container_width=True
)

st.caption(
    "게임별 긍정 비율 평균은 각 게임의 비율을 "
    "동등하게 평균낸 값이고, 군집 전체 긍정 "
    "비율은 해당 군집의 긍정 리뷰 총합을 "
    "전체 리뷰 총합으로 나눈 값입니다."
)


# ==================================================
# 9. 군집별 대표 게임
# ==================================================

st.header("7. 군집별 대표 게임")

st.write(
    "각 군집에서 전체 리뷰 수가 많은 "
    "게임 15개를 표시합니다."
)

for cluster_number in range(K):

    st.subheader(f"군집 {cluster_number}")

    display_columns = [
        c for c in [
            "appid",
            "name",
            "positive",
            "negative",
            "total_reviews",
            "positive_rate"
        ]
        if c in df.columns
    ]

    games = (
        df[df["cluster"] == cluster_number]
        .sort_values(
            "total_reviews",
            ascending=False
        )
        .head(15)
    )

    st.dataframe(
        games[display_columns],
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 10. 군집 결과 CSV 다운로드
# ==================================================

st.header("8. 군집화 결과 저장")

st.write(
    "군집 번호를 CSV로 저장하면 "
    "다음 페이지에서 분류 모델의 "
    "정답 레이블로 사용할 수 있습니다."
)

# 원본 열을 유지하고 군집 번호 추가
export_columns = [
    c for c in raw_df.columns
    if c in df.columns and c != "cluster"
]

export_df = df[
    export_columns + ["cluster"]
].copy()

st.subheader("전체 데이터 + 군집 번호")

st.dataframe(
    export_df.head(20),
    use_container_width=True
)

st.download_button(
    label="전체 군집화 결과 CSV 다운로드",
    data=export_df.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="steam_clustered_games.csv",
    mime="text/csv"
)

# 다음 페이지에서 병합하기 위한 최소 데이터
if "appid" in df.columns:

    if df["appid"].duplicated().any():
        st.warning(
            "appid 중복이 있습니다. "
            "다른 데이터와 병합하기 전에 "
            "중복 여부를 확인하세요."
        )

    label_df = df[
        ["appid", "cluster"]
    ].copy()

    st.download_button(
        label="분류 모델용 군집 레이블 CSV 다운로드",
        data=label_df.to_csv(
            index=False
        ).encode("utf-8-sig"),
        file_name="steam_cluster_labels.csv",
        mime="text/csv"
    )


# ==================================================
# 11. 분석 방법 정리
# ==================================================

st.header("9. 분석 방법")

st.markdown("""
### 분석 과정

1. Steam 게임 데이터를 불러온다.
2. 긍정·부정 리뷰가 모두 0인 게임을 제외한다.
3. 결측값, 음수, 비정상값이 있는 행을 제외한다.
4. 긍정 리뷰와 부정 리뷰에 log1p 변환을 적용한다.
5. StandardScaler로 두 변수를 표준화한다.
6. K-Means(K=3)로 게임을 군집화한다.
7. 군집별 리뷰 수와 긍정 비율을 분석한다.
8. 각 게임에 군집 번호를 부여하여 CSV로 저장한다.

### 다음 단계

분류 모델에서는 출시 전에 확인할 수 있는
가격, 장르, 개발사, 배급사 등의 정보를 입력하고,
K-Means로 생성한 군집 번호를 정답으로 사용한다.

군집화에 사용한 긍정·부정 리뷰 수는
최종 분류 모델의 입력 변수에서 제외한다.
""")
