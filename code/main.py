import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Steam 게임 K-Means 분석",
    layout="wide"
)

st.title("Steam 게임 인기 특성 군집 분석")
st.write("연구 질문: 인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?")


# ==================================================
# 1. 데이터 불러오기
# ==================================================

url = "https://raw.githubusercontent.com/dotty708/common-traits-of-successful-games/refs/heads/main/data/games_march2025_popularity_reduced.csv"


@st.cache_data
def load_data():
    return pd.read_csv(url, encoding="utf-8")


df = load_data()


st.header("1. 데이터 기본 정보")

col1, col2 = st.columns(2)

col1.metric("게임 수", f"{len(df):,}")
col2.metric("열 수", len(df.columns))

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ==================================================
# 2. estimated_owners 숫자로 변환
# ==================================================

st.header("2. 데이터 전처리")


def owner_midpoint(value):

    try:
        text = str(value).replace(",", "")

        parts = text.split("-")

        minimum = int(parts[0].strip())
        maximum = int(parts[1].strip())

        return (minimum + maximum) / 2

    except:
        return np.nan


df["owners_midpoint"] = df["estimated_owners"].apply(
    owner_midpoint
)


st.write(
    "estimated_owners의 범위에서 최솟값과 최댓값의 "
    "중앙값을 계산해 owners_midpoint를 만들었습니다."
)

st.dataframe(
    df[
        [
            "name",
            "estimated_owners",
            "owners_midpoint"
        ]
    ].head(10),
    use_container_width=True
)


# ==================================================
# 3. 군집에 사용할 특성
# ==================================================

st.header("3. K-Means에 사용할 특성")

features = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "num_reviews_total",
    "pct_pos_total",
    "average_playtime_forever",
    "metacritic_score"
]

st.write(features)


# 숫자로 변환
for column in features:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# 메타크리틱 0점은 실제 0점보다
# 정보 없음일 가능성이 높으므로 결측 처리
df["metacritic_score"] = df[
    "metacritic_score"
].replace(0, np.nan)


# ==================================================
# 4. 결측치 처리
# ==================================================

cluster_df = df[features].copy()

for column in features:

    cluster_df[column] = cluster_df[
        column
    ].fillna(
        cluster_df[column].median()
    )


st.write(
    "K-Means는 결측치를 처리할 수 없기 때문에 "
    "각 열의 결측치를 중앙값으로 채웠습니다."
)


# ==================================================
# 5. 로그 변환
# ==================================================

st.header("4. 데이터 스케일 조정")

log_columns = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "num_reviews_total",
    "average_playtime_forever"
]

for column in log_columns:

    cluster_df[column] = np.log1p(
        cluster_df[column].clip(lower=0)
    )


st.write(
    "소수의 초대형 게임이 K-Means 결과를 지배하지 않도록 "
    "가격, 소유자 수, 동시 접속자 수, 리뷰 수, 플레이타임에 "
    "log(1+x) 변환을 적용했습니다."
)


# ==================================================
# 6. 표준화
# ==================================================

scaler = StandardScaler()

X = scaler.fit_transform(
    cluster_df
)


# ==================================================
# 7. 적절한 K 확인
# ==================================================

st.header("5. 군집 개수 K 확인")

scores = []

for k in range(2, 7):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    score = silhouette_score(
        X,
        labels
    )

    scores.append({
        "K": k,
        "실루엣 점수": round(score, 4)
    })


score_df = pd.DataFrame(scores)

st.dataframe(
    score_df,
    use_container_width=True,
    hide_index=True
)


best_k = int(
    score_df.loc[
        score_df["실루엣 점수"].idxmax(),
        "K"
    ]
)

st.success(
    f"실루엣 점수가 가장 높은 군집 수는 K = {best_k} 입니다."
)


# ==================================================
# 8. K-Means 실행
# ==================================================

st.header("6. K-Means 군집화")

selected_k = st.slider(
    "군집 개수(K)",
    min_value=2,
    max_value=6,
    value=best_k
)


kmeans = KMeans(
    n_clusters=selected_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)


# ==================================================
# 9. 군집별 게임 수
# ==================================================

st.subheader("군집별 게임 수")

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_counts.columns = [
    "군집",
    "게임 수"
]

st.dataframe(
    cluster_counts,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 10. 군집별 실제 특성
# ==================================================

st.subheader("군집별 평균 특성")

profile_columns = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "num_reviews_total",
    "pct_pos_total",
    "average_playtime_forever",
    "metacritic_score"
]

cluster_profile = (
    df.groupby("cluster")[profile_columns]
    .mean()
    .round(2)
)

cluster_profile = cluster_profile.rename(
    columns={
        "price": "평균 가격",
        "owners_midpoint": "평균 추정 소유자",
        "peak_ccu": "평균 peak_ccu",
        "num_reviews_total": "평균 리뷰 수",
        "pct_pos_total": "평균 긍정 평가율",
        "average_playtime_forever": "평균 플레이타임",
        "metacritic_score": "평균 메타크리틱"
    }
)

st.dataframe(
    cluster_profile,
    use_container_width=True
)


# ==================================================
# 11. 군집별 대표 게임
# ==================================================

st.header("7. 군집별 게임 확인")

for cluster_number in range(selected_k):

    st.subheader(
        f"군집 {cluster_number}"
    )

    games = (
        df[df["cluster"] == cluster_number]
        [
            [
                "name",
                "price",
                "estimated_owners",
                "peak_ccu",
                "num_reviews_total",
                "pct_pos_total"
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


# ==================================================
# 12. 분석 안내
# ==================================================

st.header("8. 결과 해석")

st.write("""
각 군집의 평균값과 대표 게임을 비교합니다.

특히 다음을 확인합니다.

- 소유자 수가 많은 게임이 같은 군집에 모이는가?
- peak_ccu가 높은 게임들이 같은 군집에 모이는가?
- 인기 게임은 리뷰 수도 많은가?
- 인기 게임의 긍정 평가율은 높은가?
- 인기 게임의 평균 플레이타임은 긴가?
- 가격과 인기도 사이에 특징이 나타나는가?

이를 이용해 인기 게임들이 공통적으로 보이는 특징을 찾습니다.
""")
