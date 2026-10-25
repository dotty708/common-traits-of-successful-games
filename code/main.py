import streamlit as st
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Steam 게임 K-Means 분석",
    layout="wide"
)

st.title("Steam 게임 K-Means 군집 분석")
st.write("연구 질문: 인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?")


# ==================================================
# 1. 데이터 불러오기
# ==================================================

url = "https://raw.githubusercontent.com/dotty708/common-traits-of-successful-games/refs/heads/main/data/games_march2025_popularity_reduced.csv"


@st.cache_data
def load_data():
    return pd.read_csv(url, encoding="utf-8")


with st.spinner("데이터 불러오는 중..."):
    df = load_data()


st.success("데이터를 정상적으로 불러왔습니다.")

col1, col2 = st.columns(2)

col1.metric("게임 수", f"{len(df):,}")
col2.metric("열 수", len(df.columns))

st.subheader("원본 데이터")
st.dataframe(df.head(10), use_container_width=True)


# ==================================================
# 2. estimated_owners 변환
# ==================================================

def convert_owners(value):
    try:
        value = str(value).replace(",", "")
        parts = value.split("-")

        minimum = float(parts[0].strip())
        maximum = float(parts[1].strip())

        return (minimum + maximum) / 2

    except:
        return np.nan


df["owners_midpoint"] = df["estimated_owners"].apply(convert_owners)


st.header("1. estimated_owners 변환")

st.write(
    "estimated_owners의 범위에서 최솟값과 최댓값의 중앙값을 계산했습니다."
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
# 3. K-Means에 사용할 변수
# ==================================================

st.header("2. 군집화에 사용할 변수")

features = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "num_reviews_total",
    "pct_pos_total",
    "average_playtime_forever"
]

st.write("""
이번 1차 군집화에서는 다음 6개 변수를 사용합니다.

- 가격
- 추정 소유자 수
- 최대 동시 접속자 수
- 전체 리뷰 수
- 긍정 평가 비율
- 평균 플레이타임
""")


# ==================================================
# 4. 전처리
# ==================================================

X = df[features].copy()

for column in features:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )

# 결측치를 중앙값으로 처리
for column in features:
    X[column] = X[column].fillna(
        X[column].median()
    )


# 값의 차이가 매우 큰 변수 로그 변환
log_columns = [
    "price",
    "owners_midpoint",
    "peak_ccu",
    "num_reviews_total",
    "average_playtime_forever"
]

for column in log_columns:
    X[column] = np.log1p(
        X[column].clip(lower=0)
    )


# 표준화
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


st.success("전처리가 완료되었습니다.")


# ==================================================
# 5. K-Means
# ==================================================

st.header("3. K-Means 실행")

k = st.slider(
    "군집 개수 K",
    min_value=2,
    max_value=6,
    value=3
)


with st.spinner("K-Means 계산 중..."):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["cluster"] = model.fit_predict(
        X_scaled
    )


st.success(f"K={k} 군집화가 완료되었습니다.")


# ==================================================
# 6. 군집 크기
# ==================================================

st.header("4. 군집별 게임 수")

cluster_count = (
    df["cluster"]
    .value_counts()
    .sort_index()
    .reset_index()
)

cluster_count.columns = [
    "군집",
    "게임 수"
]

st.dataframe(
    cluster_count,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 7. 군집별 평균
# ==================================================

st.header("5. 군집별 특징")

cluster_profile = (
    df.groupby("cluster")[features]
    .mean()
    .round(2)
)

cluster_profile.columns = [
    "평균 가격",
    "평균 추정 소유자 수",
    "평균 peak_ccu",
    "평균 리뷰 수",
    "평균 긍정 평가율",
    "평균 플레이타임"
]

st.dataframe(
    cluster_profile,
    use_container_width=True
)


# ==================================================
# 8. 군집별 대표 게임
# ==================================================

st.header("6. 각 군집의 대표 게임")

for cluster in range(k):

    st.subheader(f"군집 {cluster}")

    cluster_games = (
        df[df["cluster"] == cluster]
        [
            [
                "name",
                "price",
                "estimated_owners",
                "peak_ccu",
                "num_reviews_total",
                "pct_pos_total",
                "average_playtime_forever"
            ]
        ]
        .sort_values(
            "peak_ccu",
            ascending=False
        )
        .head(10)
    )

    st.dataframe(
        cluster_games,
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# 9. 해석
# ==================================================

st.header("7. 결과에서 확인할 점")

st.write("""
군집별 평균값과 대표 게임을 비교해 다음 내용을 확인합니다.

1. 인기 게임들이 하나의 군집에 모이는가?
2. 소유자 수가 많은 게임은 peak_ccu도 높은가?
3. 인기 게임은 리뷰 수도 많은가?
4. 인기 게임의 긍정 평가율은 높은가?
5. 평균 플레이타임에도 차이가 나타나는가?
6. 가격에 따라 군집의 특징이 달라지는가?
""")
