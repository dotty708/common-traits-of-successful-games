import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Steam 게임 데이터 분석",
    layout="wide"
)

st.title("Steam 게임 인기 데이터 분석")
st.write("연구 질문: 인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?")

# --------------------------------------------------
# 1. 데이터 불러오기
# --------------------------------------------------

url = "import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Steam 게임 데이터 분석",
    layout="wide"
)

st.title("Steam 게임 인기 데이터 분석")
st.write("연구 질문: 인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?")

# --------------------------------------------------
# 1. 데이터 불러오기
# --------------------------------------------------

url = "https://raw.githubusercontent.com/dotty708/common-traits-of-successful-games/refs/heads/main/data/games_march2025_popularity_reduced.csv"

df = pd.read_csv(url, encoding="utf-8")

st.header("1. 데이터 기본 정보")

col1, col2 = st.columns(2)

with col1:
    st.metric("행", df.shape[0])

with col2:
    st.metric("열", df.shape[1])

st.subheader("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 2. 데이터 타입
# --------------------------------------------------

st.header("2. 열별 데이터 타입")

dtype_df = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 타입": df.dtypes.astype(str).values
})

st.dataframe(dtype_df, use_container_width=True)


# --------------------------------------------------
# 3. 결측치 확인
# --------------------------------------------------

st.header("3. 결측치")

missing = pd.DataFrame({
    "열 이름": df.columns,
    "결측치 수": df.isna().sum().values
})

missing["결측 비율(%)"] = (
    missing["결측치 수"] / len(df) * 100
).round(2)

missing = missing.sort_values(
    "결측치 수",
    ascending=False
)

st.dataframe(missing, use_container_width=True)


# --------------------------------------------------
# 4. 수치형 데이터 통계
# --------------------------------------------------

st.header("4. 수치형 데이터 요약")

st.dataframe(
    df.describe(),
    use_container_width=True
)


# --------------------------------------------------
# 5. peak_ccu 분포
# --------------------------------------------------

if "peak_ccu" in df.columns:

    st.header("5. 최대 동시 접속자 수 분포")

    peak = pd.to_numeric(
        df["peak_ccu"],
        errors="coerce"
    ).dropna()

    peak = peak[peak >= 0]

    fig, ax = plt.subplots()

    ax.hist(
        np.log1p(peak),
        bins=50
    )

    ax.set_xlabel("log(1 + peak_ccu)")
    ax.set_ylabel("게임 수")
    ax.set_title("peak_ccu 로그 변환 분포")

    st.pyplot(fig)

    st.write(
        "0인 게임 수:",
        int((peak == 0).sum())
    )

    st.write(
        "최댓값:",
        int(peak.max())
    )


# --------------------------------------------------
# 6. 리뷰 수 분포
# --------------------------------------------------

if "num_reviews_total" in df.columns:

    st.header("6. 리뷰 수 분포")

    reviews = pd.to_numeric(
        df["num_reviews_total"],
        errors="coerce"
    ).dropna()

    reviews = reviews[reviews >= 0]

    fig, ax = plt.subplots()

    ax.hist(
        np.log1p(reviews),
        bins=50
    )

    ax.set_xlabel("log(1 + num_reviews_total)")
    ax.set_ylabel("게임 수")
    ax.set_title("리뷰 수 로그 변환 분포")

    st.pyplot(fig)

    st.write(
        "리뷰가 0개인 게임:",
        int((reviews == 0).sum())
    )

    st.write(
        "리뷰가 10개 미만인 게임:",
        int((reviews < 10).sum())
    )

    st.write(
        "리뷰가 100개 미만인 게임:",
        int((reviews < 100).sum())
    )


# --------------------------------------------------
# 7. 무료/유료 게임 확인
# --------------------------------------------------

if "price" in df.columns:

    st.header("7. 무료 / 유료 게임")

    price = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    free_count = int((price == 0).sum())
    paid_count = int((price > 0).sum())

    st.write("무료 게임:", free_count)
    st.write("유료 게임:", paid_count)


# --------------------------------------------------
# 8. estimated_owners 확인
# --------------------------------------------------

if "estimated_owners" in df.columns:

    st.header("8. estimated_owners 확인")

    st.write(
        "estimated_owners는 정확한 소유자 수가 아니라 "
        "범위 형태로 저장되어 있는지 확인합니다."
    )

    st.dataframe(
        df[["name", "estimated_owners"]].head(20),
        use_container_width=True
    )

    st.write(
        "고유값 예시:",
        df["estimated_owners"]
        .dropna()
        .unique()[:20]
    )"

df = pd.read_csv(url, encoding="utf-8")

st.header("1. 데이터 기본 정보")

col1, col2 = st.columns(2)

with col1:
    st.metric("행", df.shape[0])

with col2:
    st.metric("열", df.shape[1])

st.subheader("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 2. 데이터 타입
# --------------------------------------------------

st.header("2. 열별 데이터 타입")

dtype_df = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 타입": df.dtypes.astype(str).values
})

st.dataframe(dtype_df, use_container_width=True)


# --------------------------------------------------
# 3. 결측치 확인
# --------------------------------------------------

st.header("3. 결측치")

missing = pd.DataFrame({
    "열 이름": df.columns,
    "결측치 수": df.isna().sum().values
})

missing["결측 비율(%)"] = (
    missing["결측치 수"] / len(df) * 100
).round(2)

missing = missing.sort_values(
    "결측치 수",
    ascending=False
)

st.dataframe(missing, use_container_width=True)


# --------------------------------------------------
# 4. 수치형 데이터 통계
# --------------------------------------------------

st.header("4. 수치형 데이터 요약")

st.dataframe(
    df.describe(),
    use_container_width=True
)


# --------------------------------------------------
# 5. peak_ccu 분포
# --------------------------------------------------

if "peak_ccu" in df.columns:

    st.header("5. 최대 동시 접속자 수 분포")

    peak = pd.to_numeric(
        df["peak_ccu"],
        errors="coerce"
    ).dropna()

    peak = peak[peak >= 0]

    fig, ax = plt.subplots()

    ax.hist(
        np.log1p(peak),
        bins=50
    )

    ax.set_xlabel("log(1 + peak_ccu)")
    ax.set_ylabel("게임 수")
    ax.set_title("peak_ccu 로그 변환 분포")

    st.pyplot(fig)

    st.write(
        "0인 게임 수:",
        int((peak == 0).sum())
    )

    st.write(
        "최댓값:",
        int(peak.max())
    )


# --------------------------------------------------
# 6. 리뷰 수 분포
# --------------------------------------------------

if "num_reviews_total" in df.columns:

    st.header("6. 리뷰 수 분포")

    reviews = pd.to_numeric(
        df["num_reviews_total"],
        errors="coerce"
    ).dropna()

    reviews = reviews[reviews >= 0]

    fig, ax = plt.subplots()

    ax.hist(
        np.log1p(reviews),
        bins=50
    )

    ax.set_xlabel("log(1 + num_reviews_total)")
    ax.set_ylabel("게임 수")
    ax.set_title("리뷰 수 로그 변환 분포")

    st.pyplot(fig)

    st.write(
        "리뷰가 0개인 게임:",
        int((reviews == 0).sum())
    )

    st.write(
        "리뷰가 10개 미만인 게임:",
        int((reviews < 10).sum())
    )

    st.write(
        "리뷰가 100개 미만인 게임:",
        int((reviews < 100).sum())
    )


# --------------------------------------------------
# 7. 무료/유료 게임 확인
# --------------------------------------------------

if "price" in df.columns:

    st.header("7. 무료 / 유료 게임")

    price = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    free_count = int((price == 0).sum())
    paid_count = int((price > 0).sum())

    st.write("무료 게임:", free_count)
    st.write("유료 게임:", paid_count)


# --------------------------------------------------
# 8. estimated_owners 확인
# --------------------------------------------------

if "estimated_owners" in df.columns:

    st.header("8. estimated_owners 확인")

    st.write(
        "estimated_owners는 정확한 소유자 수가 아니라 "
        "범위 형태로 저장되어 있는지 확인합니다."
    )

    st.dataframe(
        df[["name", "estimated_owners"]].head(20),
        use_container_width=True
    )

    st.write(
        "고유값 예시:",
        df["estimated_owners"]
        .dropna()
        .unique()[:20]
    )
