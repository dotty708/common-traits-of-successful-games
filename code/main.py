import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# 페이지 설정
# ==============================

st.set_page_config(
    page_title="Steam 게임 데이터 분석",
    layout="wide"
)

st.title("Steam 게임 인기 데이터 분석")
st.write("연구 질문: 인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?")


# ==============================
# 데이터 불러오기
# ==============================

url = "https://raw.githubusercontent.com/dotty708/common-traits-of-successful-games/refs/heads/main/data/games_march2025_popularity_reduced.csv"

@st.cache_data
def load_data():
    return pd.read_csv(url, encoding="utf-8")

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# ==============================
# 1. 기본 정보
# ==============================

st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("게임 수", f"{df.shape[0]:,}")
col2.metric("열 수", df.shape[1])
col3.metric("전체 결측치", f"{df.isna().sum().sum():,}")

st.subheader("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)


# ==============================
# 2. 열 정보
# ==============================

st.header("2. 데이터 열 확인")

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 타입": df.dtypes.astype(str).values,
    "결측치 수": df.isna().sum().values,
    "결측 비율(%)": (df.isna().sum().values / len(df) * 100).round(2)
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)


# ==============================
# 3. 수치형 데이터 통계
# ==============================

st.header("3. 수치형 데이터 요약")

numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 0:
    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )
else:
    st.warning("수치형 데이터가 없습니다.")


# ==============================
# 4. 최대 동시 접속자 수
# ==============================

st.header("4. 최대 동시 접속자 수 분석")

if "peak_ccu" in df.columns:

    peak = pd.to_numeric(
        df["peak_ccu"],
        errors="coerce"
    ).dropna()

    peak = peak[peak >= 0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "peak_ccu = 0",
        f"{(peak == 0).sum():,}개"
    )

    col2.metric(
        "중앙값",
        f"{peak.median():,.0f}"
    )

    col3.metric(
        "최댓값",
        f"{peak.max():,.0f}"
    )

    fig, ax = plt.subplots()

    ax.hist(
        np.log1p(peak),
        bins=50
    )

    ax.set_xlabel("log(1 + peak_ccu)")
    ax.set_ylabel("Number of Games")
    ax.set_title("Distribution of Peak CCU (Log Scale)")

    st.pyplot(fig)

    st.caption(
        "일부 게임의 동시 접속자 수가 매우 크기 때문에 "
        "log(1 + peak_ccu)로 변환하여 분포를 확인했습니다."
    )

else:
    st.warning("peak_ccu 열이 없습니다.")


# ==============================
# 5. 리뷰 수
# ==============================

st.header("5. 리뷰 수 분석")

if "num_reviews_total" in df.columns:

    reviews = pd.to_numeric(
        df["num_reviews_total"],
        errors="coerce"
    ).dropna()

    reviews = reviews[reviews >= 0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "리뷰 0개",
        f"{(reviews == 0).sum():,}"
    )

    col2.metric(
        "리뷰 10개 미만",
        f"{(reviews < 10).sum():,}"
    )

    col3.metric(
        "리뷰 100개 미만",
        f"{(reviews < 100).sum():,}"
    )

    fig, ax = plt.subplots()

    ax.hist(
        np.log1p(reviews),
        bins=50
    )

    ax.set_xlabel("log(1 + number of reviews)")
    ax.set_ylabel("Number of Games")
    ax.set_title("Distribution of Reviews (Log Scale)")

    st.pyplot(fig)

    st.caption(
        "리뷰가 매우 많은 일부 게임 때문에 로그 변환을 사용했습니다. "
        "이 결과를 이용해 리뷰가 지나치게 적은 게임의 처리 기준을 결정할 수 있습니다."
    )

else:
    st.warning("num_reviews_total 열이 없습니다.")


# ==============================
# 6. 무료 / 유료 게임
# ==============================

st.header("6. 무료 / 유료 게임")

if "price" in df.columns:

    price = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    free_count = (price == 0).sum()
    paid_count = (price > 0).sum()

    col1, col2 = st.columns(2)

    col1.metric(
        "무료 게임",
        f"{free_count:,}"
    )

    col2.metric(
        "유료 게임",
        f"{paid_count:,}"
    )

    st.caption(
        "가격이 0인 게임도 결측치가 아니라 실제 무료 게임일 수 있으므로 "
        "현재 단계에서는 삭제하지 않습니다."
    )

else:
    st.warning("price 열이 없습니다.")


# ==============================
# 7. estimated_owners
# ==============================

st.header("7. 추정 소유자 수 확인")

if "estimated_owners" in df.columns:

    st.write(
        "estimated_owners가 정확한 판매량이 아니라 "
        "범위 형태로 저장되어 있는지 확인합니다."
    )

    if "name" in df.columns:
        st.dataframe(
            df[["name", "estimated_owners"]].head(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.dataframe(
            df[["estimated_owners"]].head(20),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("고유값 예시")

    owners_example = (
        df["estimated_owners"]
        .dropna()
        .astype(str)
        .unique()[:20]
    )

    st.write(owners_example)

else:
    st.warning("estimated_owners 열이 없습니다.")


# ==============================
# 8. 현재 확인 결과
# ==============================

st.header("8. 다음 전처리 단계")

st.markdown("""
현재 데이터를 확인한 뒤 다음 사항을 결정합니다.

- 리뷰가 지나치게 적은 게임의 제외 기준
- 무료 게임을 별도 특성으로 사용할지 여부
- `estimated_owners` 범위를 숫자로 변환하는 방법
- 게임의 '인기'를 정의할 기준
- 군집 분석에 사용할 특성
- 분류 모델에 사용할 특성
""")
