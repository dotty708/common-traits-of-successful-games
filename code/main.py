import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Steam 게임 데이터 분석",
    layout="wide"
)

st.title("Steam 게임 인기 데이터 분석")
st.write("연구 질문: 인기 있는 게임은 정말 공통적인 특성을 가지고 있는가?")


# ==================================================
# 데이터 불러오기
# ==================================================

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


# ==================================================
# 1. 데이터 기본 정보
# ==================================================

st.header("1. 데이터 기본 정보")

col1, col2, col3 = st.columns(3)

col1.metric("게임 수", f"{df.shape[0]:,}")
col2.metric("열 수", df.shape[1])
col3.metric("전체 결측치", f"{df.isna().sum().sum():,}")

st.subheader("데이터 미리보기")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ==================================================
# 2. 열 / 결측치 확인
# ==================================================

st.header("2. 열과 결측치 확인")

column_info = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 타입": df.dtypes.astype(str).values,
    "결측치 수": df.isna().sum().values,
    "결측 비율(%)":
        (df.isna().sum().values / len(df) * 100).round(2)
})

st.dataframe(
    column_info,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 3. 수치형 데이터 통계
# ==================================================

st.header("3. 수치형 데이터 요약")

numeric_df = df.select_dtypes(include=np.number)

if len(numeric_df.columns) > 0:

    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )


# ==================================================
# 4. peak_ccu 분석
# ==================================================

st.header("4. 최대 동시 접속자 수 (peak_ccu)")

if "peak_ccu" in df.columns:

    peak = pd.to_numeric(
        df["peak_ccu"],
        errors="coerce"
    )

    peak = peak.dropna()
    peak = peak[peak >= 0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "CCU 0인 게임",
        f"{(peak == 0).sum():,}"
    )

    col2.metric(
        "중앙값",
        f"{peak.median():,.0f}"
    )

    col3.metric(
        "평균",
        f"{peak.mean():,.0f}"
    )

    col4.metric(
        "최댓값",
        f"{peak.max():,.0f}"
    )

    # ----------------------------------------------
    # 원본 히스토그램
    # ----------------------------------------------

    st.subheader("원본 분포")

    peak_df = pd.DataFrame({
        "peak_ccu": peak
    })

    fig_peak = px.histogram(
        peak_df,
        x="peak_ccu",
        nbins=100,
        title="peak_ccu 분포"
    )

    fig_peak.update_layout(
        xaxis_title="peak_ccu",
        yaxis_title="게임 수"
    )

    st.plotly_chart(
        fig_peak,
        use_container_width=True
    )


    # ----------------------------------------------
    # 로그 축 히스토그램
    # 0은 로그 축에 표시할 수 없으므로 양수만 사용
    # ----------------------------------------------

    st.subheader("로그 축으로 확인")

    peak_positive = peak[peak > 0]

    peak_log_df = pd.DataFrame({
        "peak_ccu": peak_positive
    })

    fig_peak_log = px.histogram(
        peak_log_df,
        x="peak_ccu",
        nbins=100,
        log_x=True,
        title="peak_ccu 분포 - 로그 축"
    )

    fig_peak_log.update_layout(
        xaxis_title="peak_ccu (로그 축)",
        yaxis_title="게임 수"
    )

    st.plotly_chart(
        fig_peak_log,
        use_container_width=True
    )

    st.info(
        "원본 그래프와 로그 축 그래프를 비교하면 "
        "소수의 매우 큰 게임에 동시 접속자 수가 몰려 있는지 확인할 수 있습니다."
    )

else:

    st.warning("peak_ccu 열이 없습니다.")


# ==================================================
# 5. 리뷰 수 분석
# ==================================================

st.header("5. 리뷰 수 분석")

if "num_reviews_total" in df.columns:

    reviews = pd.to_numeric(
        df["num_reviews_total"],
        errors="coerce"
    )

    reviews = reviews.dropna()
    reviews = reviews[reviews >= 0]

    col1, col2, col3, col4 = st.columns(4)

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

    col4.metric(
        "최대 리뷰 수",
        f"{reviews.max():,.0f}"
    )


    # ----------------------------------------------
    # 원본 히스토그램
    # ----------------------------------------------

    st.subheader("원본 분포")

    review_df = pd.DataFrame({
        "리뷰 수": reviews
    })

    fig_review = px.histogram(
        review_df,
        x="리뷰 수",
        nbins=100,
        title="게임별 리뷰 수 분포"
    )

    fig_review.update_layout(
        xaxis_title="리뷰 수",
        yaxis_title="게임 수"
    )

    st.plotly_chart(
        fig_review,
        use_container_width=True
    )


    # ----------------------------------------------
    # 로그 축
    # ----------------------------------------------

    st.subheader("로그 축으로 확인")

    reviews_positive = reviews[reviews > 0]

    review_log_df = pd.DataFrame({
        "리뷰 수": reviews_positive
    })

    fig_review_log = px.histogram(
        review_log_df,
        x="리뷰 수",
        nbins=100,
        log_x=True,
        title="게임별 리뷰 수 분포 - 로그 축"
    )

    fig_review_log.update_layout(
        xaxis_title="리뷰 수 (로그 축)",
        yaxis_title="게임 수"
    )

    st.plotly_chart(
        fig_review_log,
        use_container_width=True
    )


    # ----------------------------------------------
    # 리뷰 수 기준 후보
    # ----------------------------------------------

    st.subheader("리뷰 수 기준 확인")

    total = len(reviews)

    threshold_data = []

    for threshold in [1, 5, 10, 20, 50, 100]:

        count = int((reviews < threshold).sum())

        threshold_data.append({
            "최소 리뷰 기준": threshold,
            "기준 미달 게임 수": count,
            "전체에서 차지하는 비율(%)":
                round(count / total * 100, 2)
        })

    threshold_df = pd.DataFrame(threshold_data)

    st.dataframe(
        threshold_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "이 표를 보고 리뷰가 지나치게 적은 게임을 "
        "어디까지 제외할지 팀원과 결정하면 됩니다. "
        "기준을 먼저 임의로 정하지 않고 실제 데이터 분포를 확인한 뒤 결정합니다."
    )

else:

    st.warning("num_reviews_total 열이 없습니다.")


# ==================================================
# 6. 무료 / 유료 게임
# ==================================================

st.header("6. 무료 게임 확인")

if "price" in df.columns:

    price = pd.to_numeric(
        df["price"],
        errors="coerce"
    )

    free_count = int((price == 0).sum())
    paid_count = int((price > 0).sum())

    total_price = free_count + paid_count

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "무료 게임",
        f"{free_count:,}"
    )

    col2.metric(
        "유료 게임",
        f"{paid_count:,}"
    )

    if total_price > 0:

        free_ratio = free_count / total_price * 100

        col3.metric(
            "무료 게임 비율",
            f"{free_ratio:.1f}%"
        )

    st.info(
        "무료 게임(price = 0)은 결측치가 아니라 게임의 특성입니다. "
        "따라서 현재 탐색 단계에서는 삭제하지 않습니다."
    )

else:

    st.warning("price 열이 없습니다.")


# ==================================================
# 7. estimated_owners 확인
# ==================================================

st.header("7. estimated_owners 확인")

if "estimated_owners" in df.columns:

    st.write(
        "estimated_owners가 정확한 판매량이 아니라 "
        "범위 형태로 저장되어 있는지 확인합니다."
    )

    if "name" in df.columns:

        st.dataframe(
            df[
                ["name", "estimated_owners"]
            ].head(20),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            df[
                ["estimated_owners"]
            ].head(20),
            use_container_width=True,
            hide_index=True
        )


    st.subheader("범위 값 예시")

    owners_values = (
        df["estimated_owners"]
        .dropna()
        .astype(str)
        .value_counts()
        .reset_index()
    )

    owners_values.columns = [
        "estimated_owners",
        "게임 수"
    ]

    st.dataframe(
        owners_values,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning("estimated_owners 열이 없습니다.")


# ==================================================
# 8. 인기 기준 결정
# ==================================================

st.header("8. 인기 기준 결정")

st.write("""
현재 단계에서는 인기 기준을 임의로 확정하지 않습니다.

위에서 확인한 다음 정보를 바탕으로 팀원과 기준을 결정합니다.

- peak_ccu 분포
- 리뷰 수 분포
- 리뷰가 매우 적은 게임의 비율
- 무료 게임의 비율
- estimated_owners의 범위
""")

st.warning(
    "중요: peak_ccu나 리뷰 수를 이용해 '인기 게임'이라는 정답을 만든 경우, "
    "같은 값을 분류 모델의 입력 특성으로 다시 사용하면 데이터 누수가 발생할 수 있습니다."
)
