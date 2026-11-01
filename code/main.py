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
# 2. estimated_owners → 구간 중간값 변환
# =========================================================

raw_df = df.copy()
df = df.copy().replace(r"^\s*$", np.nan, regex=True)
original_shape = df.shape
sentinel_columns = ["pct_pos_total", "num_reviews_total", "pct_pos_recent", "num_reviews_recent"]
column_summary = pd.DataFrame({
    "열 이름": df.columns,
    "자료형": df.dtypes.astype(str).values,
    "빈 값 수(공백 포함)": df.isna().sum().values,
    "정보 없음(-1) 수": [int(pd.to_numeric(df[c], errors="coerce").eq(-1).sum())
                         if c in sentinel_columns else 0 for c in df.columns],
})

st.info("팀 질문: Steam 게임을 가격·보유자 수·리뷰·동시접속자 특성으로 묶으면 어떤 차이가 나타날까?")
st.caption("인기 지표를 비교하는 탐색적 군집 분석입니다. 군집 번호는 성공 순위가 아니며 인과관계를 뜻하지 않습니다.")

def convert_owners(value):

    try:
        text = str(value).replace(",", "").strip()

        parts = text.split("-")

        if len(parts) != 2:
            return np.nan

        minimum = float(parts[0].strip())
        maximum = float(parts[1].strip())

        if not np.isfinite([minimum, maximum]).all() or minimum < 0 or maximum < minimum:
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
    f"{int(column_summary['빈 값 수(공백 포함)'].sum()):,}"
)


st.dataframe(column_summary, use_container_width=True, hide_index=True)
st.caption("빈 값과 리뷰 관련 정보 없음(-1)은 구분해 집계합니다. 가격 0원은 무료 게임이므로 유지합니다.")
with st.expander("데이터셋 확인 — 원본"):
    st.dataframe(raw_df.head(20), use_container_width=True)

with st.expander("구간 중간값 추가 후 데이터"):

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


# 실제 빈 값만 중앙값 대체 후보로 삼습니다.
# 리뷰 정보 없음(-1)과 비정상값을 가상의 관측값으로 채우지 않습니다.
fill_actual_missing = st.checkbox("실제 빈 수치 값을 해당 열의 중앙값으로 대체", value=True)
records = []
for column in selected_features:
    original = X[column]
    actual_missing = original.isna()
    values = pd.to_numeric(original, errors="coerce")
    invalid = (~actual_missing) & (values.isna() | ~np.isfinite(values) | values.lt(0))
    if column == "owners_midpoint":
        # 범위 해석 실패는 실제 빈 값과 구분합니다.
        actual_missing = df["estimated_owners"].isna()
        invalid = (~actual_missing) & values.isna()
    values = values.mask(invalid)
    median = values.median()
    fill_mask = actual_missing & pd.notna(median) & fill_actual_missing
    values.loc[fill_mask] = median
    X[column] = values
    records.append({
        "변수": column, "실제 빈 값": int(actual_missing.sum()),
        "-1 등 비정상값": int(invalid.sum()),
        "대체 중앙값": median if fill_mask.any() else np.nan,
        "중앙값 대체 수": int(fill_mask.sum()),
        "남은 결측치": int(values.isna().sum()),
    })

valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()
analysis_df = analysis_df.loc[valid_rows].copy()
# 그래프와 평균표도 실제 분석에 사용한 대체 후 값으로 계산합니다.
analysis_df.loc[:, selected_features] = X
original_count = len(df)
used_count = len(X)
removed_count = original_count - used_count

st.subheader("결측치·비정상값 처리 결과")
st.write(
    "실제 빈 수치 값은 옵션을 켠 경우 해당 열의 유효값 중앙값으로 채웁니다. "
    "리뷰 관련 -1, 음수, 무한대, 숫자·보유자 범위 해석 실패는 비정상값으로 보고 "
    "선택한 분석 변수에 남아 있으면 해당 행을 제외합니다. "
    "무료 가격 0원과 리뷰·접속자 수 0은 유지합니다. 문자형 열과 식별자는 중앙값 대체 대상이 아닙니다."
)
st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
c1, c2, c3 = st.columns(3)
c1.metric("처리 전 행 수", f"{original_count:,}")
c2.metric("처리 후 행 수", f"{used_count:,}")
c3.metric("제외 행 수", f"{removed_count:,}")
st.caption(f"원본 {original_shape[0]:,}행 × {original_shape[1]}열 · 분석 입력 {used_count:,}행 × {len(selected_features)}개 변수. 선택하지 않은 열의 결측치 때문에 행을 제외하지 않습니다.")

if used_count < max(k, 2):
    st.error("분석 가능한 행이 부족합니다. 변수 또는 K를 조정하세요.")
    st.stop()

st.subheader("극단값 점검 및 처리")
outlier_records = []
for column in selected_features:
    q1, q3 = X[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((X[column] < lower) | (X[column] > upper)).sum())
    outlier_records.append({"변수": column, "IQR 하한": lower, "IQR 상한": upper, "범위 밖 값 수": count})
st.dataframe(pd.DataFrame(outlier_records), use_container_width=True, hide_index=True)
st.write("IQR 범위 밖 값은 극단값 후보로 집계하되 삭제하거나 상한으로 자르지 않았습니다. 인기 게임의 실제 큰 값일 수 있기 때문입니다. 선택한 변수에 log1p를 적용해 큰 값의 영향을 완화하고, 이후 표준화합니다. 로그 변환과 표준화로 행 수는 바뀌지 않습니다.")
if not log_features:
    st.caption("현재 로그 변환을 선택하지 않았으므로 극단값 완화는 적용하지 않고 표준화만 합니다.")


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

    if len(np.unique(X_scaled, axis=0)) < k:
        st.error("서로 다른 분석 행 수보다 K가 큽니다. K를 줄이세요.")
        st.stop()
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

st.write(f"**발견:** 선택 변수의 변동 중 첫 두 PCA 축이 설명하는 비율은 {explained_variance:.1f}%입니다. 2차원 그림에 담기지 않는 변동도 있으므로 겹침만으로 군집이 같다고 단정할 수 없습니다.")


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
    title="군집별 게임 수",
    labels={"군집": "군집", "게임 수": "게임 수(개)"}
)


st.plotly_chart(
    fig_count,
    use_container_width=True
)

largest = cluster_counts.loc[cluster_counts["게임 수"].idxmax()]
smallest = cluster_counts.loc[cluster_counts["게임 수"].idxmin()]
st.write(f"**발견:** 가장 큰 {largest['군집']}은 {int(largest['게임 수']):,}개({largest['게임 수']/used_count*100:.1f}%)이고, 가장 작은 {smallest['군집']}은 {int(smallest['게임 수']):,}개입니다.")


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
    xaxis_title="군집",
    yaxis_title=f"{compare_feature} 평균"
)


st.plotly_chart(
    fig_compare,
    use_container_width=True
)

highest = compare_df.loc[compare_df["평균값"].idxmax()]
lowest = compare_df.loc[compare_df["평균값"].idxmin()]
st.write(f"**발견:** {compare_feature} 평균은 {highest['군집']}이 {highest['평균값']:,.2f}로 가장 높고, {lowest['군집']}이 {lowest['평균값']:,.2f}로 가장 낮습니다. 차이는 {highest['평균값']-lowest['평균값']:,.2f}입니다.")
medians = analysis_df.groupby("cluster_name")[compare_feature].median().rename("중앙값")
st.caption("평균은 초대형 게임의 영향을 받을 수 있습니다. 아래 중앙값과 함께 비교하세요. 수치는 로그 변환 전 단위입니다.")
st.dataframe(medians.reset_index(), use_container_width=True, hide_index=True)


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

correlation = analysis_df[[x_feature, y_feature]].corr().iloc[0, 1]
if pd.notna(correlation):
    direction = "양의" if correlation > 0 else "음의" if correlation < 0 else "0의"
    st.write(f"**발견:** 전체 분석 데이터 {used_count:,}개에서 두 변수는 {direction} 선형 상관을 보이며, 피어슨 상관계수는 {correlation:.3f}입니다. 상관관계가 원인과 결과를 뜻하지는 않습니다.")
else:
    st.write("**발견:** 한 변수 이상이 상수여서 피어슨 상관계수를 계산할 수 없습니다.")
st.caption(f"산점도에는 {len(plot_df):,}개를 표시하며, 상관계수는 전체 분석 행의 로그 변환 전 값으로 계산했습니다.")


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
2. 실제 빈 수치 값은 옵션에 따라 해당 열의 중앙값으로 대체하고 정상적인 0은 유지합니다.
3. -1 등 비정상값은 중앙값으로 채우지 않고, 선택한 변수에 남은 결측치가 있는 행을 제외합니다. 극단값 후보는 IQR로 집계하되 삭제하지 않습니다.
4. 사용자가 지정한 변수에 `log1p` 변환을 적용합니다.
5. StandardScaler를 이용해 변수들의 단위를 표준화합니다.
6. K-Means를 이용하여 게임들을 K개의 군집으로 분류합니다.
7. PCA를 이용해 군집 결과를 2차원으로 시각화합니다.
8. 각 군집의 평균값과 대표 게임을 비교하여 특징을 해석합니다.
"""
)
