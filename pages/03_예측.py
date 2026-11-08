
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import joblib
import io
import ast
import re


# ==================================================
# 1. 페이지 설정
# ==================================================

st.title("신규 Steam 게임 군집 예측")

st.write(
    "게임 출시 전에 알 수 있는 정보를 입력하여 "
    "기존 Steam 게임의 3개 군집 중 "
    "어느 군집에 속할지 예측합니다."
)

st.info(
    "이 페이지는 2페이지에서 학습한 "
    "분류 모델을 사용합니다."
)

CLASS_IDS = [0, 1, 2]


# ==================================================
# 2. 모델 불러오기
# ==================================================

st.header("1. 분류 모델 불러오기")

@st.cache_resource(show_spinner=False)
def load_model_file(file_bytes):
    return joblib.load(io.BytesIO(file_bytes))


saved_result = st.session_state.get(
    "steam_classification"
)

bundle = None

if saved_result is not None:

    source = st.radio(
        "모델 불러오기 방식",
        [
            "2페이지에서 학습한 모델 사용",
            "저장된 모델 파일 업로드"
        ]
    )

    if source == "2페이지에서 학습한 모델 사용":

        best_name = saved_result["best_name"]

        bundle = {
            "version": 1,
            "model_name": best_name,
            "model": saved_result["models"][best_name],
            "selected_features":
                saved_result["selected_features"],
            "top_values":
                saved_result["top_values"],
            "class_ids": CLASS_IDS
        }

else:
    source = "저장된 모델 파일 업로드"


if source == "저장된 모델 파일 업로드":

    st.warning(
        "joblib 파일은 Python 객체를 실행할 수 있으므로 "
        "직접 생성했거나 신뢰할 수 있는 파일만 업로드하세요."
    )

    model_file = st.file_uploader(
        "2페이지에서 다운로드한 .joblib 파일",
        type=["joblib"],
        key="prediction_model"
    )

    if model_file is not None:

        try:
            bundle = load_model_file(
                model_file.getvalue()
            )

        except Exception as e:
            st.error("모델 파일을 읽지 못했습니다.")
            st.exception(e)
            st.stop()


if bundle is None:
    st.warning(
        "먼저 2페이지에서 모델을 학습하거나 "
        "저장된 모델 파일을 업로드하세요."
    )
    st.stop()


# ==================================================
# 3. 모델 검증
# ==================================================

required_keys = [
    "model",
    "model_name",
    "selected_features",
    "top_values"
]

if not isinstance(bundle, dict):
    st.error("올바른 모델 파일이 아닙니다.")
    st.stop()

missing_keys = [
    key for key in required_keys
    if key not in bundle
]

if missing_keys:
    st.error(
        "모델 파일에 필요한 정보가 없습니다: "
        + ", ".join(missing_keys)
    )
    st.stop()


model = bundle["model"]

selected_features = list(
    bundle["selected_features"]
)

top_values = bundle["top_values"]

model_name = bundle["model_name"]

if not hasattr(model, "predict_proba"):
    st.error(
        "이 모델은 예측 확률을 제공하지 않습니다."
    )
    st.stop()

st.success(
    f"모델 불러오기 완료: {model_name}"
)

c1, c2 = st.columns(2)

c1.metric("사용 모델", model_name)
c2.metric(
    "입력 특성 수",
    len(selected_features)
)

with st.expander("모델 입력 변수 확인"):
    st.write(selected_features)


# ==================================================
# 4. 전처리 함수
# 2페이지와 동일한 규칙 사용
# ==================================================

def parse_items(value):

    if isinstance(value, (list, tuple)):
        return [
            str(x).strip()
            for x in value
            if str(x).strip()
        ]

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple)):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

        except (ValueError, SyntaxError):
            pass

    return [
        x.strip()
        for x in re.split(r"[;,|]", text)
        if x.strip()
    ]


def make_feature_dicts(frame, features, top_values):

    result = []

    top_sets = {
        key: set(values)
        for key, values in top_values.items()
    }

    for values in frame[features].itertuples(
        index=False,
        name=None
    ):

        row = dict(zip(features, values))
        item = {}

        # 숫자형 특성
        for column in [
            "price",
            "release_year",
            "release_month"
        ]:

            if column not in features:
                continue

            value = pd.to_numeric(
                row.get(column),
                errors="coerce"
            )

            valid = (
                pd.notna(value)
                and np.isfinite(value)
                and value >= 0
            )

            item[column] = (
                float(value) if valid else 0.0
            )

            item[column + "_missing"] = (
                0.0 if valid else 1.0
            )

        # 장르·카테고리
        for column, prefix in [
            ("genres", "genre"),
            ("categories", "category")
        ]:

            if column not in features:
                continue

            tokens = parse_items(row.get(column))
            allowed = top_sets.get(column, set())

            matched = [
                token for token in tokens
                if token in allowed
            ]

            for token in matched:
                item[prefix + "=" + token] = 1.0

            if not tokens:
                item[prefix + "=UNKNOWN"] = 1.0

            elif not matched:
                item[prefix + "=OTHER"] = 1.0

        # 개발사·배급사
        for column, prefix in [
            ("developers", "developer"),
            ("publishers", "publisher")
        ]:

            if column not in features:
                continue

            tokens = parse_items(row.get(column))

            if not tokens:
                category = "UNKNOWN"

            elif tokens[0] in top_sets.get(
                column, set()
            ):
                category = tokens[0]

            else:
                category = "OTHER"

            item[prefix] = category

        result.append(item)

    return result


# ==================================================
# 5. 입력 UI 함수
# ==================================================

def multi_value_input(column, title):

    options = top_values.get(column, [])

    chosen = st.multiselect(
        title,
        options=options,
        key="prediction_" + column
    )

    extra = st.text_input(
        f"{title} 추가 입력",
        placeholder="목록에 없다면 직접 입력",
        key="prediction_extra_" + column
    )

    extra_values = parse_items(extra)

    return list(dict.fromkeys(
        chosen + extra_values
    ))


def single_value_input(column, title):

    options = top_values.get(column, [])

    choice = st.selectbox(
        title,
        options=[
            "정보 없음",
            "직접 입력"
        ] + options,
        key="prediction_" + column
    )

    if choice == "정보 없음":
        return np.nan

    if choice == "직접 입력":

        value = st.text_input(
            f"{title} 이름",
            placeholder="개발사 또는 배급사 이름",
            key="prediction_custom_" + column
        )

        return value if value.strip() else np.nan

    return choice


# ==================================================
# 6. 신규 게임 정보 입력
# ==================================================

st.header("2. 신규 게임 정보 입력")

st.write(
    "학습 모델이 사용한 입력 변수에 맞춰 "
    "입력 항목이 자동으로 표시됩니다."
)

with st.form("new_game_prediction_form"):

    game_name = st.text_input(
        "게임 이름",
        placeholder="출시 예정 게임 이름"
    )

    input_data = {}

    if "price" in selected_features:

        input_data["price"] = st.number_input(
            "게임 가격",
            min_value=0.0,
            value=19.99,
            step=1.0,
            help=(
                "학습 CSV의 price와 같은 단위로 "
                "입력하세요. 원본 데이터가 USD라면 "
                "달러 단위로 입력해야 합니다."
            )
        )

    if "release_year" in selected_features:

        input_data["release_year"] = st.number_input(
            "출시 예정 연도",
            min_value=1970,
            max_value=2100,
            value=2027,
            step=1
        )

    if "release_month" in selected_features:

        input_data["release_month"] = st.selectbox(
            "출시 예정 월",
            options=list(range(1, 13)),
            index=0
        )

    if "genres" in selected_features:

        input_data["genres"] = multi_value_input(
            "genres",
            "게임 장르"
        )

    if "categories" in selected_features:

        input_data["categories"] = multi_value_input(
            "categories",
            "게임 카테고리"
        )

    if "developers" in selected_features:

        input_data["developers"] = single_value_input(
            "developers",
            "개발사"
        )

    if "publishers" in selected_features:

        input_data["publishers"] = single_value_input(
            "publishers",
            "배급사"
        )

    submitted = st.form_submit_button(
        "신규 게임 군집 예측",
        type="primary",
        use_container_width=True
    )


# ==================================================
# 7. 예측 실행
# ==================================================

if submitted:

    try:

        input_df = pd.DataFrame([input_data])

        feature_dicts = make_feature_dicts(
            input_df,
            selected_features,
            top_values
        )

        predicted_cluster = int(
            model.predict(feature_dicts)[0]
        )

        probabilities = model.predict_proba(
            feature_dicts
        )[0]

        # 모델의 실제 클래스 순서 사용
        classes = [
            int(c)
            for c in model.classes_
        ]

        probability_map = {
            int(c): float(p)
            for c, p in zip(
                classes,
                probabilities
            )
        }

        st.session_state["last_prediction"] = {
            "game_name": game_name,
            "cluster": predicted_cluster,
            "probabilities": probability_map,
            "input_data": input_data,
            "model_name": model_name
        }

    except Exception as e:

        st.error("예측 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()


# ==================================================
# 8. 예측 결과
# ==================================================

if "last_prediction" in st.session_state:

    result = st.session_state["last_prediction"]

    if result["model_name"] != model_name:
        st.info(
            "모델이 변경되었습니다. "
            "새로운 예측을 실행하세요."
        )
        st.stop()

    predicted_cluster = result["cluster"]

    probability_map = result["probabilities"]

    st.header("3. 예측 결과")

    name = (
        result["game_name"]
        if result["game_name"]
        else "신규 게임"
    )

    st.subheader(name)

    st.success(
        f"예상 군집: 군집 {predicted_cluster}"
    )

    confidence = probability_map.get(
        predicted_cluster, 0.0
    )

    col1, col2 = st.columns(2)

    col1.metric(
        "예측 군집",
        f"군집 {predicted_cluster}"
    )

    col2.metric(
        "해당 군집 예측 확률",
        f"{confidence * 100:.1f}%"
    )

    # 확률 데이터
    probability_df = pd.DataFrame({
        "군집": [
            f"군집 {i}"
            for i in CLASS_IDS
        ],
        "예측 확률(%)": [
            probability_map.get(i, 0.0) * 100
            for i in CLASS_IDS
        ]
    })

    st.subheader("군집별 예측 확률")

    fig_probability = px.bar(
        probability_df,
        x="군집",
        y="예측 확률(%)",
        text_auto=".1f",
        range_y=[0, 100],
        title="신규 게임의 군집별 예측 확률"
    )

    fig_probability.update_layout(
        xaxis_title="군집",
        yaxis_title="예측 확률(%)"
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )

    st.dataframe(
        probability_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "예측 확률은 분류 모델이 추정한 "
        "군집 소속 확률입니다. "
        "실제 흥행 성공 확률을 의미하지 않습니다."
    )

    with st.expander("입력한 게임 정보 확인"):

        st.json({
            key: (
                None
                if isinstance(value, float)
                and np.isnan(value)
                else value
            )
            for key, value
            in result["input_data"].items()
        })


# ==================================================
# 9. 군집별 실제 특성 비교
# ==================================================

st.header("4. 기존 군집 특성 비교")

st.write(
    "1페이지에서 다운로드한 전체 군집화 결과 "
    "CSV를 업로드하면 각 군집의 실제 리뷰 "
    "특성을 확인할 수 있습니다."
)

cluster_file = st.file_uploader(
    "군집화 결과 CSV (선택)",
    type=["csv"],
    key="prediction_cluster_profile"
)

if cluster_file is not None:

    try:

        cluster_df = pd.read_csv(cluster_file)

        required = {
            "cluster",
            "positive",
            "negative"
        }

        if not required.issubset(
            cluster_df.columns
        ):
            st.warning(
                "군집별 특성 분석에는 cluster, "
                "positive, negative 열이 필요합니다."
            )

        else:

            for col in required:
                cluster_df[col] = pd.to_numeric(
                    cluster_df[col],
                    errors="coerce"
                )

            cluster_df = cluster_df.dropna(
                subset=list(required)
            ).copy()

            cluster_df = cluster_df[
                (cluster_df["positive"] >= 0)
                & (cluster_df["negative"] >= 0)
                & (
                    cluster_df["positive"]
                    + cluster_df["negative"] > 0
                )
            ]

            cluster_df["total_reviews"] = (
                cluster_df["positive"]
                + cluster_df["negative"]
            )

            summary = (
                cluster_df.groupby("cluster")
                .agg(
                    game_count=("cluster", "size"),
                    positive_mean=("positive", "mean"),
                    negative_mean=("negative", "mean"),
                    total_median=(
                        "total_reviews", "median"
                    ),
                    positive_sum=("positive", "sum"),
                    negative_sum=("negative", "sum")
                )
                .reset_index()
            )

            summary["positive_rate"] = (
                summary["positive_sum"]
                / (
                    summary["positive_sum"]
                    + summary["negative_sum"]
                )
                * 100
            )

            summary = summary[
                [
                    "cluster",
                    "game_count",
                    "positive_mean",
                    "negative_mean",
                    "total_median",
                    "positive_rate"
                ]
            ]

            summary.columns = [
                "군집",
                "게임 수",
                "긍정 리뷰 평균",
                "부정 리뷰 평균",
                "전체 리뷰 수 중앙값",
                "전체 긍정 비율(%)"
            ]

            st.dataframe(
                summary.round(2),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "위 통계는 기존 게임의 군집별 "
                "특성입니다. 신규 게임의 실제 "
                "리뷰 수를 예측한 값은 아닙니다."
            )

    except Exception as e:
        st.error("군집 데이터 분석에 실패했습니다.")
        st.exception(e)


# ==================================================
# 10. 분석 방법 설명
# ==================================================

st.header("5. 예측 방법")

st.markdown("""
### 예측 과정

1. 2페이지에서 학습한 최종 분류 모델을 불러옵니다.
2. 새로운 게임의 출시 전 정보를 입력합니다.
3. 학습 당시 사용한 범주형 인코딩 규칙을 적용합니다.
4. 분류 모델이 세 군집의 소속 확률을 계산합니다.
5. 예측 확률이 가장 높은 군집을 출력합니다.

### 주의 사항

- 긍정·부정 리뷰 수는 예측 입력에 사용하지 않습니다.
- 모델은 K-Means가 만든 군집 번호를 예측합니다.
- 예측 결과는 실제 판매량이나 매출을 보장하지 않습니다.
- 기존 학습 데이터에 없는 새로운 게임 특성은
  충분히 학습되지 않았을 수 있습니다.
""")
