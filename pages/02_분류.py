
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import io
import ast
import re
import joblib

from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ==================================================
# 1. 페이지 설정
# ==================================================

st.title("Steam 게임 분류 모델 비교")

st.write(
    "K-Means로 생성한 3개 군집을 정답으로 사용하여 "
    "출시 전 게임 정보만으로 군집을 예측하는 "
    "분류 모델을 학습합니다."
)

st.info(
    "비교 모델: 로지스틱 회귀 / 랜덤 포레스트 / XGBoost"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "dotty708/common-traits-of-successful-games/"
    "refs/heads/main/data/"
    "games_march2025_popularity_filtered.csv"
)

FEATURE_CANDIDATES = [
    "price",
    "release_year",
    "release_month",
    "genres",
    "categories",
    "developers",
    "publishers"
]

META_COLUMNS = [
    "price",
    "release_date",
    "genres",
    "categories",
    "developers",
    "publishers"
]

CLASS_IDS = [0, 1, 2]


# ==================================================
# 2. 데이터 불러오기
# ==================================================

st.header("1. 데이터 불러오기")

st.write(
    "1페이지에서 다운로드한 군집화 결과 CSV를 "
    "업로드하세요."
)

cluster_file = st.file_uploader(
    "군집화 결과 CSV",
    type=["csv"],
    key="classification_cluster"
)

metadata_file = st.file_uploader(
    "출시 전 게임 정보 CSV (선택)",
    type=["csv"],
    key="classification_metadata",
    help=(
        "appid, price, release_date, genres, "
        "categories, developers, publishers "
        "등의 열을 포함할 수 있습니다."
    )
)

if cluster_file is None:
    st.warning(
        "먼저 1페이지에서 군집화 결과 CSV를 "
        "다운로드해 업로드하세요."
    )
    st.stop()


@st.cache_data(ttl=300)
def load_original():
    return pd.read_csv(DATA_URL)


try:
    df = pd.read_csv(cluster_file)

except Exception as e:
    st.error("군집화 결과 CSV를 읽지 못했습니다.")
    st.exception(e)
    st.stop()


if not {"appid", "cluster"}.issubset(df.columns):
    st.error(
        "CSV에는 appid와 cluster 열이 필요합니다."
    )
    st.stop()

df["appid"] = pd.to_numeric(
    df["appid"], errors="coerce"
)

df["cluster"] = pd.to_numeric(
    df["cluster"], errors="coerce"
)

df = df.dropna(
    subset=["appid", "cluster"]
).copy()

df = df[
    df["cluster"].isin(CLASS_IDS)
].copy()

df["appid"] = df["appid"].astype("int64")
df["cluster"] = df["cluster"].astype(int)

duplicate_count = int(
    df["appid"].duplicated().sum()
)

if duplicate_count:
    st.warning(
        f"appid 중복 {duplicate_count:,}개를 "
        "제외했습니다."
    )
    df = df.drop_duplicates("appid").copy()


# 군집 레이블만 업로드한 경우 원본 데이터 연결
needed = [
    c for c in ["price", "release_date"]
    if c not in df.columns
]

if needed:
    try:
        raw = load_original()

        raw["appid"] = pd.to_numeric(
            raw["appid"], errors="coerce"
        )

        raw = raw.dropna(
            subset=["appid"]
        ).copy()

        raw["appid"] = raw["appid"].astype("int64")

        available = [
            c for c in needed
            if c in raw.columns
        ]

        raw = raw[
            ["appid"] + available
        ].drop_duplicates("appid")

        df = df.merge(
            raw,
            on="appid",
            how="left",
            validate="one_to_one"
        )

    except Exception as e:
        st.warning(
            "GitHub 원본 데이터를 자동으로 "
            "불러오지 못했습니다."
        )
        st.exception(e)


# 추가 메타데이터 병합
if metadata_file is not None:

    try:
        meta = pd.read_csv(metadata_file)

        if "appid" not in meta.columns:
            st.error(
                "추가 정보 CSV에 appid 열이 없습니다."
            )
            st.stop()

        meta["appid"] = pd.to_numeric(
            meta["appid"], errors="coerce"
        )

        meta = meta.dropna(
            subset=["appid"]
        ).copy()

        meta["appid"] = meta["appid"].astype("int64")

        available_meta = [
            c for c in META_COLUMNS
            if c in meta.columns
        ]

        meta = meta[
            ["appid"] + available_meta
        ].drop_duplicates("appid")

        df = df.merge(
            meta,
            on="appid",
            how="left",
            suffixes=("", "__meta"),
            validate="one_to_one"
        )

        for column in available_meta:

            new_column = column + "__meta"

            if new_column in df.columns:
                df[column] = (
                    df[new_column].combine_first(
                        df[column]
                    )
                )
                df.drop(
                    columns=[new_column],
                    inplace=True
                )

    except Exception as e:
        st.error("추가 정보 CSV 병합에 실패했습니다.")
        st.exception(e)
        st.stop()


# 출시일 파생 변수
if "release_date" in df.columns:

    df["_release_datetime"] = pd.to_datetime(
        df["release_date"],
        errors="coerce"
    )

    df["release_year"] = (
        df["_release_datetime"].dt.year
    )

    df["release_month"] = (
        df["_release_datetime"].dt.month
    )


# ==================================================
# 3. 데이터 확인
# ==================================================

st.header("2. 학습 데이터 확인")

counts = (
    df["cluster"]
    .value_counts()
    .reindex(CLASS_IDS, fill_value=0)
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 게임 수", f"{len(df):,}")
c2.metric("군집 0", f"{counts[0]:,}")
c3.metric("군집 1", f"{counts[1]:,}")
c4.metric("군집 2", f"{counts[2]:,}")

if (counts < 2).any():
    st.error(
        "세 군집 각각에 최소 2개 이상의 "
        "게임이 있어야 합니다."
    )
    st.stop()

fig_counts = px.bar(
    x=["군집 0", "군집 1", "군집 2"],
    y=counts.values,
    text=counts.values,
    title="분류 모델 정답 레이블 분포",
    labels={
        "x": "군집",
        "y": "게임 수"
    }
)

st.plotly_chart(
    fig_counts,
    use_container_width=True
)

with st.expander("데이터 미리보기"):
    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# ==================================================
# 4. 분류 입력 변수 선택
# ==================================================

st.header("3. 분류 모델 설정")

available_features = [
    c for c in FEATURE_CANDIDATES
    if c in df.columns
    and df[c].notna().any()
]

selected_features = st.multiselect(
    "출시 전 입력 변수 선택",
    options=available_features,
    default=available_features
)

if not selected_features:
    st.warning("입력 변수를 하나 이상 선택하세요.")
    st.stop()

missing_features = [
    c for c in FEATURE_CANDIDATES
    if c not in available_features
]

if missing_features:
    st.warning(
        "현재 데이터에 없는 변수: "
        + ", ".join(missing_features)
    )

st.caption(
    "긍정·부정 리뷰 수, 최대 동시 접속자 수, "
    "추정 소유자 수는 분류 입력에서 제외됩니다."
)

if len(selected_features) <= 2:
    st.warning(
        "사용 가능한 출시 전 특성이 적습니다. "
        "장르·카테고리·개발사·배급사 정보를 "
        "추가하면 더 다양한 관계를 학습할 수 있습니다."
    )


# ==================================================
# 5. 데이터 분할 설정
# ==================================================

split_options = ["무작위 분할"]

if (
    "_release_datetime" in df.columns
    and df["_release_datetime"].notna().sum()
    >= 30
):
    split_options.append("출시일 기준 분할")

split_mode = st.selectbox(
    "학습·테스트 데이터 분할 방식",
    options=split_options
)

test_ratio = st.slider(
    "테스트 데이터 비율",
    min_value=0.15,
    max_value=0.30,
    value=0.20,
    step=0.05
)

st.caption(
    "무작위 분할은 군집 비율을 유지합니다. "
    "출시일 기준 분할은 과거 게임으로 학습하고 "
    "이후 출시 게임을 평가하는 방식입니다."
)


# ==================================================
# 6. 범주형 데이터 처리 함수
# ==================================================

def parse_items(value):
    """
    장르·카테고리·개발사·배급사 문자열을
    항목 리스트로 변환합니다.
    """

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


def create_top_values(train_df, features):
    """
    학습 데이터만 이용해 상위 범주를 선정.
    테스트 데이터의 정보 누출을 방지합니다.
    """

    limits = {
        "genres": 30,
        "categories": 40,
        "developers": 40,
        "publishers": 40
    }

    result = {}

    for column, limit in limits.items():

        if column not in features:
            continue

        counter = Counter()

        for value in train_df[column]:
            counter.update(parse_items(value))

        result[column] = [
            name
            for name, count
            in counter.most_common(limit)
        ]

    return result


def make_feature_dicts(frame, features, top_values):
    """
    입력 데이터를 DictVectorizer가 처리할 수
    있는 딕셔너리 목록으로 변환합니다.

    3페이지에서도 동일한 변환 규칙을 사용합니다.
    """

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

        # 여러 장르·카테고리 동시 반영
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

        # 개발사·배급사는 첫 번째 항목 사용
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
# 7. 모델 학습
# ==================================================

st.header("4. 분류 모델 학습")

st.write(
    "세 모델을 동일한 데이터로 학습한 뒤 "
    "테스트 성능을 비교합니다."
)

if not XGBOOST_AVAILABLE:
    st.error(
        "XGBoost가 설치되어 있지 않습니다. "
        "requirements.txt에 xgboost를 추가하세요."
    )
    st.stop()

if st.button(
    "세 모델 학습 시작",
    type="primary",
    use_container_width=True
):

    try:

        # 데이터 분할
        if split_mode == "출시일 기준 분할":

            working = df.dropna(
                subset=["_release_datetime"]
            ).copy()

            working = working.sort_values(
                "_release_datetime"
            )

            cutoff = int(
                len(working) * (1 - test_ratio)
            )

            train_df = working.iloc[:cutoff].copy()
            test_df = working.iloc[cutoff:].copy()

        else:

            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                random_state=42,
                stratify=df["cluster"]
            )

        train_classes = set(train_df["cluster"])
        test_classes = set(test_df["cluster"])

        if (
            train_classes != set(CLASS_IDS)
            or test_classes != set(CLASS_IDS)
        ):
            st.error(
                "학습 또는 테스트 데이터에 "
                "일부 군집이 없습니다. "
                "분할 방식을 변경하세요."
            )
            st.stop()

        # 범주형 처리 기준은 학습 데이터로만 생성
        top_values = create_top_values(
            train_df,
            selected_features
        )

        X_train = make_feature_dicts(
            train_df,
            selected_features,
            top_values
        )

        X_test = make_feature_dicts(
            test_df,
            selected_features,
            top_values
        )

        y_train = train_df["cluster"].to_numpy()
        y_test = test_df["cluster"].to_numpy()

        # 모델 정의
        models = {
            "로지스틱 회귀": Pipeline([
                (
                    "vectorizer",
                    DictVectorizer(sparse=True)
                ),
                (
                    "scaler",
                    StandardScaler(with_mean=False)
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42
                    )
                )
            ]),

            "랜덤 포레스트": Pipeline([
                (
                    "vectorizer",
                    DictVectorizer(sparse=True)
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=18,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ]),

            "XGBoost": Pipeline([
                (
                    "vectorizer",
                    DictVectorizer(sparse=True)
                ),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=180,
                        max_depth=5,
                        learning_rate=0.08,
                        subsample=0.85,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        num_class=3,
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=42,
                        n_jobs=2
                    )
                )
            ])
        }

        trained_models = {}
        predictions = {}
        metrics = []

        progress = st.progress(0)

        for i, (name, model) in enumerate(
            models.items()
        ):

            with st.spinner(f"{name} 학습 중..."):

                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                trained_models[name] = model
                predictions[name] = pred

                metrics.append({
                    "모델": name,
                    "Accuracy": accuracy_score(
                        y_test, pred
                    ),
                    "Balanced Accuracy":
                        balanced_accuracy_score(
                            y_test, pred
                        ),
                    "Macro F1": f1_score(
                        y_test,
                        pred,
                        labels=CLASS_IDS,
                        average="macro",
                        zero_division=0
                    )
                })

            progress.progress((i + 1) / 3)

        metrics_df = (
            pd.DataFrame(metrics)
            .sort_values(
                ["Macro F1", "Balanced Accuracy"],
                ascending=False
            )
            .reset_index(drop=True)
        )

        best_name = metrics_df.iloc[0]["모델"]

        st.session_state["steam_classification"] = {
            "metrics": metrics_df,
            "models": trained_models,
            "predictions": predictions,
            "y_test": y_test,
            "best_name": best_name,
            "selected_features": list(selected_features),
            "top_values": top_values,
            "train_count": len(train_df),
            "test_count": len(test_df),
            "split_mode": split_mode
        }

        st.success("세 모델의 학습이 완료되었습니다.")

    except Exception as e:
        st.error("모델 학습 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()


# ==================================================
# 8. 성능 비교 결과
# ==================================================

if "steam_classification" not in st.session_state:
    st.info(
        "학습 시작 버튼을 누르면 "
        "모델별 성능 비교 결과가 표시됩니다."
    )
    st.stop()

result = st.session_state["steam_classification"]

metrics_df = result["metrics"]
best_name = result["best_name"]
y_test = result["y_test"]

st.header("5. 모델 성능 비교")

c1, c2 = st.columns(2)

c1.metric(
    "학습 게임 수",
    f"{result['train_count']:,}"
)

c2.metric(
    "테스트 게임 수",
    f"{result['test_count']:,}"
)

st.dataframe(
    metrics_df.style.format({
        "Accuracy": "{:.4f}",
        "Balanced Accuracy": "{:.4f}",
        "Macro F1": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

plot_metrics = metrics_df.melt(
    id_vars="모델",
    value_vars=[
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1"
    ],
    var_name="평가 지표",
    value_name="점수"
)

fig_metrics = px.bar(
    plot_metrics,
    x="모델",
    y="점수",
    color="평가 지표",
    barmode="group",
    range_y=[0, 1],
    title="분류 모델 성능 비교"
)

st.plotly_chart(
    fig_metrics,
    use_container_width=True
)

st.caption(
    "Macro F1은 각 군집의 F1-score를 "
    "동등하게 평균낸 값입니다. "
    "군집별 게임 수가 불균형할 때 "
    "중요한 평가 지표입니다."
)


# ==================================================
# 9. 혼동행렬
# ==================================================

st.header("6. 혼동행렬")

view_model = st.selectbox(
    "분석할 모델",
    options=metrics_df["모델"].tolist(),
    index=0
)

pred = result["predictions"][view_model]

cm = confusion_matrix(
    y_test,
    pred,
    labels=CLASS_IDS
)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=["군집 0", "군집 1", "군집 2"],
    y=["군집 0", "군집 1", "군집 2"],
    labels={
        "x": "예측 군집",
        "y": "실제 군집",
        "color": "게임 수"
    },
    color_continuous_scale="Blues",
    title=f"{view_model} 혼동행렬"
)

st.plotly_chart(
    fig_cm,
    use_container_width=True
)

st.caption(
    "대각선에 위치한 값은 올바르게 "
    "분류한 게임 수를 의미합니다."
)


# ==================================================
# 10. 군집별 성능
# ==================================================

st.header("7. 군집별 분류 성능")

report = classification_report(
    y_test,
    pred,
    labels=CLASS_IDS,
    output_dict=True,
    zero_division=0
)

report_rows = []

for class_id in CLASS_IDS:

    values = report[str(class_id)]

    report_rows.append({
        "군집": f"군집 {class_id}",
        "Precision": values["precision"],
        "Recall": values["recall"],
        "F1-score": values["f1-score"],
        "게임 수": int(values["support"])
    })

report_df = pd.DataFrame(report_rows)

st.dataframe(
    report_df.style.format({
        "Precision": "{:.3f}",
        "Recall": "{:.3f}",
        "F1-score": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)

fig_f1 = px.bar(
    report_df,
    x="군집",
    y="F1-score",
    text_auto=".3f",
    range_y=[0, 1],
    title=f"{view_model} 군집별 F1-score"
)

st.plotly_chart(
    fig_f1,
    use_container_width=True
)


# ==================================================
# 11. 최종 모델 선정
# ==================================================

st.header("8. 최종 모델 선정")

best_score = float(
    metrics_df.iloc[0]["Macro F1"]
)

st.success(
    f"선정 모델: {best_name}"
)

st.metric(
    "최종 선정 모델 Macro F1",
    f"{best_score:.4f}"
)

st.write(
    "테스트 데이터의 Macro F1-score가 "
    "가장 높은 모델을 최종 후보로 선정했습니다."
)


# ==================================================
# 12. 모델 저장
# ==================================================

st.header("9. 최종 모델 다운로드")

bundle = {
    "version": 1,
    "model_name": best_name,
    "model": result["models"][best_name],
    "selected_features": result["selected_features"],
    "top_values": result["top_values"],
    "class_ids": CLASS_IDS,
    "macro_f1": best_score,
    "split_mode": result["split_mode"]
}

buffer = io.BytesIO()

joblib.dump(bundle, buffer)

st.download_button(
    label="최종 분류 모델 다운로드 (.joblib)",
    data=buffer.getvalue(),
    file_name="steam_final_classifier.joblib",
    mime="application/octet-stream",
    use_container_width=True
)

st.caption(
    "3페이지에서 이 모델 파일을 불러와 "
    "새로운 게임의 군집을 예측합니다. "
    "신뢰할 수 있는 모델 파일만 불러와야 합니다."
)


# ==================================================
# 13. 분석 설명
# ==================================================

st.header("10. 분석 방법")

st.markdown("""
### 학습 과정

1. K-Means로 생성한 군집 번호를 정답으로 사용
2. 출시 전에 확인할 수 있는 입력 변수 선택
3. 학습·테스트 데이터 분할
4. 장르·카테고리·개발사·배급사 인코딩
5. 세 가지 분류 모델 학습
6. 테스트 데이터 예측
7. Accuracy, Balanced Accuracy, Macro F1 비교
8. 최종 모델 선정 및 저장

### 주의 사항

이 모델은 실제 흥행 성공 여부를 직접 예측하는
것이 아니라, 기존 K-Means가 정의한
리뷰 기반 군집을 예측합니다.

모델 성능은 해당 군집 레이블에 대한 예측
정확도를 의미합니다.
""")
