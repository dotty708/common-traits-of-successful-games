
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import io
import ast
import re
import joblib

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ==================================================
# 1. 기본 설정
# ==================================================

st.title("Steam 게임 분류 모델 비교")

st.write(
    "K-Means 군집화로 생성한 3개 군집을 "
    "출시 전 게임 정보만으로 예측하는 "
    "분류 모델을 학습하고 비교합니다."
)

st.info(
    "분류 모델: 로지스틱 회귀 / "
    "랜덤 포레스트 / XGBoost"
)

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "dotty708/common-traits-of-successful-games/"
    "refs/heads/main/data/"
)

DATA_URL = (
    BASE_URL
    + "games_march2025_popularity_filtered.csv"
)

# 선택 사항:
# GitHub data 폴더에 이 파일을 추가하면
# 장르, 개발사 등의 정보를 자동으로 병합합니다.
METADATA_URL = (
    BASE_URL
    + "games_march2025_prerelease_metadata.csv"
)

CLASS_IDS = [0, 1, 2]

FEATURE_CANDIDATES = [
    "price",
    "release_year",
    "release_month",
    "genres",
    "categories",
    "developers",
    "publishers"
]

METADATA_COLUMNS = [
    "price",
    "release_date",
    "genres",
    "categories",
    "developers",
    "publishers"
]


# ==================================================
# 2. 데이터 자동 로딩
# ==================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_original_data():
    return pd.read_csv(DATA_URL)


@st.cache_data(ttl=3600, show_spinner=False)
def load_optional_metadata():
    try:
        return pd.read_csv(METADATA_URL)
    except Exception:
        return None


# ==================================================
# 3. K-Means 자동 군집화
# ==================================================

@st.cache_data(ttl=3600, show_spinner=False)
def create_clustered_data():

    raw_df = load_original_data().copy()

    required = ["positive", "negative", "appid"]

    missing = [
        c for c in required
        if c not in raw_df.columns
    ]

    if missing:
        raise ValueError(
            f"필수 열이 없습니다: {missing}"
        )

    positive = pd.to_numeric(
        raw_df["positive"],
        errors="coerce"
    )

    negative = pd.to_numeric(
        raw_df["negative"],
        errors="coerce"
    )

    valid_mask = (
        np.isfinite(positive)
        & np.isfinite(negative)
        & (positive >= 0)
        & (negative >= 0)
        & ((positive + negative) > 0)
    )

    df = raw_df.loc[valid_mask].copy()

    df["positive"] = positive.loc[valid_mask]
    df["negative"] = negative.loc[valid_mask]

    df["total_reviews"] = (
        df["positive"] + df["negative"]
    )

    # 1페이지와 동일한 전처리
    X_log = np.log1p(
        df[["positive", "negative"]].to_numpy(
            dtype=float
        )
    )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X_log)

    if len(np.unique(X_scaled, axis=0)) < 3:
        raise ValueError(
            "군집화에 필요한 서로 다른 데이터가 "
            "3개 미만입니다."
        )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["_original_cluster"] = (
        kmeans.fit_predict(X_scaled)
    )

    # 1페이지와 동일한 군집 번호 정렬
    order = (
        df.groupby("_original_cluster")[
            "total_reviews"
        ]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    mapping = {
        old: new
        for new, old in enumerate(order)
    }

    df["cluster"] = (
        df["_original_cluster"]
        .map(mapping)
        .astype(int)
    )

    df.drop(
        columns=["_original_cluster"],
        inplace=True
    )

    return df


# ==================================================
# 4. 데이터 자동 연결
# ==================================================

st.header("1. 데이터 자동 연결")

try:

    if "steam_clustered_df" in st.session_state:

        df = st.session_state[
            "steam_clustered_df"
        ].copy()

        data_source = "1페이지 군집화 결과"

    else:

        with st.spinner(
            "Steam 데이터를 불러오고 "
            "K-Means 군집화를 수행하는 중..."
        ):

            df = create_clustered_data()

        st.session_state[
            "steam_clustered_df"
        ] = df.copy()

        data_source = "GitHub 원본 데이터"

except Exception as e:

    st.error("데이터 자동 연결에 실패했습니다.")
    st.exception(e)
    st.stop()


if not {"appid", "cluster"}.issubset(
    df.columns
):
    st.error(
        "군집 데이터에 appid 또는 "
        "cluster 열이 없습니다."
    )
    st.stop()


# 데이터 타입 정리
df["appid"] = pd.to_numeric(
    df["appid"],
    errors="coerce"
)

df["cluster"] = pd.to_numeric(
    df["cluster"],
    errors="coerce"
)

df = df.dropna(
    subset=["appid", "cluster"]
).copy()

df = df[
    df["cluster"].isin(CLASS_IDS)
].copy()

df["appid"] = df["appid"].astype("int64")
df["cluster"] = df["cluster"].astype(int)

df = df.drop_duplicates(
    subset=["appid"]
).copy()


# ==================================================
# 5. 원본 데이터의 출시 전 정보 보완
# ==================================================

needed_columns = [
    c for c in METADATA_COLUMNS
    if c not in df.columns
]

if needed_columns:

    try:

        original = load_original_data().copy()

        original["appid"] = pd.to_numeric(
            original["appid"],
            errors="coerce"
        )

        original = original.dropna(
            subset=["appid"]
        )

        original["appid"] = (
            original["appid"].astype("int64")
        )

        available = [
            c for c in needed_columns
            if c in original.columns
        ]

        if available:

            original = original[
                ["appid"] + available
            ].drop_duplicates("appid")

            df = df.merge(
                original,
                on="appid",
                how="left",
                validate="one_to_one"
            )

    except Exception as e:
        st.warning(
            "원본 데이터의 추가 열을 "
            "불러오지 못했습니다."
        )
        st.caption(str(e))


# ==================================================
# 6. GitHub 추가 메타데이터 자동 병합
# ==================================================

metadata = load_optional_metadata()

if metadata is not None:

    if "appid" in metadata.columns:

        metadata["appid"] = pd.to_numeric(
            metadata["appid"],
            errors="coerce"
        )

        metadata = metadata.dropna(
            subset=["appid"]
        ).copy()

        metadata["appid"] = (
            metadata["appid"].astype("int64")
        )

        available_meta = [
            c for c in METADATA_COLUMNS
            if c in metadata.columns
        ]

        metadata = metadata[
            ["appid"] + available_meta
        ].drop_duplicates("appid")

        df = df.merge(
            metadata,
            on="appid",
            how="left",
            suffixes=("", "__meta"),
            validate="one_to_one"
        )

        for column in available_meta:

            extra_col = column + "__meta"

            if extra_col in df.columns:

                df[column] = (
                    df[extra_col].combine_first(
                        df[column]
                    )
                )

                df.drop(
                    columns=[extra_col],
                    inplace=True
                )

        st.success(
            "GitHub 추가 메타데이터 자동 연결 완료"
        )

else:

    st.caption(
        "별도 메타데이터 파일이 없어 "
        "현재 Steam CSV의 출시 전 정보를 "
        "사용합니다."
    )


# ==================================================
# 7. 출시일 파생 변수
# ==================================================

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


st.success(
    f"{data_source}에서 "
    f"{len(df):,}개 게임의 군집 데이터를 "
    "자동으로 불러왔습니다."
)


# ==================================================
# 8. 데이터 분포 확인
# ==================================================

st.header("2. 학습 데이터 확인")

counts = (
    df["cluster"]
    .value_counts()
    .reindex(CLASS_IDS, fill_value=0)
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "전체 게임 수",
    f"{len(df):,}"
)

c2.metric("군집 0", f"{counts[0]:,}")
c3.metric("군집 1", f"{counts[1]:,}")
c4.metric("군집 2", f"{counts[2]:,}")

if (counts < 2).any():
    st.error(
        "각 군집에 최소 2개 이상의 "
        "게임이 필요합니다."
    )
    st.stop()


count_df = pd.DataFrame({
    "군집": [
        "군집 0",
        "군집 1",
        "군집 2"
    ],
    "게임 수": counts.values
})

fig_count = px.bar(
    count_df,
    x="군집",
    y="게임 수",
    text_auto=",",
    title="K-Means 군집별 게임 수"
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)

with st.expander("학습 데이터 미리보기"):

    st.dataframe(
        df.head(30),
        use_container_width=True
    )


# ==================================================
# 9. 분류 입력 변수 설정
# ==================================================

st.header("3. 입력 변수 설정")

available_features = [
    c for c in FEATURE_CANDIDATES
    if c in df.columns
    and df[c].notna().any()
]

selected_features = st.multiselect(
    "분류 모델 입력 변수",
    options=available_features,
    default=available_features
)

if not selected_features:
    st.warning(
        "최소 하나 이상의 입력 변수를 선택하세요."
    )
    st.stop()

missing_features = [
    c for c in FEATURE_CANDIDATES
    if c not in available_features
]

if missing_features:

    st.warning(
        "현재 데이터에서 사용할 수 없는 변수: "
        + ", ".join(missing_features)
    )

st.caption(
    "분류 모델에는 positive, negative, "
    "peak_ccu, estimated_owners 등 "
    "출시 이후에 확인되는 지표를 사용하지 않습니다."
)


# ==================================================
# 10. 데이터 분할 설정
# ==================================================

st.header("4. 학습 설정")

split_options = ["무작위 분할"]

if (
    "_release_datetime" in df.columns
    and df["_release_datetime"].notna().sum()
    >= 30
):
    split_options.insert(
        0,
        "출시일 기준 분할"
    )

split_mode = st.selectbox(
    "학습·테스트 데이터 분할",
    options=split_options
)

test_ratio = st.slider(
    "테스트 데이터 비율",
    min_value=0.15,
    max_value=0.30,
    value=0.20,
    step=0.05
)

st.write(
    "학습 데이터로 모델을 훈련하고, "
    "테스트 데이터로 예측 성능을 평가합니다."
)


# ==================================================
# 11. 범주형 데이터 전처리
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


def create_top_values(train_df, features):

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


def make_feature_dicts(
    frame,
    features,
    top_values
):

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

        # 숫자형 변수
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

        # 장르와 카테고리
        for column, prefix in [
            ("genres", "genre"),
            ("categories", "category")
        ]:

            if column not in features:
                continue

            tokens = parse_items(
                row.get(column)
            )

            allowed = top_sets.get(
                column, set()
            )

            matched = [
                token for token in tokens
                if token in allowed
            ]

            for token in matched:
                item[
                    prefix + "=" + token
                ] = 1.0

            if not tokens:
                item[
                    prefix + "=UNKNOWN"
                ] = 1.0

            elif not matched:
                item[
                    prefix + "=OTHER"
                ] = 1.0

        # 개발사와 배급사
        for column, prefix in [
            ("developers", "developer"),
            ("publishers", "publisher")
        ]:

            if column not in features:
                continue

            tokens = parse_items(
                row.get(column)
            )

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
# 12. 분류 모델 정의
# ==================================================

def create_models():

    return {

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


# ==================================================
# 13. 모델 학습
# ==================================================

st.header("5. 분류 모델 학습")

if not XGBOOST_AVAILABLE:

    st.error(
        "XGBoost가 설치되지 않았습니다. "
        "requirements.txt에 xgboost를 추가하세요."
    )
    st.stop()


# 설정 변경 시 이전 결과와 혼동 방지
current_config = (
    tuple(selected_features),
    split_mode,
    test_ratio
)

saved = st.session_state.get(
    "steam_classification"
)

if (
    saved is not None
    and saved.get("config") != current_config
):
    del st.session_state[
        "steam_classification"
    ]


if st.button(
    "세 모델 학습 시작",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "학습·테스트 데이터를 분리하는 중..."
        ):

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

                train_df = working.iloc[
                    :cutoff
                ].copy()

                test_df = working.iloc[
                    cutoff:
                ].copy()

            else:

                train_df, test_df = (
                    train_test_split(
                        df,
                        test_size=test_ratio,
                        random_state=42,
                        stratify=df["cluster"]
                    )
                )

            if (
                set(train_df["cluster"])
                != set(CLASS_IDS)
                or set(test_df["cluster"])
                != set(CLASS_IDS)
            ):

                st.error(
                    "학습 또는 테스트 데이터에 "
                    "일부 군집이 없습니다. "
                    "다른 분할 방식을 선택하세요."
                )
                st.stop()

            # 학습 데이터만으로 범주 선정
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

            y_train = (
                train_df["cluster"].to_numpy()
            )

            y_test = (
                test_df["cluster"].to_numpy()
            )

        models = create_models()

        trained_models = {}
        predictions = {}
        metrics = []

        progress = st.progress(0)

        for index, (name, model) in enumerate(
            models.items()
        ):

            with st.spinner(
                f"{name} 학습 중..."
            ):

                model.fit(
                    X_train,
                    y_train
                )

                pred = model.predict(X_test)

                trained_models[name] = model
                predictions[name] = pred

                metrics.append({

                    "모델": name,

                    "Accuracy": accuracy_score(
                        y_test,
                        pred
                    ),

                    "Balanced Accuracy":
                        balanced_accuracy_score(
                            y_test,
                            pred
                        ),

                    "Macro F1": f1_score(
                        y_test,
                        pred,
                        labels=CLASS_IDS,
                        average="macro",
                        zero_division=0
                    )
                })

            progress.progress(
                (index + 1) / len(models)
            )

        metrics_df = (
            pd.DataFrame(metrics)
            .sort_values(
                [
                    "Macro F1",
                    "Balanced Accuracy"
                ],
                ascending=False
            )
            .reset_index(drop=True)
        )

        best_name = (
            metrics_df.iloc[0]["모델"]
        )

        # 3페이지에서 자동으로 불러올 데이터
        st.session_state[
            "steam_classification"
        ] = {

            "metrics": metrics_df,

            "models": trained_models,

            "predictions": predictions,

            "y_test": y_test,

            "best_name": best_name,

            "selected_features":
                list(selected_features),

            "top_values": top_values,

            "train_count": len(train_df),

            "test_count": len(test_df),

            "split_mode": split_mode,

            "config": current_config
        }

        st.success(
            "세 모델 학습이 완료되었습니다. "
            "3페이지에서 최종 모델을 "
            "자동으로 사용할 수 있습니다."
        )

    except Exception as e:

        st.error(
            "분류 모델 학습 중 오류가 발생했습니다."
        )

        st.exception(e)
        st.stop()


# ==================================================
# 14. 학습 결과 확인
# ==================================================

if (
    "steam_classification"
    not in st.session_state
):

    st.info(
        "학습 시작 버튼을 누르면 "
        "모델 성능 비교 결과가 표시됩니다."
    )
    st.stop()


result = st.session_state[
    "steam_classification"
]

metrics_df = result["metrics"]

best_name = result["best_name"]

y_test = result["y_test"]


# ==================================================
# 15. 성능 비교
# ==================================================

st.header("6. 모델 성능 비교")

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


# ==================================================
# 16. 혼동행렬
# ==================================================

st.header("7. 혼동행렬")

view_model = st.selectbox(
    "분석할 모델 선택",
    options=metrics_df["모델"].tolist()
)

pred = result[
    "predictions"
][view_model]

cm = confusion_matrix(
    y_test,
    pred,
    labels=CLASS_IDS
)

fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=[
        "군집 0",
        "군집 1",
        "군집 2"
    ],
    y=[
        "군집 0",
        "군집 1",
        "군집 2"
    ],
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


# ==================================================
# 17. 군집별 분류 성능
# ==================================================

st.header("8. 군집별 분류 성능")

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
# 18. 최종 모델 선정
# ==================================================

st.header("9. 최종 모델 선정")

best_score = float(
    metrics_df.iloc[0]["Macro F1"]
)

st.success(
    f"최종 선정 모델: {best_name}"
)

st.metric(
    "최종 모델 Macro F1",
    f"{best_score:.4f}"
)

st.write(
    "테스트 데이터에서 Macro F1이 "
    "가장 높은 모델을 최종 모델로 선정했습니다."
)


# ==================================================
# 19. 최종 모델 자동 연결
# ==================================================

st.header("10. 3페이지 자동 연결")

st.success(
    "최종 모델이 Streamlit 세션에 저장되어 "
    "있습니다. 3페이지에서 파일 업로드 없이 "
    "바로 사용할 수 있습니다."
)

bundle = {

    "version": 1,

    "model_name": best_name,

    "model": result[
        "models"
    ][best_name],

    "selected_features":
        result["selected_features"],

    "top_values":
        result["top_values"],

    "class_ids": CLASS_IDS,

    "macro_f1": best_score,

    "split_mode":
        result["split_mode"]
}

# 선택적으로 백업 다운로드도 제공
buffer = io.BytesIO()

joblib.dump(bundle, buffer)

st.download_button(
    label="최종 모델 백업 다운로드 (.joblib)",
    data=buffer.getvalue(),
    file_name="steam_final_classifier.joblib",
    mime="application/octet-stream",
    use_container_width=True
)

st.caption(
    "다운로드는 선택 사항입니다. "
    "같은 Streamlit 세션에서는 "
    "3페이지에 모델이 자동 전달됩니다."
)


# ==================================================
# 20. 분석 방법 정리
# ==================================================

st.header("11. 분석 방법")

st.markdown("""
### 전체 분석 흐름

1. GitHub Steam 데이터 자동 로딩
2. K-Means 군집 결과 자동 연결
3. 출시 전 정보 추출
4. 학습·테스트 데이터 분할
5. 범주형 변수 인코딩
6. 로지스틱 회귀 학습
7. 랜덤 포레스트 학습
8. XGBoost 학습
9. 분류 모델 성능 비교
10. 최적 모델 선정
11. 3페이지로 모델 자동 전달

### 분류 모델의 목적

새로운 게임의 출시 전 정보를 기반으로
기존 Steam 게임에서 관찰된
리뷰 기반 군집을 예측합니다.

### 분석 시 유의점

- 군집은 긍정·부정 리뷰 수로 생성되었습니다.
- 분류 모델에는 출시 후 리뷰 지표를 넣지 않습니다.
- 군집 예측 확률은 실제 흥행 성공 확률이 아닙니다.
- 1페이지에서 생성한 전체 데이터 기반 군집을
  정답으로 사용하므로, 테스트 성능은 해당
  군집 레이블의 재현 성능으로 해석해야 합니다.
""")
