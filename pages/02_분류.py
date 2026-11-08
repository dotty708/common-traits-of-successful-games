
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import ast
import re
import io
import joblib

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer

from sklearn.dummy import DummyClassifier
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

st.title("Steam 게임 인기 군집 분류")

st.write(
    "Steam 게임의 출시 전 정보를 이용하여 "
    "K-Means로 생성한 3개 군집을 예측합니다."
)

st.info(
    "베이스라인, 로지스틱 회귀, 랜덤 포레스트, "
    "XGBoost를 동일한 조건에서 비교합니다."
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
# 2. 데이터 로딩
# ==================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return pd.read_csv(DATA_URL)


@st.cache_data(ttl=3600, show_spinner=False)
def load_metadata():
    try:
        return pd.read_csv(METADATA_URL)
    except Exception:
        return None


# ==================================================
# 3. K-Means 군집 생성
# ==================================================

@st.cache_data(ttl=3600, show_spinner=False)
def create_clustered_data():

    raw = load_data().copy()

    required = ["appid", "positive", "negative"]

    missing = [
        col for col in required
        if col not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"필수 열이 없습니다: {missing}"
        )

    positive = pd.to_numeric(
        raw["positive"],
        errors="coerce"
    )

    negative = pd.to_numeric(
        raw["negative"],
        errors="coerce"
    )

    valid = (
        np.isfinite(positive)
        & np.isfinite(negative)
        & (positive >= 0)
        & (negative >= 0)
        & ((positive + negative) > 0)
    )

    df = raw.loc[valid].copy()

    df["positive"] = positive.loc[valid]
    df["negative"] = negative.loc[valid]

    df["total_reviews"] = (
        df["positive"] + df["negative"]
    )

    X_log = np.log1p(
        df[["positive", "negative"]].to_numpy(
            dtype=float
        )
    )

    X_scaled = StandardScaler().fit_transform(
        X_log
    )

    if len(np.unique(X_scaled, axis=0)) < 3:
        raise ValueError(
            "서로 다른 데이터가 3개 미만입니다."
        )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["_original_cluster"] = (
        kmeans.fit_predict(X_scaled)
    )

    # 리뷰 수 중앙값 기준으로 군집 번호 정렬
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
# 4. 데이터 연결
# ==================================================

st.header("1. 데이터 연결")

try:

    if (
        "steam_clustered_df" in st.session_state
        and "cluster" in st.session_state[
            "steam_clustered_df"
        ].columns
    ):

        df = st.session_state[
            "steam_clustered_df"
        ].copy()

        data_source = "1페이지 군집화 결과"

    else:

        with st.spinner(
            "Steam 데이터를 불러오고 군집화하는 중..."
        ):
            df = create_clustered_data()

        st.session_state[
            "steam_clustered_df"
        ] = df.copy()

        data_source = "GitHub 데이터 재군집화"

except Exception as e:

    st.error("데이터 연결에 실패했습니다.")
    st.exception(e)
    st.stop()


# ==================================================
# 5. 데이터 정리
# ==================================================

required = {"appid", "cluster"}

if not required.issubset(df.columns):
    st.error("appid 또는 cluster 열이 없습니다.")
    st.stop()

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
# 6. 출시 전 메타데이터 보완
# ==================================================

missing_columns = [
    col for col in METADATA_COLUMNS
    if col not in df.columns
]

if missing_columns:

    try:

        original = load_data().copy()

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
            col for col in missing_columns
            if col in original.columns
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
            "기본 CSV의 추가 정보를 "
            "병합하지 못했습니다."
        )
        st.caption(str(e))


metadata = load_metadata()

if (
    metadata is not None
    and "appid" in metadata.columns
):

    metadata = metadata.copy()

    metadata["appid"] = pd.to_numeric(
        metadata["appid"],
        errors="coerce"
    )

    metadata = metadata.dropna(
        subset=["appid"]
    )

    metadata["appid"] = (
        metadata["appid"].astype("int64")
    )

    available_meta = [
        col for col in METADATA_COLUMNS
        if col in metadata.columns
    ]

    if available_meta:

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

        for col in available_meta:

            extra = col + "__meta"

            if extra in df.columns:

                df[col] = (
                    df[extra].combine_first(df[col])
                )

                df.drop(
                    columns=[extra],
                    inplace=True
                )

    st.success("추가 메타데이터 연결 완료")

else:

    st.caption(
        "추가 메타데이터가 없어 "
        "기본 CSV의 변수를 사용합니다."
    )


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


st.success(
    f"{data_source}에서 "
    f"{len(df):,}개 게임을 불러왔습니다."
)


# ==================================================
# 7. 데이터 분포
# ==================================================

st.header("2. 학습 데이터 확인")

counts = (
    df["cluster"]
    .value_counts()
    .reindex(CLASS_IDS, fill_value=0)
)

c1, c2, c3, c4 = st.columns(4)

c1.metric("전체 게임", f"{len(df):,}")
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
    "군집": [f"군집 {i}" for i in CLASS_IDS],
    "게임 수": counts.values
})

fig_count = px.bar(
    count_df,
    x="군집",
    y="게임 수",
    text="게임 수",
    title="K-Means 군집별 게임 수"
)

fig_count.update_layout(
    xaxis_title="군집",
    yaxis_title="게임 수"
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
# 8. 입력 변수 설정
# ==================================================

st.header("3. 입력 변수 설정")

available_features = []

for col in FEATURE_CANDIDATES:

    if col not in df.columns:
        continue

    series = df[col]

    if series.notna().any():
        available_features.append(col)


selected_features = st.multiselect(
    "분류에 사용할 입력 변수",
    options=available_features,
    default=available_features
)

if not selected_features:
    st.warning(
        "최소 하나의 입력 변수를 선택하세요."
    )
    st.stop()


unavailable = [
    col for col in FEATURE_CANDIDATES
    if col not in available_features
]

if unavailable:
    st.warning(
        "현재 사용할 수 없는 변수: "
        + ", ".join(unavailable)
    )

st.caption(
    "출시 이후에 확인되는 positive, negative, "
    "peak_ccu, recommendations 등은 "
    "분류 입력 변수로 사용하지 않습니다."
)


missing_table = pd.DataFrame({
    "변수": selected_features,
    "결측치 수": [
        int(df[col].isna().sum())
        for col in selected_features
    ],
    "결측 비율(%)": [
        round(df[col].isna().mean() * 100, 2)
        for col in selected_features
    ]
})

with st.expander("선택 변수의 결측치 확인"):
    st.dataframe(
        missing_table,
        hide_index=True,
        use_container_width=True
    )

st.write(
    "숫자형 변수의 결측치는 0과 결측 여부 "
    "표시 변수로 처리합니다. 범주형 변수는 "
    "UNKNOWN 또는 OTHER로 처리합니다."
)


# ==================================================
# 9. 학습 설정
# ==================================================

st.header("4. 학습 설정")

split_options = ["무작위 분할"]

if (
    "_release_datetime" in df.columns
    and df["_release_datetime"].notna().sum() >= 30
):
    split_options.insert(
        0,
        "출시일 기준 분할"
    )

split_mode = st.selectbox(
    "학습·시험 데이터 분할 방식",
    options=split_options
)

test_ratio = st.slider(
    "시험 데이터 비율",
    min_value=0.15,
    max_value=0.30,
    value=0.20,
    step=0.05
)

st.caption(
    "무작위 분할은 군집 비율을 유지하는 "
    "계층화 분할을 사용합니다. "
    "난수는 42로 고정합니다."
)


# ==================================================
# 10. 범주형 데이터 전처리
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

    for col, limit in limits.items():

        if col not in features:
            continue

        counter = Counter()

        for value in train_df[col]:
            counter.update(parse_items(value))

        result[col] = [
            name
            for name, _ in counter.most_common(limit)
        ]

    return result


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

        # 숫자형 변수
        for col in [
            "price",
            "release_year",
            "release_month"
        ]:

            if col not in features:
                continue

            value = pd.to_numeric(
                row.get(col),
                errors="coerce"
            )

            valid = (
                pd.notna(value)
                and np.isfinite(value)
                and value >= 0
            )

            item[col] = (
                float(value) if valid else 0.0
            )

            item[col + "_missing"] = (
                0.0 if valid else 1.0
            )

        # 장르 및 카테고리
        for col, prefix in [
            ("genres", "genre"),
            ("categories", "category")
        ]:

            if col not in features:
                continue

            tokens = parse_items(row.get(col))
            allowed = top_sets.get(col, set())

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

        # 개발사 및 배급사
        for col, prefix in [
            ("developers", "developer"),
            ("publishers", "publisher")
        ]:

            if col not in features:
                continue

            tokens = parse_items(row.get(col))

            if not tokens:
                category = "UNKNOWN"

            elif tokens[0] in top_sets.get(
                col, set()
            ):
                category = tokens[0]

            else:
                category = "OTHER"

            item[prefix] = category

        result.append(item)

    return result


# ==================================================
# 11. 모델 정의
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


def calculate_metrics(name, y_true, y_pred):

    return {
        "모델": name,

        "Accuracy": accuracy_score(
            y_true, y_pred
        ),

        "Balanced Accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),

        "Macro F1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_IDS,
            average="macro",
            zero_division=0
        )
    }


# ==================================================
# 12. 모델 학습
# ==================================================

st.header("5. 모델 학습")

if not XGBOOST_AVAILABLE:

    st.error(
        "XGBoost가 설치되어 있지 않습니다. "
        "requirements.txt에 xgboost를 추가하세요."
    )
    st.stop()


current_config = (
    tuple(selected_features),
    split_mode,
    test_ratio
)

saved = st.session_state.get(
    "steam_classification"
)

if saved is not None:

    old_metrics = saved.get("metrics")

    has_baseline = (
        isinstance(old_metrics, pd.DataFrame)
        and "베이스라인"
        in old_metrics["모델"].values
    )

    if (
        saved.get("config") != current_config
        or not has_baseline
    ):
        del st.session_state[
            "steam_classification"
        ]


if st.button(
    "4개 모델 학습 및 비교 시작",
    type="primary",
    use_container_width=True
):

    try:

        # ------------------------------------------
        # 학습·시험 데이터 분할
        # ------------------------------------------

        if split_mode == "출시일 기준 분할":

            working = df.dropna(
                subset=["_release_datetime"]
            ).copy()

            working = working.sort_values(
                ["_release_datetime", "appid"]
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

            train_df, test_df = train_test_split(
                df,
                test_size=test_ratio,
                random_state=42,
                stratify=df["cluster"]
            )

        if (
            set(train_df["cluster"]) != set(CLASS_IDS)
            or set(test_df["cluster"]) != set(CLASS_IDS)
        ):

            st.error(
                "학습 또는 시험 데이터에 "
                "일부 군집이 없습니다. "
                "분할 방식을 변경하세요."
            )
            st.stop()

        # 범주형 상위 항목은 학습 데이터에서만 선정
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

        trained_models = {}
        predictions = {}
        metrics = []

        progress = st.progress(0)

        # ------------------------------------------
        # 베이스라인
        # ------------------------------------------

        with st.spinner("베이스라인 평가 중..."):

            baseline = DummyClassifier(
                strategy="most_frequent"
            )

            X_dummy_train = np.zeros(
                (len(y_train), 1)
            )

            X_dummy_test = np.zeros(
                (len(y_test), 1)
            )

            baseline.fit(
                X_dummy_train,
                y_train
            )

            baseline_pred = baseline.predict(
                X_dummy_test
            )

            predictions["베이스라인"] = (
                baseline_pred
            )

            metrics.append(
                calculate_metrics(
                    "베이스라인",
                    y_test,
                    baseline_pred
                )
            )

        progress.progress(0.25)

        # ------------------------------------------
        # 분류 모델 3개
        # ------------------------------------------

        models = create_models()

        for index, (name, model) in enumerate(
            models.items()
        ):

            with st.spinner(f"{name} 학습 중..."):

                model.fit(X_train, y_train)

                pred = model.predict(X_test)

                trained_models[name] = model
                predictions[name] = pred

                metrics.append(
                    calculate_metrics(
                        name,
                        y_test,
                        pred
                    )
                )

            progress.progress(
                (index + 2) / 4
            )

        # ------------------------------------------
        # 결과 정리
        # ------------------------------------------

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

        # 최종 모델은 베이스라인 제외
        candidates = metrics_df[
            metrics_df["모델"] != "베이스라인"
        ]

        best_name = (
            candidates.iloc[0]["모델"]
        )

        # 실제 사례 분석용 시험 데이터
        example_columns = [
            col for col in [
                "appid",
                "name",
                "cluster",
                "price",
                "release_date",
                "positive",
                "negative"
            ]
            if col in test_df.columns
        ]

        test_examples = (
            test_df[example_columns]
            .reset_index(drop=True)
            .copy()
        )

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

            "config": current_config,

            "test_examples": test_examples,

            "train_class_counts": (
                train_df["cluster"]
                .value_counts()
                .reindex(CLASS_IDS, fill_value=0)
                .to_dict()
            )
        }

        st.success(
            "베이스라인 및 3개 분류 모델의 "
            "학습과 평가가 완료되었습니다."
        )

    except Exception as e:

        st.error("모델 학습 중 오류가 발생했습니다.")
        st.exception(e)
        st.stop()


# ==================================================
# 13. 학습 결과 불러오기
# ==================================================

if (
    "steam_classification"
    not in st.session_state
):

    st.info(
        "학습 버튼을 누르면 성능 비교, "
        "혼동행렬, 오분류 분석이 표시됩니다."
    )
    st.stop()


result = st.session_state[
    "steam_classification"
]

metrics_df = result["metrics"]
best_name = result["best_name"]
y_test = result["y_test"]


# ==================================================
# 14. 모델 성능 비교
# ==================================================

st.header("6. 모델 성능 비교")

c1, c2 = st.columns(2)

c1.metric(
    "학습 게임 수",
    f"{result['train_count']:,}"
)

c2.metric(
    "시험 게임 수",
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

plot_df = metrics_df.melt(
    id_vars="모델",
    value_vars=[
        "Accuracy",
        "Balanced Accuracy",
        "Macro F1"
    ],
    var_name="평가지표",
    value_name="점수"
)

fig_metrics = px.bar(
    plot_df,
    x="모델",
    y="점수",
    color="평가지표",
    barmode="group",
    range_y=[0, 1],
    title="베이스라인 및 분류 모델 성능 비교"
)

fig_metrics.update_layout(
    xaxis_title="모델",
    yaxis_title="평가지표 점수"
)

st.plotly_chart(
    fig_metrics,
    use_container_width=True
)


# ==================================================
# 15. 베이스라인 대비 성능
# ==================================================

st.subheader("베이스라인 대비 성능 향상")

baseline_f1 = float(
    metrics_df.loc[
        metrics_df["모델"] == "베이스라인",
        "Macro F1"
    ].iloc[0]
)

best_f1 = float(
    metrics_df.loc[
        metrics_df["모델"] == best_name,
        "Macro F1"
    ].iloc[0]
)

improvement = best_f1 - baseline_f1

c1, c2, c3 = st.columns(3)

c1.metric(
    "베이스라인 Macro F1",
    f"{baseline_f1:.4f}"
)

c2.metric(
    "최종 모델 Macro F1",
    f"{best_f1:.4f}"
)

c3.metric(
    "베이스라인 대비 차이",
    f"{improvement:+.4f}"
)

if improvement > 0:

    st.success(
        "최종 모델이 베이스라인보다 "
        "높은 Macro F1을 기록했습니다."
    )

else:

    st.warning(
        "최종 모델이 베이스라인보다 "
        "높은 Macro F1을 기록하지 못했습니다."
    )

st.caption(
    "베이스라인은 입력 변수를 사용하지 않고 "
    "학습 데이터에서 가장 많은 군집만 예측합니다."
)


# ==================================================
# 16. 혼동행렬
# ==================================================

st.header("7. 혼동행렬")

model_names = metrics_df["모델"].tolist()

view_model = st.selectbox(
    "분석할 모델",
    options=model_names,
    index=model_names.index(best_name)
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
    x=[f"군집 {i}" for i in CLASS_IDS],
    y=[f"군집 {i}" for i in CLASS_IDS],
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
    "대각선은 올바르게 분류한 게임, "
    "대각선 밖은 잘못 분류한 게임입니다."
)


# ==================================================
# 17. 군집별 성능
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
    text="F1-score",
    range_y=[0, 1],
    title=f"{view_model} 군집별 F1-score"
)

fig_f1.update_traces(
    texttemplate="%{text:.3f}"
)

fig_f1.update_layout(
    xaxis_title="군집",
    yaxis_title="F1-score"
)

st.plotly_chart(
    fig_f1,
    use_container_width=True
)


# ==================================================
# 18. 실제 오분류 사례 분석
# ==================================================

st.header("9. 실제 게임 오분류 분석")

examples = result["test_examples"].copy()

examples["실제 군집"] = y_test
examples["예측 군집"] = pred

examples["분류 결과"] = np.where(
    examples["실제 군집"]
    == examples["예측 군집"],
    "정답",
    "오분류"
)

wrong = examples[
    examples["분류 결과"] == "오분류"
].copy()

wrong_count = len(wrong)

wrong_rate = (
    wrong_count / len(examples) * 100
    if len(examples) else 0
)

c1, c2 = st.columns(2)

c1.metric(
    "오분류 게임 수",
    f"{wrong_count:,}"
)

c2.metric(
    "오분류 비율",
    f"{wrong_rate:.2f}%"
)

if wrong_count > 0:

    st.subheader("가장 많이 혼동한 군집")

    confusion_pairs = (
        wrong.groupby(
            ["실제 군집", "예측 군집"]
        )
        .size()
        .reset_index(name="게임 수")
        .sort_values(
            "게임 수",
            ascending=False
        )
    )

    st.dataframe(
        confusion_pairs,
        hide_index=True,
        use_container_width=True
    )

    top_pair = confusion_pairs.iloc[0]

    st.write(
        f"가장 많은 오분류는 실제 군집 "
        f"{int(top_pair['실제 군집'])}을 "
        f"군집 {int(top_pair['예측 군집'])}으로 "
        f"예측한 경우이며, "
        f"{int(top_pair['게임 수']):,}개입니다."
    )

    st.subheader("실제 오분류 게임")

    st.dataframe(
        wrong.head(20),
        hide_index=True,
        use_container_width=True
    )

else:

    st.success(
        "시험 데이터에서 오분류가 발생하지 않았습니다."
    )


# ==================================================
# 19. 실제 게임 예측 사례
# ==================================================

st.header("10. 실제 게임 예측 사례")

correct = examples[
    examples["분류 결과"] == "정답"
]

if not correct.empty:

    st.write("올바르게 예측한 실제 게임 사례")

    st.dataframe(
        correct.head(5),
        hide_index=True,
        use_container_width=True
    )

if not wrong.empty:

    st.write("잘못 예측한 실제 게임 사례")

    st.dataframe(
        wrong.head(5),
        hide_index=True,
        use_container_width=True
    )

st.caption(
    "위 사례는 학습에 사용하지 않은 "
    "시험 데이터의 실제 게임입니다."
)


# ==================================================
# 20. 최종 모델 선정
# ==================================================

st.header("11. 최종 모델 선정")

st.success(
    f"최종 선정 모델: {best_name}"
)

st.metric(
    "최종 모델 Macro F1",
    f"{best_f1:.4f}"
)

st.write(
    "베이스라인을 제외한 세 가지 분류 모델 중 "
    "시험 데이터의 Macro F1이 가장 높은 "
    "모델을 최종 모델로 선정했습니다."
)

st.caption(
    "동점일 경우 Balanced Accuracy가 "
    "높은 모델을 우선합니다."
)


# ==================================================
# 21. 3페이지 자동 연결
# ==================================================

st.header("12. 신규 게임 예측 연결")

st.success(
    "최종 모델이 Streamlit 세션에 저장되었습니다. "
    "3페이지에서 자동으로 불러올 수 있습니다."
)

bundle = {
    "version": 1,

    "model_name": best_name,

    "model": result["models"][best_name],

    "selected_features":
        result["selected_features"],

    "top_values":
        result["top_values"],

    "class_ids": CLASS_IDS,

    "macro_f1": best_f1,

    "split_mode":
        result["split_mode"]
}

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
    "백업 파일은 신뢰할 수 있는 환경에서만 "
    "불러오세요. 같은 Streamlit 세션에서는 "
    "별도 업로드 없이 3페이지로 이동할 수 있습니다."
)


# ==================================================
# 22. 분석 방법 및 해석
# ==================================================

st.header("13. 분석 방법 정리")

st.markdown("""
### 연구 목적

게임의 출시 전 정보를 활용해
기존 Steam 게임의 리뷰 기반 군집을
예측할 수 있는지 확인합니다.

### 모델 비교 과정

1. Steam 게임 데이터 불러오기
2. 긍정·부정 리뷰를 이용한 K-Means 군집 생성
3. 출시 전 정보만 입력 변수로 선택
4. 학습·시험 데이터 분리
5. 학습 데이터 기준 범주형 변수 인코딩
6. 최빈 군집 베이스라인 평가
7. 로지스틱 회귀 학습
8. 랜덤 포레스트 학습
9. XGBoost 학습
10. Accuracy, Balanced Accuracy, Macro F1 비교
11. 혼동행렬과 실제 오분류 사례 분석
12. 최종 모델 선정 및 신규 게임 예측 연결

### 주요 평가지표

- **Accuracy:** 전체 게임 중 정확하게 분류한 비율
- **Balanced Accuracy:** 군집별 재현율의 평균
- **Macro F1:** 군집별 F1-score의 단순 평균

군집별 게임 수가 다를 수 있으므로
Macro F1을 주요 평가지표로 사용합니다.

### 분석의 한계

- 군집은 긍정·부정 리뷰 수로 만들어졌습니다.
- 군집 번호 자체가 흥행 성공을 의미하지는 않습니다.
- 출시 이후의 리뷰 수와 동시 접속자 수는
  분류 모델의 입력 변수로 사용하지 않습니다.
- 군집 레이블은 전체 데이터를 이용해
  생성했으므로, 분류 성능은 해당 레이블의
  재현 성능으로 해석해야 합니다.
- 예측 확률은 실제 흥행 성공 확률이 아닙니다.
- 게임의 품질이나 시장 환경 등 데이터에
  없는 요인은 충분히 반영되지 않을 수 있습니다.
""")
