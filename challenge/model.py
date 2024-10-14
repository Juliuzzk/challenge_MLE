from datetime import datetime
from typing import List, Tuple, Union
from sklearn.model_selection import train_test_split

import pandas as pd
import xgboost as xgb
import numpy as np
import pickle as ple

THRESHOLD = 15
FILENAME = "model.pkl"


class DelayModel:
    def __init__(self):
        self._model = xgb.XGBClassifier()
        self._features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]
        # self._model = self.load_model(FILENAME)

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        valid_features = list(set(self._features).intersection(set(features.columns)))
        missing_features = list(set(self._features).difference(set(features.columns)))

        # get valid features and fill missin with  0 due to one-hot encoding
        features = features[valid_features]
        features[missing_features] = 0

        if target_column is None:
            return features[self._features]

        else:
            data["min_diff"] = data.apply(self.get_min_diff, axis=1)
            data["delay"] = np.where(data["min_diff"] > THRESHOLD, 1, 0)

            return (features[self._features], data["delay"].to_frame())

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        n_y0 = int((target == 0).sum())
        n_y1 = int((target == 1).sum())
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=0.01, scale_pos_weight=scale
        )
        self._model.fit(x_train, y_train)

        self.save_model(FILENAME)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """

        self._model = self.load_model(FILENAME)

        predictions = self._model.predict(features)

        return predictions.tolist()

    @staticmethod
    def is_high_season(fecha):
        fecha_año = int(fecha.split("-")[0])
        fecha = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=fecha_año)
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=fecha_año)
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=fecha_año)
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=fecha_año)
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=fecha_año)
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=fecha_año)
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=fecha_año)
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=fecha_año)

        if (
            (fecha >= range1_min and fecha <= range1_max)
            or (fecha >= range2_min and fecha <= range2_max)
            or (fecha >= range3_min and fecha <= range3_max)
            or (fecha >= range4_min and fecha <= range4_max)
        ):
            return 1
        else:
            return 0

    @staticmethod
    def get_rate_from_column(data, column):
        delays = {}
        for _, row in data.iterrows():
            if row["delay"] == 1:
                if row[column] not in delays:
                    delays[row[column]] = 1
                else:
                    delays[row[column]] += 1
        total = data[column].value_counts().to_dict()

        rates = {}
        for name, total in total.items():
            if name in delays:
                rates[name] = round(total / delays[name], 2)
            else:
                rates[name] = 0

        return pd.DataFrame.from_dict(data=rates, orient="index", columns=["Tasa (%)"])

    @staticmethod
    def get_min_diff(data):
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    @staticmethod
    def get_period_day(date):
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("4:59", "%H:%M").time()

        if date_time > morning_min and date_time < morning_max:
            return "mañana"
        elif date_time > afternoon_min and date_time < afternoon_max:
            return "tarde"
        elif (date_time > evening_min and date_time < evening_max) or (
            date_time > night_min and date_time < night_max
        ):
            return "noche"

    def save_model(self, filename):
        with open(filename, "wb") as fp:
            ple.dump(self._model, fp)

    def load_model(self, filename: str):
        try:
            with open(filename, "rb") as fp:
                return ple.load(fp)
        except FileNotFoundError:
            return None


# For testing purposes
def main():
    model = DelayModel()
    data = pd.read_csv(filepath_or_buffer="../data/data.csv", low_memory=False)

    features = model.preprocess(data)

    # save
    # model.save_model("models")


if __name__ == "__main__":
    main()
