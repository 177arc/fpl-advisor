import pandas as pd
import tensorflow as tf
import numpy as np
import re

def df_to_ds(df: pd.DataFrame, pred_col: str, shuffle: bool = False, batch_size: int = 32) -> tf.data.Dataset:
    """
    Converts the given data frame into a TensorFlow data set.

    Args:
        df: The data frame to be converted.
        pred_col: The name of the column with the value to predict, i.e. the label column.
        shuffle: Whether to shuffle the data set or not.
        batch_size: The batch size when shuffling the data set.

    Returns:
        The converted TensorFlow data set
    """
    df = df.copy()
    labels = df.pop(pred_col)
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))

    return ds.batch(batch_size)


def nn_norm(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    return df.apply(lambda col: (col-col.mean()) / col.std() if col.name != pred_col and np.issubdtype(col.dtype, np.number) else col)


def nn_sys_name(name: str) -> str:
    return re.sub(r'\W', '_', name.lower())


def nn_prep_ds(player_fixture_stats: pd.DataFrame, pred_col: str) -> tf.data.Dataset:
    return (player_fixture_stats
            .rename(columns=lambda col: nn_sys_name(col))
            .pipe(nn_norm, nn_sys_name(pred_col))
            .pipe(df_to_ds, nn_sys_name(pred_col)))


def nn_split(df: pd.DataFrame, frac: float) -> (pd.DataFrame, pd.DataFrame):
    train_df = df.sample(frac=frac, random_state=0)
    test_df = df.drop(train_df.index).sample(frac=1)

    return (train_df, test_df)


def calc_mae(df: pd.DataFrame, predicted_col: str, actual_col: str):
    return df.apply(lambda row: abs(row[predicted_col]-row[actual_col]), axis=1).mean()


def calc_mse(df: pd.DataFrame, predicted_col: str, actual_col: str):
    return df.apply(lambda row: (row[predicted_col]-row[actual_col])**2, axis=1).mean()