import pandas as pd
from typing import List


def load_titanic(path: str) -> pd.DataFrame:
    """
    加载 Titanic 数据集并返回 pandas DataFrame。
    """
    data = None
    return data


def get_k(df: pd.DataFrame, quasi_identifiers: List[str]) -> int:
    """
    计算在给定准标识符下的 k 值（最小等价类大小）。

    参数:
        df (pd.DataFrame): 输入数据集。
        quasi_identifiers (List[str]): 准标识符列名列表，如 ["Sex", "Pclass"]。

    返回:
        int: 最小的等价类大小（k 值）。
    """
    k = None
    return k


def anonymization_by_age(data: pd.DataFrame) -> pd.DataFrame:
    """
    对年龄进行分箱操作，使得每个年龄段至少包含5条记录，从而满足5-匿名性。
    
    参数:
        data (pd.DataFrame): 原始数据集。

    返回:
        pd.DataFrame: 添加了 'Age_Binned' 列的匿名化数据集。
    """

    df_sorted = None
    return df_sorted


def anonymization_by_age_and_pclass(data: pd.DataFrame) -> pd.DataFrame:
    """
    对 Age 进行分箱，并对 Pclass 进行泛化处理，使得在准标识符 ["Age_Binned", "Pclass"] 下满足 5-匿名性。
    
    参数:
        data (pd.DataFrame): 原始数据集。

    返回:
        pd.DataFrame: 包含 'Age_Binned' 和泛化后的 'Pclass' 的匿名化数据集。
    """
    df = None
    
    return df


if __name__ == "__main__":
    path = "data/train.csv"
    data = load_titanic(path)
    # 尝试不同准标识符组合
    quasi_identifier_sets = [
        ["Survived"],
        ["Sex"],
        ["Pclass"],
        ["Sex", "Pclass"],
        ["Age"],
        ["Sex", "Pclass", "Fare"],
    ]

    for attrs in quasi_identifier_sets:
        k = get_k(data, attrs)
        print(f"\n准标识符: {attrs} => k = {k}")
    data_anonymized = anonymization_by_age(data)
    k = get_k(data_anonymized, ["Age_Binned"])
    print(f"\n5-匿名性下的最小等价类大小 k = {k}")
    data_anonymized_age_pclass = anonymization_by_age_and_pclass(data_anonymized)
    k = get_k(data_anonymized_age_pclass, ["Age_Binned", "Pclass"])
    print(f"\n在 Age_Binned 和泛化后的 Pclass 下的 k 值: {k}")