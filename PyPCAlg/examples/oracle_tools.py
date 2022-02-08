import pandas
import pandas as pd


def _read_true_independence_relationships(filename: str) -> pandas.DataFrame:

    df = pd.read_csv(
        filename,
        header=0,
        sep=';'
    )

    expected_columns = [
        'X',
        'Y',
        'Conditioning Set',
        '(Conditional) Independence Holds'
    ]

    assert list(df.columns) == expected_columns, f'{filename} invalid format !'

    return df


def generate_oracle_independence_relationships(filename: str) -> dict:

    df = _read_true_independence_relationships(filename)

    def format_conditioning_set(cond_set):

        without_brackets = cond_set[1:-1]

        if len(without_brackets) == 0:
            return tuple()
        else:
            return tuple(elt.strip() for elt in without_brackets.split(','))

    oracle = dict()

    for i in range(df.shape[0]):
        key = (
            df.loc[i, 'X'],
            df.loc[i, 'Y'],
            format_conditioning_set(df.loc[i, 'Conditioning Set'])
        )
        oracle[key] = df.loc[i, '(Conditional) Independence Holds']

    return oracle


def check_independence_from_oracle(oracle: dict, data: pd.DataFrame, x: int,
                                   y: int, z: list[int]) -> bool:

    column_names = list(data.columns)

    # Because when we manually list independence relationship we would write
    # only X_2 _||_ X_3, not X_3 _||_ X_2, for example.
    if x < y:
        key_part_1 = column_names[x]
        key_part_2 = column_names[y]
    else:
        key_part_1 = column_names[y]
        key_part_2 = column_names[x]
    key_part_3 = tuple(column_names[i] for i in sorted(z))
    key = (key_part_1, key_part_2, key_part_3)

    return oracle[key]


def oracle_independence_test(oracle: dict, data: pd.DataFrame, x: int, y: int,
                             level: float) -> bool:

    return check_independence_from_oracle(
        oracle=oracle,
        data=data,
        x=x,
        y=y,
        z=[]
    )


def oracle_conditional_independence_test(oracle: dict, data: pd.DataFrame,
                                         x: int, y: int, z: list[int],
                                         level: float) -> bool:

    return check_independence_from_oracle(
        oracle=oracle,
        data=data,
        x=x,
        y=y,
        z=z
    )
