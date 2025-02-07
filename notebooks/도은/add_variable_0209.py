import pandas as pd
import numpy as np

def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    특정 조건을 만족하는 데이터를 필터링하는 함수.
    - 배란 자극을 하지 않은 사람 제거
    - DI 시술을 한 사람 제거
    - PGD/PGS 검사를 받은 사람 제거
    
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: 필터링된 데이터프레임
    """
    df = df.drop('ID', axis=1, errors='ignore')  # ID 컬럼이 있으면 삭제
    df = df[df['배란 자극 여부'] == 1]  # 배란 자극을 한 사람만 포함
    df = df[df['시술 유형'] != 'DI']  # DI 시술을 한 사람 제외
    df = df[(df['PGD 시술 여부'] != 1) | (df['PGS 시술 여부'] != 1)]  # PGD, PGS 검사를 받은 사람 제외
    return df


def add_oocyte_ratio(df: pd.DataFrame, remove_outliers=True) -> pd.DataFrame:
    """
    혼합할만한 난자 비율을 계산하고 구간화를 수행하는 함수.
    
    Parameters:
        df (pd.DataFrame): 데이터프레임
        remove_outliers (bool): 비율이 1보다 큰 행을 제거할지 여부
    
    Returns:
        pd.DataFrame: 혼합할만한 난자 비율이 추가된 데이터프레임
    """
    df = df[(df["수집된 신선 난자 수"] + df['해동 난자 수']) != 0].copy()  # 0인 경우 제외

    # 혼합할만한 난자 비율 계산
    df["혼합할만한 난자 비율"] = (df['저장된 신선 난자 수'] + df['혼합된 난자 수']) / (df["수집된 신선 난자 수"] + df['해동 난자 수'])

    # 이상치 제거 옵션
    if remove_outliers:
        df = df[df['혼합할만한 난자 비율'] <= 1]

    # 비율 구간화
    df["혼합할만한 난자 비율 구간"] = pd.cut(
        df["혼합할만한 난자 비율"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["낮음", "중간", "높음"],
        include_lowest=True
    )
    
    return df


def add_embryo_creation_ratio(df: pd.DataFrame, remove_outliers=True) -> pd.DataFrame:
    """
    배아 생성 비율을 추가하고 구간화를 수행하는 함수.

    Parameters:
        df (pd.DataFrame): 데이터프레임
        remove_outliers (bool): 비율이 1보다 큰 행을 제거할지 여부
    
    Returns:
        pd.DataFrame: 배아 생성 비율이 추가된 데이터프레임
    """
    df['배아 생성 비율'] = df['총 생성 배아 수'] / df['혼합된 난자 수']
    
    # 이상치 제거 옵션
    if remove_outliers:
        df = df[df['배아 생성 비율'] <= 1]

    # 배아 생성 비율 구간화
    df['배아 생성 비율 구간'] = pd.qcut(df["배아 생성 비율"], q=3, labels=["낮음", "중간", "높음"])
    
    return df


def add_implantable_embryo_ratio(df: pd.DataFrame, remove_outliers=True) -> pd.DataFrame:
    """
    착상할만한 배아 생성 비율을 추가하고 구간화를 수행하는 함수.

    Parameters:
        df (pd.DataFrame): 데이터프레임
        remove_outliers (bool): 비율이 1보다 큰 행을 제거할지 여부
    
    Returns:
        pd.DataFrame: 착상할만한 배아 생성 비율이 추가된 데이터프레임
    """
    df['착상할만한 배아 생성 비율'] = (df['이식된 배아 수'] + df['저장된 배아 수']) / df['총 생성 배아 수']
    df['착상할만한 배아 생성 비율'] = df['착상할만한 배아 생성 비율'].replace([np.inf, -np.inf], 0)

    # 이상치 제거 옵션
    if remove_outliers:
        df = df[df['착상할만한 배아 생성 비율'] <= 1]

    # 착상할만한 배아 생성 비율 구간화
    df['착상할만한 배아 생성 비율 구간'] = pd.qcut(df["착상할만한 배아 생성 비율"], q=3, labels=["낮음", "중간", "높음"])
    
    return df


def add_ivf_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    IVF 성공 확률을 계산하는 함수.
    
    Parameters:
        df (pd.DataFrame): 원본 데이터프레임
    
    Returns:
        pd.DataFrame: IVF 성공 확률 컬럼이 추가된 데이터프레임
    """
    # IVF 관련 컬럼을 숫자로 변환 (첫 번째 문자만 추출 후 변환)
    numeric_IVF = df[['IVF 시술 횟수', 'IVF 임신 횟수', 'IVF 출산 횟수']].apply(
        lambda x: pd.to_numeric(x.str[0], errors='coerce')
    )

    # IVF 성공 확률 계산 (0으로 나누는 경우 NaN 처리)
    df['IVF 성공 확률'] = numeric_IVF['IVF 임신 횟수'] / numeric_IVF['IVF 시술 횟수'].replace(0, np.nan)

    return df


'''
# 데이터 불러오기
df = pd.read_csv('../../data/raw/train.csv')

# 1️⃣ 데이터 필터링 적용
df = filter_data(df)

# 2️⃣ 혼합할만한 난자 비율 추가 (이상치 제거 옵션 포함)
df = add_oocyte_ratio(df, remove_outliers=True)

# 3️⃣ 배아 생성 비율 추가 (이상치 제거 옵션 포함)
df = add_embryo_creation_ratio(df, remove_outliers=True)

# 4️⃣ 착상할만한 배아 생성 비율 추가 (이상치 제거 옵션 포함)
df = add_implantable_embryo_ratio(df, remove_outliers=True)

# 결과 확인
print(df.head())

'''