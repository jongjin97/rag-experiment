import os
import pandas as pd
from pathlib import Path

from src.config import DATA_DIR

def merge_csv_datasets():
    """
    data 디렉토리에 있는 eval_dataset_v3_1.csv ~ eval_dataset_v3_4.csv 
    파일들을 하나로 병합하여 eval_dataset_v3_merged.csv 로 저장합니다.
    """
    print("병합 대상 파일을 찾는 중입니다...")
    
    # 합칠 파일들의 경로 패턴 설정 (사용자 요청에 따라 1~4번 인덱스)
    # 실제 파일명은 v3_x.csv로 존재하므로 이들을 타겟팅합니다.
    file_list = [DATA_DIR / f"eval_dataset_v3_{i}.csv" for i in range(1, 5)]
    
    # 존재하는 파일만 선택
    valid_files = [f for f in file_list if f.exists()]
    
    if not valid_files:
        print("경고: 병합할 파일이 발견되지 않았습니다. 파일 이름을 다시 확인해주세요.")
        return

    print(f"총 {len(valid_files)}개의 파일을 발견했습니다. 병합을 시작합니다.")
    for file in valid_files:
        print(f" - {file.name}")

    # 데이터프레임 병합
    dataframes = []
    for file in valid_files:
        try:
            df = pd.read_csv(file, encoding='utf-8-sig') # 저장 시 사용했던 utf-8-sig 인코딩으로 불러오기
            dataframes.append(df)
            print(f"[{file.name}] 로드 완료 (데이터 수: {len(df)}개)")
        except Exception as e:
            print(f"[{file.name}] 로드 중 오류 발생: {e}")

    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        
        # 중복 데이터가 혹시라도 있을 경우 대비 (테스트셋 생성 과정상 거의 없겠지만)
        initial_count = len(merged_df)
        # 보통 질문(user_input/question)이 동일하면 중복으로 간주할 수 있음
        subset_col = 'question' if 'question' in merged_df.columns else 'user_input'
        
        if subset_col in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=[subset_col])
            dup_count = initial_count - len(merged_df)
            if dup_count > 0:
                print(f"중복된 데이터 {dup_count}개를 제거했습니다.")

        merged_file_path = DATA_DIR / "eval_dataset_v3_merged.csv"
        
        # 저장
        merged_df.to_csv(merged_file_path, index=False, encoding='utf-8-sig')
        print(f"\n최종 완료! 총 {len(merged_df)}개의 데이터가 {merged_file_path.name} 로 병합 및 저장되었습니다.")
        
        # 병합 데이터 샘플 출력
        print("\n--- [병합된 데이터 첫 3행 샘플] ---")
        print(merged_df.head(3))
    else:
        print("병합할 데이터가 존재하지 않습니다.")

if __name__ == "__main__":
    merge_csv_datasets()
