# PaliGemma CAST Training and Inference

이 디렉토리는 CAST 데이터셋을 사용하여 PaliGemma-3B 모델을 학습하고 평가하는 코드를 포함합니다.

## 설치

필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

## 파일 구조

- `action_tokenizer.py`: xy 좌표 델타를 토큰으로 변환하는 액션 토크나이저
- `cast_dataset.py`: CAST 데이터셋 로더 및 전처리
- `train_paligemma.py`: PaliGemma 모델 학습 스크립트
- `inference_paligemma.py`: 학습된 모델로 추론하고 결과를 CSV로 저장
- `evaluate.py`: 상세한 평가 메트릭 및 시각화
- `config.yaml`: 학습 및 추론 설정

## 사용 방법

### 1. 모델 학습

기본 설정으로 학습:

```bash
python train_paligemma.py --config config.yaml
```

Weights & Biases 로깅과 함께 학습:

```bash
python train_paligemma.py --config config.yaml --wandb
```

### 2. 모델 추론

학습된 모델로 추론하고 결과를 CSV로 저장:

```bash
python inference_paligemma.py --config config.yaml --checkpoint ./checkpoints/best_model
```

### 3. 결과 평가

추론 결과를 상세히 분석:

```bash
python evaluate.py --results ./results/predictions.csv
```

## 설정 파일 (config.yaml)

주요 설정 항목:

- `model.name`: 사용할 PaliGemma 모델 ("google/paligemma-3b-mix-224")
- `training.batch_size`: 학습 배치 크기
- `training.learning_rate`: 학습률
- `training.num_epochs`: 학습 에포크 수
- `action.vocab_size`: 액션 토크나이저 어휘 크기
- `action.num_steps`: 예측할 액션 스텝 수 (8 steps * 2D = 16 tokens)

## 데이터셋

CAST 데이터셋은 HuggingFace에서 자동으로 다운로드됩니다:
- 데이터셋: `catglossop/CAST-dataset`
- 형태: (이미지, 언어 지시, 액션) 튜플
- 액션: 8 스텝의 xy 좌표 델타값

## 출력 결과

### 학습 후 생성되는 파일:
- `./checkpoints/best_model/`: 최고 성능 모델 체크포인트
- `./logs/`: 학습 로그
- `./checkpoints/*/tokenizers.pt`: 액션 토크나이저 및 정규화기

### 추론 후 생성되는 파일:
- `./results/predictions.csv`: 예측 결과 CSV 파일
- `./results/images/`: 추론에 사용된 이미지들
- CSV 컬럼:
  - `sample_id`: 샘플 ID
  - `image_filename`: 저장된 이미지 파일명
  - `image_path`: 이미지 파일 경로
  - `instruction`: 언어 지시
  - `pred_x_0` ~ `pred_x_7`: 예측된 x 좌표 (8 스텝)
  - `pred_y_0` ~ `pred_y_7`: 예측된 y 좌표 (8 스텝)
  - `target_x_0` ~ `target_x_7`: 정답 x 좌표 (8 스텝)
  - `target_y_0` ~ `target_y_7`: 정답 y 좌표 (8 스텝)
  - `overall_mse`, `overall_mae` 등: 전체 평가 메트릭

### 평가 후 생성되는 파일:
- `./results/evaluation/evaluation_report.txt`: 상세 평가 리포트
- `./results/evaluation/error_distributions.png`: 오차 분포 시각화
- `./results/evaluation/trajectory_examples.png`: 궤적 예시 비교
- `./results/evaluation/instruction_performance.png`: 명령어별 성능 분석

## 주요 특징

1. **액션 토크나이저**: 연속적인 xy 좌표를 이산적인 토큰으로 변환
2. **반사실적 데이터**: CAST 데이터셋의 반사실적 레이블 활용
3. **종합적인 평가**: MSE, MAE, 성공률 등 다양한 메트릭
4. **이미지 저장**: 각 추론 샘플의 이미지를 개별 저장
5. **상세한 시각화**: 궤적 비교, 오차 분포 등 다양한 분석

## 하드웨어 요구사항

- GPU: NVIDIA GPU with 16GB+ VRAM (권장)
- RAM: 32GB+ (권장)
- Storage: 50GB+ 여유 공간

## 문제 해결

1. **메모리 부족**: `batch_size`를 줄이거나 `gradient_accumulation_steps`를 늘리세요
2. **데이터셋 다운로드 실패**: `cache_dir`을 확인하고 인터넷 연결을 확인하세요
3. **토크나이저 오류**: `vocab_size`와 `action_bounds` 설정을 확인하세요