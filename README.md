# 📘 Criminal Law AI Evaluation

## 🔍 프로젝트 개요
Criminal Law AI Evaluation은 OpenAI의 GPT 모델을 활용하여 형법 관련 문제를 해결하고 평가하는 프로젝트입니다. 
주어진 형법 관련 질문에 대해 AI가 정답을 예측하며, PDF 문서를 기반으로 검색 및 응답 생성을 수행합니다.

## 📂 프로젝트 구조
```
📦 프로젝트 폴더
├── main.py               # 메인 실행 파일
├── input.jsonl           # AI에 입력될 질문 데이터
├── output.jsonl          # AI의 응답 결과
├── benchmark_result.txt  # 벤치마크 결과
├── embedding/            # PDF 임베딩 저장 폴더
│   ├── 형사소송법.pdf      # 법률 문서 파일
│   ├── batch_형소법_cache.pkl  # FAISS 임베딩 캐시 파일
├── pyproject.toml        # Poetry 의존성 관리 파일
├── poetry.lock           # Poetry 종속성 버전 잠금 파일
├── Dockerfile            # Docker 이미지 구성 파일
├── docker-compose.yml    # Docker Compose 설정 파일
└── README.md             # 프로젝트 설명 파일
```

## ⚙️ 주요 기능
- **법률 문서 분석**: PDF 문서에서 텍스트를 추출하고 정제하여 AI 모델의 학습 데이터로 사용
- **문서 검색 및 임베딩**: FAISS를 이용한 빠른 문서 검색 기능 제공
- **AI를 통한 법률 문제 풀이**: OpenAI API를 활용한 형법 문제 풀이 및 평가 수행
- **배치 평가**: 다수의 질문을 한 번에 처리하고 정확도를 평가하는 기능 포함
- **Docker 컨테이너 지원**: Docker 및 Docker Compose를 이용하여 손쉽게 실행 가능

## 🛠️ 설치 및 실행 방법
### 1️⃣ 환경 설정
Python 및 Docker가 설치되어 있어야 하며, 필요한 패키지를 Poetry를 이용해 설치합니다.
```sh
pip install poetry
docker-compose up --build -d
```

### 2️⃣ 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 추가하세요.
```
OPENAI_API_KEY=your_openai_api_key
```

### 3️⃣ 프로젝트 실행 (Docker 활용)
```sh
docker-compose up --build -d
```

## 📌 주요 코드 설명
### 📜 `main.py`
- `evaluate()` : AI의 응답을 평가하여 정확도를 계산하는 함수
- `clean_text()` : 텍스트 전처리를 수행하는 함수
- `get_embedding()` : OpenAI API를 활용하여 텍스트 임베딩을 생성하는 함수
- `extract_text_from_pdf()` : PDF 문서에서 텍스트를 추출하는 함수
- `split_text_into_chunks()` : 문서를 일정 크기로 나누는 함수
- `CriminalLawAgent` 클래스
  - `build_retrieval_corpus()` : PDF 문서의 검색 인덱스를 구축하는 메서드
  - `retrieve_context()` : 질의어와 관련된 문서를 검색하는 메서드
  - `generate_prompt()` : AI가 응답할 프롬프트를 생성하는 메서드
  - `run_evaluation()` : AI의 문제 풀이 및 평가 수행

## 📊 벤치마크 결과 예시
성능 평가 결과는 `benchmark_result.txt` 파일에 저장됩니다. 

```json
{
 "batch_id": "batch_xxxxx",
 "duration_seconds": 2696.1948013305664,
 "accuracy_percentage": "38.5%"
}
```

## 🐳 Docker 사용 방법
### 1️⃣ Docker 이미지 빌드 및 실행
```sh
docker-compose up --build -d
```

### 2️⃣ 컨테이너 상태 확인
```sh
docker ps
```

### 3️⃣ 컨테이너 종료
```sh
docker-compose down
```

## 📌 사용 기술
- **Python**: 프로젝트 전체 구현
- **OpenAI API**: GPT 모델을 이용한 응답 생성
- **FAISS**: 문서 검색 및 유사도 분석
- **PyPDF2**: PDF 문서 처리
- **Poetry**: Python 의존성 관리
- **Docker & Docker Compose**: 컨테이너화된 실행 환경 제공

## 🏗️ 향후 개선 방향
- 다양한 법률 문서 추가 지원
- 문서 검색 최적화 및 속도 개선
- AI 모델의 성능 향상 및 피드백 반영
- 컨테이너 최적화 및 경량화


