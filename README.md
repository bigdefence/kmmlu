# Criminal Law Agent

이 프로젝트는 한국어 형사법 시험 문제를 OpenAI 임베딩과 Faiss 인덱스를 이용해 풀어보는 예시입니다.  
Poetry를 사용하여 의존성을 관리하고, Docker 컨테이너에서 스크립트를 실행합니다.

---

## 구성 요소

1. **main.py**  
   - 형사소송법 PDF를 읽어 Faiss 인덱스 구축
   - KMMLU 데이터셋(형법 영역)을 사용하여 테스트
   - 배치 형태로 OpenAI API 호출 (예시)
   - 정확도(Accuracy) 계산 및 결과 저장

2. **pyproject.toml / poetry.lock**  
   - Poetry로 Python 의존성 관리

3. **Dockerfile**  
   - Python:3.10 기반 이미지
   - Poetry 설치 후 `main.py` 실행

4. **docker-compose.yml**  
   - `docker-compose`로 Docker 빌드 및 실행 자동화
   - `.env` 파일에 있는 환경변수를 컨테이너 내로 전달

5. **.env**  
   - `OPENAI_API_KEY` 환경 변수를 포함해야 합니다.
   - 예: `OPENAI_API_KEY=sk-xxxxx`

6. **embedding/형사소송법.pdf**  
   - 스크립트에서 참조할 PDF 파일  
   - 만약 다른 파일명이면 `main.py`에서 pdf_list 경로를 적절히 수정해야 함

---

## 사전 준비

1. **OpenAI API Key 발급**  
   - [OpenAI](https://platform.openai.com/)에서 API Key를 발급받고, `.env` 파일에 `OPENAI_API_KEY`로 설정

2. **Docker 및 docker-compose 설치**  
   - [Docker 설치 가이드](https://docs.docker.com/get-docker/) 참조
   - 보통 최신 Docker Desktop 설치 시 `docker compose`(V2)도 같이 설치됩니다.

3. **프로젝트 구조**  
   - 위에 제시된 디렉토리 구조대로 파일들을 배치
   - `.env` 파일(환경변수) 준비

---

## 설치 및 실행 방법

### 1) Docker 이미지 빌드

```bash
docker-compose up --build -d
