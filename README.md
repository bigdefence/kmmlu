# 프로젝트 설정 및 실행 방법

## 1. 필수 환경 변수 설정
프로젝트 실행 전 `.env` 파일을 생성하여 `OPENAI_API_KEY` 값을 설정하세요.

```sh
OPENAI_API_KEY=your_openai_api_key_here
```

## 2. Docker 컨테이너 실행

아래 명령어를 실행하여 Docker 컨테이너를 빌드 및 실행합니다.

```sh
docker-compose up --build -d
```

## 3. 컨테이너 로그 확인

```sh
docker logs -f criminal-law-agent
```

## 4. 컨테이너 종료

```sh
docker-compose down
```

## 5. 개발 환경에서 실행 (Docker 없이)

```sh
poetry install
python main.py
```