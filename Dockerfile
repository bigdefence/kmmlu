FROM python:3.10

# 필수 빌드 도구 및 curl 설치
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

# Poetry 설치 (버전은 필요에 따라 조정)
ENV POETRY_VERSION=1.4.2
RUN curl -sSL https://install.python-poetry.org | python3 - --version $POETRY_VERSION
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
# Poetry 설정: 가상환경 없이 시스템 환경에 설치
RUN poetry config virtualenvs.create false

# pyproject.toml 및 poetry.lock 복사 후 의존성 설치
COPY pyproject.toml poetry.lock* ./
RUN poetry install --no-interaction --no-ansi

# 소스 전체 복사
COPY . .

CMD ["python", "main.py"]

