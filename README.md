# Luxia Translation - MD 논문 번역 시스템

대용량 마크다운 논문 파일을 자체 LLM API를 사용하여 영어→한국어로 번역하는 시스템입니다.

## 주요 기능

- ✅ **대용량 처리**: 400페이지 논문도 안정적으로 처리
- ✅ **스마트 청킹**: 10,000자 단위로 문단 경계에서 분할
- ✅ **구조 보존**: 코드 블록, 수식, 링크, 이미지 경로 자동 보존
- ✅ **순차 처리**: 모델 과부하 방지를 위한 순차적 번역
- ✅ **자동 재시도**: API 오류 시 exponential backoff로 최대 3회 재시도
- ✅ **진행률 표시**: tqdm 진행률 바로 실시간 상태 확인

## 시스템 요구사항

- Python 3.8 이상
- OpenAI 호환 LLM API 엔드포인트

## 설치

```bash
# 저장소 클론 또는 프로젝트 디렉토리로 이동
cd translation

# 의존성 설치
pip install -r requirements.txt
```

## 설정

### 방법 1: .env 파일 사용 (권장)

1. `.env.example`을 복사하여 `.env` 파일 생성:

```bash
cp .env.example .env
```

2. `.env` 파일을 열어 API 정보 입력:

```bash
# .env
LLM_API_KEY=your-api-key-here
LLM_API_ENDPOINT=https://your-api-endpoint.com/v1/chat/completions
LLM_MODEL_NAME=your-model-name

# 선택사항: 커스텀 리퀘스트 바디 템플릿 (JSON 형식)
# LLM_REQUEST_TEMPLATE={"model": "{model}", "messages": [{"role": "system", "content": "{system_prompt}"}, {"role": "user", "content": "{user_content}"}], "temperature": {temperature}}
```

**커스텀 리퀘스트 바디 설정:**

자체 개발 모델의 API 형식이 OpenAI와 다른 경우, `LLM_REQUEST_TEMPLATE`을 설정하세요:

```bash
# 예시: 커스텀 API 형식
LLM_REQUEST_TEMPLATE={"model": "{model}", "prompt": "{user_content}", "system": "{system_prompt}", "temp": {temperature}}
```

사용 가능한 변수:
- `{model}`: 모델 이름
- `{system_prompt}`: 시스템 프롬프트 (번역 지시사항)
- `{user_content}`: 번역할 텍스트
- `{temperature}`: 온도 값 (0.3)

### 방법 2: config.yaml 수정

`.env` 파일 대신 `config.yaml`에 직접 설정할 수도 있습니다:

```yaml
api:
  endpoint: "https://your-model-api.com/v1/chat/completions"
  model: "your-model-name"
  api_key: "your-api-key-here"

translation:
  chunk_size: 10000
  output_dir: "translated"
```

**우선순위**: `.env` 환경변수 > `config.yaml` 설정

## 사용법

### 기본 사용

```bash
python main.py --input paper.md
```

출력 파일은 `translated/paper.md`에 자동 생성됩니다.

### 출력 경로 지정

```bash
python main.py --input paper.md --output translated_paper.md
```

### 커스텀 설정 파일 사용

```bash
python main.py --input paper.md --config custom_config.yaml
```

## 실행 예시

```bash
$ python main.py --input research_paper.md

📄 Reading file: research_paper.md
   File size: 85.3K chars

🔍 Extracting preserved elements (code blocks, math, links)...
   Preserved 127 elements
   - code_block: 15
   - math_block: 8
   - inline_math: 42
   - inline_code: 35
   - link: 27

✂️  Chunking content (target size: 10000 chars)...
   Created 9 chunks

🌐 Translating 9 chunks...
   Model: your-model-name

Translation progress: 100%|██████████| 9/9 [01:32<00:00, 10.25s/chunk]

🔗 Merging translated chunks...

💾 Writing output: translated/research_paper.md

✅ Translation complete!
   Input:  research_paper.md (85.3K chars)
   Output: translated/research_paper.md (103.7K chars)
```

## 프로젝트 구조

```
translation/
├── prompts/
│   └── translation_system.txt  # 번역 시스템 프롬프트 (한글)
├── src/
│   ├── __init__.py
│   ├── chunker.py      # 스마트 청킹 로직
│   ├── translator.py   # API 번역 엔진
│   ├── merger.py       # 청크 병합
│   └── utils.py        # 보존 요소 처리
├── config.yaml         # 설정 파일
├── main.py            # CLI 진입점
├── requirements.txt   # 의존성
└── README.md          # 본 문서
```

## 보존 요소

번역 시 다음 요소들은 원문 그대로 유지됩니다:

- **코드 블록**: ````python ... ````
- **수식 블록**: `$$...$$`
- **인라인 수식**: `$...$`
- **인라인 코드**: `` `code` ``
- **링크**: `[text](url)`
- **이미지**: `![alt](path)`

## 처리 과정

1. **읽기**: 원본 MD 파일 로드
2. **추출**: 보존 요소를 플레이스홀더로 치환
3. **청킹**: 10,000자 단위로 문단 경계에서 분할
4. **번역**: 각 청크를 순차적으로 번역 (재시도 포함)
5. **병합**: 번역된 청크 결합
6. **복원**: 플레이스홀더를 원본 보존 요소로 복원
7. **저장**: 번역된 파일 출력

## 오류 처리

- API 호출 실패 시 자동으로 최대 3회 재시도
- Exponential backoff: 1초 → 2초 → 4초
- 재시도 실패 시 해당 청크는 원문 유지

## 성능

- **400페이지 논문** (약 100,000자):
  - 청크 수: ~10개
  - 예상 시간: 1~2분
  - 청크당 평균: 5~10초

## 문제 해결

### API 키 오류
```
❌ Error: API key not found
```
→ 환경변수 `LLM_API_KEY`를 설정하거나 `config.yaml`에 직접 입력하세요.

### 설정 파일 없음
```
❌ Error: Configuration file not found
```
→ `config.yaml` 파일이 프로젝트 루트에 있는지 확인하세요.

### 플레이스홀더 누락 경고
```
⚠️  Warning: Missing placeholder {{CODE_BLOCK_0}}
```
→ 번역 중 일부 플레이스홀더가 손실되었습니다. 해당 청크는 원문으로 유지됩니다.

## 프롬프트 관리

번역 시스템 프롬프트는 `prompts/translation_system.txt` 파일에서 한글로 관리됩니다.

**프롬프트 수정 방법:**
1. `prompts/translation_system.txt` 파일을 텍스트 에디터로 열기
2. 번역 지시사항 수정
3. 저장 후 바로 적용 (재시작 불필요)

**커스텀 프롬프트 사용:**
```bash
# config.yaml에 추가
translation:
  prompt_file: "prompts/custom_prompt.txt"

# 또는 환경변수 설정
export LLM_PROMPT_FILE=prompts/custom_prompt.txt
```

## 추후 개선 사항

- [ ] 중간 저장 및 재개 기능
- [ ] 병렬 번역 옵션 (모델 허용 시)
- [ ] 번역 품질 검증 기능
- [ ] 다국어 지원 (한→영, 중→한 등)

## 라이선스

MIT License

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.
