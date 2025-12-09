# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

마크다운 논문 파일을 자체 LLM API를 사용해 영어→한국어로 번역하는 시스템입니다. 400페이지급 대용량 문서를 안정적으로 처리하며, 코드 블록, 수식, 링크 등의 마크다운 구조를 보존합니다.

**프롬프트 관리:** 모든 번역 프롬프트는 `prompts/` 폴더에서 txt 파일로 관리되어 언제든 수정 가능합니다.

## 핵심 명령어

### 개발 환경 설정
```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일 생성)
cp .env.example .env
# .env 파일에서 LLM_API_KEY, LLM_API_ENDPOINT, LLM_MODEL_NAME 설정
```

### 번역 실행
```bash
# 기본 사용 (출력: translated/<input_filename>)
python main.py --input paper.md

# 출력 경로 지정
python main.py --input paper.md --output result.md

# 커스텀 설정 파일 사용
python main.py --input paper.md --config custom_config.yaml
```

### 테스트
```bash
# 샘플 파일로 테스트
python main.py --input test_sample.md
```

## 아키텍처

### 처리 파이프라인 (main.py)
1. **추출** (utils.py): 보존할 요소를 플레이스홀더로 치환
2. **청킹** (chunker.py): 10,000자 단위로 문단 경계에서 분할
3. **번역** (translator.py): 각 청크를 순차적으로 API 번역 (재시도 포함)
4. **병합** (merger.py): 번역된 청크 결합
5. **복원** (utils.py): 플레이스홀더를 원본 요소로 복원

### 디렉토리 구조

```
translation/
├── prompts/                    # 프롬프트 관리 폴더
│   └── translation_system.txt  # 번역 시스템 프롬프트 (한글)
├── src/                        # 핵심 모듈
│   ├── chunker.py
│   ├── translator.py
│   ├── merger.py
│   └── utils.py
├── main.py                     # CLI 진입점
├── config.yaml                 # 설정 파일
└── .env                        # 환경 변수
```

### 핵심 모듈 설명

**prompts/translation_system.txt** - 번역 시스템 프롬프트
- 한글로 작성된 번역 지시사항
- 언제든 txt 파일을 수정하여 프롬프트 변경 가능
- 환경변수 `LLM_PROMPT_FILE`로 커스텀 프롬프트 파일 지정 가능

**src/utils.py** - 보존 요소 처리
- `extract_preserved_elements()`: 코드/수식/링크를 정규식으로 추출하고 플레이스홀더(`{{CODE_BLOCK_0}}`)로 치환
- `restore_preserved_elements()`: 번역 후 플레이스홀더를 원본으로 복원
- 처리 순서 중요: 추출은 큰 요소부터(code_block→inline_code), 복원은 작은 요소부터

**src/chunker.py** - 스마트 청킹
- `chunk_markdown()`: 10,000자 목표로 문단 경계에서 분할
- `find_split_point()`: 우선순위 - 이중 개행(문단) > 단일 개행(줄) > 강제 분할
- 허용 오차 1,000자 내에서 최적 분할점 탐색

**src/translator.py** - API 번역 엔진
- `Translator`: aiohttp 기반 비동기 번역기
- `_load_prompt()`: txt 파일에서 시스템 프롬프트 로드 (기본: `prompts/translation_system.txt`)
- `_build_request_body()`: 커스텀 리퀘스트 템플릿 지원 (`LLM_REQUEST_TEMPLATE` 환경변수)
- `translate_chunk()`: tenacity로 exponential backoff 재시도 (최대 3회)
- `translate_chunks()`: 순차 처리 (0.5초 지연으로 API 과부하 방지)

**src/merger.py** - 청크 병합
- `merge_chunks()`: 청크를 `\n\n`로 결합 후 플레이스홀더 복원
- `verify_placeholders()`: 번역 중 플레이스홀더 손실 검증

### 설정 우선순위
환경변수 (.env) > config.yaml

**필수 환경변수:**
- `LLM_API_KEY`: API 인증 키
- `LLM_API_ENDPOINT`: API 엔드포인트 URL
- `LLM_MODEL_NAME`: 사용할 모델명

**선택 환경변수:**
- `LLM_REQUEST_TEMPLATE`: 커스텀 API 리퀘스트 바디 (JSON 형식, 변수: `{model}`, `{system_prompt}`, `{user_content}`, `{temperature}`)
- `LLM_PROMPT_FILE`: 커스텀 프롬프트 파일 경로 (기본값: `prompts/translation_system.txt`)

## 중요 구현 세부사항

### 플레이스홀더 시스템
- 형식: `{{ELEMENT_TYPE_INDEX}}` (예: `{{CODE_BLOCK_0}}`, `{{INLINE_MATH_5}}`)
- 번역 모델에게 "플레이스홀더를 그대로 유지하라"는 system prompt 전달
- 번역 후 누락 검증 (`verify_placeholders`)으로 손실 감지

### 재시도 로직
- tenacity 라이브러리로 exponential backoff 구현
- 대기 시간: 1초 → 2초 → 4초 (최대 10초)
- 3회 실패 시 원본 청크 유지

### 커스텀 API 포맷 지원
OpenAI 호환 API가 아닌 경우 `LLM_REQUEST_TEMPLATE`로 리퀘스트 바디 커스터마이징:
```bash
LLM_REQUEST_TEMPLATE={"model": "{model}", "prompt": "{user_content}", "system": "{system_prompt}"}
```

### 비동기 처리
- aiohttp로 API 호출 비동기화
- 하지만 `translate_chunks()`는 순차 처리 (병렬은 모델 과부하 위험)
- 청크 간 0.5초 지연으로 rate limit 회피

## 일반적인 문제 해결

### API 키 오류
`.env` 파일 확인 또는 환경변수 `LLM_API_KEY` 설정. 자체 모델의 경우 API 키 없이도 작동 가능.

### 번역이 원문과 번역문을 함께 출력하는 경우
`prompts/translation_system.txt` 파일을 수정하여 "오직 한국어 번역문만 반환" 지시를 더 명확하게 작성

### 플레이스홀더 누락 경고
번역 모델이 플레이스홀더를 제거/변경한 경우 발생. `prompts/translation_system.txt`에서 플레이스홀더 보존 지시 강화

### 응답 파싱 실패
API 응답 형식이 OpenAI와 다른 경우:
1. `LLM_REQUEST_TEMPLATE`로 리퀘스트 커스터마이징
2. translator.py의 `translate_chunk()` 응답 파싱 로직 수정 (choices[0].message.content → content/text 등)

### 커스텀 프롬프트 사용
`prompts/` 폴더에 새로운 txt 파일 생성 후:
- config.yaml: `translation.prompt_file: "prompts/custom_prompt.txt"`
- 또는 환경변수: `LLM_PROMPT_FILE=prompts/custom_prompt.txt`
