#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# upload_to_github.sh
# GitHub 레포지토리에 PET 코드를 업로드하는 스크립트
#
# 사전 준비:
#   1. GitHub 계정 생성 및 로그인
#   2. GitHub CLI(gh) 설치:  sudo apt install gh  또는  brew install gh
#      (없을 경우 수동 remote 설정 방법도 안내됩니다)
#   3. 인증:  gh auth login
# ─────────────────────────────────────────────────────────────────────────────

set -e  # 오류 발생 시 즉시 중단

# ── 색상 출력 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 인자 파싱 ─────────────────────────────────────────────────────────────────
REPO_NAME="${1:-PET-SplitLearning}"
REPO_DESC="Personal Embedding Transformation for Privacy-Preserving Split Learning (IEEE Access 2026)"
VISIBILITY="${2:-public}"   # public 또는 private

# ── 위치 확인 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
info "작업 디렉터리: $SCRIPT_DIR"

# ── git 초기화 확인 ───────────────────────────────────────────────────────────
if [ ! -d ".git" ]; then
    info "git init 실행 중 …"
    git init
fi

# ── .gitignore 확인 ───────────────────────────────────────────────────────────
if [ ! -f ".gitignore" ]; then
    warn ".gitignore 파일이 없습니다. 업로드 전 확인 권장."
fi

# ── 스테이징 & 커밋 ───────────────────────────────────────────────────────────
info "파일 스테이징 중 …"
git add .

# 변경 사항이 없으면 커밋 건너뜀
if git diff --cached --quiet; then
    warn "커밋할 변경 사항이 없습니다. 이미 최신 상태입니다."
else
    info "커밋 생성 중 …"
    git commit -m "Initial commit: PET + CRS split learning implementation

- T5ClientModel with PET (128 amplify + 128 suppress + 256 free dims)
- Supervised contrastive loss (CRS, InfoNCE-style)
- FORA-style DRA: mimic client + MK-MMD alignment + inversion model
- Baselines: vanilla SL, DP, PET-only ablation
- Dataset: AGNews (auto-downloaded via HuggingFace datasets)
- Metrics: accuracy, cosine similarity, BLEU, ROUGE-1/2/L"
fi

# ── GitHub CLI 확인 ───────────────────────────────────────────────────────────
if ! command -v gh &> /dev/null; then
    warn "GitHub CLI(gh)가 설치되어 있지 않습니다."
    echo ""
    echo "  ① gh 설치 방법:"
    echo "     sudo apt install gh          # Ubuntu/Debian"
    echo "     brew install gh              # macOS"
    echo ""
    echo "  ② 설치 후 인증:"
    echo "     gh auth login"
    echo ""
    echo "  ③ 또는 수동으로 remote를 설정할 수 있습니다:"
    echo "     git remote add origin https://github.com/<USERNAME>/${REPO_NAME}.git"
    echo "     git branch -M main"
    echo "     git push -u origin main"
    echo ""
    error "gh CLI 설치 후 다시 실행해 주세요."
fi

# ── gh 인증 확인 ─────────────────────────────────────────────────────────────
if ! gh auth status &> /dev/null; then
    warn "GitHub 인증이 필요합니다."
    echo ""
    echo "  다음 명령어를 실행한 후 이 스크립트를 다시 실행하세요:"
    echo "     gh auth login"
    echo ""
    error "인증 필요."
fi

GITHUB_USER=$(gh api user --jq '.login' 2>/dev/null)
info "GitHub 사용자: ${GITHUB_USER}"

# ── remote 확인 ───────────────────────────────────────────────────────────────
if git remote get-url origin &> /dev/null; then
    REMOTE_URL=$(git remote get-url origin)
    info "기존 remote origin 발견: ${REMOTE_URL}"
    echo ""
    read -r -p "  기존 remote에 push하시겠습니까? [Y/n] " CONFIRM
    CONFIRM="${CONFIRM:-Y}"
    if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
        warn "취소됨. remote를 직접 수정 후 재실행하세요."
        exit 0
    fi
else
    # ── GitHub 레포지토리 생성 ────────────────────────────────────────────────
    echo ""
    echo "  레포지토리 이름 : ${REPO_NAME}"
    echo "  공개 여부       : ${VISIBILITY}"
    echo ""
    read -r -p "  위 설정으로 GitHub 레포지토리를 생성하시겠습니까? [Y/n] " CONFIRM
    CONFIRM="${CONFIRM:-Y}"
    if [[ "$CONFIRM" =~ ^[Nn]$ ]]; then
        info "취소됨."
        exit 0
    fi

    info "GitHub 레포지토리 생성 중: ${GITHUB_USER}/${REPO_NAME}"
    gh repo create "${REPO_NAME}" \
        --description "${REPO_DESC}" \
        --"${VISIBILITY}" \
        --source=. \
        --remote=origin
fi

# ── push ─────────────────────────────────────────────────────────────────────
info "GitHub에 push 중 …"
git branch -M main
git push -u origin main

# ── 완료 ─────────────────────────────────────────────────────────────────────
REPO_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  업로드 완료!${NC}"
echo -e "${GREEN}  URL: ${REPO_URL}${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
echo ""
