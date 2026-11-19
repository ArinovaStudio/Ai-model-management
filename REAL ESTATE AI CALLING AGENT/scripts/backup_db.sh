#!/bin/sh
# ==============================================================================
# Automated PostgreSQL + pgvector Database Backup Script
# Retains vector embeddings, schema, and transactional history with 7-day retention.
# ==============================================================================

set -eu

# Portable script directory resolution (POSIX sh compatible)
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)

# Configuration
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_ROOT}/storage/backups}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILE="${BACKUP_DIR}/realestate_ai_${TIMESTAMP}.dump"
BACKUP_SQL="${BACKUP_DIR}/realestate_ai_${TIMESTAMP}.sql.gz"

POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-realestate_ai}"
export PGPASSWORD="${POSTGRES_PASSWORD:-postgres}"

mkdir -p "${BACKUP_DIR}"

echo "=================================================================="
echo " Starting Database Backup for ${POSTGRES_DB} at ${TIMESTAMP}"
echo "=================================================================="

# 1. Custom binary format dump with compression (fastest for pg_restore)
echo "[1/3] Creating compressed custom pg_dump archive..."
pg_dump -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    -F c -b -v -f "${BACKUP_FILE}"

# 2. Text SQL dump with gzip (human-readable disaster recovery fallback)
echo "[2/3] Creating plain text gzipped SQL snapshot..."
pg_dump -h "${POSTGRES_HOST}" -p "${POSTGRES_PORT}" -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" \
    --clean --if-exists | gzip -9 > "${BACKUP_SQL}"

# Check sizes
FILE_SIZE=$(du -h "${BACKUP_FILE}" | cut -f1)
SQL_SIZE=$(du -h "${BACKUP_SQL}" | cut -f1)
echo "  * Custom dump created : ${BACKUP_FILE} (${FILE_SIZE})"
echo "  * Gzip SQL snapshot   : ${BACKUP_SQL} (${SQL_SIZE})"

# 3. Apply retention policy
echo "[3/3] Pruning backups older than ${RETENTION_DAYS} days..."
find "${BACKUP_DIR}" -type f \( -name "*.dump" -o -name "*.sql.gz" \) -mtime +"${RETENTION_DAYS}" -exec rm -f {} +

echo "=================================================================="
echo " Backup Completed Successfully at $(date +"%Y-%m-%d %H:%M:%S")"
echo "=================================================================="
