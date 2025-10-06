#!/usr/bin/env bash
set -euo pipefail

# download_alphafold_weights.sh
# Download and extract AlphaFold2 parameters archive used by BindCraftPeptide
# Usage: ./download_alphafold_weights.sh [-d DEST_DIR] [-f]
#   -d DEST_DIR  Destination directory (default: ./params)
#   -f           Force re-download and overwrite existing files

DEST_DIR="./params"
FORCE=0

while getopts ":d:f" opt; do
  case ${opt} in
    d ) DEST_DIR="$OPTARG" ;;
    f ) FORCE=1 ;;
    \? ) echo "Usage: $0 [-d DEST_DIR] [-f]"; exit 1 ;;
  esac
done

ARCHIVE_NAME="alphafold_params_2022-12-06.tar"
ARCHIVE_PATH="${DEST_DIR}/${ARCHIVE_NAME}"
DOWNLOAD_URL="https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"

echo "Destination directory: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

if [ -f "${ARCHIVE_PATH}" ] && [ ${FORCE} -ne 1 ]; then
  echo "Archive already exists at ${ARCHIVE_PATH}. Use -f to force re-download."
else
  echo "Downloading AlphaFold2 parameters to ${ARCHIVE_PATH}..."
  if command -v curl >/dev/null 2>&1; then
    curl -fSL "${DOWNLOAD_URL}" -o "${ARCHIVE_PATH}"
  else
    wget -O "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
  fi
fi

if [ ! -s "${ARCHIVE_PATH}" ]; then
  echo "Error: download failed or archive is empty: ${ARCHIVE_PATH}" >&2
  exit 1
fi

echo "Verifying archive integrity (listing contents)..."
if ! tar tf "${ARCHIVE_PATH}" >/dev/null 2>&1; then
  echo "Error: archive is corrupted or cannot be read: ${ARCHIVE_PATH}" >&2
  exit 1
fi

echo "Extracting archive to ${DEST_DIR} (this may take a while)..."
tar -xvf "${ARCHIVE_PATH}" -C "${DEST_DIR}"

# Verify expected file exists after extraction
EXPECTED_FILE="${DEST_DIR}/params_model_5_ptm.npz"
if [ ! -f "${EXPECTED_FILE}" ]; then
  echo "Warning: expected AlphaFold parameter file not found: ${EXPECTED_FILE}"
  echo "List of files in ${DEST_DIR}:"
  ls -lah "${DEST_DIR}" || true
  exit 1
fi

echo "AlphaFold2 parameters successfully downloaded and extracted to ${DEST_DIR}"
echo "You can now run BindCraft and the ColabDesign models that require these params."

exit 0
