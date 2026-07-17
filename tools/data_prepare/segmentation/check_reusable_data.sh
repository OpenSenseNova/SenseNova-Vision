#!/usr/bin/env bash
# Copyright 2026 SenseTime Group Inc. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -u

usage() {
  echo "Usage: $0 TARGET ARCHIVE [TARGET ARCHIVE ...]" >&2
}

path_ready() {
  local target_path="$1"
  if [ -d "${target_path}" ]; then
    find "${target_path}" -maxdepth 1 -type f -print -quit | grep -q .
  else
    [ -s "${target_path}" ]
  fi
}

if [ "$#" -eq 0 ] || [ $(( $# % 2 )) -ne 0 ]; then
  usage
  exit 2
fi

status=0
while [ "$#" -gt 0 ]; do
  target_path="$1"
  archive_path="$2"

  if path_ready "${target_path}"; then
    echo "[READY] ${target_path}"
  elif [ ! -f "${archive_path}" ]; then
    echo "[MISSING] ${target_path}"
    status=1
  elif ! command -v unzip >/dev/null 2>&1; then
    echo "[UNAVAILABLE] unzip is required to verify ${archive_path}" >&2
    status=1
  elif unzip -tq "${archive_path}" >/dev/null; then
    echo "[ARCHIVE READY] ${archive_path}"
  else
    echo "[INVALID ARCHIVE] ${archive_path}" >&2
    status=1
  fi

  shift 2
done

exit "${status}"
