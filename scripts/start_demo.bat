@echo off
setlocal
cd /d "%~dp0\.."

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%PY%" set "PY=python"

if not exist data\index.json (
  powershell -ExecutionPolicy Bypass -File scripts\install_local.ps1
  set "PY=%CD%\.venv\Scripts\python.exe"
  if not exist "%PY%" set "PY=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
  if not exist "%PY%" set "PY=python"
)
"%PY%" -m rag_search.web
