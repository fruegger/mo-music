@echo off
setlocal
set "LIBRARY_DIR=%~1"
if "%LIBRARY_DIR%"=="" set "LIBRARY_DIR=C:\Users\frueg\Desktop\music_studio\1_preprocessing\FJ\repertoire"
set "OP=%~2"
if "%OP%"=="" (
    python "%~dp0song_eval.py" --library-dir "%LIBRARY_DIR%"
) else (
    python "%~dp0song_eval.py" --library-dir "%LIBRARY_DIR%" --op "%OP%"
)
