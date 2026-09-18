@echo off
    setlocal
set "LIBRARY_DIR=%~1"
if "%LIBRARY_DIR%"=="" set "LIBRARY_DIR=C:\Users\frueg\Desktop\music_studio\1_preprocessing\FJ\repertoire"
python "%~dp0song_eval.py" --library-dir "%LIBRARY_DIR%"
