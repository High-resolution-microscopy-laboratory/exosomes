set COMMAND=%1
set ROOT=%2
set IN_DIR=%3

if "%~4"=="" (
    set OUT_DIR=""
) else (
    set OUT_DIR="data/%4"
)

docker container run -v %ROOT%:/app/data/ exactly/vesicles %COMMAND% data/%IN_DIR% %OUT_DIR%
