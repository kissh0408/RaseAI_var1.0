@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDACXX=%CUDA_PATH%\bin\nvcc.exe"
pip uninstall lightgbm -y
pip install lightgbm==4.6.0 --no-binary lightgbm --no-cache-dir ^
  --config-settings=cmake.define.USE_CUDA=ON ^
  --config-settings=cmake.define.CMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe" ^
  --config-settings=cmake.define.CMAKE_CXX_FLAGS="/utf-8" ^
  --config-settings=cmake.define.CMAKE_CUDA_FLAGS="-Xcompiler=/utf-8" ^
  --config-settings=cmake.args=-GNinja
exit /b %ERRORLEVEL%
