@echo off
setlocal enabledelayedexpansion

rem Specify the directory containing the folders you want to analyze
set "directory_path=motionnet\scheduler"
set "config_file=config.yaml"
set "ptr_file=ptr.yaml"
set "dest_dir_config=motionnet\configs"
set "dest_dir_ptr=motionnet\configs\method"

rem Iterate over each folder in the specified directory
for /d %%i in ("%directory_path%\*") do (
    rem Copy the config file in the directory d to the destination directory
    copy "%%i\%config_file%" "%dest_dir_config%\%config_file%"
    rem Copy the ptr file in the directory d to the destination directory
    copy "%%i\%ptr_file%" "%dest_dir_ptr%\%ptr_file%"

    rem Activate the virtual environment in the directory .venv/Scripts/activate
    call ".venv\Scripts\activate"

    rem Move to the motionnet directory
    cd motionnet

    rem Run the train.py script
    python train.py

    rem Get back to the parent directory
    cd ..

    rem Deactivate the virtual environment
    call ".venv\Scripts\deactivate"
)

endlocal