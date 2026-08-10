$env:CUDA_VISIBLE_DEVICES = "1"
$env:PYTHONUNBUFFERED = "1"
Set-Location "C:\Users\joeyh\Documents\GitHub\latin"
& ".\latinvenv\Scripts\python.exe" "scripts\stylize_library.py" "--skip-poetry" `
    *> "data\_stylize_shard0.log"
