param($envName = "LaGCNEnv")
Write-Output "EnvName is $envName"; 

Write-Output "Creating ENV..."
python -m venv $envName
Write-Output "OK!"

Write-Output "Activating Env..."
powershell -noexit "./$envName/Scripts/Activate.ps1"
Write-Output "OK!"