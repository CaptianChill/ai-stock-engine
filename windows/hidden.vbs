' Runs a .cmd file with no visible window and passes back its exit code.
' The scheduled tasks call:  wscript.exe //B //Nologo hidden.vbs "C:\...\windows\tick.cmd"
' (Started directly, a .cmd flashes a black terminal on screen every time the task runs.)
If WScript.Arguments.Count < 1 Then WScript.Quit 2
Set sh = CreateObject("WScript.Shell")
WScript.Quit sh.Run("""" & WScript.Arguments(0) & """", 0, True)
