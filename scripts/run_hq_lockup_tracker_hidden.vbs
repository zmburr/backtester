' Hidden launcher for run_hq_lockup_tracker.bat — no console window pops up.
' Task Scheduler action: wscript.exe //B //Nologo "<this file>"
' Waits for the bat and propagates its exit code so LastResult still reflects failures.
Dim sh, code
Set sh = CreateObject("WScript.Shell")
code = sh.Run("""C:\Users\zmbur\PycharmProjects\backtester\scripts\run_hq_lockup_tracker.bat""", 0, True)
WScript.Quit code
