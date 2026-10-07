@echo off
if not exist out mkdir out
javac -d out src\vectordb\*.java
if errorlevel 1 exit /b 1
java -cp out vectordb.Main %1