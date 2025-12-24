@ECHO off 
SET msg= 
SET /P msg="�������ύ��Ϣ: " 
IF "%msg%"=="" ( 
SET msg="auto commit" 
) 



git add .

git commit --no-verify -m %msg%
git pull
git push -u origin main
pause