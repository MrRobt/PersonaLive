@ECHO off 
SET msg="auto commit"  

git add .

git commit --no-verify -m %msg%
git pull
git push
