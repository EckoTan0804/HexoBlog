cd source/_posts

fileName=$1
title=$2

if [ -n $fileName ] 
then
  if [ -e $fileName.md ]
  then
    echo "${fileName} already exists."
  else
    touch ${fileName}.md
    cat Template.md >> ${fileName}.md
    open -a typora ${fileName}.md
  fi
else 
  echo "Please enter file name"
fi






  


