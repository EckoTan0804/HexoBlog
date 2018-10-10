cd source/_posts

fileName=$1
title=$2
template="
--- \n
title: ${title} \n
tags: \n
- \n
\n
--- \n
"


if [ -n $fileName ] 
then
  if [ -e $fileName.md ]
  then
    echo "${fileName} already exists."
  else
    touch ${fileName}.md
    echo ${template} >> ${fileName}.md
    open -a typora ${fileName}.md
  fi
else 
  echo "Please enter file name"
fi






  


