
fileName=$1

git checkout -b ${fileName}

cd source/_posts

if [ -n $fileName ] 
then
  if [ -e $fileName.md ]
  then
    echo "${fileName} already exists."
  else
    touch ${fileName}.md
    cat ../Template/Template.md >> ${fileName}.md
    open -a typora ${fileName}.md
  fi
else 
  echo "Please enter file name"
fi






  


