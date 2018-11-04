---
title: Hello World
tags:
- HelloWorld
- General


---
![Bildergebnis fÃ¼r Hello World](https://cdn.instructables.com/FNF/7PUG/IRAVYHIC/FNF7PUGIRAVYHIC.LARGE.jpg)

<!--more-->



## Why I Start Blogging

> “I learned very early the difference between knowing the name of something and knowing something.” - Richard Feynman

When I read the famous [***"Surely You're Joking, Mr. Feynman!"*: Adventures of a Curious Character**](https://en.wikipedia.org/wiki/Surely_You%27re_Joking,_Mr._Feynman!) by [Richard Feynman](https://en.wikipedia.org/wiki/Richard_Feynman),

![img](https://upload.wikimedia.org/wikipedia/commons/e/eb/Richard_Feynman_Los_Alamos_ID_badge.jpg)



yes, this handsome guy, one of the greatest physicist, I was quiet impressed by the **Feynman technique**. 

> I couldn’t reduce it to the freshman level. That means we don’t really understand it.”

That's quote that’s often attributed to Albert Einstein which goes:

> “If you can’t explain it simply, you don’t understand it well enough.”

The idea behind Feynman technique is that **try to explain something as simply** 
**(in plain, simple language) as you can**. During the explaining, you can quickly locate where you stuck in and find out the points that you didn't actually understand well. In other words, this is a "self test and fix" procedure. In addition, the explanation using plain, simple language will help you to understand some abstract concepts more concretely.

Therefore, based on this technique, I will try to write articles to explain what I have learned using plain and simple language, under the [**KISS** (Keep It Simple Stupid)](https://en.wikipedia.org/wiki/KISS_principle) principle. As a developer, I am a super fan of this principle. Additionally, there will be some post to mark down the bugs and fix I have met during my development as well. I will also share some of my setups and configurations of my development tools.



## Basic Hexo Operation

### Create a new post

``` bash
$ hexo new "My New Post"
```

More info: [Writing](https://hexo.io/docs/writing.html)

### Run server

``` bash
$ hexo server
```

More info: [Server](https://hexo.io/docs/server.html)

### Generate static files

``` bash
$ hexo generate
```

More info: [Generating](https://hexo.io/docs/generating.html)

### Deploy to remote sites

``` bash
$ hexo deploy
```

More info: [Deployment](https://hexo.io/docs/deployment.html)



## My Custom Script for Hexo Blogging

Every script below locates directly under the **source** folder. 

### Serve locally

`serveLocal.sh`

~~~bash
hexo clean
hexo g
hexo s -open
~~~

### Write new article

This is the structure of my **source** folder:

~~~
source
- Template
- _posts
- about
- images
- tags
~~~

In the **Template** folder I have my custom template for every new article:

~~~markdown
---
title: 
tags:
- 
- 
- 


---

<!--- Overview or first paragraph --->

<!--more-->

<!--- Your article --->
~~~

Assume that you are using [git](https://git-scm.com/) for version control (If you are not, I strongly recommend you to use it!). `newArticle.sh` do the following things:

+ Create and switch to the new branch using `${fileName}`
+ Create a new empty Markdown named `${fileName}.md`
+ Inject the template into the new article

~~~bash

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

~~~

The first argument of this script is the file name as well as the branch name and it should be in [`Delimeter-Seperated`](https://en.wikipedia.org/wiki/Naming_convention_%28programming%29#Delimiter-separated_words) format. For instance, if you want to write an article name "Hello World", you should execute the script like this:

~~~shell
$ ./newArticle.sh hello-world
~~~

### How to use?

1. Go to **source** folder

2. Create the script

   1. For example, if you use [vim](https://www.vim.org/):

   ~~~shell
   $ vim ${script} 
   ~~~

   2. Write the script
      1. Hit `i`
      2. Copy and paste 
      3. Hit `Esc`
      4. `:` + `wq`

3. Make the script executable:

```shell
$ chmod +x ${script}
```

4. Execute the script

```shell
$ ./${script} [arguments]
```

For Example:

```shell
$ chmod +x serveLocal.sh
$ serveLocal.sh
```

