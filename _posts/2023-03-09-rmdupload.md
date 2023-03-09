---
lyaout: single
title: 'Rmarkdown을 gitblog에 업로드하기'
categories: coding
tag: [R, blog, jekyll]
author_profile: false
sidebar:
    nav: "docs"
---


git 블로그를 생성하고 구색을 갖춘건 좋은데, 정작 포스팅을 하려고 하니까
주로 사용하는 rmarkdown에서 knit을 html이나 tex을 사용한 pdf가 아닌 md
파일을 사용해야 하고 다른 설정값을 몇 개 만질 필요가 있어 오늘은 그에
대한 글을 써보려고 한다.

크게 어렵지 않다.

일단 rstudio에서 rmarkdown파일을 만들고 기존의

``` tex
title: "제목"
author: 작성자
date: "2023-03-09"
output: html_document
```

이렇게 되있는 설정에서 output을 한칸 뛰어쓰고 *github_document:* 로만
바꿔주면 된다.

``` tex
title: "제목"
author: 작성자
date: "2023-03-09"
output: 
    github_document:
```

그 뒤 file제목을 `2023-03-09-rmdupload` 이런식으로 해준 뒤 본인의
github.io폴더의 \_post폴더에 넣어주면 된다.

참고자료 : <https://rmarkdown.rstudio.com/github_document_format.html>
