---
lyaout: single
title: 'Rmarkdown을 gitblog에 업로드하기'
categories: coding
tag: [R, blog, jekyll]
author_profile: false
sidebar:
    nav: "docs"
---

이전 포스팅에서 rmarkdown을 md확장자로 knit해서 gitblog에 업로드하는
법을 알아봤다.

그래서 공부한 내용을 업로드 하려고 하는데 이미지가 같은 방식으로하니
이미지가 업로드가 되지 않는 문제가 발생했다.

쉬운 예시로 $y=x$\$의 직선을 그려서 이것을 gitblog에 업로드 하고 싶다고
해보자.

``` r
x <- seq(-1, 1, 0.0001)
y = x
plot(x, y)
```

![](2023-03-09-upload_images_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->
