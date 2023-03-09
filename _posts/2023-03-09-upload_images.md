---
lyaout: single
title: 'Rmarkdown에서 이미지를 gitblog에 업로드하기'
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

쉬운 예시로 $y=x$ 의 직선을 그려서 이것을 gitblog에 업로드 하고 싶다고
해보자.

``` r
x <- seq(-1, 1, 0.0001)
y = x
plot(x, y)
```

![](2023-03-09-upload_images_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

이렇게해서 md확장자로 만들고 이후에 블로그에 업로드하고 확인을 해보니
코드청크는 맞게 들어갔지만 output의 line이 출력되지 않은 문제가
발생한다.

![](/images/2023-03-09-upload_images_files/figure-gfm/plot_md.png)

이 이유는 우리가 md파일로 변환하면 저 코드청크의 출력에 대해서 결과물이

`![](2023-03-09-upload_images_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->`
이런식으로 주어지고 rmd파일을 저장한 폴더에
*2023-03-09-upload_images_files* 이런 파일이 생겨서 해당하는 이미지들이
다 저기에 저장되는 형식으로 주어지기 때문이다.

그래서 저 이미지 파일을 github.io 레포지토리 폴더에 넣어주는 과정이
필요하다.

레포지토리 최상단에 이미지 파일을 넣어줘도 지장이 없지만, 이럴시에 해당
파일이 많이질 경우 정리가 어려울 수 있으므로 레포지토리 최상단에 images
폴더를 만들고, 그곳에 정리하도록 하자.

이럴시 주의사항은 이미지 경로를
`![](images/2023-03-09-upload_images_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->`
이런식으로 수정해 줘야한다.

그런 뒤에 다시 위의 코드를 실행해보면 맞게 이미지가 들어감을 확인할 수
있다.

``` r
x <- seq(-1, 1, 0.0001)
y = x
plot(x, y)
```

![](/images/2023-03-09-upload_images_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->