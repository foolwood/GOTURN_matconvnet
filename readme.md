# GOTURN_*matconvnet*

A *matlab* implementation of GOTURN. **Rather Fast in CPU(20FPS).**


GOTURN appeared in this paper:

**[Learning to Track at 100 FPS with Deep Regression Networks](http://davheld.github.io/GOTURN/GOTURN.html)**,
European Conference on Computer Vision (ECCV), 2016

GOTURN addresses the problem of **single target tracking**: given a bounding box label of an object in the first frame of the video, we track that object through the rest of the video.  

Here is a brief overview of how GOTURN works:

<img src="https://github.com/davheld/GOTURN/blob/master/imgs/pull7f-web_e2.png" width=85%>



## Pretrained model

**Download** from [**google drive**](https://drive.google.com/open?id=0BwWEXCnRCqJ-M1FFRVNTQ0JEXzQ) or [**BaiduYun**](https://pan.baidu.com/s/1nuW8llR)

or

**Convert** caffe models to matconvnet by yourself.

- Download caffemodel

```
cd <GOTURN_matconvnet>
cd model
wget http://cs.stanford.edu/people/davheld/public/GOTURN/trained_model/tracker.caffemodel
```

- MATLAB

```
cd model
Converter_caffe_matconvnet
make_goturn_net
```


## Track

- Compile [matconvnet](https://github.com/vlfeat/matconvnet)

```
cd <GOTURN_matconvnet>
git clone https://github.com/vlfeat/matconvnet.git
cd matconvnet
run matlab/vl_compilenn ;
```

- Download [VOT2015](http://data.votchallenge.net/vot2015/vot2015.zip)

```
cd <GOTURN_matconvnet>
cd data
wget http://data.votchallenge.net/vot2015/vot2015.zip
unzip -n vot2015.zip -d ./VOT15
```

- Track in MATLAB

```
cd <GOTURN_matconvnet>
cd track
show_tracker_test();
```

### VOT2015 speed 

**Ubuntu16.04 MATLAB2016b matconvnet-1.0-beta24 **

**CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz**

**GPU: GTX 1080ti**

|   index   |    Name     | CPU(fps) | GPU(fps) |
| :-------: | :---------: | :------: | :------: |
| Video: 1  |     bag     |  19.752  |  87.459  |
| Video: 2  |    ball1    |  18.36   |  92.336  |
| Video: 3  |    ball2    |  18.841  | 100.206  |
| Video: 4  | basketball  |  20.557  |  89.386  |
| Video: 5  |   birds1    |  18.177  |  96.058  |
| Video: 6  |   birds2    |  20.574  |  95.602  |
| Video: 7  |   blanket   |  20.751  |  98.822  |
| Video: 8  |     bmx     |  17.976  |  79.232  |
| Video: 9  |    bolt1    |  19.737  |  99.359  |
| Video: 10 |    bolt2    |  19.728  |  99.905  |
| Video: 11 |    book     |  19.929  |  96.446  |
| Video: 12 |  butterfly  |   19.1   |  94.681  |
| Video: 13 |    car1     |  18.178  |  67.749  |
| Video: 14 |    car2     |  20.272  |  96.047  |
| Video: 15 |  crossing   |  19.109  |  92.823  |
| Video: 16 |  dinosaur   |  19.865  |  92.37   |
| Video: 17 |  fernando   |  19.033  |  78.656  |
| Video: 18 |    fish1    |  20.49   |  95.689  |
| Video: 19 |    fish2    |  20.609  |  97.977  |
| Video: 20 |    fish3    |  20.411  |  94.673  |
| Video: 21 |    fish4    |  19.897  |  96.342  |
| Video: 22 |    girl     |  20.181  |  94.281  |
| Video: 23 |    glove    |  19.159  |  94.516  |
| Video: 24 |  godfather  |  19.934  |  101.74  |
| Video: 25 |  graduate   |  20.496  |  99.465  |
| Video: 26 | gymnastics1 |  20.224  |  95.631  |
| Video: 27 | gymnastics2 |  17.944  |  94.272  |
| Video: 28 | gymnastics3 |  17.924  |  88.965  |
| Video: 29 | gymnastics4 |  19.763  |  93.196  |
| Video: 30 |    hand     |  20.212  |  98.163  |
| Video: 31 |  handball1  |  20.486  |  99.822  |
| Video: 32 |  handball2  |  20.04   |  94.673  |
| Video: 33 | helicopter  |  19.653  |  87.207  |
| Video: 34 | iceskater1  |  20.101  |  96.558  |
| Video: 35 | iceskater2  |  19.575  |  91.674  |
| Video: 36 |   leaves    |  18.942  |  99.633  |
| Video: 37 |  marching   |  18.284  |  88.932  |
| Video: 38 |   matrix    |  19.811  |  95.412  |
| Video: 39 | motocross1  |  20.769  |  96.067  |
| Video: 40 | motocross2  |  16.784  |  78.495  |
| Video: 41 |   nature    |  19.877  |  88.762  |
| Video: 42 |   octopus   |  16.946  |  75.872  |
| Video: 43 | pedestrian1 |  21.029  |  98.191  |
| Video: 44 | pedestrian2 |  20.381  |  96.385  |
| Video: 45 |   rabbit    |  20.271  |  97.809  |
| Video: 46 |   racing    |  19.932  |  96.254  |
| Video: 47 |    road     |  18.91   |  96.028  |
| Video: 48 |   shaking   |  20.484  |  95.895  |
| Video: 49 |    sheep    |   21.3   | 101.453  |
| Video: 50 |   singer1   |  20.772  |  95.205  |
| Video: 51 |   singer2   |  20.427  |  91.472  |
| Video: 52 |   singer3   |  20.131  |  83.639  |
| Video: 53 |   soccer1   |  20.029  |  89.014  |
| Video: 54 |   soccer2   |  19.094  |  96.23   |
| Video: 55 |   soldier   |  17.937  |  92.634  |
| Video: 56 |   sphere    |  20.265  |  92.279  |
| Video: 57 |    tiger    |  19.663  |  96.815  |
| Video: 58 |   traffic   |  18.328  |  97.919  |
| Video: 59 |   tunnel    |  20.135  |  98.278  |
| Video: 60 |    wiper    |  20.404  |  97.771  |



#### And more?

If you have problem, email [@foolwood](wangqiang2015@ia.ac.cn).