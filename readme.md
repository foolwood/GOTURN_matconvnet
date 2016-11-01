# GOTURN_CASIA

## Track

```
show_tracker_test
```
### result in VOT2015	

**Ubuntu16.04 MATLAB2015b matconvnet-1.0-beta23 CPU**

|    index   |     Name     |     fps    |
|:----------:|:------------:|:----------:|
|  Video:  1 |          bag | fps:14.478 |
|  Video:  2 |        ball1 | fps:14.507 |
|  Video:  3 |        ball2 | fps:15.156 |
|  Video:  4 |   basketball | fps:15.766 |
|  Video:  5 |       birds1 | fps:16.397 |
|  Video:  6 |       birds2 | fps:15.472 |
|  Video:  7 |      blanket | fps:14.715 |
|  Video:  8 |          bmx | fps:15.699 |
|  Video:  9 |        bolt1 | fps:16.077 |
|  Video: 10 |        bolt2 | fps:16.644 |
|  Video: 11 |         book | fps:16.599 |
|  Video: 12 |    butterfly | fps:15.753 |
|  Video: 13 |         car1 | fps:14.030 |
|  Video: 14 |         car2 | fps:15.509 |
|  Video: 15 |     crossing | fps:15.137 |
|  Video: 16 |     dinosaur | fps:14.692 |
|  Video: 17 |     fernando | fps:14.734 |
|  Video: 18 |        fish1 | fps:14.843 |
|  Video: 19 |        fish2 | fps:14.386 |
|  Video: 20 |        fish3 | fps:14.739 |
|  Video: 21 |        fish4 | fps:15.308 |
|  Video: 22 |         girl | fps:15.655 |
|  Video: 23 |        glove | fps:16.648 |
|  Video: 24 |    godfather | fps:15.932 |
|  Video: 25 |     graduate | fps:16.895 |
|  Video: 26 |  gymnastics1 | fps:16.196 |
|  Video: 27 |  gymnastics2 | fps:17.015 |
|  Video: 28 |  gymnastics3 | fps:17.080 |
|  Video: 29 |  gymnastics4 | fps:16.369 |
|  Video: 30 |         hand | fps:16.260 |
|  Video: 31 |    handball1 | fps:16.126 |
|  Video: 32 |    handball2 | fps:15.329 |
|  Video: 33 |   helicopter | fps:15.710 |
|  Video: 34 |   iceskater1 | fps:16.125 |
|  Video: 35 |   iceskater2 | fps:13.874 |
|  Video: 36 |       leaves | fps:17.077 |
|  Video: 37 |     marching | fps:15.758 |
|  Video: 38 |       matrix | fps:15.762 |
|  Video: 39 |   motocross1 | fps:13.975 |
|  Video: 40 |   motocross2 | fps:12.651 |
|  Video: 41 |       nature | fps:13.502 |
|  Video: 42 |      octopus | fps:14.168 |
|  Video: 43 |  pedestrian1 | fps:15.917 |
|  Video: 44 |  pedestrian2 | fps:14.111 |
|  Video: 45 |       rabbit | fps:15.683 |
|  Video: 46 |       racing | fps:14.060 |
|  Video: 47 |         road | fps:14.941 |
|  Video: 48 |      shaking | fps:15.249 |
|  Video: 49 |        sheep | fps:15.366 |
|  Video: 50 |      singer1 | fps:14.336 |
|  Video: 51 |      singer2 | fps:15.419 |
|  Video: 52 |      singer3 | fps:15.329 |
|  Video: 53 |      soccer1 | fps:14.843 |
|  Video: 54 |      soccer2 | fps:15.942 |
|  Video: 55 |      soldier | fps:16.232 |
|  Video: 56 |       sphere | fps:15.640 |
|  Video: 57 |        tiger | fps:15.955 |
|  Video: 58 |      traffic | fps:16.229 |
|  Video: 59 |       tunnel | fps:16.492 |
|  Video: 60 |        wiper | fps:16.077 |


## Train

### Experiment（bbox）
OS:OSX MATLAB matconvnet cpu


#### opts.version = 1
**bbox: minmax**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  |
|:---------:|:-----:| ------------:| ---------:|
|  VOT15    |  train|  21395       |  1+0      |
|  VOT14    |  val  |  10188       |  1+0      |

#### opts.version = 2
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  |
|:---------:|:-----:| ------------:| ---------:|
|  VOT15    |  train|  21395       |  1+0      |
|  VOT14    |  val  |  10188       |  1+0      |



### Experiment（augment）
OS:WIN8.1 MATLAB matconvnet GPU

#### opts.version = 3
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  | 
|:---------:|:-----:| ------------:| ---------:| 
|  VOT15    |  train|  21395       |  1+0      | 
|  VOT14    |  val  |  10188       |  1+0      |
|  NUS_PRO  |  train|  26090       |  1+0      |

#### opts.version = 4
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  | 
|:---------:|:-----:| ------------:| ---------:| 
|  VOT15    |  train|  21395       |  1+9     | 
|  VOT14    |  val  |  10188       |  1+9     |
|  NUS_PRO  |  train|  26090       |  1+9     |

#### opts.version = 5
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  | 
|:---------:|:-----:| ------------:| ---------:| 
|  VOT15    |  train|  21395       |  1+19     | 
|  VOT14    |  val  |  10188       |  1+19     |
|  NUS_PRO  |  train|  26090       |  1+19     |

#### opts.version = 6
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  | 
|:---------:|:-----:| ------------:| ---------:| 
|  VOT15    |  train|  21395       |  1+29     | 
|  VOT14    |  val  |  10188       |  1+29     |
|  NUS_PRO  |  train|  26090       |  1+29     |

#### opts.version = 7
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  | 
|:---------:|:-----:| ------------:| ---------:| 
|  VOT15    |  train|  21395       |  1+39     | 
|  VOT14    |  val  |  10188       |  1+39     |
|  NUS_PRO  |  train|  26090       |  1+39     |

#### opts.version = 8
**bbox: axis_aligned**(batch:50)

|  DATASET  |  set  |  image_pair  |  augment  | 
|:---------:|:-----:| ------------:| ---------:| 
|  VOT15    |  train|  21395       |  1+49     | 
|  VOT14    |  val  |  10188       |  1+49     |
|  NUS_PRO  |  train|  26090       |  1+49     |




