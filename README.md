## Cooperative Exploration for Multi-Agent Deep Reinforcement Learning #
### ICML 2021
#### [[Project Website]](https://ioujenliu.github.io/CMAE/) [[PDF]](http://proceedings.mlr.press/v139/liu21j/liu21j.pdf)

[Iou-Jen Liu](https://ioujenliu.github.io/), [Unnat Jain](https://unnat.github.io/), [Raymond A. Yeh](http://raymondyeh07.github.io/), [Alexander G. Schwing](http://www.alexander-schwing.de/) <br/>
University of Illinois at Urbana-Champaign<br/>


The repository contains Python implementation of Cooperative Exploration for Multi-Agent Deep Reinforcement Learning (CMAE) with Q-Learning on the discrete multi-agent particle environments (MPE).

If you used this code for your experiments or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{LiuICML2021,
  author = {I.-J. Liu and U. Jain and R.~A. Yeh and A.~G. Schwing},
  title = {{Cooperative Exploration for Multi-Agent Deep Reinforcement Learning}},
  booktitle = {Proc. ICML},
  year = {2021},
}
</pre>

### Platform and Dependencies:
* Platform: Ubuntu 16.04
* Conda 4.8.3
* Python 3.6.3
* Numpy 1.19.2

### Training
    cd script
    sh run_came_push_box.sh
    sh run_came_room.sh
    sh run_came_secret_room.sh
Training log will be dumped to `log/CMAE/`.


### License
CMAE is licensed under the MIT License