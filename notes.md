# initial repro run
followed instructions on repo to `pip install -r requirements.txt`

created conda environment `conda create -n py27 python=2.7`

install torch 0.2.0 on that env `conda install pytorch=0.2.0 torchvision cuda90 -c pytorch`

training of sqlnet took about 31 hours and produced similar results to that in the paper (59% validation)
https://www.comet.ml/wronnyhuang/sqlnet/a317394af04242bd8e0490831b0a9d13

# how to run for demo

first switch to py27 conda environment in tricky's docker

`conda activate py27`

then run the following

`python infer.py --ca --gpu=3`

