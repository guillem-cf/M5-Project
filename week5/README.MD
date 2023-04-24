# Week 5

The *main.py* file contains the pipeline for all tasks. You can choose the task to run in the parameter:

## Run a sweep in wandb: adapt the sweep_config dictionary below and run:

```
python main.py --task_a --train True --sweep True --network_image RESNET50 --network_text FastText --batch_size 64
```

## Run task_a train:

```
python main.py --task task_a --train True --sweep False --network_image RESNET50 --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```

```
python main.py --task task_a --train True --sweep False --network_image RESNET101 --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```

```
python main.py --task task_a --train True --sweep False --network_image fasterRCNN --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
```

# Run task_a test:

```
python main.py --task task_a --train False --sweep False --network_image RESNET50 --network_text FastText --weights_model /ghome/group03/M5-Project/week5/results/task_a/task_aRESNET_dim_out_fc_as_image_margin_0.1_lr_0.0001/task_a_triplet_10.pth
```

## Run task_b:

```
python main.py --task task_b --train True --sweep False --network_image RESNET50 --network_text FastText --batch_size 32 --lr 1e-4 --margin 0.1
```

```
python main.py --task task_b --train True --sweep False --network_image RESNET101 --network_text FastText --batch_size 32 --lr 1e-4 --margin 0.1
```

```
python main.py --task task_b --train True --sweep False --network_image fasterRCNN --network_text FastText --batch_size 32 --lr 1e-4 --margin 0.1
```

## Run task_c_a:
Same as task_a but using --network_text BERT