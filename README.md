

# CSE 5522 Final Project



# File Directory

	|
	|--code
	|
	|
	|--data
		|
		|--dataset
		|	|
		|	|--train
		|	|   |
		|	|   |--plots
		|	|   |
		|	|   |--jsons
		|	|
		|	|--test
		|	    |
		|	    |--plots
		|	    |
		|	    |--jsons
		|
		|--test_results		

# Environment
Ubuntu, python3.6+


# Train

```
cd code/
python main.py
```


# Test
```
cd code
python main.py --phase test --resume  ../data/models/best.ckpt
```

-	The generated results will be saved in data/test\_results/.

# Important parameters
## Important parameters for training
-	num-hard: default=32, select 32 hard negative samples for each image
-	anchors: can be modified in the code/main.py (line 45 - 62)
-	pos-iou: defautl = 0.5, When iou > 0.5, the corrsponding bounding box will be selected as postive sample.
-	neg-iou: defautl = 0.16, When iou < 0.16, the corrsponding bounding box will be selected as negative sample.

## Important parameters for test
-	neg-th: default = 0.8, when predicted probability > 0.8, the found object will be used to generate final output
-	nms-ol: default = 0.1, if two bounding boxes have overlap > 0.1, the bounding box with lower probability will be removed.

