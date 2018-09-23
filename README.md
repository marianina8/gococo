# gococo
This `gococo` package runs object detection on a webcam stream by splitting the input frame by frame and running against a tensorflow model trained on the COCO dataset. This is a fork of https://github.com/ActiveState/gococo.  

## Installing

Before you begin, you'll of course need the Go programming language installed as well as [GoCV](https://gocv.io/)

1. Clone this repo.
2. Run `dep ensure`
3. Build the application with `go build`.
4. Download one of the COCO models from the [TensorFlow model zoo](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md).
5. Run the program according the usage below.

## Usage

`gococo -dir=<model folder> [-deviceID=0] [-labels=<labels.txt>]`

The default device ID is 0 for the default webcam on your computer.

## Using Pre-Trained Models with TensorFlow in Go

One of the challenges with machine learning is figuring out how to deploy trained models into production environments. After training your model, you can 'freeze' the weights in place and export it to be used in a production environment, potentially deployed to any number of server instances depending on your application.

For many common use cases, we're beginning to see organizations sharing their trained models in ready-to-use forms - and there are already a number of the most common models available for use in the TensorFlow [model repo](https://github.com/tensorflow/models).

For many building large scale web services, Go has become a language of choice. Go also has a growing data science community, but some of the tools are still missing documentation or features when compared to other languages like Python.

In this package we use one of the pre-trained models for TensorFlow and set it up to be executed in Go. In this case, we'll use the newly released TensorFlow Object Detection model, which is trained on the [COCO](http://mscoco.org) (Common Objects in Context) dataset.

We'll build a small command line application that takes in a web cam stream then will place a label and bounding box around any identified objects found within the frame. You can find all of the code from this post and the full application in the [following repo](https://github.com/marianina8/gococo).

One of the first places to start is by looking at the [example application](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/go/example_inception_inference_test.go) included in the Go TensorFlow binding, which uses the Inception model to do object identification - without extensive documentation, this example can give us valuable clues into how to use the bindings with the other pre-trained models, which are similar but not exactly the same.

For ours, we'll use the multi-object detection model trained on the COCO dataset. You can find that model on [GitHub](https://github.com/tensorflow/models/blob/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection/g3doc/detection_model_zoo.md). You can choose any of the models to download. We’ll trade off a bit of accuracy for speed and use the mobile one [ssd_mobilenet_v1_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz).

After extracting the model, the first step in our program is to load the frozen graph to a model folder so that we can use it to identify objects in our camera stream.

```Go
    // Load a frozen graph to use for queries
    modelpath := filepath.Join(*modeldir, "frozen_inference_graph.pb")
    model, err := ioutil.ReadFile(modelpath)
    if err != nil {
        log.Fatal(err)
    }

    // Construct an in-memory graph from the serialized form.
    graph := tf.NewGraph()
    if err := graph.Import(model, ""); err != nil {
        log.Fatal(err)
    }

    // Create a session for inference over graph.
    session, err := tf.NewSession(graph, nil)
    if err != nil {
        log.Fatal(err)
    }
    defer session.Close()
```

Thankfully, as you can see, we can just feed the protocol buffer file to the `NewGraph` function and it will decode it and build the graph. Then we just set up a session using this graph, and we can move on to the next step.

In order to start identifying objects within images with a tensorflow trained model, we'll need to know what the input and output nodes of the frozen graph is that we're using as well as it's `Shape`. Finding this information is not straight forward or easy, as there isn't much documentation online on this information.  Luckily, it was published online for the tensorflow model trained on COCO.

### Step 1: Identify the input and output nodes of the graph

The nodes for the COCO trained tensorflow model are as follows:

| Node Name         | Input/Output | Shape     | Data Description                                                                                         |
|-------------------|--------------|-----------|----------------------------------------------------------------------------------------------------------|
| image_tensor      | Input        | [1,?,?,3] | RGB pixel values as uint8 in a square format (Width, Height). The first column represent the batch size. |
| detection_boxes   | Output       | [?][4]    | Array of boxes for each detected object in the format [yMin, xMin, yMax, xMax]                           |
| detection_scores  | Output       | [?]       | Array of probability scores for each detected object between 0..1                                        |
| detection_classes | Output       | [?]       | Array of object class indices for each object detected based on COCO objects                             |
| num_detections    | Output       | [1]       | Number of detections                                                                                     | 

I would suggest that it would be best practice when publishing models to include this information as part of the documentation. Once you get the names of the nodes associated with the input/output, you can use the `Shape` method to display the shape of these inputs. In our case the input shape is similar to the one used in the Inception example referred to earlier.

From here, we can now work towards loading our image and transforming it into a format that we can use in our graph.

### Step 2: Load the webcam and transform it into a tensor

The next thing we need to do is to capture the webcam stream and load the images to analyze frame by frame.  We'll be using Ron Evan's [GoCV package](https://gocv.io/) to handle capture input from the webcam and altering it to later display in a new window. 

```Go
    // open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	// create new window
	window := gocv.NewWindow("Tensorflow Classifier")
	defer window.Close()

	// create new mat to hold a frame's image as numerical data
	mat := gocv.NewMat()
	defer mat.Close()
	
	for {
        // extract frame from webcam and populate
		if ok := webcam.Read(&mat); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
        }
        
        // check if mat is empty, continue since no analysis to be done
		if mat.Empty() {
			continue
        }
        
        // convert mat to jpeg encoding byte string
		bImg, err := gocv.IMEncode(".jpg", mat)
		if err != nil {
			fmt.Println("Err encoding images")
		}

		// DecodeJpeg uses a scalar String-valued tensor as input.
		tensor, i, err := makeTensorFromImage(bImg)
		if err != nil {
			log.Fatal(err)
		}

        // Execute the COCO Graph to identify objects (see below #3)

        // Display results with label and bounding box (see below #4)

        // display image inside the new window after labelling and bounding boxes are applied
		window.IMShow(mat)
		if window.WaitKey(1) >= 0 {
			break
		}

	}
```

### Step 3: Execute the COCO Graph to identify objects

We've now got our image transformed into a tensor, and we've identified all of our input and output nodes on the COCO graph. Now, if we execute that graph in a session, we'll get back a list of probabilities of objects detected in the image:

```Go
    output, err := session.Run(
        map[tf.Output]*tf.Tensor{
            inputop.Output(0): tensor,
        },
        []tf.Output{
            o1.Output(0),
            o2.Output(0),
            o3.Output(0),
            o4.Output(0),
        },
        nil)
```

The variable `tensor` above is the output from the previous `DecodeJpeg` graph we constructed. The list of outputs (`o1,o2,o3,o4`) are the various outputs outlined in the table above. And at this stage we can parse the results of the output.

There are a couple of notes to keep in mind when parsing the results:

- You probably want to set a threshold below which you want to ignore the results since the algorithm will try to detect things even with very low probability. I filtered out everything below around 40% confidence.
- The `detection_scores` list is sorted by probability, and each corresponding array is also identically sorted. So, for example, index 0 will have the highest probability object detected. And `detection_boxes` will contain the coordinates of its bounding box, and `detection_classes` will contain the class label for the object (ie. the name of the object: 'dog', 'person', etc.).
- The box coordinates are normalized, so you need to make sure you cache the width and height of the original JPG if you want to translate them into pixel coordinates in the image.

### Step 4: Visualizing the output

We're just going to use the GoCV package to add labels and bounding boxes to the image before displaying them in a new window.  We capture the output from the graph which gives us probabilities, classes and boxes.  By looping through the probabilities of each object in the frame, we extract the bounding box for that object, and the label (class).  Using GoCV we create a rectangle using the box information with `gocv.Rectangle`, and label with `gocv.PutText`.  I suggest you take a look at the [gocv documentation]() which is a Go wrapper for GoCV (using CGo binding for OpenCV (C/C++)) to see all the fun things you can do with OpenCV!

```Go
        // Outputs
	probabilities := output[1].Value().([][]float32)[0]
	classes := output[2].Value().([][]float32)[0]
	boxes := output[0].Value().([][][]float32)[0]

	// Draw a box around the objects
	curObj := 0

	// 0.4 is an arbitrary threshold, below this the results get a bit random
	for probabilities[curObj] > 0.4 {
		x1 := float32(img.Bounds().Max.X) * boxes[curObj][1]
		x2 := float32(img.Bounds().Max.X) * boxes[curObj][3]
		y1 := float32(img.Bounds().Max.Y) * boxes[curObj][0]
		y2 := float32(img.Bounds().Max.Y) * boxes[curObj][2]

		Rect(img, int(x1), int(y1), int(x2), int(y2), 4, colornames.Map[colornames.Names[int(classes[curObj])]])
		class := classes[curObj]
		col := colornames.Map[colornames.Names[int(class)]]
		// add bounding box to image
		gocv.Rectangle(&mat, image.Rect(int(x1), int(y1), int(x2), int(y2)), col, 3)
		// add label to image
		gocv.PutText(&mat, getLabel(curObj, probabilities, classes), image.Pt(int(x1), int(y1)), 				gocv.FontHersheyPlain, 1.2, color.RGBA{0, 255, 0, 0}, 2)

		curObj++
	}
```

We get the label for the object using an extra method `getLabel`.  

```Go
    func getLabel(idx int, probabilities []float32, classes []float32) string {
        index := int(classes[idx])
        label := fmt.Sprintf("%s (%2.0f%%)", labels[index], probabilities[idx]*100.0)

        return label
    }
```

With labels loaded into a slice, and armed with the tools in Go's standard library for image processing we can fairly easily iterate through the results and output an image with each object identified.

### Wrap-up

And that's it! We've now got a small Go program that can take any webcam and identify objects within it's stream using the popular COCO TensorFlow models supplied by Google. All of the code for this program, including all of the snippets from this post on GitHub in the [gococo](https://github.com/marianina8/gococo) repo.

*Further Reading*
[Building an ML Powered AI using TensorFlow in Go](http://gopherdata.io/post/build_ml_powered_game_ai_tensorflow/)
