/*
Convolutional Neural Network using the Multilayer Perceptron and the Back-Propagation Algorithm.
This is a web application that uses the html/template package to create the HTML.
The URL is http://127.0.0.1:8080/imageCNN.  There are two phases of
operation:  the training phase and the testing phase.  Epochs consising of
a sequence of examples are used to train the nn.  Each example consists
of an input vector of (x,y) coordinates and a desired class output.  The nn
itself consists of an input layer of nodes, one or more hidden layers of nodes,
and an output layer of nodes.  The nodes are connected by weighted links.  The
weights are trained by back propagating the output layer errors forward to the
input layer.  The chain rule of differential calculus is used to assign credit
for the errors in the output to the weights in the hidden layers.
The output layer outputs are subtracted from the desired to obtain the error.
The user trains first and then tests.
*/

package main

import (
	"bufio"
	"fmt"
	"html/template"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"strconv"
	"strings"
)

const (
	addr               = "127.0.0.1:8080"             // http server listen address
	fileTrainingCNN    = "templates/trainingCNN.html" // html for training CNN
	fileTestingCNN     = "templates/testingCNN.html"  // html for testing CNN
	patternTrainingCNN = "/imageCNN"                  // http handler for training the CNN
	patternTestingCNN  = "/imageCNNtest"              // http handler for testing the CNN
	xlabels            = 11                           // # labels on x axis
	ylabels            = 11                           // # labels on y axis
	fileweights        = "weights.csv"                // cnn weights
	filechars          = "encoded_chars.csv"          // encoded characters
	a                  = 1.7159                       // activation function const
	b                  = 2.0 / 3.0                    // activation function const
	K1                 = b / a
	K2                 = a * a
	dataDir            = "data/" // directory for the weights
	maxClasses         = 25
	classes            = 64      // number of classes and the number of training samples
	imageSize          = 81      // character image size 9 x 9
	testingSamples     = 33 * 33 // int(300/9) * int(300/9)
)

// Type to contain all the HTML template actions
type PlotT struct {
	Grid         []string  // plotting grid
	Status       string    // status of the plot
	Xlabel       []string  // x-axis labels
	Ylabel       []string  // y-axis labels
	HiddenLayers string    // number of hidden layers
	LayerDepth   string    // number of Nodes in hidden layers
	Classes      string    // constant number of classes = 64
	LearningRate string    // size of weight update for each iteration
	Momentum     string    // previous weight update scaling factor
	Epochs       string    // number of epochs
	TestResults  []Results // tabulated statistics of testing
	TotalCount   string    // Results tabulation
	TotalCorrect string
}

// Type to hold the minimum and maximum data values of the MSE in the Learning Curve
type Endpoints struct {
	xmin float64
	xmax float64
	ymin float64
	ymax float64
}

// graph node
type Node struct {
	y     float64 // output of this node for forward prop
	delta float64 // local gradient for backward prop
}

// graph links
type Link struct {
	wgt      float64 // weight
	wgtDelta float64 // previous weight update used in momentum
}

type Stats struct {
	correct    []int // % correct classifcation
	classCount []int // #samples in each class
}

// training examples
type Sample struct {
	encChar   []int8 // encoded character consisting of 81 (-1,1)
	desired   int    // true class of the character
	character string // alpha-numeric character and / and +
}

// Primary data structure for holding the CNN Backprop state
type CNN struct {
	plot         *PlotT   // data to be distributed in the HTML template
	Endpoints             // embedded struct
	link         [][]Link // links in the graph
	node         [][]Node // nodes in the graph
	samples      []Sample
	statistics   Stats
	mse          []float64 // mean square error in output layer per epoch used in Learning Curve
	epochs       int       // number of epochs
	learningRate float64   // learning rate parameter
	momentum     float64   // delta weight scale constant
	hiddenLayers int       // number of hidden layers
	desired      []float64 // desired output of the sample
	layerDepth   int       // hidden layer number of nodes
	rows         int       // #rows in grid
	columns      int       // #columns in grid
	charTest     []int     // testing examples desired classes
	grid         []int8    // grid containing the image
}

// test statistics that are tabulated in HTML
type Results struct {
	Class     string // int
	Correct   string // int      percent correct
	Character string // alpha-numeric character
	Count     string // int      number of training examples in the class
}

// global variables for parse and execution of the html template
var (
	tmplTrainingCNN *template.Template
	tmplTestingCNN  *template.Template
)

// calculateMSE calculates the MSE at the output layer every
func (cnn *CNN) calculateMSE(epoch int) {
	// loop over the output layer nodes
	var err float64 = 0.0
	outputLayer := cnn.hiddenLayers + 1
	for n := 0; n < len(cnn.node[outputLayer]); n++ {
		// Calculate (desired[n] - cnn.node[L][n].y)^2 and store in cnn.mse[n]
		//fmt.Printf("n = %d, desired = %f, y = %f\n", n, cnn.desired[n], cnn.node[outputLayer][n].y)
		err = float64(cnn.desired[n]) - cnn.node[outputLayer][n].y
		err2 := err * err
		cnn.mse[epoch] += err2
	}
	cnn.mse[epoch] /= float64(classes)

	// calculate min/max mse
	if cnn.mse[epoch] < cnn.ymin {
		cnn.ymin = cnn.mse[epoch]
	}
	if cnn.mse[epoch] > cnn.ymax {
		cnn.ymax = cnn.mse[epoch]
	}
}

// determineClass determines testing example class given sample
func (cnn *CNN) determineClass(sample Sample) error {
	// At output layer, classify example, increment class count, %correct

	// convert node outputs to the class; zero is the threshold
	class := 0
	for i, output := range cnn.node[cnn.hiddenLayers+1] {
		if output.y > 0.0 {
			class |= (1 << i)
		}
	}

	// Assign Stats.correct, Stats.classCount
	cnn.statistics.classCount[sample.desired]++
	if class == sample.desired {
		cnn.statistics.correct[class]++
	}

	return nil
}

// class2desired constructs the desired output from the given class
func (cnn *CNN) class2desired(class int) {
	// tranform int to slice of -1 and 1 representing the 0 and 1 bits
	for i := 0; i < len(cnn.desired); i++ {
		if class&1 == 1 {
			cnn.desired[i] = 1
		} else {
			cnn.desired[i] = -1
		}
		class >>= 1
	}
}

func (cnn *CNN) propagateForward(samp Sample) error {
	// Assign sample to input layer
	for i := 1; i < len(cnn.node[0]); i++ {
		cnn.node[0][i].y = float64(samp.encChar[i-1])
	}

	// calculate desired from the class
	cnn.class2desired(samp.desired)

	// Loop over layers: cnn.hiddenLayers + output layer
	// input->first hidden, then hidden->hidden,..., then hidden->output
	for layer := 1; layer <= cnn.hiddenLayers; layer++ {
		// Loop over nodes in the layer, d1 is the layer depth of current
		d1 := len(cnn.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Each node in previous layer is connected to current node because
			// the network is fully connected.  d2 is the layer depth of previous
			d2 := len(cnn.node[layer-1])
			// Loop over weights to get v
			v := 0.0
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				v += cnn.link[layer-1][i2*(d1-1)+i1-1].wgt * cnn.node[layer-1][i2].y
			}
			// compute output y = Phi(v)
			cnn.node[layer][i1].y = a * math.Tanh(b*v)
		}
	}

	// last layer is different because there is no bias node, so the indexing is different
	layer := cnn.hiddenLayers + 1
	d1 := len(cnn.node[layer])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		// Each node in previous layer is connected to current node because
		// the network is fully connected.  d2 is the layer depth of previous
		d2 := len(cnn.node[layer-1])
		// Loop over weights to get v
		v := 0.0
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			v += cnn.link[layer-1][i2*d1+i1].wgt * cnn.node[layer-1][i2].y
		}
		// compute output y = Phi(v)
		cnn.node[layer][i1].y = a * math.Tanh(b*v)
	}

	return nil
}

func (cnn *CNN) propagateBackward() error {

	// output layer is different, no bias node, so the indexing is different
	// Loop over nodes in output layer
	layer := cnn.hiddenLayers + 1
	d1 := len(cnn.node[layer])
	for i1 := 0; i1 < d1; i1++ { // this layer loop
		//compute error e=d-Phi(v)
		cnn.node[layer][i1].delta = cnn.desired[i1] - cnn.node[cnn.hiddenLayers+1][i1].y
		// Multiply error by this node's Phi'(v) to get local gradient.
		cnn.node[layer][i1].delta *= K1 * (K2 - cnn.node[layer][i1].y*cnn.node[layer][i1].y)
		// Send this node's local gradient to previous layer nodes through corresponding link.
		// Each node in previous layer is connected to current node because the network
		// is fully connected.  d2 is the previous layer depth
		d2 := len(cnn.node[layer-1])
		for i2 := 0; i2 < d2; i2++ { // previous layer loop
			cnn.node[layer-1][i2].delta += cnn.link[layer-1][i2*d1+i1].wgt * cnn.node[layer][i1].delta
			// Compute weight delta, Update weight with momentum, y, and local gradient
			wgtDelta := cnn.learningRate * cnn.node[layer][i1].delta * cnn.node[layer-1][i2].y
			cnn.link[layer-1][i2*d1+i1].wgt +=
				wgtDelta + cnn.momentum*cnn.link[layer-1][i2*d1+i1].wgtDelta
			// update weight delta
			cnn.link[layer-1][i2*d1+i1].wgtDelta = wgtDelta

		}
		// Reset this local gradient to zero for next training example
		cnn.node[layer][i1].delta = 0.0
	}

	// Loop over layers in backward direction, starting at the last hidden layer
	for layer := cnn.hiddenLayers; layer > 0; layer-- {
		// Loop over nodes in this layer, d1 is the current layer depth
		d1 := len(cnn.node[layer])
		for i1 := 1; i1 < d1; i1++ { // this layer loop
			// Multiply deltas propagated from past node by this node's Phi'(v) to get local gradient.
			cnn.node[layer][i1].delta *= K1 * (K2 - cnn.node[layer][i1].y*cnn.node[layer][i1].y)
			// Send this node's local gradient to previous layer nodes through corresponding link.
			// Each node in previous layer is connected to current node because the network
			// is fully connected.  d2 is the previous layer depth
			d2 := len(cnn.node[layer-1])
			for i2 := 0; i2 < d2; i2++ { // previous layer loop
				cnn.node[layer-1][i2].delta += cnn.link[layer-1][i2*(d1-1)+i1-1].wgt * cnn.node[layer][i1].delta
				// Compute weight delta, Update weight with momentum, y, and local gradient
				// anneal learning rate parameter: cnn.learnRate/(epoch*layer)
				// anneal momentum: momentum/(epoch*layer)
				wgtDelta := cnn.learningRate * cnn.node[layer][i1].delta * cnn.node[layer-1][i2].y
				cnn.link[layer-1][i2*(d1-1)+i1-1].wgt +=
					wgtDelta + cnn.momentum*cnn.link[layer-1][i2*(d1-1)+i1-1].wgtDelta
				// update weight delta
				cnn.link[layer-1][i2*(d1-1)+i1-1].wgtDelta = wgtDelta

			}
			// Reset this local gradient to zero for next training example
			cnn.node[layer][i1].delta = 0.0
		}
	}
	return nil
}

// runEpochs performs forward and backward propagation over each sample
func (cnn *CNN) runEpochs() error {

	// Initialize the weights

	// input layer
	// initialize the wgt and wgtDelta randomly, zero mean, normalize by fan-in
	for i := range cnn.link[0] {
		cnn.link[0][i].wgt = 2.0 * (rand.ExpFloat64() - .5) / float64(imageSize+1)
		cnn.link[0][i].wgtDelta = 2.0 * (rand.ExpFloat64() - .5) / float64(imageSize+1)
	}

	// output layer links
	for i := range cnn.link[cnn.hiddenLayers] {
		cnn.link[cnn.hiddenLayers][i].wgt = 2.0 * (rand.Float64() - .5) / float64(cnn.layerDepth)
		cnn.link[cnn.hiddenLayers][i].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(cnn.layerDepth)
	}

	// hidden layers
	for lay := 1; lay < len(cnn.link)-1; lay++ {
		for link := 0; link < len(cnn.link[lay]); link++ {
			cnn.link[lay][link].wgt = 2.0 * (rand.Float64() - .5) / float64(cnn.layerDepth)
			cnn.link[lay][link].wgtDelta = 2.0 * (rand.Float64() - .5) / float64(cnn.layerDepth)
		}
	}
	for n := 0; n < cnn.epochs; n++ {
		//fmt.Printf("epoch %d\n", n)
		// Loop over the training examples
		for _, samp := range cnn.samples {
			// Forward Propagation
			err := cnn.propagateForward(samp)
			if err != nil {
				return fmt.Errorf("forward propagation error: %s", err.Error())
			}

			// Backward Propagation
			err = cnn.propagateBackward()
			if err != nil {
				return fmt.Errorf("backward propagation error: %s", err.Error())
			}
		}

		// At the end of each epoch, loop over the output nodes and calculate mse
		// This is the so-called Learning Curve
		cnn.calculateMSE(n)

		// Shuffle training examples to train in a different sample order
		rand.Shuffle(len(cnn.samples), func(i, j int) {
			cnn.samples[i], cnn.samples[j] = cnn.samples[j], cnn.samples[i]
		})

	}

	return nil
}

// init parses the html template files
func init() {
	tmplTrainingCNN = template.Must(template.ParseFiles(fileTrainingCNN))
	tmplTestingCNN = template.Must(template.ParseFiles(fileTestingCNN))
}

// createExamples creates a slice of training examples
func (cnn *CNN) createExamples() error {
	// read in encoded characters from encoded characters .csv file
	f, err := os.Open(path.Join(dataDir, filechars))
	if err != nil {
		fmt.Printf("Error opening file %s: %v\n", filechars, err.Error())
		return fmt.Errorf("open file %s error: %v", filechars, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	class := 0
	for scanner.Scan() {
		line := scanner.Text()
		items := strings.Split(line, ",")
		cnn.samples[class] = Sample{
			encChar:   make([]int8, len(items)-1),
			desired:   class,
			character: items[0],
		}
		for i, strval := range items[1:] {
			intval, err := strconv.Atoi(strval)
			if err != nil {
				fmt.Printf("String %s to int conversion error: %v\n", strval, err.Error())
				return fmt.Errorf("string %s to int conversion error: %v", strval, err.Error())
			}
			cnn.samples[class].encChar[i] = int8(intval)
		}
		class++
	}
	if err := scanner.Err(); err != nil {
		fmt.Printf("bufio.Scanner error: %v\n", err.Error())
		return fmt.Errorf("bufio.Scanner error: %v", err.Error())
	}
	return nil
}

// newCNN constructs an CNN instance for training
func newCNN(r *http.Request, hiddenLayers int, plot *PlotT) (*CNN, error) {
	// Read the training parameters in the HTML Form

	txt := r.FormValue("layerdepth")
	layerDepth, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("layerdepth int conversion error: %v\n", err)
		return nil, fmt.Errorf("layerdepth int conversion error: %s", err.Error())
	}

	txt = r.FormValue("learningrate")
	learningRate, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("learningrate float conversion error: %v\n", err)
		return nil, fmt.Errorf("learningrate float conversion error: %s", err.Error())
	}

	txt = r.FormValue("momentum")
	momentum, err := strconv.ParseFloat(txt, 64)
	if err != nil {
		fmt.Printf("momentum float conversion error: %v\n", err)
		return nil, fmt.Errorf("momentum float conversion error: %s", err.Error())
	}

	txt = r.FormValue("epochs")
	epochs, err := strconv.Atoi(txt)
	if err != nil {
		fmt.Printf("epochs int conversion error: %v\n", err)
		return nil, fmt.Errorf("epochs int conversion error: %s", err.Error())
	}

	cnn := CNN{
		rows:         300,
		columns:      300,
		hiddenLayers: hiddenLayers,
		layerDepth:   layerDepth,
		epochs:       epochs,
		learningRate: learningRate,
		momentum:     momentum,
		plot:         plot,
		Endpoints: Endpoints{
			ymin: math.MaxFloat64,
			ymax: -math.MaxFloat64,
			xmin: 0,
			xmax: float64(epochs - 1)},
		samples: make([]Sample, classes),
	}

	// construct link that holds the weights and weight deltas
	cnn.link = make([][]Link, hiddenLayers+1)

	// input layer
	cnn.link[0] = make([]Link, (imageSize+1)*layerDepth)

	// outer layer nodes
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer links
	cnn.link[len(cnn.link)-1] = make([]Link, olnodes*(layerDepth+1))

	// hidden layer links
	for i := 1; i < len(cnn.link)-1; i++ {
		cnn.link[i] = make([]Link, (layerDepth+1)*layerDepth)
	}

	// construct node, init node[i][0].y to 1.0 (bias)
	cnn.node = make([][]Node, hiddenLayers+2)

	// input layer
	cnn.node[0] = make([]Node, imageSize+1)
	// set first node in the layer (bias) to 1
	cnn.node[0][0].y = 1.0

	// output layer, which has no bias node
	cnn.node[hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= hiddenLayers; i++ {
		cnn.node[i] = make([]Node, layerDepth+1)
		// set first node in the layer (bias) to 1
		cnn.node[i][0].y = 1.0
	}

	// construct desired from classes, binary representation
	cnn.desired = make([]float64, olnodes)

	// mean-square error
	cnn.mse = make([]float64, epochs)

	return &cnn, nil
}

// gridFillInterp inserts the data points in the grid and draws a straight line between points
func (cnn *CNN) gridFillInterp() error {
	var (
		x            float64 = 0.0
		y            float64 = cnn.mse[0]
		prevX, prevY float64
		xscale       float64
		yscale       float64
	)

	// Mark the data x-y coordinate online at the corresponding
	// grid row/column.

	// Calculate scale factors for x and y
	xscale = float64(cnn.columns-1) / (cnn.xmax - cnn.xmin)
	yscale = float64(cnn.rows-1) / (cnn.ymax - cnn.ymin)

	cnn.plot.Grid = make([]string, cnn.rows*cnn.columns)

	// This cell location (row,col) is on the line
	row := int((cnn.ymax-y)*yscale + .5)
	col := int((x-cnn.xmin)*xscale + .5)
	cnn.plot.Grid[row*cnn.columns+col] = "online"

	prevX = x
	prevY = y

	// Scale factor to determine the number of interpolation points
	lenEPy := cnn.ymax - cnn.ymin
	lenEPx := cnn.xmax - cnn.xmin

	// Continue with the rest of the points in the file
	for i := 1; i < cnn.epochs; i++ {
		x++
		// ensemble average of the mse
		y = cnn.mse[i]

		// This cell location (row,col) is on the line
		row := int((cnn.ymax-y)*yscale + .5)
		col := int((x-cnn.xmin)*xscale + .5)
		cnn.plot.Grid[row*cnn.columns+col] = "online"

		// Interpolate the points between previous point and current point

		/* lenEdge := math.Sqrt((x-prevX)*(x-prevX) + (y-prevY)*(y-prevY)) */
		lenEdgeX := math.Abs((x - prevX))
		lenEdgeY := math.Abs(y - prevY)
		ncellsX := int(float64(cnn.columns) * lenEdgeX / lenEPx) // number of points to interpolate in x-dim
		ncellsY := int(float64(cnn.rows) * lenEdgeY / lenEPy)    // number of points to interpolate in y-dim
		// Choose the biggest
		ncells := ncellsX
		if ncellsY > ncells {
			ncells = ncellsY
		}

		stepX := (x - prevX) / float64(ncells)
		stepY := (y - prevY) / float64(ncells)

		// loop to draw the points
		interpX := prevX
		interpY := prevY
		for i := 0; i < ncells; i++ {
			row := int((cnn.ymax-interpY)*yscale + .5)
			col := int((interpX-cnn.xmin)*xscale + .5)
			cnn.plot.Grid[row*cnn.columns+col] = "online"
			interpX += stepX
			interpY += stepY
		}

		// Update the previous point with the current point
		prevX = x
		prevY = y
	}
	return nil
}

// insertLabels inserts x- an y-axis labels in the plot
func (cnn *CNN) insertLabels() {
	cnn.plot.Xlabel = make([]string, xlabels)
	cnn.plot.Ylabel = make([]string, ylabels)
	// Construct x-axis labels
	incr := (cnn.xmax - cnn.xmin) / (xlabels - 1)
	x := cnn.xmin
	// First label is empty for alignment purposes
	for i := range cnn.plot.Xlabel {
		cnn.plot.Xlabel[i] = fmt.Sprintf("%.2f", x)
		x += incr
	}

	// Construct the y-axis labels
	incr = (cnn.ymax - cnn.ymin) / (ylabels - 1)
	y := cnn.ymin
	for i := range cnn.plot.Ylabel {
		cnn.plot.Ylabel[i] = fmt.Sprintf("%.2f", y)
		y += incr
	}
}

// handleTraining performs forward and backward propagation to calculate the weights
func handleTrainingCNN(w http.ResponseWriter, r *http.Request) {

	var (
		plot PlotT
		cnn  *CNN
	)

	// Get the number of hidden layers
	txt := r.FormValue("hiddenlayers")
	// Need hidden layers to continue
	if len(txt) > 0 {
		hiddenLayers, err := strconv.Atoi(txt)
		if err != nil {
			fmt.Printf("Hidden Layers int conversion error: %v\n", err)
			plot.Status = fmt.Sprintf("Hidden Layers conversion to int error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// create CNN instance to hold state
		cnn, err = newCNN(r, hiddenLayers, &plot)
		if err != nil {
			fmt.Printf("newMLP() error: %v\n", err)
			plot.Status = fmt.Sprintf("newMLP() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Create training examples by reading in the encoded characters
		err = cnn.createExamples()
		if err != nil {
			fmt.Printf("createExamples error: %v\n", err)
			plot.Status = fmt.Sprintf("createExamples error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Loop over the Epochs
		err = cnn.runEpochs()
		if err != nil {
			fmt.Printf("runEpochs() error: %v\n", err)
			plot.Status = fmt.Sprintf("runEpochs() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// Put MSE vs Epoch in PlotT, the Learning Curve
		err = cnn.gridFillInterp()
		if err != nil {
			fmt.Printf("gridFillInterp() error: %v\n", err)
			plot.Status = fmt.Sprintf("gridFillInterp() error: %v", err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}

		// insert x-labels and y-labels in PlotT
		cnn.insertLabels()

		// At the end of all epochs, insert form previous control items in PlotT
		cnn.plot.HiddenLayers = strconv.Itoa(cnn.hiddenLayers)
		cnn.plot.LayerDepth = strconv.Itoa(cnn.layerDepth)
		cnn.plot.Classes = strconv.Itoa(classes)
		cnn.plot.LearningRate = strconv.FormatFloat(cnn.learningRate, 'f', 3, 64)
		cnn.plot.Momentum = strconv.FormatFloat(cnn.momentum, 'f', 3, 64)
		cnn.plot.Epochs = strconv.Itoa(cnn.epochs)

		// Save hidden layers, hidden layer depth, and weights to csv file, one layer per line
		f, err := os.Create(path.Join(dataDir, fileweights))
		if err != nil {
			fmt.Printf("os.Create() file %s error: %v\n", path.Join(fileweights), err)
			plot.Status = fmt.Sprintf("os.Create() file %s error: %v", path.Join(fileweights), err.Error())
			// Write to HTTP using template and grid
			if err := tmplTrainingCNN.Execute(w, plot); err != nil {
				log.Fatalf("Write to HTTP output using template with error: %v\n", err)
			}
			return
		}
		defer f.Close()
		fmt.Fprintf(f, "%d,%d\n",
			cnn.hiddenLayers, cnn.layerDepth)
		for _, layer := range cnn.link {
			for _, node := range layer {
				fmt.Fprintf(f, "%f,", node.wgt)
			}
			fmt.Fprintln(f)
		}

		cnn.plot.Status = "Learning Curve plotted: MSE vs Epoch"

		// Execute data on HTML template
		if err = tmplTrainingCNN.Execute(w, cnn.plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
	} else {
		plot.Status = "Enter Convolutional Neural Network (CNN) training parameters."
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}

	}
}

// Classify test examples and display test results
func (cnn *CNN) runClassification() error {
	// Loop over the training examples
	cnn.plot.Grid = make([]string, cnn.rows*cnn.columns)
	cnn.statistics =
		Stats{correct: make([]int, classes), classCount: make([]int, classes)}

	const (
		startRow  = 1  // padding
		startCol  = 1  // padding
		imageSize = 81 // 9 x 9 cells (2px)
	)
	endRow := cnn.rows
	endCol := cnn.columns
	imageRow := int(math.Sqrt(imageSize))
	imageCol := int(math.Sqrt(imageSize))
	stride := int(math.Sqrt(imageSize))
	charNumber := 0
	sample := Sample{
		desired: 0,
		encChar: make([]int8, imageSize),
	}

	// loop over rows, start at 1, end at cnn.rows, stride
	for row := startRow; row < endRow; row += stride {
		// loop over columns, start at 1, end at cnn.columns, stride
		for col := startCol; col < endCol; col += stride {
			current := row*cnn.columns + col
			// insert encoded character in a Sample
			k := 0
			for i := 0; i < imageRow; i++ {
				for j := 0; j < imageCol; j++ {
					sample.desired = cnn.charTest[charNumber]
					sample.encChar[k] = cnn.grid[current+k]
					k++
				}
				current += cnn.columns
			}
			// Forward Propagation
			err := cnn.propagateForward(sample)
			if err != nil {
				return fmt.Errorf("forward propagation error: %s", err.Error())
			}
			// At output layer, classify example, increment class count, %correct
			// Convert node output y to class
			err = cnn.determineClass(sample)
			if err != nil {
				return fmt.Errorf("determineClass error: %s", err.Error())
			}
			charNumber++
		}
	}

	cnn.plot.TestResults = make([]Results, classes)

	totalCount := 0
	totalCorrect := 0
	// tabulate TestResults by converting numbers to string in Results
	for i := range cnn.plot.TestResults {
		totalCount += cnn.statistics.classCount[i]
		totalCorrect += cnn.statistics.correct[i]
		cnn.plot.TestResults[i] = Results{
			Class:     strconv.Itoa(i),
			Character: cnn.samples[i].character,
			Count:     strconv.Itoa(cnn.statistics.classCount[i]),
			Correct:   strconv.Itoa(cnn.statistics.correct[i] * 100 / cnn.statistics.classCount[i]),
		}
	}
	cnn.plot.TotalCount = strconv.Itoa(totalCount)
	cnn.plot.TotalCorrect = strconv.Itoa(totalCorrect * 100 / totalCount)

	cnn.plot.HiddenLayers = strconv.Itoa(cnn.hiddenLayers)
	cnn.plot.LayerDepth = strconv.Itoa(cnn.layerDepth)
	cnn.plot.Classes = strconv.Itoa(classes)

	cnn.plot.Status = "Results completed."

	return nil
}

// drawCharacters draws the alpha-numeric characters that are in the grid.
// The grid is 300x300 cells, each cell is 2px.  Each character occupies
// 9x9=81 cells.  There are 33*33=1089 characters in the grid.
func (cnn *CNN) drawCharacters() error {

	const (
		startRow  = 1  // padding
		startCol  = 1  // padding
		imageSize = 81 // 9 x 9 cells (2px per cell)
	)
	endRow := cnn.rows
	endCol := cnn.columns
	imageRow := int(math.Sqrt(imageSize))
	imageCol := int(math.Sqrt(imageSize))
	stride := int(math.Sqrt(imageSize))

	// loop over rows, start at 1, end at cnn.rows, stride
	for row := startRow; row < endRow; row += stride {
		// loop over columns, start at 1, end at cnn.columns, stride
		for col := startCol; col < endCol; col += stride {
			// insert this character in TestResults
			current := row*cnn.columns + col
			k := 0
			for i := 0; i < imageRow; i++ {
				for j := 0; j < imageCol; j++ {
					// This cell is part of the character
					if cnn.grid[current+k] == 1 {
						cnn.plot.Grid[current+j] = "online"
					}
					k++
				}
				current += cnn.columns
			}
		}
	}
	return nil
}

// createExamplesTesting creates testing examples
func (cnn *CNN) createExamplesTesting() error {

	// get the training examples using the encoded characters
	cnn.createExamples()

	const (
		startRow  = 1  // padding
		startCol  = 1  // padding
		imageSize = 81 // 9 x 9 cells (2px)
	)
	endRow := cnn.rows
	endCol := cnn.columns
	imageRow := int(math.Sqrt(imageSize))
	imageCol := int(math.Sqrt(imageSize))
	stride := int(math.Sqrt(imageSize))
	charNumber := 0

	// loop over rows, start at 1, end at cnn.rows, stride
	for row := startRow; row < endRow; row += stride {
		// loop over columns, start at 1, end at cnn.columns, stride
		for col := startCol; col < endCol; col += stride {
			current := row*cnn.columns + col
			// insert random encoded character in the grid
			k := 0
			for i := 0; i < imageRow; i++ {
				for j := 0; j < imageCol; j++ {
					// randomly choose a char (class) and insert the encoded char
					class := rand.Intn(classes)
					cnn.charTest[charNumber] = class
					cnn.grid[current+k] = int8(cnn.samples[class].encChar[k])
					k++
				}
				current += cnn.columns
			}
			charNumber++
		}
	}
	fmt.Printf("Test characters generated = %d\n", charNumber)

	return nil
}

// newTestingCNN constructs a CNN from the saved cnn weights and parameters
func newTestingCNN(plot *PlotT) (*CNN, error) {
	// Read in weights from csv file, ordered by layers, and CNN parameters
	f, err := os.Open(path.Join(dataDir, fileweights))
	if err != nil {
		fmt.Printf("Open file %s error: %v", fileweights, err)
		return nil, fmt.Errorf("open file %s error: %s", fileweights, err.Error())
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	// get the parameters
	scanner.Scan()
	line := scanner.Text()

	items := strings.Split(line, ",")
	hiddenLayers, err := strconv.Atoi(items[0])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v", items[0], err)
		return nil, err
	}
	hidLayersDepth, err := strconv.Atoi(items[1])
	if err != nil {
		fmt.Printf("Conversion to int of %s error: %v", items[1], err)
		return nil, err
	}

	// construct the cnn
	cnn := CNN{
		rows:         300,
		columns:      300,
		hiddenLayers: hiddenLayers,
		layerDepth:   hidLayersDepth,
		plot:         plot,
		samples:      make([]Sample, classes),
		charTest:     make([]int, testingSamples),
		grid:         make([]int8, 300*300),
	}

	// retrieve the weights
	rows := 0
	for scanner.Scan() {
		rows++
		line = scanner.Text()
		weights := strings.Split(line, ",")
		weights = weights[:len(weights)-1]
		temp := make([]Link, len(weights))
		for i, wtStr := range weights {
			wt, err := strconv.ParseFloat(wtStr, 64)
			if err != nil {
				fmt.Printf("ParseFloat of %s error: %v", wtStr, err)
				continue
			}
			temp[i] = Link{wgt: wt, wgtDelta: 0}
		}
		cnn.link = append(cnn.link, temp)
	}
	if err = scanner.Err(); err != nil {
		fmt.Printf("scanner error: %s", err.Error())
	}

	fmt.Printf("hidden layer depth = %d, hidden layers = %d, classes = %d\n",
		cnn.layerDepth, cnn.hiddenLayers, classes)

	// construct node, init node[i][0].y to 1.0 (bias)
	cnn.node = make([][]Node, cnn.hiddenLayers+2)

	// input layer
	cnn.node[0] = make([]Node, imageSize+1)
	// set first node in the layer (bias) to 1
	cnn.node[0][0].y = 1.0

	// outer layer nodes
	olnodes := int(math.Ceil(math.Log2(float64(classes))))

	// output layer, which has no bias node
	cnn.node[cnn.hiddenLayers+1] = make([]Node, olnodes)

	// hidden layers
	for i := 1; i <= cnn.hiddenLayers; i++ {
		cnn.node[i] = make([]Node, cnn.layerDepth+1)
		// set first node in the layer (bias) to 1
		cnn.node[i][0].y = 1.0
	}

	// construct desired from classes, binary representation
	cnn.desired = make([]float64, olnodes)

	return &cnn, nil
}

// handleTesting performs pattern classification of the test data
func handleTestingCNN(w http.ResponseWriter, r *http.Request) {
	var (
		plot PlotT
		cnn  *CNN
		err  error
	)
	// Construct CNN instance containing CNN state
	cnn, err = newTestingCNN(&plot)
	if err != nil {
		fmt.Printf("newTestingMLP() error: %v\n", err)
		plot.Status = fmt.Sprintf("newTestingMLP() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Create testing examples by reading in the encoded characters and distributing them
	// randomly in the 300 x 300 grid
	err = cnn.createExamplesTesting()
	if err != nil {
		fmt.Printf("createExamplesTesting error: %v\n", err)
		plot.Status = fmt.Sprintf("createExamplesTesting error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// At end of all examples tabulate TestingResults
	// Convert numbers to string in Results
	err = cnn.runClassification()
	if err != nil {
		fmt.Printf("runClassification() error: %v\n", err)
		plot.Status = fmt.Sprintf("runClassification() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Draw the characters to show what is being classified
	err = cnn.drawCharacters()
	if err != nil {
		fmt.Printf("drawCharacters() error: %v\n", err)
		plot.Status = fmt.Sprintf("drawCharacters() error: %v", err.Error())
		// Write to HTTP using template and grid
		if err := tmplTrainingCNN.Execute(w, plot); err != nil {
			log.Fatalf("Write to HTTP output using template with error: %v\n", err)
		}
		return
	}

	// Execute data on HTML template
	if err = tmplTestingCNN.Execute(w, cnn.plot); err != nil {
		log.Fatalf("Write to HTTP output using template with error: %v\n", err)
	}
}

// executive creates the HTTP handlers, listens and serves
func main() {
	// Set up HTTP servers with handlers for training and testing the CNN Neural Network

	// Create HTTP handler for training
	http.HandleFunc(patternTrainingCNN, handleTrainingCNN)
	// Create HTTP handler for testing
	http.HandleFunc(patternTestingCNN, handleTestingCNN)
	fmt.Printf("Convolutional Neural Network Server listening on %v.\n", addr)
	http.ListenAndServe(addr, nil)
}
