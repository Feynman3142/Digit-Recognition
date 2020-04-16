import java.io.*;
import java.util.Arrays;
import java.util.Random;

enum Loss {
    MSE,
    CROSS_ENTROPY;

    static double getLossOf(Loss loss, double[] actual, double[] pred) {
        double res = 0.0;
        switch(loss) {
            case MSE:
                for (int elem = 0; elem < actual.length; ++elem) {
                    res += Math.pow(pred[elem] - actual[elem], 2);
                }
                return res;
            case CROSS_ENTROPY:
                for (int elem = 0; elem < actual.length; ++elem) {
                    res += actual[elem] * Math.log10(pred[elem]);
                }
                res = -1 * res;
                return res;
            default:
                throw new IllegalArgumentException(String.format("Loss function does not exist / not yet supported!. Please try %s", values().toString()));
        }
    }

    static double[] getDerivLossOf(Loss loss, double[] actual, double[] pred) {
        int numNeurons = actual.length;
        double[] res = new double[numNeurons];
        switch(loss) {
            case MSE:
                for (int elem  = 0; elem < numNeurons; ++elem) {
                    res[elem] = 2 * (pred[elem] - actual[elem]);
                }
                return res;
            case CROSS_ENTROPY:
                double constant = Math.log10(Math.E);
                for (int elem = 0; elem < numNeurons; ++elem) {
                    res[elem] = (pred[elem] - actual[elem]) * constant;//(-1) * actual[elem] * (1.0 / pred[elem]) * constant;
                }
                return res;
            default:
                throw new IllegalArgumentException(String.format("Loss function does not exist / not yet supported!. Please try %s", values().toString()));
        }
    }
}

enum ActivFunc {
    SIGMOID,
    SOFTMAX,
    RELU,
    LEAKY_RELU;

    static double[] getActivFuncOf(ActivFunc activFunc, double[] in) {
        int numElems = in.length;
        double[] res = new double[numElems];
        switch(activFunc) {
            case SIGMOID:
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = 1.0 / (1 + Math.exp(-in[elem]));
                }
                return res;
            case SOFTMAX:
                double maxElem = getMaxOf(in);
                //double[] shiftedIn = new double[numElems];
                double[] expShiftedIn = new double[numElems];
                double sumElems = 0.0;
                for (int elem = 0; elem < numElems; ++elem) {
                    //shiftedIn[elem] = in[elem] - maxElem;
                    expShiftedIn[elem] = Math.exp(in[elem] - maxElem);
                    sumElems += expShiftedIn[elem];
                }
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = expShiftedIn[elem] / sumElems;
                }
                return res;
            case LEAKY_RELU:
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = in[elem] > 0.0 ? in[elem] : (0.01 * in[elem]);
                }
                return res;
            case RELU:
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = in[elem] > 0.0 ? in[elem] : 0.0;
                }
                return res;
            default:
                throw new IllegalArgumentException(String.format("Loss function does not exist / not yet supported!. Please try %s", values().toString()));
        }
    }

    static double[] getDerivActivFuncOf(ActivFunc activFunc, double[] in) {
        int numElems = in.length;
        double[] res = new double[numElems];
        double[] funcOut;
        switch(activFunc) {
            case SIGMOID:
                funcOut = getActivFuncOf(SIGMOID, in);
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = funcOut[elem] * (1 - funcOut[elem]);
                }
                return res;
            case SOFTMAX:
                funcOut = getActivFuncOf(SOFTMAX, in);
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = funcOut[elem] * (1 - funcOut[elem]);
                }
                return res;
            case LEAKY_RELU:
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = in[elem] > 0.0 ? 1.0 : 0.01;
                }
                return res;
            case RELU:
                for (int elem = 0; elem < numElems; ++elem) {
                    res[elem] = in[elem] > 0.0 ? in[elem] : 0.0;
                }
                return res;
            default:
                throw new IllegalArgumentException(String.format("Loss function does not exist / not yet supported!. Please try %s", values().toString()));
        }
    }

    static double getMaxOf(double[] in) {
        double max = in[0];
        for (int elem = 1; elem < in.length; ++elem) {
            if (in[elem] > max) {
                max = in[elem];
            }
        }
        return max;
    }
}

class Layer implements Serializable {

    private static final long serialVersionUID = 1L;
    private int numNeurons;
    private int numWeights;
    private double[][] W;
    private double[][] Wt;
    private double[] B;
    private double learnRate;

    private Layer(int numNeurons, int numWeights, double[][] W, double[][] Wt, double[] B, double learnRate) {
        this.numNeurons = numNeurons;
        this.numWeights = numWeights;
        this.W = W;
        this.Wt = Wt;
        this.B = B;
        this.learnRate = learnRate;
    }

    int getNumNeurons() {
        return numNeurons;
    }

    int getNumWeights() {
        return numWeights;
    }

    double[][] getW() {
        return W;
    }

    double[][] getWt() {
        return Wt;
    }

    double[] getB() {
        return B;
    }

    double getLearnRate() {
        return learnRate;
    }

    double[] feedforward(double[] X){
        return Matrix.add(Matrix.makeVector(Matrix.multiply(X, W, numWeights, numNeurons), 1, numNeurons), B);
    }

    double[] backpropagate(double[] dEdY, double[] X) {
        double[] dEdB = dEdY;
        double[][] dEdW = Matrix.multiply(X, dEdY);
        W = Matrix.subtract(W, Matrix.multiplyScalar(dEdW, numWeights, numNeurons, learnRate), numWeights, numNeurons);
        Wt = Matrix.transpose(W, numWeights, numNeurons);
        B = Matrix.subtract(B, Matrix.multiplyScalar(dEdB, learnRate));
        double[] dEdX = Matrix.makeVector(Matrix.multiply(dEdY, Wt, numNeurons, numWeights), 1, numWeights);
        return dEdX;
    }

    static Layer createRandomGaussianLayer(int numNeurons, int numWeights, double learnRate) {
        if (numNeurons < 1) {
            throw new IllegalArgumentException(String.format("Cannot have < 1 (%d) neuron in a layer", numNeurons));
        } else if (numWeights < 1) {
            throw new IllegalArgumentException(String.format("Cannot have < 1 (%d) input to a neuron", numWeights));
        } else if (!(learnRate > 0 && learnRate <= 10.0)) {
            throw new IllegalArgumentException(String.format("Learning rate should be in range (0, 10] (%f)", learnRate));
        }
        Random randGen = new Random();
        double[][] W_init = new double[numWeights][numNeurons];
        for (int row = 0; row < numWeights; ++row) {
            Arrays.setAll(W_init[row], i -> randGen.nextGaussian());
        }
        double[][] Wt_init = Matrix.transpose(W_init, numWeights, numNeurons);

        double[] B_init = new double[numNeurons];
        Arrays.setAll(B_init, i -> randGen.nextGaussian());
        return new Layer(numNeurons, numWeights, W_init, Wt_init, B_init, learnRate);
    }
}

class Network implements Serializable {

    private static final long serialVersionUID = 2L;
    private int numLayers;
    private Loss lossType;
    private ActivFunc[] activFuncType;
    private Layer[] layers;

    private Network(int numLayers, Layer[] layers, Loss lossType, ActivFunc[] activFuncType) {
        this.numLayers = numLayers;
        this.layers = layers;
        this.lossType = lossType;
        this.activFuncType = activFuncType;
    }

    int getNumLayers() {
        return numLayers;
    }

    Loss getLossType() {
        return lossType;
    }

    ActivFunc[] getActivFuncType() {
        return activFuncType;
    }

    Layer[] getLayers() {
        return layers;
    }

    void train(String inFileName, int epochs) throws IOException {

        boolean append = false;
        String outFileName = "results.txt";

        File datasetFile = new File(inFileName);

        if (!datasetFile.exists()) {
            System.out.println("Dataset does not exist!");
        } else {
            double[] in = new double[784];
            double[] actual = new double[10];
            double loss;
            int ans;

            if (datasetFile.isFile()) {
                ans = textToArr(datasetFile, in, 28, 28, true);
                oneHotEncode(ans, actual);
                loss = learn(in, actual);
                System.out.printf("Digit: %d | Loss: %.7f\n", ans, loss);
            } else {
                File[] trainFiles = datasetFile.listFiles();
                if (trainFiles.length == 0) {
                    System.out.println("Dataset does not have any training examples!");
                } else {
                    int numFiles = trainFiles.length;
                    int index;
                    File tempFile;
                    Random randGen = new Random();

                    for (int epoch = 0; epoch < epochs; ++epoch, append = true) {
//                        try (PrintWriter writer = new PrintWriter(new FileWriter(outFileName, append))) {
//                            writer.printf("Epoch %d:\n", epoch + 1);
                        for (int file = numFiles - 1; file >= 0; --file) {
                            index = randGen.nextInt(file + 1);
                            if (trainFiles[index].isDirectory()) {
                                continue;
                            }
                            ans = textToArr(trainFiles[index], in, 28, 28, true);
                            oneHotEncode(ans, actual);
                            loss = learn(in, actual);
//                                writer.printf("#%d %s %d %f\n", numFiles - file, trainFiles[index].getName(), ans, loss);
                            System.out.printf("\rEpoch (%d/%d): Trained files (%d/%d) Loss : %.7f", epoch + 1, epochs, numFiles - file, numFiles, loss);
                            tempFile = trainFiles[index];
                            trainFiles[index] = trainFiles[file];
                            trainFiles[file] = tempFile;
                        }
//                        } catch (FileNotFoundException e) {
//                            throw new FileNotFoundException();
//                        }
                    }
                    System.out.println();
                }
            }
        }
    }

    void test(String inFileName) throws IOException {

        File datasetFile = new File(inFileName);
        File[] trainFiles;

        if (!datasetFile.exists()) {
            System.out.println("Dataset does not exist!");
        } else {
            double[] in = new double[784];
            if (datasetFile.isFile()) {
                textToArr(datasetFile, in, 28, 28, false);
                int pred = displayAns(in);
                System.out.printf("This number is %d\n", pred);
            } else {
                trainFiles = datasetFile.listFiles();
                if (trainFiles.length == 0) {
                    System.out.println("Dataset does not have any training examples!");
                } else {
                    int numFiles = trainFiles.length;
                    int numCorrect = 0;
                    for (int file = 0; file < numFiles; ++file) {
                        int ans = textToArr(trainFiles[file], in, 28, 28, true);
                        int pred = displayAns(in);
                        if (pred == ans) {
                            ++numCorrect;
                        }
                        System.out.printf("\rThe network prediction accuracy: %d/%d, %.2f%%", numCorrect, file + 1, (numCorrect * 100) / (double) (file + 1));
                    }
                    System.out.println();
                }
            }
        }
    }

    private int textToArr(File file, double[] in, int height, int width, boolean readAns) throws IOException {
        int ans = -1;
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            for (int row = 0; row < height; ++row) {
                double[] arrTemp = Arrays.stream(reader.readLine().split("\t")).mapToDouble(num -> (Double.parseDouble(num) / 255.0)).toArray();
                System.arraycopy(arrTemp, 0, in, width * row, width);
            }
            if (readAns) {
                ans = Integer.parseInt(reader.readLine());
            }
        } catch (FileNotFoundException e) {
            throw new FileNotFoundException();
        } finally {
            return ans;
        }
    }

    private void oneHotEncode(int ans, double[] arr) {
        assert (ans < arr.length && ans >= 0) : "Invalid answer for training example";
        Arrays.fill(arr, 0.0);
        arr[ans] = 1.0;
    }

    private double learn(double[] in, double[] actual) {
        double[] inOut = in;
        double[][] outsLin = new double[numLayers][];
        double[][] outsActFunc = new double[numLayers][];

        for (int layer = 0; layer < numLayers; ++layer) {
            inOut = layers[layer].feedforward(inOut);
            outsLin[layer] = inOut;
            inOut = ActivFunc.getActivFuncOf(activFuncType[layer], inOut);
            outsActFunc[layer] = inOut;
        }

        double loss = Loss.getLossOf(lossType, actual, inOut);
        double[] dEdYActFunc = Loss.getDerivLossOf(lossType, actual, inOut);
        double[] dEdY;
        double[] X;
        for (int layer = numLayers - 1; layer > 0; --layer) {
            dEdY = (layer == numLayers - 1) ? dEdYActFunc : Matrix.multiplyElemWise(dEdYActFunc, ActivFunc.getDerivActivFuncOf(activFuncType[layer], outsLin[layer]));
            X = outsActFunc[layer - 1];
            dEdYActFunc = layers[layer].backpropagate(dEdY, X);
        }
        dEdY = Matrix.multiplyElemWise(dEdYActFunc, ActivFunc.getDerivActivFuncOf(activFuncType[0], outsLin[0]));
        X = in;
        layers[0].backpropagate(dEdY, X);

        return loss;
    }

    int displayAns(double[] in) {

        double[] inOut = in;

        for (int layer = 0; layer < numLayers; ++layer) {
            inOut = layers[layer].feedforward(inOut);
            inOut = ActivFunc.getActivFuncOf(activFuncType[layer], inOut);
        }

        int maxNeuron = 0;

        for (int neuron = 1; neuron < inOut.length; ++neuron) {
            if (inOut[neuron] > inOut[maxNeuron]) {
                maxNeuron = neuron;
            }
        }
        return maxNeuron;
    }

    static Network createRandomGaussianNetwork(int[] layerSizes, Loss lossType, ActivFunc[] activFuncType, int numInputs, double learnRate) {
        int numLayers = layerSizes.length;
        if (numLayers < 1) {
            throw new IllegalArgumentException(String.format("Cannot have < 1 layer (%d) in network", numLayers));
        }
        Layer[] layers_init = new Layer[numLayers];
        layers_init[0] = Layer.createRandomGaussianLayer(layerSizes[0], numInputs, learnRate);
        for (int layer = 1; layer < numLayers; ++layer) {
            layers_init[layer] = Layer.createRandomGaussianLayer(layerSizes[layer], layerSizes[layer - 1], learnRate);
        }
        return new Network(numLayers, layers_init, lossType, activFuncType);
    }
}
