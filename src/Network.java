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

    static Layer createLayer(int numNeurons, int numWeights, double learnRate, boolean shouldXavInit) {
        if (numNeurons < 1) {
            throw new IllegalArgumentException(String.format("Cannot have < 1 (%d) neuron in a layer", numNeurons));
        } else if (numWeights < 1) {
            throw new IllegalArgumentException(String.format("Cannot have < 1 (%d) input to a neuron", numWeights));
        } else if (!(learnRate > 0 && learnRate <= 10.0)) {
            throw new IllegalArgumentException(String.format("Learning rate should be in range (0, 10] (%f)", learnRate));
        }
        Random randGen = new Random();
        double xavier_init = shouldXavInit ? Math.sqrt(6.0 / (numNeurons + numWeights)) : 1.0;
        double[][] W_init = new double[numWeights][numNeurons];
        for (int row = 0; row < numWeights; ++row) {
            Arrays.setAll(W_init[row], i -> randGen.nextGaussian() * xavier_init);
        }
        double[][] Wt_init = Matrix.transpose(W_init, numWeights, numNeurons);

        double[] B_init = new double[numNeurons];
        Arrays.fill(B_init, 0.0);
        return new Layer(numNeurons, numWeights, W_init, Wt_init, B_init, learnRate);
    }
}

class ScaleMethods implements Serializable {

    private static final long serialVersionUID = 2L;
    private double MEAN;
    private double STD;
    private double SCALE_FACTOR;
    private String scalingMethod;

    String getScalingMethod() {
        return scalingMethod;
    }

    double getMEAN() {
        return MEAN;
    }

    double getSCALE_FACTOR() {
        return SCALE_FACTOR;
    }

    double getSTD() {
        return STD;
    }

    double identity(double num) {
        return num;
    }

    double standardize(double num) {
        return ((num - this.MEAN) / this.STD);
    }

    double normalize(double num) {
        return (num * this.SCALE_FACTOR);
    }

    double centre(double num) {
        return (num - this.MEAN);
    }

    ScaleMethods(double mean, double std) {
        this.MEAN = mean;
        this.STD = std;
        this.scalingMethod = "standardize";
    }

    ScaleMethods(double val, boolean isScaleFactor) {
        if (isScaleFactor) {
            this.SCALE_FACTOR = val;
            this.scalingMethod = "normalize";
        } else {
            this.MEAN = val;
            this.scalingMethod = "centre";
        }
    }

    ScaleMethods() {
        this.scalingMethod = "identity";
    }
}

abstract class Scaler implements Serializable {

    private static final long serialVersionUID = 2L;
    private double mean;
    private double std;
    private double scaleFactor;
    private String scaleMethodType;

    double getMean() {
        return mean;
    }

    double getScaleFactor() {
        return scaleFactor;
    }

    double getStd() {
        return std;
    }

    String getScaleMethodType() {
        return scaleMethodType;
    }

    void setMean(double mean) {
        this.mean = mean;
    }

    void setScaleFactor(double scaleFactor) {
        this.scaleFactor = scaleFactor;
    }

    void setScaleMethodType(String scaleMethodType) {
        this.scaleMethodType = scaleMethodType;
    }

    void setStd(double std) {
        this.std = std;
    }

    abstract double scale(double num);
}

class Identity extends Scaler {

    Identity() {
        setScaleMethodType("identity");
    }

    @Override
    double scale(double num) {
        return num;
    }
}

class Normalizer extends Scaler {

    Normalizer(double scaleFactor) {
        setScaleFactor(scaleFactor);
        setScaleMethodType("normalize");
    }

    @Override
    double scale(double num) {
        return (num * getScaleFactor());
    }
}

class Standardizer extends Scaler {

    Standardizer(double mean, double std) {
        setMean(mean);
        setStd(std);
        setScaleMethodType("standardize");
    }

    @Override
    double scale(double num) {
        return ((num - getMean()) / getStd());
    }
}

class Centralizer extends Scaler {

    Centralizer(double mean) {
        setMean(mean);
        setScaleMethodType("centralize");
    }

    @Override
    double scale(double num) {
        return (num - getMean());
    }
}

class Network implements Serializable {

    private static final long serialVersionUID = 3L;
    private int numLayers;
    private Loss lossType;
    private ActivFunc[] activFuncType;
    private Layer[] layers;
    private File datasetFile;
    private Scaler scaler;

    private Network(int numLayers, Layer[] layers, Loss lossType, ActivFunc[] activFuncType, File datasetFile, Scaler scaler) {
        this.numLayers = numLayers;
        this.layers = layers;
        this.lossType = lossType;
        this.activFuncType = activFuncType;
        this.datasetFile = datasetFile;
        this.scaler = scaler;
    }

    void setDatasetFile(File datasetFile) {
        this.datasetFile = datasetFile;
    }

    void train(int epochs) throws IOException {

        if (!datasetFile.exists()) {
            System.out.println("Dataset does not exist!");
        } else {
            double[] in = new double[layers[0].getNumWeights()];
            double[] actual = new double[layers[numLayers - 1].getNumNeurons()];
            double loss;

            File[] trainFiles = datasetFile.listFiles();
            if (trainFiles.length == 0) {
                System.out.println("Dataset does not have any training examples!");
            } else {
                int numFiles = trainFiles.length;
                int index;
                File tempFile;
                Random randGen = new Random();

                for (int epoch = 0; epoch < epochs; ++epoch) {
                    for (int file = numFiles - 1; file >= 0; --file) {
                        index = randGen.nextInt(file + 1);
                        processSample(trainFiles[index], in, actual);
                        loss = learn(in, actual);
                        System.out.printf("\rEpoch (%d/%d): Trained files (%d/%d) Loss : %.7f", epoch + 1, epochs, numFiles - file, numFiles, loss);
                        tempFile = trainFiles[index];
                        trainFiles[index] = trainFiles[file];
                        trainFiles[file] = tempFile;
                    }
                }
                System.out.println();
            }
        }
    }

    void test(String inFileName) throws IOException {

        File datasetFile;

        if ("".equals(inFileName)) {
            datasetFile = this.datasetFile;
        } else {
            datasetFile = new File(inFileName);
        }

        if (!datasetFile.exists()) {
            System.out.println("Dataset does not exist!");
        } else {
            double[] in = new double[784];
            if (datasetFile.isFile()) {
                readSample(datasetFile, in);
                int pred = displayAns(in);
                System.out.printf("This number is %d\n", pred);
            } else {
                File[] trainFiles = datasetFile.listFiles();
                if (trainFiles.length == 0) {
                    System.out.println("Dataset does not have any training examples!");
                } else {
                    int numFiles = trainFiles.length;
                    int numCorrect = 0;
                    for (int file = 0; file < numFiles; ++file) {
                        int ans = readSample(trainFiles[file], in);
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

    private int readSample(File file, double[] in) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            for (int row = 0; row < 28; ++row) {
                double[] arrTemp = Arrays.stream(reader.readLine().split("\t")).mapToDouble(num -> scaler.scale(Double.parseDouble(num))).toArray();
                System.arraycopy(arrTemp, 0, in, 28 * row, 28);
            }
            return Integer.parseInt(reader.readLine());
        } catch (FileNotFoundException e) {
            throw new FileNotFoundException();
        }
    }

    private void processSample(File file, double[] in, double[] actual) throws IOException {
        int ans = readSample(file, in);
        oneHotEncode(ans, actual);
    }

    private static void oneHotEncode(int ans, double[] arr) {
        assert (ans < arr.length && ans >= 0) : String.format("Invalid answer (%s) for training example with output vector of length %d", ans, arr.length);
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

    private int displayAns(double[] in) {

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

    static Network createNetwork(int[] layerSizes, Loss lossType, ActivFunc[] activFuncType, int numInputs, double learnRate, String inputDataPath, int scaleMethodChoice, double scaleFactor) throws IOException {
        int numLayers = layerSizes.length;
        if (numLayers < 1) {
            System.out.printf("Cannot have < 1 layer (%d) in network\n", numLayers);
            return null;
        } else if (numInputs < 1) {
            System.out.printf("Cannot have < 1 input (%d) to network\n", numInputs);
            return null;
        } else {
            boolean shouldXavInit = scaleMethodChoice == 3;
            Layer[] layers_init = new Layer[numLayers];
            layers_init[0] = Layer.createLayer(layerSizes[0], numInputs, learnRate, shouldXavInit);
            for (int layer = 1; layer < numLayers; ++layer) {
                layers_init[layer] = Layer.createLayer(layerSizes[layer], layerSizes[layer - 1], learnRate, shouldXavInit);
            }

            ScaleMethods scaleMethod;
            File inputDataFile = new File(inputDataPath);
            Scaler scaler;

            if (scaleMethodChoice == 0) {
                scaler = new Identity();
            } else if (scaleMethodChoice == 1) {
                scaler = new Normalizer(scaleFactor);
            } else {
                if (inputDataFile.exists() && inputDataFile.isDirectory()) {
                    File[] inputDataSampArr = inputDataFile.listFiles();
                    int numFiles = inputDataSampArr.length;
                    if (numFiles == 0) {
                        System.out.println("Dataset has no training samples!");
                        return null;
                    } else {
                        double mean = findMean(inputDataSampArr);
                        if (scaleMethodChoice == 3) {
                            double std = findSTD(mean, inputDataSampArr);
                            scaler = new Standardizer(mean, std);
                        } else {
                            scaler = new Centralizer(mean);
                        }
                    }
                } else {
                    System.out.println("Dataset is a file/non-existent!");
                    return null;
                }
            }
            return new Network(numLayers, layers_init, lossType, activFuncType, inputDataFile, scaler);
        }
    }

    private static double findMean(File ... filePaths) throws IOException {
        int numFiles = filePaths.length;
        double temp = 0.0;
        double avg = 0.0;
        double total = numFiles * 784;

        for (int file = 0; file < numFiles; ++file) {
            try (BufferedReader reader = new BufferedReader(new FileReader(filePaths[file]))) {
                for (int row = 0; row < 28; ++row) {
                    temp += Arrays.stream(reader.readLine().split("\t")).mapToDouble(Double::parseDouble).sum();
                }
            } catch (FileNotFoundException e) {
                throw new FileNotFoundException(String.format("%s", e.toString()));
            }
            System.out.printf("\rCalculating parameters [MEAN] (%d/%d) files", file + 1, numFiles);
            avg += temp / total;
            temp = 0.0;
        }
        System.out.println("\rCompleted calculation [MEAN]");
        return avg;
    }

    private static double findSTD(double mean, File ... filePaths) throws IOException {
        int numFiles = filePaths.length;
        double temp = 0.0;
        double avg = 0.0;
        double total = numFiles * 784;

        for (int file = 0; file < numFiles; ++file) {
            try (BufferedReader reader = new BufferedReader(new FileReader(filePaths[file]))) {
                for (int row = 0; row < 28; ++row) {
                    temp += Arrays.stream(reader.readLine().split("\t")).mapToDouble(num -> Math.pow(Double.parseDouble(num) - mean, 2)).sum();
                }
            } catch (FileNotFoundException e) {
                throw new FileNotFoundException();
            }
            System.out.printf("\rCalculating parameters [STD] (%d/%d) files", file + 1, numFiles);
            avg += temp / total;
            temp = 0.0;
        }
        System.out.println("\rCompleted calculation [STD]");
        return Math.sqrt(avg);
    }

    @Override
    public String toString() {
        StringBuilder string = new StringBuilder();
        string.append(String.format("Learning rate: %.3f\nLoss function: %s\n", layers[0].getLearnRate(), lossType.name()));
        switch (scaler.getScaleMethodType()) {
            case "identity":
                string.append("Scaling Methods: None used!\nParameters of dataset: None calculated\n");
                break;
            case "normalize":
                string.append(String.format("Scaling Methods: Normalization [SCALE FACTOR = %.6f]\nParameters of dataset: None calculated\n", scaler.getScaleFactor()));
                break;
            case "centralize":
                string.append(String.format("Scaling Methods: Centralizing\nParameters of dataset: [MEAN = %.6f]\n", scaler.getMean()));
                break;
            case "standardize":
                string.append(String.format("Scaling Methods: Standardizing\nParameters of dataset: [MEAN = %.6f] [STD = %.6f]\n", scaler.getMean(), scaler.getStd()));
                break;
            default:
                break;
        }
        for (int layer = 0; layer < numLayers; ++layer) {
            string.append(String.format("Layer %d: %d -> %d -> %s\n", layer + 1, layers[layer].getNumWeights(), layers[layer].getNumNeurons(), activFuncType[layer].name()));
        }
        return string.toString();
    }
}
