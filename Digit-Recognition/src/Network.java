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
    SOFTMAX;

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
        String outFileName = "results-" + inFileName;
        for (int epoch = 0; epoch < epochs; ++epoch, append = true) {
            try (BufferedReader reader = new BufferedReader(new FileReader(inFileName))) {
                try (PrintWriter writer = new PrintWriter(new FileWriter(outFileName, append))) {
                    int sample = 1;
                    writer.printf("Epoch %d:\n", epoch);
                    while (reader.ready()) {
                        double[] in = Arrays.stream(reader.readLine().split("")).mapToDouble(ch -> ch.equals("X") ? 1.0 : 0.0).toArray();
                        double[] actual = Arrays.stream(reader.readLine().split(" ")).mapToDouble(Double::parseDouble).toArray();
                        double loss = learn(in, actual);
                        writer.printf("Sample %d: Loss = %f\n", sample, loss);
                        ++sample;
                    }
                    writer.println();
                } catch (FileNotFoundException e) {
                    System.out.printf("File: %s not found!", outFileName);
                }
            } catch (FileNotFoundException e) {
                System.out.printf("File: %s not found!", inFileName);
            }
        }
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

    double[] displayAns(double[] in) {
        double[] inOut = in;
        for (int layer = 0; layer < numLayers; ++layer) {
            inOut = layers[layer].feedforward(inOut);
            inOut = ActivFunc.getActivFuncOf(activFuncType[layer], inOut);
        }
        return inOut;
    }

    static Network createRandomGaussianNetwork(int[] layerSizes, Loss lossType, int numInputs, double learnRate) {
        int numLayers = layerSizes.length;
        if (numLayers < 1) {
            throw new IllegalArgumentException(String.format("Cannot have < 1 layer (%d) in network", numLayers));
        }
        Layer[] layers_init = new Layer[numLayers];
        layers_init[0] = Layer.createRandomGaussianLayer(layerSizes[0], numInputs, learnRate);
        for (int layer = 1; layer < numLayers; ++layer) {
            layers_init[layer] = Layer.createRandomGaussianLayer(layerSizes[layer], layerSizes[layer - 1], learnRate);
        }
        ActivFunc[] activFuncType = new ActivFunc[numLayers];
        Arrays.fill(activFuncType, 0, numLayers - 1, ActivFunc.SIGMOID);
        activFuncType[numLayers - 1] = ActivFunc.SOFTMAX;
        return new Network(numLayers, layers_init, lossType, activFuncType);
    }
}
