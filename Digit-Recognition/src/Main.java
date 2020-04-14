import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        Scanner scanner = new Scanner(System.in);

        String networkConfigPath = "network-configs.dat";
        String inputDataPath = "inputs.txt";

        int numInputs = 15;
        double learnRate = 0.5;
        int[] layerSizes;
        Loss lossType = Loss.CROSS_ENTROPY;
        int numEpochs = 1;
        Network network;

        boolean shouldExit = false;

        do {
            System.out.println("1. Learn the network");
            System.out.println("2. Guess a number");
            System.out.println("3. Exit");
            System.out.print("Your choice: ");
            try {
                int choice = Integer.parseInt(scanner.nextLine());
                switch(choice) {
                    case 1:
                        File file = new File(networkConfigPath);
                        if (file.exists()) {
                            network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        } else {
                            System.out.print("Enter the sizes of the layers: ");
                            layerSizes = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
                            network = Network.createRandomGaussianNetwork(layerSizes, lossType, numInputs, learnRate);
                        }
                        System.out.println("Learning...");
                        network.train(inputDataPath, numEpochs);
                        SerializationUtils.serialize(network, networkConfigPath);
                        System.out.println("Done! Saved to the file.");
                        break;
                    case 2:
                        StringBuilder inpStr = new StringBuilder();
                        System.out.println("Input grid:");
                        while (scanner.hasNextLine()) {
                            String tempStr = scanner.nextLine();
                            if ("".equals(tempStr)) {
                                break;
                            }
                            inpStr.append(tempStr);
                        }
                        double[] inputs = Arrays.stream(inpStr.toString().split("")).mapToDouble(ch -> ch.equals("X") ? 1.0 : 0.0).toArray();
                        network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        //int ans = network.displayAns(inputs);
                        double[] ans = network.displayAns(inputs);
                        int maxNeuron = 0;
                        for (int neuron = 1; neuron < ans.length; ++neuron) {
                            if (ans[neuron] > ans[maxNeuron]) {
                                maxNeuron = neuron;
                            }
                        }
                        System.out.printf("This number is %d\n", maxNeuron);
                        System.out.println();
                        for (int elem = 0; elem < ans.length; ++elem) {
                            System.out.printf("%f ", ans[elem]);
                        }
                        System.out.println();
                        break;
                    case 3:
                        shouldExit = true;
                        break;
                    case 4:
                        network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        String outPath = "network-configs.txt";
                        try (PrintWriter writer = new PrintWriter(outPath)) {
                            writer.println(network.getNumLayers());
                            writer.println(network.getLossType());
                            printArray(network.getActivFuncType(), writer);
                            Layer[] layers = network.getLayers();
                            for (int layer = 0; layer < network.getNumLayers(); ++layer) {
                                writer.println(layers[layer].getNumWeights());
                                writer.println(layers[layer].getNumNeurons());
                                writer.println(layers[layer].getLearnRate());
                                printArray(layers[layer].getB(), writer);
                                printMatrix(layers[layer].getW(), writer);
                                printMatrix(layers[layer].getWt(), writer);
                            }
                            System.out.println("Did the deed ;)");
                        } catch (FileNotFoundException e) {
                            System.out.println("Could not do the deed!");
                        }
                        break;
                    default:
                        System.out.println("Invalid choice! Try again");
                }
            } catch (NumberFormatException e) {
                System.out.println("Please enter a number 1 or 2");
            }
        } while (!shouldExit);
    }

    private static void printArray(Object[] arr, PrintWriter writer) {
        for (int elem = 0; elem < arr.length; ++elem) {
            writer.print(arr[elem]);
            if (elem != arr.length - 1) {
                writer.print(" ");
            }
        }
        writer.println();
    }

    private static void printArray(double[] arr, PrintWriter writer) {
        for (int elem = 0; elem < arr.length; ++elem) {
            writer.print(arr[elem]);
            if (elem != arr.length - 1) {
                writer.print(" ");
            }
        }
        writer.println();
    }

    private static void printMatrix(double[][] mat, PrintWriter writer) {
        for (int elem = 0; elem < mat.length; ++elem) {
            printArray(mat[elem], writer);
        }
    }
}
