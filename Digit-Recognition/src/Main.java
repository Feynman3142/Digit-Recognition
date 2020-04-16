import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);

        String networkConfigPath;
        String inputDataPath;
        Network network = null;
        int numEpochs;

        boolean shouldExit = false;

        do {
            System.out.println("0. Train new network");
            System.out.println("1. Train existing network");
            System.out.println("2. Guess all the numbers");
            System.out.println("3. Guess number from text file");
            System.out.println("4. Save network configuration to file");
            System.out.println("5. Exit");
            System.out.print("Your choice: ");
            try {
                int choice = Integer.parseInt(scanner.nextLine());
                switch(choice) {
                    case 0:
                        double learnRate;
                        int[] layerSizes;
                        Loss lossType;
                        System.out.print("Enter the number of inputs to network: ");
                        int numInputs = Integer.parseInt(scanner.nextLine());
                        System.out.print("Enter the sizes of the layers: ");
                        layerSizes = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
                        System.out.print("Enter the learning rate: ");
                        learnRate = Double.parseDouble(scanner.nextLine());
                        System.out.printf("Enter the loss function %s:\n", Arrays.toString(Loss.values()));
                        switch (scanner.nextLine().toLowerCase()) {
                            case "mse":
                                lossType = Loss.MSE;
                                break;
                            case "cross_entropy":
                                lossType = Loss.CROSS_ENTROPY;
                                break;

                            default:
                                System.out.println("Loss function not supported/existent. Default CROSS_ENTROPY chosen");
                                lossType = Loss.CROSS_ENTROPY;
                                break;
                        }
                        ActivFunc[] activFuncType = new ActivFunc[layerSizes.length];
                        for (int elem = 0; elem < layerSizes.length; ++elem) {
                            System.out.printf("Enter the activation function for layer #%d %s:\n", elem + 1, Arrays.toString(ActivFunc.values()));
                            switch (scanner.nextLine().toLowerCase()) {
                                case "sigmoid":
                                    activFuncType[elem] = ActivFunc.SIGMOID;
                                    break;
                                case "softmax":
                                    activFuncType[elem] = ActivFunc.SOFTMAX;
                                    break;
                                case "leaky_relu":
                                    activFuncType[elem] = ActivFunc.LEAKY_RELU;
                                    break;
                                case "relu":
                                    activFuncType[elem] = ActivFunc.RELU;
                                    break;
                                default:
                                    System.out.println("Activation function not supported/existent. Default SIGMOID chosen");
                                    activFuncType[elem] = ActivFunc.SIGMOID;
                                    break;
                            }
                        }
                        network = Network.createRandomGaussianNetwork(layerSizes, lossType, activFuncType, numInputs, learnRate);
                        System.out.print("Enter the path to the dataset: ");
                        inputDataPath = scanner.nextLine();
                        System.out.print("Enter the number of epochs for training: ");
                        numEpochs = Integer.parseInt(scanner.nextLine());
                        System.out.println("Enter the path to the network configuration file:");
                        networkConfigPath = scanner.nextLine();
                        System.out.println("Learning...");
                        network.train(inputDataPath, numEpochs);
                        SerializationUtils.serialize(network, networkConfigPath);
                        break;
                    case 1:
                        System.out.println("Enter the path to the network configuration file:");
                        networkConfigPath = scanner.nextLine();
                        network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        if (network != null) {
                            System.out.print("Enter the path to the dataset: ");
                            inputDataPath = scanner.nextLine();
                            System.out.print("Enter the number of epochs for training: ");
                            numEpochs = Integer.parseInt(scanner.nextLine());
                            System.out.println("Learning...");
                            network.train(inputDataPath, numEpochs);
                            SerializationUtils.serialize(network, networkConfigPath);
                        }
                        break;
                    case 2:
                    case 3:
                        System.out.println("Enter the path to the dataset / training example: ");
                        inputDataPath = scanner.nextLine();
                        System.out.println("Enter the path to the network configuration file:");
                        networkConfigPath = scanner.nextLine();
                        network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        if (network != null) {
                            network.test(inputDataPath);
                        }
                        break;
                    case 4:
                        System.out.println("Enter the path to the network configuration file:");
                        networkConfigPath = scanner.nextLine();
                        if (network == null) {
                            System.out.println("No existing network running!");
                        } else {
                            SerializationUtils.serialize(network, networkConfigPath);
                        }
                        break;
                    case 5:
                        shouldExit = true;
                        break;
                    default:
                        System.out.println("Invalid choice! Try again");
                }
            } catch (NumberFormatException e) {
                System.out.println("Please enter a number 1 or 2");
            }
        } while (!shouldExit);
    }
}
