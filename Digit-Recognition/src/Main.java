import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {
        Scanner scanner = new Scanner(System.in);

        String networkConfigPath = null;
        String inputDataPath = null;
        Network network = null;

        boolean shouldExit = false;
        boolean networkSaved = false;

        menu:
        do {
            System.out.println("1. Create new network");
            System.out.println("2. Load existing network");
            System.out.println("3. Train network");
            System.out.println("4. Test network");
            System.out.println("5. Set network training dataset");
            System.out.println("6. Save network configuration");
            System.out.println("7. Display network parameters");
            System.out.println("8. Exit");
            System.out.print("Your choice: ");
            try {
                int choice = Integer.parseInt(scanner.nextLine());
                switch(choice) {
                    case 1:
                        double learnRate;
                        int[] layerSizes;
                        Loss lossType;
                        System.out.print("Enter the number of inputs to network: ");
                        int numInputs = Integer.parseInt(scanner.nextLine());
                        System.out.print("Enter the sizes of the layers separated by spaces: ");
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
                                System.out.println("Loss function not supported/existent. Exiting network creation.");
                                continue menu;
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
                                    System.out.println("Activation function not supported/existent. Exiting network creation.");
                                    continue menu;
                            }
                        }

                        System.out.print("Enter the path to the training dataset: ");
                        String inputDataFile = scanner.nextLine();

                        System.out.print("Enter the scaling methods for the dataset\n0. No scaling method\n1. Normalize\n2. Centre\n3. Standardize\nEnter your choice: ");
                        int scaleMethodChoice = Integer.parseInt(scanner.nextLine());
                        double scaleFactor = 1.0;
                        if (scaleMethodChoice == 1) {
                            System.out.println("Enter the scale factor for normalization");
                            scaleFactor = Double.parseDouble(scanner.nextLine());
                        } else if (scaleMethodChoice > 3 || scaleMethodChoice < 0) {
                            System.out.println("Invalid choice. Exiting network creation.");
                            continue menu;
                        }

                        network = Network.createNetwork(layerSizes, lossType, activFuncType, numInputs, learnRate, inputDataFile, scaleMethodChoice, scaleFactor);
                        if (network == null) {
                            System.out.println("Exiting network creation.");
                        } else {
                            networkConfigPath = null;
                            networkSaved = false;
                            System.out.println("Network successfully created!");
                            System.out.println(network);
                        }
                        break;
                    case 2:
                        System.out.println("Enter the path to the network configuration file:");
                        networkConfigPath = scanner.nextLine();
                        network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        if (network != null) {
                            System.out.println("Successfully loaded network configurations!");
                            networkSaved = true;
                        }
                        break;
                    case 3:
                        if (network != null) {
                            System.out.print("Enter the number of epochs for training: ");
                            int numEpochs = Integer.parseInt(scanner.nextLine());
                            System.out.println("Learning...");
                            network.train(numEpochs);
                            networkSaved = false;
                        } else {
                            System.out.println("No network loaded!");
                        }
                        break;
                    case 4:
                        if (network != null) {
                            System.out.println("Enter the path to the testing dataset / training example: ");
                            inputDataPath = scanner.nextLine();
                            network.test(inputDataPath);
                        } else {
                            System.out.println("No network loaded!");
                        }
                        break;
                    case 5:
                        if (network != null) {
                            System.out.println("Enter the path to the training dataset:");
                            inputDataPath = scanner.nextLine();
                            network.setDatasetFile(new File(inputDataPath));
                            networkSaved = false;
                        } else {
                            System.out.println("No network loaded!");
                        }
                        break;
                    case 6:
                        if (network != null) {
                            System.out.println("Enter the path to the network configuration file:");
                            networkConfigPath = scanner.nextLine();
                            networkSaved = SerializationUtils.serialize(network, networkConfigPath);
                        } else {
                            System.out.println("No network loaded!");
                        }
                        break;
                    case 7:
                        if (network != null) {
                            System.out.println(network);
                        } else {
                            System.out.println("No network loaded!");
                        }
                        break;
                    case 8:
                        if (network != null && !networkSaved) {
                            System.out.println("Network configurations not saved! Last warning issued!");
                            networkSaved = true;
                        } else {
                            shouldExit = true;
                        }
                        break;
                    default:
                        System.out.println("Invalid choice! Try again");
                        break;
                }
            } catch (NumberFormatException e) {
                System.out.printf("Could not parse number. Returning to menu!\n[DETAILS: (%s)]", e.toString());
            }
        } while (!shouldExit);
    }
}
