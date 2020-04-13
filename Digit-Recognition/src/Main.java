import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException, ClassNotFoundException {
        Scanner scanner = new Scanner(System.in);

        String networkConfigPath = "network-configs.dat";
        String inputDataPath = "inputs.txt";

        int numNeurons = 10;
        int numInputs = 15;
        double learnRate = 0.5;
        int numLayers = 1;
        int[] layerSizes = new int[numLayers];
        layerSizes[0] = numNeurons;
        Loss lossType = Loss.MSE;
        ActivFunc activFuncType = ActivFunc.SOFTMAX;
        int numEpochs = 1000;

        boolean shouldExit = false;

        do {
            System.out.println("1. Learn the network");
            System.out.println("2. Guess a number");
            System.out.print("Your choice: ");
            try {
                int choice = Integer.parseInt(scanner.nextLine());
                switch(choice) {
                    case 1:
                        System.out.println("Learning...");
                        Network network = Network.createRandomGaussianNetwork(layerSizes, lossType, activFuncType, numInputs, learnRate);
                        network.train(inputDataPath, numEpochs);
                        SerializationUtils.serialize(network, networkConfigPath);
                        System.out.println("Done! Saved to the file.");
                        break;
                    case 2:
                        StringBuilder inpStr = new StringBuilder();
                        System.out.println("Input grid:");
                        while (scanner.hasNext()) {
                            inpStr.append(scanner.nextLine());
                        }
                        double[] inputs = Arrays.stream(inpStr.toString().split("")).mapToDouble(ch -> ch.equals("X") ? 1.0 : 0.0).toArray();
                        network = (Network) SerializationUtils.deserialize(networkConfigPath);
                        int ans = network.displayAns(inputs);
                        System.out.printf("This number is %d\n", ans);
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
