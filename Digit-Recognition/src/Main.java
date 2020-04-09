import java.util.Arrays;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
        StringBuilder inpStr = new StringBuilder();
        Scanner scanner = new Scanner(System.in);
        while (scanner.hasNext()) {
            inpStr.append(scanner.nextLine());
        }
        int[] inputs = Arrays.stream(inpStr.toString().split("")).mapToInt(ch -> ch.equals("_") ? 0 : 1).toArray();
        DigitNetwork3x5 network = new DigitNetwork3x5();
        int ans = network.displayAns(inputs);
        System.out.printf("This number is %d\n", ans);
    }
}
