import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Scanner;

public class CoffeeMachine {

    private Map<String, Part> parts;
    private Item[] items;
    private int amount;

    private CoffeeMachine(Map<String, Part> parts, Item[] items, int amount) {
        this.parts = parts;
        this.items = items;
        this.amount = amount;
    }

    private void display() {
        System.out.println("\nThe coffee machine has:");
        for (String part : parts.keySet()) {
            System.out.printf("%d of %s\n", parts.get(part).quantity, part);
        }
        System.out.printf("$%d of money\n", amount);
    }

    private void fill(Scanner scanner) {

        for (String prt : parts.keySet()) {
            boolean isValid = false;
            Part part = parts.get(prt);
            System.out.printf("Write how many %s%s do you want to add:\n", ("".equals(part.unit)) ? "" : part.unit + " of ", part.name);
            do {
                try {
                    int quantity = Integer.parseInt(scanner.nextLine());
                    if (quantity >= 0) {
                        part.quantity += quantity;
                        parts.put(part.name, part);
                        isValid = true;
                    } else {
                        System.out.println("Please enter a whole number!");
                    }
                } catch (NumberFormatException e) {
                    System.out.println("Please enter a whole number!");
                }
            } while (!isValid);
        }
    }

    private void buy(Scanner scanner) {

        System.out.print("\nWhat do you want to buy? ");
        for (int item = 0; item < items.length; ++item) {
            System.out.printf("%d - %s, ", item + 1, items[item].name);
        }
        System.out.println("back - to main menu");

        boolean isValid = false;
        do {
            try {
                String choiceStr = scanner.nextLine();
                if ("back".equals(choiceStr)) {
                    return;
                }
                int choice = Integer.parseInt(choiceStr);
                if (choice >= 1 && choice <= items.length) {

                    Map<String, Part> itemParts = items[choice - 1].parts;

                    for (String itemPartName : itemParts.keySet()) {
                        Part machinePart = this.parts.get(itemPartName);
                        Part itemPart = itemParts.get(itemPartName);
                        if (machinePart.quantity < itemPart.quantity) {
                            System.out.printf("Sorry, not enough %s!\n", itemPartName);
                            return;
                        }
                    }
                    System.out.println("I have enough resources, making you a coffee!");
                    for (String itemPartName : itemParts.keySet()) {
                        Part machinePart = this.parts.get(itemPartName);
                        Part itemPart = itemParts.get(itemPartName);
                        machinePart.quantity -= itemPart.quantity;
                        this.parts.put(itemPartName, machinePart);
                    }

                    this.amount += items[choice - 1].cost;
                    isValid = true;

                } else {
                    System.out.println("Please enter a valid number");
                }
            } catch (NumberFormatException e) {
                System.out.println("Please enter a valid number");
            }
        } while (!isValid);
    }

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        Item item1 = new Item("espresso", 4, new HashMap<>(Map.of("water", new Part("water", 250, "ml"), "coffee beans", new Part("coffee beans", 16, "grams"), "disposable cups", new Part("disposable cups", 1, ""))));
        Item item2 = new Item("latte", 7, new HashMap<>(Map.of("water", new Part("water", 350, "ml"), "coffee beans", new Part("coffee beans", 20, "grams"), "milk", new Part("milk", 75, "ml"), "disposable cups", new Part("disposable cups", 1, ""))));
        Item item3 = new Item("cappuccino", 6, new HashMap<>(Map.of("water", new Part("water", 200, "ml"), "coffee beans", new Part("coffee beans", 12, "grams"), "milk", new Part("milk", 100, "ml"), "disposable cups", new Part("disposable cups", 1, ""))));

        Map<String, Part> parts = new LinkedHashMap<>();
        parts.put("water", new Part("water", 400, "ml"));
        parts.put("milk", new Part("milk", 540, "ml"));
        parts.put("coffee beans", new Part("coffee beans", 120, "grams"));
        parts.put("disposable cups", new Part("disposable cups", 9, ""));

        CoffeeMachine machine = new CoffeeMachine(parts, new Item[]{item1, item2, item3}, 550);

        boolean shouldExit = false;
        do {
            System.out.println("\nWrite action (buy, fill, take, remaining, exit):");

            String choice = scanner.nextLine();

            switch(choice.toLowerCase()) {
                case "buy":
                    machine.buy(scanner);
                    break;
                case "fill":
                    System.out.println();
                    machine.fill(scanner);
                    break;
                case "take":
                    System.out.printf("\nI gave you $%d\n", machine.amount);
                    machine.amount = 0;
                    break;
                case "remaining":
                    machine.display();
                    break;
                case "exit":
                    shouldExit = true;
                    break;
                default:
                    System.out.println("Invalid action! Try again");
                    break;
            }
        } while (!shouldExit);
    }
}
