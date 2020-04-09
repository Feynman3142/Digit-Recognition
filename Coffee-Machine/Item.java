class Item {

    String name;
    int cost;
    Map<String, Part> parts;

    Item(String name, int cost, Map<String, Part> parts) {
        this.name = name;
        this.cost = cost;
        this.parts = parts;
    }
}