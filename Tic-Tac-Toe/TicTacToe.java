import java.io.*

enum State {
    WIN,
    DRAW,
    UNFINISHED//,
    //IMPOSSIBLE
}

class Status {

    State state;
    char player;

    Status(State state, char player) {
        this.state = state;
        this.player = player;
    }
}

class Track {
    char lastSymbol;
    int count;

    Track(char lastSymbol, int count) {
        this.lastSymbol = lastSymbol;
        this.count = count;
    }
}

public class TicTacToe {

    private char[][] field;
    private int size;
    private char[] symbols;
    private int turn;
    private int numToWin;
    private Status status;
    //private Map<Character, Integer> playersMap;
    private Map<Integer, Track> rowTrackMap;
    private Map<Integer, Track> colTrackMap;
    private Map<Integer, Track> leftRightDiagTrackMap;
    private Map<Integer, Track> rightLeftDiagTrackMap;

    private static char[][] createField(int size) {
        char [][] field = new char[size][size];
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                field[i][j] = '_';
            }
        }
        return field;
    }

    private static char[] createSymbols(int numPlayers) {
        if (numPlayers == 2) {
            return new char[]{'X', 'O'};
        } else {
            char symbol = '0';
            char[] symbols = new char[numPlayers];
            for (int i = 0; i < numPlayers; ++i) {
                symbols[i] = symbol;
                ++symbol;
            }
            return symbols;
        }
    }

    private TicTacToe(char[][] field, int size, int numPlayers, int numToWin) {
        this.field = field;
        this.size = size;
        this.numToWin = numToWin;
        turn = 0;
        symbols = createSymbols(numPlayers);
        status = new Status(State.UNFINISHED, '_');
        //playersMap = new LinkedHashMap<>();
        rowTrackMap = new LinkedHashMap<>();
        colTrackMap = new LinkedHashMap<>();
        leftRightDiagTrackMap = new LinkedHashMap<>();
        rightLeftDiagTrackMap = new LinkedHashMap<>();
    }

    private static void checkParams(int size, int numPlayers, int numToWin) {
        if (size < 2) {
            throw new IllegalArgumentException(String.format("Cannot have a grid of size (given: %dx%d) less than 2x2!\n", size, size));
        } else if (numPlayers < 2 || numPlayers > 10) {
            throw new IllegalArgumentException(String.format("Cannot have less than 2 or more than 10 (given: %d) players!\n", numPlayers));
        } else if (numToWin < 2) {
            throw new IllegalArgumentException(String.format("Cannot have less than 2 (given: %d) consecutive symbols to win!\n", numToWin));
        } else if (numPlayers > size) {
            throw new IllegalArgumentException(String.format("Cannot have %d players for a %dx%d field!\n", numPlayers, size, size));
        } else if (numToWin > size) {
            throw new IllegalArgumentException(String.format("Cannot have %d consecutive symbols to win in a %dx%d grid!\n", numToWin, size, size));
        }
    }

    /*static TicTacToe createGame(char[][] field, int size, int numPlayers, int numToWin) {

        checkParams(size, numPlayers, numToWin);

        int length = field.length;
        if (length != size) {
            throw new IllegalArgumentException(String.format("Field given is not equal to dimensions given (%dx%d)", size, size));
        } else {
            for (char[] row : field) {
                if (row.length != size) {
                    throw new IllegalArgumentException(String.format("Field given is not equal to dimensions given (%dx%d)", size, size));
                }
            }
        }

        return new TicTacToe(field, size, numPlayers, numToWin);
    }*/

    static TicTacToe createGame(int size, int numPlayers, int numToWin) {

        checkParams(size, numPlayers, numToWin);
        return new TicTacToe(createField(size), size, numPlayers, numToWin);
    }

    Status nextTurn(Scanner scanner) {
        boolean isAlright = false;
        do {
            System.out.print("Enter the coordinates: ");
            try {
                int[] coords = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
                int x = coords[0];
                int y = coords[1];
                if (!(x >= 1 && x <= size && y >= 1 && y <= size)){
                    System.out.printf("Coordinates should be from 1 to %d\n", size);
                } else if (field[size - y][x - 1] != '_') {
                    System.out.println("This cell is occupied! Choose another one!");
                } else {
                    field[size - y][x - 1] = symbols[turn];
                    ++turn;
                    if (turn == symbols.length) {
                        turn = 0;
                    }
                    displayField();
                    isAlright = true;
                }
            } catch (NumberFormatException e) {
                System.out.println("You should enter numbers!");
            }
        } while (!isAlright);
        return displayStatus();
    }

    void displayField() {
        System.out.println("---------");
        for (int i = 0; i < size; ++i) {
            System.out.print("| ");
            for (int j = 0; j < size; ++j) {
                System.out.printf("%c ", field[i][j]);
            }
            System.out.println("|");
        }
        System.out.println("---------");
    }

    private static void fillTrackMap(Map<Integer, Track> trackMap, int lowLim, int highLim) {
        for (int i = lowLim; i <= highLim; ++i) {
            trackMap.put(i, new Track('_', 0));
        }
    }

    private boolean updateTrackMap(Map<Integer, Track> trackMap, char symbol, int key) {
        boolean isWin = false;
        Track track = trackMap.get(key);
        if (symbol == track.lastSymbol) {
            ++track.count;
            if (track.count == numToWin) {
                isWin = true;
            }
        } else {
            track.lastSymbol = symbol;
            track.count = 1;
        }
        return isWin;
    }

    /*private boolean checkImpossible() {
        boolean isImpossible = false;
        List<Character> playerList = new ArrayList<>(playersMap.keySet());
        for (int player = 0; player < playerList.size() - 1 && !isImpossible; ++player) {
            for (int otherPlayer = player + 1; otherPlayer < playerList.size(); ++otherPlayer) {
                int playerCount = playersMap.get(playerList.get(player));
                int otherPlayerCount = playersMap.get(playerList.get(otherPlayer));
                if (Math.abs(playerCount - otherPlayerCount) > 1) {
                    isImpossible = true;
                    break;
                }
            }
        }
        return isImpossible;
    }*/

    private Status displayStatus() {

        boolean isFinished = true;
        boolean isWon = false;
        //boolean wasWon = false;
        char player = '_';

        fillTrackMap(rowTrackMap, 0, size - 1);
        fillTrackMap(colTrackMap, 0, size - 1);
        fillTrackMap(leftRightDiagTrackMap, -(size - 1), size - 1);
        fillTrackMap(rightLeftDiagTrackMap, 0, 2 * (size - 1));

        //playersMap.clear();

        for (int i = 0; i < size && !isWon; ++i) {
            for (int j = 0; j < size && !isWon; ++j) {
                char symbol = field[i][j];
                if (symbol == '_') {
                    isFinished = false;
                } else {
                    //System.out.println(symbol);
                    //playersMap.put(symbol, playersMap.getOrDefault(symbol, 0) + 1);
                    boolean rowWin = updateTrackMap(rowTrackMap, symbol, i);
                    boolean colWin = updateTrackMap(colTrackMap, symbol, j);
                    boolean lrdWin = updateTrackMap(leftRightDiagTrackMap, symbol, j - i);
                    boolean rldWin = updateTrackMap(rightLeftDiagTrackMap, symbol, j + i);
                    isWon = rowWin || colWin || lrdWin || rldWin;
                    if (isWon) {
                        player = symbol;
                        isFinished = true;
                        /*if (!wasWon) {
                            wasWon = true;
                            isWon = false;
                        }*/
                    }
                }
            }
        }

        //boolean isDraw = isFinished && !wasWon;
        boolean isDraw = isFinished && !isWon;
        //boolean isImpossible = checkImpossible() || (wasWon && isWon);

        /*for (Character ch : playersMap.keySet()) {
            int count = playersMap.get(ch);
            System.out.printf("%c: %d\n", ch, count);
        }*/

        /*if (isImpossible) {
            status.state = State.IMPOSSIBLE;
        } else if (wasWon) {
            status.state = State.WIN;
            status.player = player;
        } else if (isDraw) {
            status.state = State.DRAW;
        } else {
            status.state = State.UNFINISHED;
        }*/

        if (isWon) {
            status.state = State.WIN;
            status.player = player;
        } else if (isDraw) {
            status.state = State.DRAW;
        } else {
            status.state = State.UNFINISHED;
        }

        return status;
    }
}
