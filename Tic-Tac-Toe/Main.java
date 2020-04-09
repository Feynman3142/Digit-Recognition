import java.util.*;

public class Main {
    public static void main(String[] args) {
        // write your code here
        Scanner scanner = new Scanner(System.in);
        //System.out.print("Enter cells: ");
        //char[] valArr = scanner.nextLine().toCharArray();
        //char[][] field = new char[3][3];
        int size = 3;
        int numPlayers = 2;
        int numToWin = 3;
        /*int count = 0;
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                field[i][j] = valArr[count];
                ++count;
            }
        }*/
        TicTacToe game = TicTacToe.createGame(size, numPlayers, numToWin);
        game.displayField();
        Status status;
        do {
            status = game.nextTurn(scanner);
        } while (status.state == State.UNFINISHED);

        switch (status.state) {
            case WIN:
                System.out.printf("%c wins\n", status.player);
                break;
            case DRAW:
                System.out.println("Draw");
                break;
            /*case IMPOSSIBLE:
                System.out.println("Impossible");
                break;*/
            default:
                System.out.println("Error in determining state");
                break;
        }
    }

}