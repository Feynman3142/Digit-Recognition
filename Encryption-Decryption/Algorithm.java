import java.io.*;

abstract class Algorithm {

    private void encryptFile(BufferedReader reader, PrintWriter writer) throws IOException {
        while (reader.ready()) {
            String msg = reader.readLine();
            System.out.println(msg);
            String encMsg = encryptMsg(msg);
            System.out.println(encMsg);
            writer.println(encMsg);
        }
    }

    void encryptFile(String path) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            PrintWriter writer = new PrintWriter(System.out);
            encryptFile(reader, writer);
        } catch (FileNotFoundException e) {
            System.out.println("Error: Input file not found");
        }
    }

    void encryptFile(String inPath, String outPath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(inPath))) {
            try (PrintWriter writer = new PrintWriter(outPath)) {
                encryptFile(reader, writer);
            } catch (FileNotFoundException e) {
                System.out.println("Error: Output file not found");
            }
        } catch (FileNotFoundException e) {
            System.out.println("Error: Input file not found");
        }
    }

    private void decryptFile(BufferedReader reader, PrintWriter writer) throws IOException {
        while (reader.ready()) {
            String msg = reader.readLine();
            String decMsg = decryptMsg(msg);
            writer.println(decMsg);
        }
    }

    void decryptFile(String path) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            PrintWriter writer = new PrintWriter(System.out);
            decryptFile(reader, writer);
        } catch (FileNotFoundException e) {
            System.out.println("Error: Input file not found");
        }
    }

    void decryptFile(String inPath, String outPath) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(inPath))) {
            try (PrintWriter writer = new PrintWriter(outPath)) {
                decryptFile(reader, writer);
            } catch (FileNotFoundException e) {
                System.out.println("Error: Output file not found");
            }
        } catch (FileNotFoundException e) {
            System.out.println("Error: Input file not found");
        }
    }

    public abstract String encryptMsg(String msg);
    public abstract String decryptMsg(String msg);
}

class Config {

    private int key;

    void setKey(int key) {
        if (key < 0) {
            throw new IllegalArgumentException(String.format("Error: key (%d) < 0\n", key));
        } else {
            this.key = key;
        }
    }

    int getKey() {
        return this.key;
    }
}

class SysFactory {

    static Algorithm createSystem(String sys, Config cfg) {
        switch (sys.toLowerCase()) {
            case "shift":
                return new CaesarCipher(cfg.getKey());
            case "unicode":
                return new UnicodeCipher(cfg.getKey());
            default:
                throw new IllegalArgumentException(String.format("Invalid algorithm %s\n", sys));
        }
    }
}

class CaesarCipher extends Algorithm {

    private int key;

    CaesarCipher(int key) {
        this.key = key;
    }

    @Override
    public String encryptMsg(String msg) {
        if (msg.length() == 0) {
            return msg;
        } else {
            char[] encMsg = new char[msg.length()];
            for (int ind = 0; ind < msg.length(); ++ind) {
                char letter = msg.charAt(ind);
                if (letter >= 'a' && letter <= 'z') {
                    encMsg[ind] = (char) ('a' + ((letter - 'a' + key) % 26));
                } else if (letter >= 'A' && letter <= 'Z') {
                    encMsg[ind] = (char) ('A' + ((letter - 'A' + key) % 26));
                } else {
                    encMsg[ind] = letter;
                }
            }
            return new String(encMsg);
        }
    }

    @Override
    public String decryptMsg(String msg) {
        if (msg.length() == 0) {
            return msg;
        } else {
            char[] decMsg = new char[msg.length()];
            for (int ind = 0; ind < msg.length(); ++ind) {
                char letter = msg.charAt(ind);
                if (letter >= 'a' && letter <= 'z') {
                    int temp = letter - 'a' - key;
                    decMsg[ind] = (char) ('a' + (char) ((temp < 0) ? temp + 26 : temp));
                } else if (letter >= 'A' && letter <= 'Z') {
                    int temp = letter - 'A' - key;
                    decMsg[ind] = (char) ('A' + (char) ((temp < 0) ? temp + 26 : temp));
                } else {
                    decMsg[ind] = letter;
                }
            }
            return new String(decMsg);
        }
    }
}

class UnicodeCipher extends Algorithm {

    private int key;

    UnicodeCipher(int key) {
        this.key = key;
    }

    @Override
    public String encryptMsg(String msg) {
        if (msg.length() == 0) {
            return msg;
        } else {
            char[] encMsg = new char[msg.length()];
            for (int ind = 0; ind < msg.length(); ++ind) {
                char letter = msg.charAt(ind);
                encMsg[ind] = (char) ((letter + key) % 65536);
            }
            return new String(encMsg);
        }
    }

    @Override
    public String decryptMsg(String msg) {
        if (msg.length() == 0) {
            return msg;
        } else {
            char[] decMsg = new char[msg.length()];
            int actKey = key % 65536;
            for (int ind = 0; ind < msg.length(); ++ind) {
                char letter = msg.charAt(ind);
                decMsg[ind] = (char) ((letter - actKey < 0) ? letter - actKey + 65536 : letter - actKey);
            }
            return new String(decMsg);
        }
    }
}
