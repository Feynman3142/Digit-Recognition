import java.io.*

public class Main {
    public static void main(String[] args) throws IOException {
        String op = "enc";
        String input = "";
        int shift = 0;
        boolean isFile = false;
        boolean gotData = false;
        String outPath = null;
        String algo = "shift";
        for (int ind = 0; ind < args.length; ++ind) {
            if ("-mode".equals(args[ind])) {
                op = args[ind + 1].toLowerCase();
                if (!"enc".equals(op) && !"dec".equals(op)) {
                    throw new IllegalArgumentException(String.format("Invalid operation: %s", args[ind + 1]));
                }
            } else if ("-key".equals(args[ind])) {
                try {
                    shift = Integer.parseInt(args[ind + 1]);
                } catch (NumberFormatException e) {
                    throw new NumberFormatException(String.format("Invalid key value: %s", args[ind + 1]));
                }
            } else if ("-data".equals(args[ind])) {
                input = args[ind + 1];
                gotData = true;
            } else if (("-in".equals(args[ind])) && !gotData) {
                input = args[ind + 1];
                isFile = true;
            } else if ("-out".equals(args[ind])) {
                outPath = args[ind + 1];
            } else if ("-algo".equals(args[ind])) {
                algo = args[ind + 1].toLowerCase();
                if (!"shift".equals(op) && !"unicode".equals(op)) {
                    throw new IllegalArgumentException(String.format("Invalid operation: %s", args[ind + 1]));
                }
            }
        }

        System.out.println(String.format("%s %s %d %s %s %s", algo, op, shift, input, outPath, isFile));
        Config cfg = new Config();
        cfg.setKey(shift);
        Algorithm cryptoSys = SysFactory.createSystem(algo, cfg);

        boolean opIsEnc = "enc".equals(op);
        boolean outPathExists = outPath != null;

        if (!opIsEnc && !isFile && !outPathExists) {
            String decMsg = cryptoSys.decryptMsg(input);
            System.out.println(decMsg);
        } else if (!opIsEnc && !isFile && outPathExists) {
            String decMsg = cryptoSys.decryptMsg(input);
            try (PrintWriter writer = new PrintWriter(outPath)) {
                writer.print(decMsg);
            } catch (FileNotFoundException e) {
                System.out.println("Error: Output file not found");
            }
        } else if (!opIsEnc && isFile && !outPathExists) {
            cryptoSys.decryptFile(input);
        } else if (!opIsEnc && isFile && outPathExists) {
            cryptoSys.decryptFile(input, outPath);
        } else if (opIsEnc && !isFile && !outPathExists) {
            String encMsg = cryptoSys.encryptMsg(input);
            System.out.println(encMsg);
        } else if (opIsEnc && !isFile && outPathExists) {
            String encMsg = cryptoSys.encryptMsg(input);
            try (PrintWriter writer = new PrintWriter(outPath)) {
                writer.print(encMsg);
            } catch (FileNotFoundException e) {
                System.out.println("Error: Output file not found");
            }
        } else if (opIsEnc && isFile && !outPathExists) {
            cryptoSys.encryptFile(input);
        } else if (opIsEnc && isFile && outPathExists) {
            System.out.println("yay got the right choice!");
            cryptoSys.encryptFile(input, outPath);
        }
    }
}