import java.io.*;

class SerializationUtils {

    static void serialize(Object obj, String fileName) {
        try {
            FileOutputStream fos = new FileOutputStream(fileName);
            BufferedOutputStream bos = new BufferedOutputStream(fos);
            ObjectOutputStream oos = new ObjectOutputStream(bos);
            oos.writeObject(obj);
            oos.close();
            System.out.println("Done! Saved to the file.");
        } catch (IOException e) {
            System.out.printf("Error writing to file! (%s)\n", e.getMessage());
        }
    }

    static Object deserialize(String fileName) {
        try {
            FileInputStream fis = new FileInputStream(fileName);
            BufferedInputStream bis = new BufferedInputStream(fis);
            ObjectInputStream ois = new ObjectInputStream(bis);
            Object obj = ois.readObject();
            ois.close();
            return obj;
        } catch (IOException e) {
            System.out.printf("Error in reading from file! (%s)", e.getMessage());
            return null;
        } catch (ClassNotFoundException e) {
            System.out.printf("Could not find the right class! (%s)", e.getMessage());
            return null;
        }
    }
}
