class Neuron {
    private int[] weights;
    private int bias;

    Neuron(int[] weights, int bias) {
        this.weights = weights;
        this.bias = bias;
    }

    void setWeights(int[] weights) {
        this.weights = weights;
    }

    void setBias(int bias) {
        this.bias = bias;
    }

    int feedForward(int[] inputs) {
        if (inputs.length != weights.length) {
            throw new IllegalArgumentException(String.format("# of inputs to neuron (%d) is not equal to # of weights of neuron (%d)\n", inputs.length, weights.length));
        } else {
            int res = 0;
            for (int ind = 0; ind < inputs.length; ++ind) {
                res += inputs[ind] * weights[ind];
            }
            res += bias;
            return res;
        }
    }
}

class ZeroOneNetwork {

    Neuron neuron;

    ZeroOneNetwork() {
        neuron = new Neuron(new int[]{2, 1, 2, 4, -4, 4, 2, -1, 2}, -5);
    }

    int displayAns(int[] inputs) {
        int res = neuron.feedForward(inputs);
        int ans = (res > 0) ? 0 : 1;
        return ans;
    }
}

class DigitNetwork3x5 {

    Neuron[] layer;

    DigitNetwork3x5() {
        layer = new Neuron[10];
        layer[0] = new Neuron(new int[]{1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, 1}, -1);
        layer[1] = new Neuron(new int[]{-1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1}, 6);
        layer[2] = new Neuron(new int[]{1, 1, 1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1}, 1);
        layer[3] = new Neuron(new int[]{1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1}, 0);
        layer[4] = new Neuron(new int[]{1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1}, 2);
        layer[5] = new Neuron(new int[]{1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1}, 0);
        layer[6] = new Neuron(new int[]{1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1}, -1);
        layer[7] = new Neuron(new int[]{1, 1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1}, 3);
        layer[8] = new Neuron(new int[]{1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1}, -2);
        layer[9] = new Neuron(new int[]{1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1}, -1);
    }

    int displayAns(int[] inputs) {
        int[] res = new int[10];
        int maxNeuron = 0;
        for (int neuron = 0; neuron < 10; ++neuron) {
            res[neuron] = layer[neuron].feedForward(inputs);
            if (res[neuron] > res[maxNeuron]) {
                maxNeuron = neuron;
            }
        }
        return maxNeuron;
    }
}
