class Matrix {

    static double[][] makeMatrix(double[] vect, int m, int n) {
        if ((m != 1 && n != 1) || vect.length != Math.max(m, n)) {
            throw new IllegalArgumentException(String.format("Cannot make a matrix (%dx%d) of vector (size:%d) with the following parameters\n", m, n, vect.length));
        }
        double[][] res = new double[m][n];
        int elem = 0;
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                res[row][col] = vect[elem];
                ++elem;
            }
        }
        return res;
    }

    static double[] makeVector(double[][] mat, int m, int n) {
        if (m != 1 && n != 1) {
            throw new IllegalArgumentException(String.format("Cannot make a vector from matrix (%dx%d)\n", m, n));
        }
        double[] res = new double[Math.max(m, n)];
        int elem = 0;
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                res[elem] = mat[row][col];
                ++elem;
            }
        }
        return res;
    }

    static double[][] multiply(double[] vect, double[][] mat, int m, int n) {
        try {
            double[][] matFromVect = makeMatrix(vect, 1, m);
            return multiply(matFromVect, 1, m, mat, m, n);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(String.format("Cannot multiply matrices MAT1(1x%d) X MAT2(%dx%d)\n", vect.length, m, n));
        }
    }

    static double[][] multiply(double[][] mat, int m, int n, double[] vect) {
        try {
            double[][] matFromVect = makeMatrix(vect, n, 1);
            return multiply(mat, m, n, matFromVect, n, 1);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(String.format("Cannot multiply matrices MAT1(%dx%d) X MAT2(%dx1)\n", m, n, vect.length));
        }
    }

    static double[][] multiply(double[] vect1, double[] vect2) {
        double[][] mat1 = makeMatrix(vect1, vect1.length, 1);
        double[][] mat2 = makeMatrix(vect2, 1, vect2.length);
        return  multiply(mat1, vect1.length, 1, mat2, 1, vect2.length);
    }

    static double dotProduct(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException(String.format("Cannot perform dot product on vectors with different sizes! (%d, %d)\n", vect1.length, vect2.length));
        }
        double res = 0.0;
        for (int elem = 0; elem < vect1.length; ++elem) {
            res += vect1[elem] * vect2[elem];
        }
        return res;
    }

    static double[][] multiply(double[][] mat1, int m1, int n1, double[][] mat2, int m2, int n2) {
        if (n1 != m2) {
            throw new IllegalArgumentException(String.format("Cannot multiply matrices MAT1(%dx%d) X MAT2(%dx%d)\n", m1, n1, m2, n2));
        }
        double[][] res = new double[m1][n2];
        for (int row = 0; row < m1; ++row) {
            for (int col = 0; col < n2; ++col) {
                res[row][col] = multiplyForElem(mat1, row, mat2, col, n1);
            }
        }
        return res;
    }

    private static double multiplyForElem(double[][] mat1, int row, double[][] mat2, int col, int num) {
        double res = 0.0;
        for (int elem = 0; elem < num; ++elem) {
            res += mat1[row][elem] * mat2[elem][col];
        }
        return res;
    }

    static double[][] multiplyElemWise(double[][] mat1, double[][] mat2, int m, int n) {
        double[][] res = new double[m][n];
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                res[row][col] = mat1[row][col] * mat2[row][col];
            }
        }
        return res;
    }

    static double[] multiplyElemWise(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException(String.format("Cannot perform element-wise multiplication for unequal vectors! (%d, %d)", vect1.length, vect2.length));
        }
        double[] res = new double[vect1.length];
        for (int elem = 0; elem < vect1.length; ++elem) {
            res[elem] = vect1[elem] * vect2[elem];
        }
        return res;
    }

    static double[][] transpose(double[][] mat, int m, int n) {
        double[][] res = new double[n][m];
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < m; ++col) {
                res[row][col] = mat[col][row];
            }
        }
        return res;
    }

    static double[] transpose(double[] vect) {
        return vect.clone();
    }

    static double[][] multiplyScalar(double[][] mat, int m , int n, double scalar) {
        double[][] res = new double[m][n];
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                res[row][col] = scalar * mat[row][col];
            }
        }
        return res;
    }

    static double[] multiplyScalar(double[] vect, double scalar){
        double[] res = new double[vect.length];
        for (int elem = 0; elem < vect.length; ++elem) {
            res[elem] = scalar * vect[elem];
        }
        return res;
    }

    static double[][] add(double[][] mat1, double[][] mat2, int m, int n) {
        double[][] res = new double[m][n];
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                res[row][col] = mat1[row][col] + mat2[row][col];
            }
        }
        return res;
    }

    static double[] add(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException(String.format("Cannot add vectors of different sizes! (%d, %d)", vect1.length, vect2.length));
        }
        double[] res = new double[vect1.length];
        for (int elem = 0; elem < vect1.length; ++elem) {
            res[elem] = vect1[elem] + vect2[elem];
        }
        return res;
    }

    static double[] add(double[] vect, double[][] mat) {
        double[] vectFromMat = makeVector(mat, mat.length, mat[0].length);
        return add(vect, vectFromMat);
    }

    static double[] add(double[][] mat, double[] vect) {
        double[] vectFromMat = makeVector(mat, mat.length, mat[0].length);
        return add(vect, vectFromMat);
    }

    static double[][] subtract(double[][] mat1, double[][] mat2, int m, int n) {
        double[][] res = new double[m][n];
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                res[row][col] = mat1[row][col] - mat2[row][col];
            }
        }
        return res;
    }

    static double[] subtract(double[] vect1, double[] vect2) {
        if (vect1.length != vect2.length) {
            throw new IllegalArgumentException(String.format("Cannot add vectors of different sizes! (%d, %d)", vect1.length, vect2.length));
        }
        double[] res = new double[vect1.length];
        for (int elem = 0; elem < vect1.length; ++elem) {
            res[elem] = vect1[elem] - vect2[elem];
        }
        return res;
    }

    static double[] subtract(double[] vect, double[][] mat) {
        double[] vectFromMat = makeVector(mat, mat.length, mat[0].length);
        return subtract(vect, vectFromMat);
    }

    static double[] subtract(double[][] mat, double[] vect) {
        double[] vectFromMat = makeVector(mat, mat.length, mat[0].length);
        return subtract(vect, vectFromMat);
    }
}