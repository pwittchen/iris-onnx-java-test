package com.github.pwittchen;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class Main {

    public static void main(String[] args) {

        float[][] testInputs = new float[][]{
                {5.9f, 3.0f, 4.2f, 1.5f}, {6.2f, 3.4f, 5.4f, 2.3f}, {6.0f, 2.7f, 5.1f, 1.6f}, {6.4f, 2.8f, 5.6f, 2.2f},
                {5.7f, 4.4f, 1.5f, 0.4f}, {6.1f, 2.8f, 4.7f, 1.2f}, {6.3f, 2.7f, 4.9f, 1.8f}, {7.7f, 3.8f, 6.7f, 2.2f},
                {6.7f, 3.3f, 5.7f, 2.1f}, {5.0f, 3.6f, 1.4f, 0.2f}, {4.8f, 3.1f, 1.6f, 0.2f}, {6.2f, 2.9f, 4.3f, 1.3f},
                {6.7f, 3.0f, 5.0f, 1.7f}, {5.9f, 3.2f, 4.8f, 1.8f}, {6.1f, 2.6f, 5.6f, 1.4f}, {5.5f, 2.6f, 4.4f, 1.2f},
                {7.0f, 3.2f, 4.7f, 1.4f}, {6.7f, 3.3f, 5.7f, 2.5f}, {7.2f, 3.0f, 5.8f, 1.6f}, {5.0f, 2.0f, 3.5f, 1.0f},
                {5.7f, 2.8f, 4.1f, 1.3f}, {6.3f, 3.3f, 4.7f, 1.6f}, {5.6f, 2.7f, 4.2f, 1.3f}, {7.2f, 3.2f, 6.0f, 1.8f},
                {5.5f, 4.2f, 1.4f, 0.2f}, {5.8f, 4.0f, 1.2f, 0.2f}, {6.3f, 2.5f, 5.0f, 1.9f}, {5.3f, 3.7f, 1.5f, 0.2f},
                {6.5f, 2.8f, 4.6f, 1.5f}, {4.4f, 3.0f, 1.3f, 0.2f}, {6.8f, 2.8f, 4.8f, 1.4f}, {5.4f, 3.4f, 1.5f, 0.4f},
                {5.1f, 2.5f, 3.0f, 1.1f}, {5.2f, 3.5f, 1.5f, 0.2f}, {4.6f, 3.4f, 1.4f, 0.3f}, {6.3f, 3.4f, 5.6f, 2.4f},
                {4.9f, 3.1f, 1.5f, 0.1f}, {5.4f, 3.9f, 1.7f, 0.4f}
        };


        int inputSize = 4;
        float[] inputArray = new float[inputSize];

        // run predictions using random forest model
        final String modelFileNameRf = "rf_iris.onnx";

        //noinspection DuplicatedCode
        for (float[] row : testInputs) {
            inputArray[0] = row[0];
            inputArray[1] = row[1];
            inputArray[2] = row[2];
            inputArray[3] = row[3];
            long result = predict(modelFileNameRf, inputArray);
            System.out.print(result + " ");
        }

        System.out.println();

        // run predictions using logistic regression model
        final String modelFileNameLogReg = "logreg_iris.onnx";

        //noinspection DuplicatedCode
        for (float[] row : testInputs) {
            inputArray[0] = row[0];
            inputArray[1] = row[1];
            inputArray[2] = row[2];
            inputArray[3] = row[3];
            long result = predict(modelFileNameLogReg, inputArray);
            System.out.print(result + " ");
        }
    }

    /**
     * Predict iris type basing on the input values
     *
     * @param modelFileName onnx model file name
     * @param inputArray    array including input values for prediction
     * @return classified iris result 0,1 or 2 and in the case of error returns -1
     */
    private static long predict(String modelFileName, float[] inputArray) {
        byte[] bytes;
        try {
            bytes = loadModel(modelFileName);
        } catch (IOException e) {
            System.err.println(e.getMessage());
            return -1;
        }

        var env = OrtEnvironment.getEnvironment();

        try (var session = env.createSession(bytes, new OrtSession.SessionOptions())) {


            try (var tensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(inputArray), new long[]{1, inputArray.length}
            )) {

                Map<String, OnnxTensor> inputs = Map.of(
                        String.valueOf(session.getInputNames().toArray()[0]),
                        tensor
                );

                try (var results = session.run(inputs)) {
                    Set<String> outputNames = session.getOutputNames();
                    Optional<OnnxValue> onnxValue = results.get((String) outputNames.toArray()[1]);
                    if (onnxValue.isPresent()) {
                        //noinspection rawtypes
                        Map<?, ?> result = ((OnnxMap) ((List) onnxValue.get().getValue()).get(0)).getValue();

                        //noinspection unchecked
                        Optional<Long> classifiedResult = classifyIrisResult((Map<Long, Float>) result);
                        if (classifiedResult.isPresent()) {
                            return classifiedResult.get();
                        }
                    }
                }
            }

        } catch (OrtException e) {
            System.err.println(e.getMessage());
        }
        return -1;
    }

    private static byte[] loadModel(String modelFileName) throws IOException {
        byte[] bytes = new byte[0];
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        try (var is = classloader.getResourceAsStream(modelFileName)) {
            if (is != null) {
                bytes = is.readAllBytes();
            }
        }
        return bytes;
    }

    private static Optional<Long> classifyIrisResult(Map<Long, Float> result) {
        Float max = Collections.max(result.values());
        return result
                .entrySet()
                .stream()
                .filter(entry -> entry.getValue().equals(max))
                .map(Map.Entry::getKey)
                .findFirst();
    }
}