package com.github.pwittchen;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

public class Main {

    //todo 1: use all sample data in a batch processing like in the python sample
    //todo 2: preprocess  result data to choose correct index in the output like in the python example
    //todo 3: refactor code in the end

    public static void main(String[] args) {
        var env = OrtEnvironment.getEnvironment();
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        String modelFileName = "rf_iris.onnx"; // or: logreg_iris.onnx

        try (var is = classloader.getResourceAsStream(modelFileName)) {

            byte[] bytes = new byte[0];
            if (is != null) {
                bytes = is.readAllBytes();
            }

            if (bytes.length == 0) {
                System.err.println("Loaded bytes array from the onnx model is empty.");
                return;
            }

            // sample inputs from python code:
            //
            // [[5.9 3.  4.2 1.5], [6.2 3.4 5.4 2.3], [6.  2.7 5.1 1.6], [6.4 2.8 5.6 2.2],
            // [5.7 4.4 1.5 0.4], [6.1 2.8 4.7 1.2], [6.3 2.7 4.9 1.8], [7.7 3.8 6.7 2.2],
            // [6.7 3.3 5.7 2.1], [5.  3.6 1.4 0.2], [4.8 3.1 1.6 0.2], [6.2 2.9 4.3 1.3],
            // [6.7 3.  5.  1.7], [5.9 3.2 4.8 1.8], [6.1 2.6 5.6 1.4], [5.5 2.6 4.4 1.2],
            // [7.  3.2 4.7 1.4], [6.7 3.3 5.7 2.5], [7.2 3.  5.8 1.6], [5.  2.  3.5 1. ],
            // [5.7 2.8 4.1 1.3], [6.3 3.3 4.7 1.6], [5.6 2.7 4.2 1.3], [7.2 3.2 6.  1.8],
            // [5.5 4.2 1.4 0.2], [5.8 4.  1.2 0.2], [6.3 2.5 5.  1.9], [5.3 3.7 1.5 0.2],
            // [6.5 2.8 4.6 1.5], [4.4 3.  1.3 0.2], [6.8 2.8 4.8 1.4], [5.4 3.4 1.5 0.4],
            // [5.1 2.5 3.  1.1], [5.2 3.5 1.5 0.2], [4.6 3.4 1.4 0.3], [6.3 3.4 5.6 2.4],
            // [4.9 3.1 1.5 0.1], [5.4 3.9 1.7 0.4]]

            OrtSession session = env.createSession(bytes, new OrtSession.SessionOptions());
            int inputSize = 4;

            float[] inputArray = new float[inputSize];

            inputArray[0] = 5.9f;
            inputArray[1] = 3.0f;
            inputArray[2] = 4.2f;
            inputArray[3] = 1.5f;

            OnnxTensor inputTensorOne = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(inputArray),
                    new long[]{1, inputSize}
            );

            Map<String, OnnxTensor> inputs = Map.of("float_input", inputTensorOne);

            try (var results = session.run(inputs)) {
                Set<String> outputNames = session.getOutputNames();
                Optional<OnnxValue> onnxValue = results.get((String) outputNames.toArray()[1]);
                if (onnxValue.isPresent()) {
                    //noinspection rawtypes
                    Map<?, ?> result = ((OnnxMap) ((List) onnxValue.get().getValue()).get(0)).getValue();
                    System.out.println(result);
                }
            } finally {
                inputTensorOne.close();
                session.close();
            }

        } catch (OrtException | IOException e) {
            System.err.println(e.getMessage());
        }
    }
}