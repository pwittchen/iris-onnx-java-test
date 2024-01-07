# iris-onnx-java-test

verifying prediction in java using model generated in python

## requirements

Java 17

## building

to build project, execute:

```
./gradlew clean build
```

to build fat jar, execute:

```
./gradlew createFatJar
```

## running

to run compiled fat jar app, execute:

```
java -jar build/libs/iris-java-1.0-SNAPSHOT.jar
```

## references
- https://github.com/getaicube/iris-onnx-python-test
- https://onnxruntime.ai/docs/get-started/with-java.html
- https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/java/src/test/java/ai/onnxruntime/InferenceTest.java#L66
