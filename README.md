# iris-onnx-java-test

verifying prediction in java using IRIS model generated in python

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

## IRIS dataset

### Features

The Iris dataset contains 150 instances, each with 4 features:
- sepal length
- sepal width
- petal length
- petal width. 

### Ouput

The Iris dataset contains three different species of Iris flowers:
- Iris setosa: Known for its relatively small and straight petals. It's usually the easiest to distinguish among the three species due to its unique flower characteristics.
- Iris versicolor: Often referred to as the blue flag, it's intermediate in terms of petal and sepal size between Iris setosa and Iris virginica.
- Iris virginica: Known as the Virginia iris, it typically has the largest petals and sepals among the three species.

### Scoring & Analysis

Once we have trained model, we can perform scoring with given 4 features and we receive an array with 3 items with probability of the given species from 0 to 1. Now, we can choose the highest value and that's the index of predicted flower.

## references
- https://github.com/getaicube/iris-onnx-python-test
- https://onnxruntime.ai/docs/get-started/with-java.html
- https://github.com/microsoft/onnxruntime-openenclave/blob/openenclave-public/java/src/test/java/ai/onnxruntime/InferenceTest.java#L66
