const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const { ImageDataGenerator } = require('@tensorflow/tfjs-data');

// Paths for the dataset
const dataDir = 'path_to_dataset'; // Change this to the dataset path
const trainDir = path.join(dataDir, 'train');
const validationDir = path.join(dataDir, 'validation');
const testDir = path.join(dataDir, 'test');

// Data preprocessing using TensorFlow.js
const createDataGenerator = (dir, isTraining = false) => {
  const generator = new ImageDataGenerator({
    rescale: 1.0 / 255,
    ...(isTraining && {
      rotationRange: 20,
      widthShiftRange: 0.2,
      heightShiftRange: 0.2,
      shearRange: 0.2,
      zoomRange: 0.2,
      horizontalFlip: true,
    }),
  });
  return generator.flowFromDirectory(dir, {
    targetSize: [150, 150],
    batchSize: 32,
    classMode: 'categorical',
    shuffle: isTraining,
  });
};

const trainGenerator = createDataGenerator(trainDir, true);
const validationGenerator = createDataGenerator(validationDir);
const testGenerator = createDataGenerator(testDir);

// Building the model
const model = tf.sequential();
model.add(tf.layers.conv2d({
  inputShape: [150, 150, 3],
  filters: 32,
  kernelSize: 3,
  activation: 'relu',
}));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

model.add(tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

model.add(tf.layers.conv2d({ filters: 128, kernelSize: 3, activation: 'relu' }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
model.add(tf.layers.dropout({ rate: 0.5 }));
model.add(tf.layers.dense({ units: trainGenerator.numClasses, activation: 'softmax' }));

// Compiling the model
model.compile({
  optimizer: tf.train.adam(),
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

// Training the model
(async () => {
  const history = await model.fitDataset(trainGenerator, {
    epochs: 20,
    validationData: validationGenerator,
    callbacks: tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: 5,
      restoreBestWeights: true,
    }),
  });

  // Evaluate on the test dataset
  const [testLoss, testAccuracy] = await model.evaluateDataset(testGenerator);
  console.log(`Test Accuracy: ${(testAccuracy * 100).toFixed(2)}%`);

  // Save the model
  await model.save('file://./plant_disease_detection_model');

  // Generate classification report and confusion matrix
  const yTrue = testGenerator.labels;
  const yPred = tf.argMax(await model.predict(testGenerator).dataSync(), 1).arraySync();

  console.log('Classification Report:', classificationReport(yTrue, yPred, testGenerator.classIndices));
  console.log('Confusion Matrix:', confusionMatrix(yTrue, yPred));
})();

// Helper functions for classification report and confusion matrix
function classificationReport(yTrue, yPred, classIndices) {
  const classes = Object.keys(classIndices);
  const report = {};

  classes.forEach((cls, idx) => {
    const truePositive = yTrue.filter((val, i) => val === idx && yPred[i] === idx).length;
    const falsePositive = yPred.filter((val, i) => val === idx && yTrue[i] !== idx).length;
    const falseNegative = yTrue.filter((val, i) => val === idx && yPred[i] !== idx).length;
    const precision = truePositive / (truePositive + falsePositive);
    const recall = truePositive / (truePositive + falseNegative);

    report[cls] = { precision, recall, f1Score: 2 * ((precision * recall) / (precision + recall)) };
  });

  return report;
}

function confusionMatrix(yTrue, yPred) {
  const matrix = Array.from(new Set(yTrue)).map(() => Array.from(new Set(yPred)).fill(0));

  yTrue.forEach((val, i) => {
    matrix[val][yPred[i]] += 1;
  });

  return matrix;
}
